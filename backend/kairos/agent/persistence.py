"""Durable run artefacts.

The in-process cache is fast but dies with the worker. That is fine during a run
and unacceptable afterwards: restart the API and every cost curve, SHAP ranking
and what-if simulation for a finished run silently becomes an empty state. It
also breaks ``make seed``, where the whole point is that a run baked earlier is
ready the moment the stack comes up.

So a completed run writes everything the Decision Layer needs to re-decide —
predictions, the split slices they index into, the champion model, and the
metadata tying them together — and the cache rehydrates from disk on first miss.

Deliberately NOT stored: every candidate's fitted model. Only the champion is
kept, because that is the only one anything downstream explains or serves, and a
random forest over 257 features is tens of megabytes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dataclasses import fields as dataclass_fields

from kairos.data.loaders import data_dir
from kairos.decision.cost import CostConfig

log = logging.getLogger(__name__)


def run_dir(run_id: str) -> Path:
    return data_dir() / "runs" / run_id


def save(run_id: str, store: Any) -> bool:
    """Persist a finished run. Returns False on failure rather than raising."""
    try:
        target = run_dir(run_id)
        target.mkdir(parents=True, exist_ok=True)
        split = store.split
        if split is None:
            return False

        split.val.to_parquet(target / "val.parquet")
        split.test.to_parquet(target / "test.parquet")

        arrays: dict[str, np.ndarray] = {}
        trials: list[dict[str, Any]] = []
        for model_id, trial in store.models.items():
            if trial.val_prediction is not None:
                arrays[f"{model_id}__val"] = np.asarray(trial.val_prediction)
            if trial.test_prediction is not None:
                arrays[f"{model_id}__test"] = np.asarray(trial.test_prediction)
            trials.append({
                "model_id": model_id,
                "candidate_key": trial.candidate_key,
                "model_name": trial.model_name,
                "family": trial.family,
                "task_type": trial.task_type,
                "metrics": trial.metrics,
                "cv_metrics": trial.cv_metrics,
                "fit_seconds": trial.fit_seconds,
                "supports_shap": trial.supports_shap,
                "calibration": trial.calibration.as_dict() if trial.calibration else None,
            })
        if arrays:
            np.savez_compressed(target / "predictions.npz", **arrays)

        champion_id = store.leaderboard.best.model_id if store.leaderboard else None
        if champion_id and champion_id in store.models:
            try:
                import joblib

                joblib.dump(store.models[champion_id].model, target / "champion.joblib")
            except Exception as exc:  # noqa: BLE001 - a missing model degrades, not fails
                log.warning("could not persist champion model: %s", exc)

        (target / "meta.json").write_text(json.dumps({
            "run_id": run_id,
            "feature_names": store.feature_names,
            "cost_config": store.cost_config.__dict__ if store.cost_config else None,
            "champion_id": champion_id,
            "trials": trials,
            "split": {
                "strategy": split.strategy,
                "group_col": split.group_col,
                "target": split.target,
            },
            "shap_global": store.extra.get("shap_global"),
            "leaderboard": store.leaderboard.as_dict() if store.leaderboard else None,
        }, indent=2, default=str))
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to persist run %s: %s", run_id, exc)
        return False


def load(run_id: str, store: Any) -> bool:
    """Rehydrate a run into an empty cache entry. Returns False if nothing on disk."""
    source = run_dir(run_id)
    meta_path = source / "meta.json"
    if not meta_path.exists():
        return False
    try:
        from kairos.data.splits import Split
        from kairos.ml.calibrate import CalibrationReport
        from kairos.ml.train import TrialResult

        meta = json.loads(meta_path.read_text())
        val = pd.read_parquet(source / "val.parquet")
        test = pd.read_parquet(source / "test.parquet")
        store.split = Split(
            train=test.iloc[0:0], val=val, test=test,
            strategy=meta["split"]["strategy"],
            group_col=meta["split"]["group_col"],
            target=meta["split"]["target"],
        )
        store.feature_names = meta["feature_names"]
        if meta.get("cost_config"):
            store.cost_config = CostConfig(**meta["cost_config"])

        arrays = (
            np.load(source / "predictions.npz")
            if (source / "predictions.npz").exists()
            else {}
        )
        champion_model = None
        if (source / "champion.joblib").exists():
            try:
                import joblib

                champion_model = joblib.load(source / "champion.joblib")
            except Exception as exc:  # noqa: BLE001
                log.warning("could not load champion model for %s: %s", run_id, exc)

        def _calibration(payload: dict[str, Any] | None) -> CalibrationReport | None:
            """as_dict() emits computed properties too; the constructor takes fields only."""
            if not payload:
                return None
            allowed = {f.name for f in dataclass_fields(CalibrationReport)}
            return CalibrationReport(**{k: v for k, v in payload.items() if k in allowed})

        for entry in meta["trials"]:
            model_id = entry["model_id"]
            calibration = entry.get("calibration")
            trial = TrialResult(
                model_id=model_id,
                candidate_key=entry["candidate_key"],
                model_name=entry["model_name"],
                task_type=entry["task_type"],
                family=entry["family"],
                metrics=entry.get("metrics") or {},
                cv_metrics=entry.get("cv_metrics") or {},
                fit_seconds=entry.get("fit_seconds", 0.0),
                feature_names=meta["feature_names"],
                supports_shap=entry.get("supports_shap", False),
                model=champion_model if model_id == meta.get("champion_id") else None,
                calibration=_calibration(calibration),
                val_prediction=arrays.get(f"{model_id}__val"),
                test_prediction=arrays.get(f"{model_id}__test"),
            )
            store.models[model_id] = trial
        store.trials = list(store.models.values())

        if meta.get("shap_global"):
            store.extra["shap_global"] = meta["shap_global"]
        if meta.get("leaderboard"):
            store.extra["leaderboard_snapshot"] = meta["leaderboard"]
        store.extra["rehydrated"] = True
        log.info("rehydrated run %s from disk (%d models)", run_id, len(store.models))
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to rehydrate run %s: %s", run_id, exc)
        return False


def exists(run_id: str) -> bool:
    return (run_dir(run_id) / "meta.json").exists()
