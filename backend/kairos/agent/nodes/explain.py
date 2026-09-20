"""Node 8: explain — real SHAP or nothing (PROJECT_BRIEF.md §11).

TreeExplainer only. KernelExplainer is model-agnostic and far too slow to sit in a
demo path, so a non-tree champion gets an honest "not available" rather than a
fabricated attribution.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from kairos.agent.runtime import artifacts, step
from kairos.agent.state import KairosState

log = logging.getLogger(__name__)

#: Rows sampled for the global explanation. SHAP is superlinear in rows and the
#: global ranking is stable well before the full test set.
GLOBAL_SAMPLE = 500
#: How many riskiest assets get an individual explanation for the work orders.
TOP_N_ASSETS = 10


def build_explainer(model: Any):
    """A TreeExplainer over the estimator, unwrapped from any pipeline."""
    import shap

    steps = getattr(model, "named_steps", {})
    inner = steps.get("clf") or steps.get("reg") or model
    return shap.TreeExplainer(inner)


def shap_for(explainer: Any, frame) -> np.ndarray | None:
    """SHAP values shaped (rows, features), whichever API version is installed."""
    try:
        values = explainer.shap_values(frame, check_additivity=False)
        if isinstance(values, list):          # older API returns one array per class
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:                  # (rows, features, classes)
            values = values[..., -1]
        return values
    except Exception as exc:  # noqa: BLE001
        log.warning("SHAP evaluation failed: %s", exc)
        return None


def _tree_shap(model: Any, frame) -> np.ndarray | None:
    try:
        return shap_for(build_explainer(model), frame)
    except Exception as exc:  # noqa: BLE001
        log.warning("TreeExplainer unavailable: %s", exc)
        return None


def node_explain(state: KairosState) -> dict[str, Any]:
    run_id = state["run_id"]
    store = artifacts(run_id)
    best = state.get("best") or {}
    trial = store.models.get(best.get("model_id"))

    with step(run_id, "explain", inputs=best.get("model_id")) as box:
        if trial is None or store.split is None:
            box["summary"] = "No champion model to explain."
            return {"explanations": None}

        if not trial.supports_shap:
            note = (
                f"{trial.model_name} is not a tree model, and Kairos uses TreeExplainer only "
                f"— KernelExplainer is too slow for an interactive path. No SHAP values are "
                f"shown rather than approximated ones."
            )
            box["summary"] = "SHAP unavailable for this model family."
            box["reasoning"] = note
            return {
                "explanations": {"available": False, "reason": note},
                "findings": [*state.get("findings", []), note],
            }

        features = store.feature_names
        test = store.split.test
        sample = test.sample(min(GLOBAL_SAMPLE, len(test)), random_state=0)
        values = _tree_shap(trial.model, sample[features])

        if values is None:
            note = "SHAP computation failed; explanations are unavailable for this run."
            box["summary"] = note
            return {
                "explanations": {"available": False, "reason": note},
                "findings": [*state.get("findings", []), note],
            }

        importance = np.abs(values).mean(axis=0)
        order = np.argsort(-importance)
        global_ranking = [
            {"feature": features[i], "mean_abs_shap": float(importance[i])}
            for i in order[:20]
        ]

        store.extra["shap_values"] = values
        store.extra["shap_sample"] = sample
        store.extra["shap_global"] = global_ranking
        # Kept so the scheduler can explain an individual asset that did not happen
        # to fall inside the global sample. Without this, work orders cite no
        # drivers, which is exactly the thing the brief forbids faking.
        try:
            store.extra["shap_explainer"] = build_explainer(trial.model)
        except Exception:  # noqa: BLE001
            store.extra["shap_explainer"] = None

        box["summary"] = (
            f"SHAP over {len(sample)} rows; top driver is {global_ranking[0]['feature']}."
        )
        box["reasoning"] = (
            "Top five drivers: "
            + ", ".join(f["feature"] for f in global_ranking[:5])
        )
        box["payload"] = {"global": global_ranking[:10], "n_rows": len(sample)}

        return {
            "explanations": {
                "available": True,
                "model_id": trial.model_id,
                "model_name": trial.model_name,
                "global": global_ranking,
                "n_rows_explained": int(len(sample)),
            }
        }
