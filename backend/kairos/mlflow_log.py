"""Best-effort MLflow tracking.

MLflow is a SOFT dependency (PROJECT_BRIEF.md §4 notes, and hard experience): a
tracking server that is slow, migrating or simply down must never take a run with
it. Every call here is wrapped, failures are logged once and then suppressed, and
training continues regardless.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from kairos.config import get_settings

log = logging.getLogger(__name__)

_lock = threading.Lock()
_disabled = False
_experiment_ready = False


def _mlflow():
    """Import and configure lazily; MLflow is slow to import and often unused."""
    global _experiment_ready
    import mlflow

    if not _experiment_ready:
        settings = get_settings()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment)
        _experiment_ready = True
    return mlflow


def disable(reason: str) -> None:
    global _disabled
    if not _disabled:
        log.warning("MLflow logging disabled: %s", reason)
    _disabled = True


def log_trial(run_id: str, trial: Any, attempt: int = 1) -> str | None:
    """Log one candidate as a nested-style MLflow run. Returns the run id, or None."""
    global _disabled
    if _disabled:
        return None
    try:
        with _lock:
            mlflow = _mlflow()
            with mlflow.start_run(run_name=f"{trial.candidate_key}-attempt{attempt}") as active:
                mlflow.set_tags({
                    "kairos_run_id": run_id,
                    "candidate": trial.candidate_key,
                    "family": trial.family,
                    "task_type": trial.task_type,
                    "attempt": str(attempt),
                })
                numeric = {
                    k: float(v) for k, v in (trial.metrics or {}).items()
                    if isinstance(v, (int, float))
                }
                numeric.update({
                    k: float(v) for k, v in (trial.cv_metrics or {}).items()
                    if isinstance(v, (int, float))
                })
                numeric["fit_seconds"] = float(trial.fit_seconds)
                if numeric:
                    mlflow.log_metrics(numeric)
                if trial.calibration:
                    mlflow.log_params({"calibration_method": trial.calibration.method})
                mlflow.log_params({"n_features": len(trial.feature_names)})
                return active.info.run_id
    except Exception as exc:  # noqa: BLE001
        disable(f"{type(exc).__name__}: {exc}")
        return None


def log_decision(run_id: str, payload: dict[str, Any]) -> None:
    """Record the chosen operating point and its cost alongside the trials."""
    if _disabled:
        return
    try:
        with _lock:
            mlflow = _mlflow()
            with mlflow.start_run(run_name=f"decision-{run_id[:8]}"):
                mlflow.set_tags({"kairos_run_id": run_id, "stage": "decision"})
                mlflow.log_metrics({
                    k: float(v) for k, v in payload.items()
                    if isinstance(v, (int, float))
                })
    except Exception as exc:  # noqa: BLE001
        disable(f"{type(exc).__name__}: {exc}")
