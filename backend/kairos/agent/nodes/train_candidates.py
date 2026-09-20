"""Node 5: train_candidates — time-budgeted loop, every trial streamed."""

from __future__ import annotations

from typing import Any

from sqlmodel import Session

from kairos.agent.runtime import artifacts, step
from kairos.agent.state import KairosState
from kairos.db.models import Trial
from kairos.db.session import engine
from kairos.events import bus
from kairos.ml.train import TrialResult, train_all
from kairos.mlflow_log import log_trial


def node_train_candidates(state: KairosState) -> dict[str, Any]:
    run_id = state["run_id"]
    plan = state.get("plan") or {}
    store = artifacts(run_id)
    split, features = store.split, store.feature_names
    if split is None:
        raise ValueError("no split prepared; engineer_features must run first")

    attempt = state.get("replan_count", 0) + 1
    target = plan.get("target") or split.target
    task_type = plan.get("task_type") or state.get("task_type")
    budget = float(store.extra.get("time_budget_s", 120.0))

    with step(run_id, "train_candidates", inputs={"attempt": attempt, "budget": budget}) as box:
        def on_trial(trial: TrialResult) -> None:
            """Stream each finished candidate so the timeline fills as it trains."""
            bus.publish(run_id, {
                "type": "trial",
                "run_id": run_id,
                "model_id": trial.model_id,
                "model_name": trial.model_name,
                "ok": trial.ok,
                "fit_seconds": round(trial.fit_seconds, 2),
                "metrics": trial.metrics,
                "error": trial.error,
            })
            try:
                with Session(engine) as session:
                    session.add(Trial(
                        run_id=run_id,
                        model_id=trial.model_id,
                        candidate_key=trial.candidate_key,
                        model_name=trial.model_name,
                        family=trial.family,
                        attempt=attempt,
                        metrics=trial.metrics,
                        cv_metrics=trial.cv_metrics,
                        calibration=trial.calibration.as_dict() if trial.calibration else None,
                        fit_seconds=trial.fit_seconds,
                        error=trial.error,
                    ))
                    session.commit()
            except Exception:  # noqa: BLE001 - a lost row must not stop training
                pass
            log_trial(run_id, trial, attempt=attempt)

        report = train_all(
            split,
            features,
            target,
            task_type,
            group_col=split.group_col,
            budget_seconds=budget,
            only=plan.get("candidate_models") or None,
            on_trial=on_trial,
        )
        store.trials = report.successful
        for trial in report.successful:
            store.models[trial.model_id] = trial

        offered = len(plan.get("candidate_models") or []) or len(report.trials)
        skipped = max(offered - len(report.trials), 0)
        box["summary"] = (
            f"Attempt {attempt}: trained {len(report.successful)} of {offered} candidates "
            f"in {report.elapsed_seconds:.1f}s"
            + (f" ({skipped} skipped, budget exhausted)." if skipped else ".")
        )
        box["reasoning"] = " ".join(report.findings) or "All candidates trained cleanly."
        box["payload"] = report.as_dict()

        return {
            "trials": [t.as_dict() for t in report.trials],
            "findings": [*state.get("findings", []), *report.findings],
        }
