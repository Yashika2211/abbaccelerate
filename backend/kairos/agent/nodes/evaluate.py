"""Node 6: evaluate — apply the quality gate that drives the replan branch."""

from __future__ import annotations

from typing import Any

from kairos.agent.runtime import artifacts, step
from kairos.agent.state import KairosState
from kairos.ml.evaluate import beats_baseline


def node_evaluate(state: KairosState) -> dict[str, Any]:
    run_id = state["run_id"]
    store = artifacts(run_id)
    task_type = state.get("task_type") or "binary_classification"
    trials = store.trials

    with step(run_id, "evaluate", inputs={"n_trials": len(trials)}) as box:
        if not trials:
            box["summary"] = "No candidate trained successfully."
            return {
                "quality_gate_passed": False,
                "replan_reason": "No candidate trained successfully.",
            }

        baseline = next((t for t in trials if t.family == "baseline"), None)
        baseline_metrics = baseline.metrics if baseline else {}
        scored = [t for t in trials if t.family != "baseline"] or trials

        if task_type == "binary_classification":
            best = max(scored, key=lambda t: t.metrics.get("pr_auc", 0.0))
            headline = f"PR-AUC {best.metrics.get('pr_auc', 0):.4f}"
        else:
            best = min(scored, key=lambda t: t.metrics.get("rmse", float("inf")))
            headline = f"RMSE {best.metrics.get('rmse', float('nan')):.3f}"

        passed = beats_baseline(best.metrics, baseline_metrics, task_type)
        reason = None
        if not passed:
            reason = (
                f"Best model {best.model_name} reached {headline}, below the quality gate "
                f"(PR-AUC >= 0.6 for classification, better-than-mean RMSE for RUL)."
            )

        box["summary"] = f"Best so far: {best.model_name} at {headline}. Gate {'passed' if passed else 'FAILED'}."
        box["reasoning"] = reason or "Best candidate clears the quality gate; proceeding to decide."
        box["payload"] = {
            "best_model_id": best.model_id,
            "best_model_name": best.model_name,
            "metrics": best.metrics,
            "baseline_metrics": baseline_metrics,
            "quality_gate_passed": passed,
        }

        findings = [*state.get("findings", [])]
        if not passed:
            findings.append(reason)
        return {
            "best": {"model_id": best.model_id, "model_name": best.model_name, "metrics": best.metrics},
            "quality_gate_passed": passed,
            "replan_reason": reason,
            "findings": findings,
        }
