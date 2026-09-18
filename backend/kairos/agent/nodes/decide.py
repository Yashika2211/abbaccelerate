"""Node 7: decide — the Decision Layer. This is the node the product exists for.

Thresholds or lead times, cost curves, baselines and the cost-ranked leaderboard.
Also the place where Kairos says out loud that the economics, not the model, are
making the decision — the degenerate-economics branch from PROJECT_BRIEF.md §7.
"""

from __future__ import annotations

from typing import Any

from sqlmodel import Session, select

from kairos.agent.runtime import artifacts, step
from kairos.agent.state import KairosState
from kairos.db.models import Trial
from kairos.db.session import engine
from kairos.decision.presets import MILLING_PLANT, TURBOFAN_PLANT
from kairos.decision.simulate import (
    ClassifierPredictions,
    RulPredictions,
    simulate_classification,
    simulate_rul,
)
from kairos.mlflow_log import log_decision


def failure_cycles(frame) -> dict[Any, float]:
    """True failure cycle per unit on a complete run-to-failure trajectory."""
    return {
        int(u): float(g.cycle.max() + g.rul.min())
        for u, g in frame.groupby("unit_id")
    }


def default_cost_config(task_type: str):
    return MILLING_PLANT if task_type == "binary_classification" else TURBOFAN_PLANT


def node_decide(state: KairosState) -> dict[str, Any]:
    run_id = state["run_id"]
    store = artifacts(run_id)
    split, trials = store.split, store.trials
    task_type = state.get("task_type") or "binary_classification"
    cfg = store.cost_config or default_cost_config(task_type)
    store.cost_config = cfg
    target = (state.get("plan") or {}).get("target") or split.target

    with step(run_id, "decide", inputs={"n_models": len(trials)}) as box:
        if task_type == "binary_classification":
            models = [
                ClassifierPredictions(
                    model_id=t.model_id, model_name=t.model_name,
                    val_y=split.val[target].to_numpy(), val_prob=t.val_prediction,
                    test_y=split.test[target].to_numpy(), test_prob=t.test_prediction,
                    metrics=t.metrics,
                )
                for t in trials
            ]
            board = simulate_classification(models, cfg)
            metric, higher = "pr_auc", True
        else:
            models = [
                RulPredictions(
                    model_id=t.model_id, model_name=t.model_name,
                    val_units=split.val.unit_id.to_numpy(),
                    val_cycles=split.val.cycle.to_numpy(),
                    val_pred=t.val_prediction,
                    val_failure=failure_cycles(split.val),
                    test_units=split.test.unit_id.to_numpy(),
                    test_cycles=split.test.cycle.to_numpy(),
                    test_pred=t.test_prediction,
                    test_failure=failure_cycles(split.test),
                    metrics=t.metrics,
                )
                for t in trials
            ]
            board = simulate_rul(models, cfg)
            metric, higher = "rmse", False

        store.leaderboard = board
        inversion = board.inversion(metric, higher_is_better=higher)
        winner = board.best

        _persist_costs(run_id, board)
        log_decision(run_id, {
            "best_cost_per_asset_year": winner.cost_per_asset_year,
            "best_operating_point": winner.operating_point,
        })

        findings = [*state.get("findings", [])]
        if inversion:
            findings.append(
                f"{inversion['metric_winner']} wins on {metric}, but {inversion['cost_winner']} "
                f"is cheaper by {cfg.currency} {inversion['annual_cost_penalty']:,.0f} per asset "
                f"per year. Kairos ranks by cost, so it picks {inversion['cost_winner']}."
            )
        else:
            findings.append(
                f"The best-{metric} model is also the cheapest, so there is no trade-off to "
                f"report here."
            )
        if winner.degenerate:
            findings.append(f"DEGENERATE ECONOMICS: {winner.degenerate_reason}")
        if board.baselines and not board.baselines.kairos_wins:
            findings.append(board.baselines.verdict())

        box["summary"] = (
            f"{winner.model_name} is cheapest at {cfg.currency} "
            f"{winner.cost_per_asset_year:,.0f} per asset per year "
            f"({winner.operating_point_kind} {winner.operating_point:.3f})."
        )
        box["reasoning"] = findings[-1]
        box["payload"] = {
            "leaderboard": board.as_dict(),
            "inversion": inversion,
            "cost_config": cfg.__dict__,
        }

        return {
            "cost_analysis": {
                "leaderboard": board.as_dict(),
                "inversion": inversion,
                "degenerate": winner.degenerate,
            },
            "best": {
                "model_id": winner.model_id,
                "model_name": winner.model_name,
                "operating_point": winner.operating_point,
                "cost_per_asset_year": winner.cost_per_asset_year,
                "metrics": winner.metrics,
            },
            "findings": findings,
        }


def _persist_costs(run_id: str, board) -> None:
    """Write the decided costs back onto the trial rows."""
    try:
        with Session(engine) as session:
            rows = session.exec(select(Trial).where(Trial.run_id == run_id)).all()
            by_model = {r.model_id: r for r in rows}
            for i, entry in enumerate(board.rows):
                row = by_model.get(entry.model_id)
                if row is None:
                    continue
                row.operating_point = entry.operating_point
                row.total_cost = entry.total_cost
                row.cost_per_asset_year = entry.cost_per_asset_year
                row.cost_regret = entry.cost_regret
                row.is_champion = i == 0
                session.add(row)
            session.commit()
    except Exception:  # noqa: BLE001 - persistence is not worth a failed run
        pass
