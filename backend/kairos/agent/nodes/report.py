"""Node 10: report — the executive impact summary."""

from __future__ import annotations

from typing import Any

from sqlmodel import Session

from kairos.agent.runtime import artifacts, step
from kairos.agent.state import KairosState
from kairos.db.models import Run
from kairos.db.session import engine


def node_report(state: KairosState) -> dict[str, Any]:
    run_id = state["run_id"]
    store = artifacts(run_id)
    board = store.leaderboard
    cfg = store.cost_config

    with step(run_id, "report", inputs=run_id) as box:
        if board is None:
            box["summary"] = "No leaderboard to summarise."
            return {}

        winner = board.best
        baselines = board.baselines
        impact: dict[str, Any] = {
            "champion": winner.model_name,
            "operating_point": winner.operating_point,
            "operating_point_kind": winner.operating_point_kind,
            "cost_per_asset_year": winner.cost_per_asset_year,
            "currency": cfg.currency if cfg else "INR",
            "n_models_compared": len(board.rows),
            "llm_used": bool(state.get("llm_used")),
            "findings": state.get("findings", []),
        }
        if baselines:
            impact["baselines"] = baselines.as_dict()
            impact["verdict"] = baselines.verdict()
            impact["savings_per_asset_year"] = baselines.savings_per_asset_year
            impact["kairos_wins"] = baselines.kairos_wins
        if cfg:
            impact["annualisation_note"] = (
                f"Costs are annualised assuming {cfg.cycles_per_year:,.0f} cycles per asset "
                f"per year for RUL work and {cfg.observations_per_asset_year:,.0f} observations "
                f"per asset-year for classification. Both are editable in Cost Studio."
            )

        try:
            with Session(engine) as session:
                run = session.get(Run, run_id)
                if run:
                    run.leaderboard = board.as_dict()
                    run.impact = impact
                    run.findings = state.get("findings", [])
                    run.llm_used = bool(state.get("llm_used"))
                    session.add(run)
                    session.commit()
        except Exception:  # noqa: BLE001
            pass

        box["summary"] = impact.get("verdict", f"Champion: {winner.model_name}.")
        box["reasoning"] = impact.get("annualisation_note")
        box["payload"] = impact
        return {"cost_analysis": {**(state.get("cost_analysis") or {}), "impact": impact}}
