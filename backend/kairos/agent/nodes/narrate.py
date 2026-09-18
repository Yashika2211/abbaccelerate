"""Node 9: narrate — prose from computed values only (PROJECT_BRIEF.md §9).

The LLM receives a JSON payload of numbers that already exist and is told to use
only those. If it is unavailable, a deterministic template says the same thing
less gracefully. Either way the numbers are identical, because neither path
produces them.
"""

from __future__ import annotations

from typing import Any

from kairos.agent import llm
from kairos.agent.runtime import artifacts, step
from kairos.agent.state import KairosState


def _template(payload: dict[str, Any]) -> str:
    """Deterministic narration. Reads a little flat; every figure is real."""
    currency = payload.get("currency", "INR")
    drivers = payload.get("top_drivers", [])
    driver_text = ", ".join(d["feature"] for d in drivers[:3]) or "no driver data available"
    return (
        f"{payload['asset_label']} is flagged for maintenance. "
        f"Recommended action: {payload['recommended_action']} by {payload['act_by']}. "
        f"Acting now is estimated at {currency} {payload['cost_if_acted']:,.0f}; leaving it "
        f"is estimated at {currency} {payload['cost_if_ignored']:,.0f}. "
        f"Primary drivers: {driver_text}."
    )


def node_narrate(state: KairosState) -> dict[str, Any]:
    run_id = state["run_id"]
    store = artifacts(run_id)
    decisions = state.get("decisions") or []

    with step(run_id, "narrate", inputs={"n_decisions": len(decisions)}) as box:
        if not decisions:
            box["summary"] = "No work orders to narrate."
            return {}

        system = llm.load_prompt("narrate")
        narrated: list[dict[str, Any]] = []
        used_llm = False
        for decision in decisions[:10]:
            payload = {
                "asset_label": decision.get("asset_label"),
                "recommended_action": decision.get("recommended_action"),
                "act_by": decision.get("act_by"),
                "cost_if_acted": decision.get("expected_cost_if_acted"),
                "cost_if_ignored": decision.get("expected_cost_if_ignored"),
                "risk": decision.get("risk"),
                "top_drivers": decision.get("top_drivers", []),
                "currency": decision.get("currency", "INR"),
            }
            outcome = llm.prose(system, payload, fallback=_template(payload))
            used_llm = used_llm or outcome.used_llm
            narrated.append({**decision, "narrative": outcome.value, "narration_note": outcome.note})

        store.extra["work_orders"] = narrated
        box["summary"] = f"Wrote {len(narrated)} work order(s)."
        box["reasoning"] = (
            "Narrated by Claude from computed values."
            if used_llm
            else "LLM unavailable; deterministic templates used. Figures are identical either way."
        )
        box["payload"] = {"count": len(narrated), "llm_used": used_llm}
        return {
            "decisions": narrated,
            "llm_used": bool(state.get("llm_used")) or used_llm,
        }
