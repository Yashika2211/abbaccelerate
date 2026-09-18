"""Node 3: human_checkpoint — a real LangGraph interrupt, not a modal.

Execution genuinely stops here. State is checkpointed to Postgres and the call
returns only when a separate HTTP request resumes the thread.

The 'awaiting approval' event is published by the graph driver rather than from
inside this node: on resume LangGraph re-executes the node from the top, so
anything published before interrupt() would fire a second time and the timeline
would show the checkpoint twice. That was measured in the Phase 0 spike.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from kairos.agent.runtime import record
from kairos.agent.state import KairosState


def node_human_checkpoint(state: KairosState) -> dict[str, Any]:
    plan = state.get("plan") or {}
    decision = interrupt({
        "question": "Approve this plan?",
        "plan": plan,
        "findings": state.get("findings", []),
    })

    approved = bool(decision.get("approved", True))
    edited_plan = decision.get("plan") or plan

    record(
        state["run_id"], "human_checkpoint", "done",
        summary="Operator approved the plan." if approved else "Operator rejected the plan.",
        reasoning=decision.get("note"),
        payload={"approved": approved, "edited": edited_plan != plan},
    )

    findings = [*state.get("findings", [])]
    if edited_plan != plan:
        findings.append("Operator edited the plan before approving it.")
    if not approved:
        findings.append("Operator rejected the plan; run halted before training.")

    return {
        "plan": edited_plan,
        "approved": approved,
        "awaiting_approval": False,
        "findings": findings,
        "error": None if approved else "halted: operator rejected the plan",
    }
