"""Recovery node — PROJECT_BRIEF.md §7: nothing kills a demo like a traceback.

Reached whenever a node raises. Records the failure as a visible finding and lets
the run end in a reportable state instead of a stack trace on a projector.
"""

from __future__ import annotations

from typing import Any

from kairos.agent.runtime import record, set_run_status
from kairos.agent.state import KairosState


def node_recover(state: KairosState) -> dict[str, Any]:
    run_id = state["run_id"]
    error = state.get("error") or "unknown error"
    message = (
        f"The run stopped early: {error}. Everything computed before this point is "
        f"still shown below; nothing displayed is estimated or filled in."
    )
    record(run_id, "recover", "done", summary="Recovered from a node failure.", reasoning=message)
    set_run_status(run_id, "failed", error=error)
    return {"findings": [*state.get("findings", []), message]}
