"""Phase 0 risk spike: LangGraph ``interrupt`` + Postgres checkpointer + SSE resume.

This is the thin slice of the Phase 4 architecture, proven early because it is the
only part of the stack nobody on this team has run before:

    graph runs in a BackgroundTask thread
      -> hits interrupt(), parks its state in Postgres
      -> browser sees the pause over SSE
      -> POST /resume feeds a decision back in via Command(resume=...)
      -> graph continues in a NEW request, from checkpointed state

If the Postgres checkpointer misbehaves we fall back to MemorySaver and say so in
the payload, rather than losing the human-in-the-loop demo beat entirely.

Delete this module once agent/graph.py carries the real thing.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, TypedDict

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from kairos.config import get_settings
from kairos.events import bus, sse_source

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/spike", tags=["spike"])


class SpikeState(TypedDict, total=False):
    thread_id: str
    plan: str
    approved: bool
    edited_plan: str
    result: str


# --------------------------------------------------------------------------- nodes


def node_plan(state: SpikeState) -> dict[str, Any]:
    plan = "train 4 candidates, 120s budget"
    bus.publish(state["thread_id"], {"type": "node", "node": "plan", "status": "done", "plan": plan})
    return {"plan": plan}


def node_human_checkpoint(state: SpikeState) -> dict[str, Any]:
    bus.publish(
        state["thread_id"],
        {"type": "awaiting_approval", "node": "human_checkpoint", "plan": state["plan"]},
    )
    # Execution stops here. The graph state is checkpointed; this call returns only
    # when someone resumes the thread with a payload.
    decision = interrupt({"question": "Approve this plan?", "plan": state["plan"]})
    approved = bool(decision.get("approved", False))
    edited = str(decision.get("edited_plan") or state["plan"])
    bus.publish(
        state["thread_id"],
        {"type": "node", "node": "human_checkpoint", "status": "resumed", "approved": approved},
    )
    return {"approved": approved, "edited_plan": edited}


def node_execute(state: SpikeState) -> dict[str, Any]:
    result = (
        f"executed: {state.get('edited_plan')}"
        if state.get("approved")
        else "halted: operator rejected the plan"
    )
    bus.publish(
        state["thread_id"], {"type": "node", "node": "execute", "status": "done", "result": result}
    )
    return {"result": result}


# --------------------------------------------------------------------------- graph

_graph = None
_checkpointer_kind = "none"


def _build_checkpointer():
    """Postgres if it will have us, MemorySaver if not. Never fail the app over this."""
    global _checkpointer_kind
    settings = get_settings()
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg import Connection
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        pool = ConnectionPool(
            conninfo=settings.checkpointer_url,
            max_size=5,
            open=False,
            kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0},
            connection_class=Connection,
        )
        pool.open(wait=True, timeout=10)
        saver = PostgresSaver(pool)
        saver.setup()
        _checkpointer_kind = "postgres"
        return saver
    except Exception as exc:  # noqa: BLE001 - degrade, never crash the demo
        log.warning("Postgres checkpointer unavailable (%s); using MemorySaver", exc)
        _checkpointer_kind = "memory"
        return MemorySaver()


def get_graph():
    global _graph
    if _graph is None:
        builder = StateGraph(SpikeState)
        builder.add_node("plan", node_plan)
        builder.add_node("human_checkpoint", node_human_checkpoint)
        builder.add_node("execute", node_execute)
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "human_checkpoint")
        builder.add_edge("human_checkpoint", "execute")
        builder.add_edge("execute", END)
        _graph = builder.compile(checkpointer=_build_checkpointer())
    return _graph


def checkpointer_kind() -> str:
    get_graph()
    return _checkpointer_kind


# --------------------------------------------------------------------------- runner


def _drive(thread_id: str, payload: Any) -> None:
    """Run the graph until it interrupts or finishes. Safe to call from a worker thread."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        for chunk in get_graph().stream(payload, config=config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                bus.publish(thread_id, {"type": "interrupted", "node": "human_checkpoint"})
                return  # stream stays open; the browser waits for the operator
        state = get_graph().get_state(config)
        bus.publish(thread_id, {"type": "complete", "state": dict(state.values)})
    except Exception as exc:  # noqa: BLE001 - no traceback reaches the projector
        log.exception("spike graph failed")
        bus.publish(thread_id, {"type": "error", "message": str(exc)})
    finally:
        if not _is_interrupted(thread_id):
            bus.close(thread_id)


def _is_interrupted(thread_id: str) -> bool:
    config = {"configurable": {"thread_id": thread_id}}
    try:
        return bool(get_graph().get_state(config).next)
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- routes


class ResumeRequest(BaseModel):
    approved: bool = True
    edited_plan: str | None = None


@router.post("/start")
def start(background: BackgroundTasks) -> dict[str, Any]:
    thread_id = f"spike-{uuid.uuid4().hex[:8]}"
    bus.publish(thread_id, {"type": "run_started", "thread_id": thread_id})
    background.add_task(_drive, thread_id, {"thread_id": thread_id})
    return {
        "thread_id": thread_id,
        "checkpointer": checkpointer_kind(),
        "stream_url": f"/api/spike/{thread_id}/stream",
    }


@router.post("/{thread_id}/resume")
def resume(thread_id: str, body: ResumeRequest, background: BackgroundTasks) -> dict[str, Any]:
    if not _is_interrupted(thread_id):
        raise HTTPException(status_code=409, detail="thread is not awaiting approval")
    background.add_task(
        _drive,
        thread_id,
        Command(resume={"approved": body.approved, "edited_plan": body.edited_plan}),
    )
    return {"thread_id": thread_id, "resumed": True}


@router.get("/{thread_id}/stream")
async def stream(thread_id: str) -> StreamingResponse:
    return StreamingResponse(
        sse_source(thread_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.get("/{thread_id}/state")
def state(thread_id: str) -> dict[str, Any]:
    snapshot = get_graph().get_state({"configurable": {"thread_id": thread_id}})
    return {
        "values": dict(snapshot.values),
        "next": list(snapshot.next),
        "awaiting_approval": bool(snapshot.next),
    }
