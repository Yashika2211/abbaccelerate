"""Run endpoints — start a run, stream it, approve the plan, read the results."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from sqlmodel import Session, select

from kairos.agent.graph import checkpointer_kind, get_graph
from kairos.agent.runtime import artifacts, clear_artifacts, set_run_status
from kairos.agent.state import initial_state
from kairos.api.schemas import ApprovalIn, RunCreate
from kairos.db.models import AuditEvent, Dataset, Run, Trial
from kairos.db.session import engine
from kairos.decision.presets import MILLING_PLANT, TURBOFAN_PLANT
from kairos.events import bus, sse_source

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/runs", tags=["runs"])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


def _config(run_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": run_id}}


def _awaiting(run_id: str) -> bool:
    try:
        return bool(get_graph().get_state(_config(run_id)).next)
    except Exception:  # noqa: BLE001
        return False


def _drive(run_id: str, payload: Any) -> None:
    """Run the graph until it interrupts or finishes. Executes in a worker thread."""
    try:
        for chunk in get_graph().stream(payload, config=_config(run_id), stream_mode="updates"):
            if "__interrupt__" in chunk:
                # Published here, not inside the node: on resume LangGraph re-runs
                # the node from the top, so a publish inside it fires twice.
                state = get_graph().get_state(_config(run_id)).values
                set_run_status(run_id, "awaiting_approval")
                bus.publish(run_id, {
                    "type": "awaiting_approval",
                    "run_id": run_id,
                    "plan": state.get("plan"),
                    "findings": state.get("findings", []),
                })
                return                      # stream stays open for the operator
        final = get_graph().get_state(_config(run_id)).values
        status = "failed" if final.get("error") else "complete"
        set_run_status(run_id, status, error=final.get("error"))
        bus.publish(run_id, {
            "type": "complete",
            "run_id": run_id,
            "status": status,
            "findings": final.get("findings", []),
            "best": final.get("best"),
        })
    except Exception as exc:  # noqa: BLE001 - never a traceback on the demo path
        log.exception("run %s failed", run_id)
        set_run_status(run_id, "failed", error=str(exc))
        bus.publish(run_id, {"type": "error", "run_id": run_id, "message": str(exc)})
    finally:
        if not _awaiting(run_id):
            bus.close(run_id)


@router.post("")
def create_run(body: RunCreate, background: BackgroundTasks) -> dict[str, Any]:
    with Session(engine) as session:
        dataset = session.get(Dataset, body.dataset_id)
        if dataset is None:
            raise HTTPException(404, "dataset not found")
        # Read what we need while the instance is still attached; everything below
        # runs after the session closes.
        dataset_name = dataset.name
        task_type = dataset.task_type
        default = MILLING_PLANT if task_type == "binary_classification" else TURBOFAN_PLANT
        cfg = body.cost_config.to_config() if body.cost_config else default
        run = Run(
            dataset_id=body.dataset_id,
            status="running",
            task_type=task_type,
            time_budget_s=body.time_budget_s,
            cost_config=cfg.__dict__,
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        run_id = run.id

    store = artifacts(run_id)
    store.cost_config = cfg
    store.extra["time_budget_s"] = body.time_budget_s

    bus.publish(run_id, {"type": "run_started", "run_id": run_id, "dataset": dataset_name})
    background.add_task(_drive, run_id, initial_state(run_id, body.dataset_id))
    return {
        "run_id": run_id,
        "status": "running",
        "checkpointer": checkpointer_kind(),
        "stream_url": f"/api/runs/{run.id}/stream",
    }


@router.get("/{run_id}/stream")
async def stream_run(run_id: str) -> StreamingResponse:
    return StreamingResponse(sse_source(run_id), media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("/{run_id}/approve")
def approve(run_id: str, body: ApprovalIn, background: BackgroundTasks) -> dict[str, Any]:
    if not _awaiting(run_id):
        raise HTTPException(409, "run is not awaiting approval")
    set_run_status(run_id, "running")
    background.add_task(
        _drive, run_id,
        Command(resume={"approved": body.approved, "plan": body.plan, "note": body.note}),
    )
    return {"run_id": run_id, "resumed": True, "approved": body.approved}


@router.get("")
def list_runs() -> list[dict[str, Any]]:
    with Session(engine) as session:
        rows = session.exec(select(Run).order_by(Run.created_at.desc())).all()
    return [
        {
            "run_id": r.id, "dataset_id": r.dataset_id, "status": r.status,
            "task_type": r.task_type, "created_at": r.created_at, "error": r.error,
        }
        for r in rows
    ]


@router.get("/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    with Session(engine) as session:
        run = session.get(Run, run_id)
        if run is None:
            raise HTTPException(404, "run not found")
        trials = session.exec(select(Trial).where(Trial.run_id == run_id)).all()
    return {
        "run_id": run.id,
        "status": run.status,
        "task_type": run.task_type,
        "awaiting_approval": _awaiting(run_id),
        "cost_config": run.cost_config,
        "plan": run.plan,
        "findings": run.findings or [],
        "impact": run.impact,
        "error": run.error,
        "llm_used": run.llm_used,
        "n_trials": len(trials),
    }


@router.get("/{run_id}/timeline")
def get_timeline(run_id: str) -> list[dict[str, Any]]:
    """The audit trail — the 'show your work' panel."""
    with Session(engine) as session:
        events = session.exec(
            select(AuditEvent).where(AuditEvent.run_id == run_id).order_by(AuditEvent.sequence)
        ).all()
    return [
        {
            "sequence": e.sequence, "node": e.node, "status": e.status,
            "summary": e.summary, "reasoning": e.reasoning,
            "duration_ms": e.duration_ms, "inputs_hash": e.inputs_hash,
            "created_at": e.created_at,
        }
        for e in events
    ]


@router.get("/{run_id}/leaderboard")
def get_leaderboard(run_id: str) -> dict[str, Any]:
    with Session(engine) as session:
        run = session.get(Run, run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    if not run.leaderboard:
        return {"run_id": run_id, "status": run.status, "rows": [], "empty_reason":
                "This run has not reached the decision stage yet."}
    return {"run_id": run_id, **run.leaderboard}


@router.get("/{run_id}/workorders")
def get_workorders(run_id: str) -> dict[str, Any]:
    schedule = artifacts(run_id).extra.get("schedule")
    orders = artifacts(run_id).extra.get("work_orders")
    if schedule is None:
        return {"run_id": run_id, "orders": [], "empty_reason": "No schedule computed yet."}
    payload = schedule.as_dict()
    if orders:
        by_id = {o["asset_id"]: o for o in orders}
        payload["orders"] = [{**o, **by_id.get(o["asset_id"], {})} for o in payload["orders"]]
    return {"run_id": run_id, **payload}


@router.get("/{run_id}/impact")
def get_impact(run_id: str) -> dict[str, Any]:
    with Session(engine) as session:
        run = session.get(Run, run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    if not run.impact:
        return {"run_id": run_id, "empty_reason": "This run has not produced an impact summary."}
    return {"run_id": run_id, **run.impact}


@router.delete("/{run_id}/cache")
def drop_cache(run_id: str) -> dict[str, Any]:
    clear_artifacts(run_id)
    return {"run_id": run_id, "cleared": True}
