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
from kairos.decision.rebuild import ensure_leaderboard
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


def _champion_id(store) -> str | None:
    board = ensure_leaderboard(store)
    return board.best.model_id if board else None


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


@router.get("/{run_id}/cost-curve")
def get_cost_curve(run_id: str) -> dict[str, Any]:
    """Cost vs the decision variable, with the optimum marked.

    Classification gives cost vs threshold; RUL gives the U-shaped cost vs lead
    time. Both are recomputed from cached predictions, so this is cheap.
    """
    store = artifacts(run_id)
    split, cfg = store.split, store.cost_config
    board = ensure_leaderboard(store)
    if board is None or split is None or cfg is None:
        return {"run_id": run_id, "points": [], "empty_reason": "This run has no decision yet."}

    champion = board.best
    trial = store.models.get(champion.model_id)
    if trial is None:
        return {"run_id": run_id, "points": [], "empty_reason": "Champion model not in cache."}

    from kairos.decision.policy import optimal_lead_time, optimal_threshold

    if champion.operating_point_kind == "threshold":
        target = (
            store.extra.get("plan").target
            if hasattr(store.extra.get("plan"), "target")
            else split.target
        )
        decision = optimal_threshold(split.val[target].to_numpy(), trial.val_prediction, cfg)
        return {
            "run_id": run_id,
            "kind": "threshold",
            "model_name": champion.model_name,
            "optimum": decision.threshold,
            "bayes_threshold": decision.bayes_threshold,
            "calibration_warning": decision.calibration_warning,
            "degenerate": decision.degenerate,
            "degenerate_reason": decision.degenerate_reason,
            "currency": cfg.currency,
            "chosen_on": decision.chosen_on,
            "points": [
                {"x": p.threshold, "cost": p.total_cost, "cost_per_asset_year": p.cost_per_asset_year,
                 "tp": p.tp, "fp": p.fp, "fn": p.fn, "tn": p.tn}
                for p in decision.curve
            ],
        }

    from kairos.agent.nodes.decide import failure_cycles

    decision = optimal_lead_time(
        split.val.unit_id.to_numpy(), split.val.cycle.to_numpy(), trial.val_prediction,
        failure_cycles(split.val), cfg,
    )
    return {
        "run_id": run_id,
        "kind": "lead_time",
        "model_name": champion.model_name,
        "optimum": decision.lead_time,
        "degenerate": decision.degenerate,
        "degenerate_reason": decision.degenerate_reason,
        "currency": cfg.currency,
        "chosen_on": decision.chosen_on,
        "points": [
            {"x": r.lead_time, "cost": r.total_cost, "cost_per_asset_year": r.cost_per_asset_year,
             "caught": r.caught, "missed": r.missed, "wasted_life_cycles": r.wasted_life_cycles}
            for r in decision.curve
        ],
    }


@router.get("/{run_id}/baselines")
def get_baselines(run_id: str) -> dict[str, Any]:
    """The four named policies, computed on the same test set."""
    store = artifacts(run_id)
    board = ensure_leaderboard(store)
    if board is None or board.baselines is None:
        with Session(engine) as session:
            run = session.get(Run, run_id)
        stored = (run.leaderboard or {}).get("baselines") if run else None
        if stored:
            return {"run_id": run_id, **stored}
        return {"run_id": run_id, "baselines": [], "empty_reason": "No baselines computed yet."}
    return {"run_id": run_id, **board.baselines.as_dict()}


@router.get("/{run_id}/calibration")
def get_calibration(run_id: str) -> dict[str, Any]:
    """Reliability-diagram data for every trained classifier.

    This is the answer to 'why should I trust 0.71?' — without it, a cost-optimal
    threshold is arithmetic on numbers that do not mean what they claim.
    """
    store = artifacts(run_id)
    if not store.models:
        return {"run_id": run_id, "models": [], "empty_reason": "No trained models in cache."}
    models = [
        {
            "model_id": trial.model_id,
            "model_name": trial.model_name,
            "is_champion": bool(_champion_id(store) == mid),
            **trial.calibration.as_dict(),
        }
        for mid, trial in store.models.items()
        if trial.calibration is not None
    ]
    if not models:
        return {
            "run_id": run_id, "models": [],
            "empty_reason": "Calibration applies to classifiers; this run is a regression task.",
        }
    return {"run_id": run_id, "models": models}


@router.get("/{run_id}/explain")
def get_explanations(run_id: str) -> dict[str, Any]:
    """Global SHAP ranking for the champion. Real values or an honest absence."""
    store = artifacts(run_id)
    ranking = store.extra.get("shap_global")
    board = ensure_leaderboard(store)
    if not ranking:
        return {
            "run_id": run_id, "global": [],
            "empty_reason": (
                "No SHAP values for this run. Kairos uses TreeExplainer only — a non-tree "
                "champion gets no explanation rather than an approximated one."
            ),
        }
    return {
        "run_id": run_id,
        "model_name": board.best.model_name if board else None,
        "global": ranking,
        "n_rows_explained": len(store.extra.get("shap_sample", [])),
    }


@router.get("/{run_id}/degradation")
def get_degradation(run_id: str, limit: int = 6) -> dict[str, Any]:
    """Predicted vs true RUL trajectories for the riskiest units."""
    store = artifacts(run_id)
    split = store.split
    board = ensure_leaderboard(store)
    if board is None or split is None or board.best.operating_point_kind != "lead_time":
        return {"run_id": run_id, "units": [], "empty_reason": "Degradation applies to RUL runs."}
    trial = store.models.get(board.best.model_id)
    if trial is None:
        return {"run_id": run_id, "units": [], "empty_reason": "Champion model not in cache."}

    frame = split.test.assign(_pred=trial.test_prediction)
    units = []
    for unit, group in list(frame.groupby("unit_id"))[:limit]:
        group = group.sort_values("cycle")
        units.append({
            "unit_id": int(unit),
            "points": [
                {"cycle": int(c), "true_rul": float(t), "pred_rul": float(p)}
                for c, t, p in zip(group.cycle, group.rul, group._pred)
            ],
        })
    return {
        "run_id": run_id,
        "lead_time": board.best.operating_point,
        "model_name": board.best.model_name,
        "units": units,
    }
