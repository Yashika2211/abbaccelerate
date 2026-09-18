"""The Kairos agent graph — PROJECT_BRIEF.md §7.

What makes this an agent rather than a script with an LLM attached is that it
branches on what it observes:

* **Quality gate.** If the best model fails the gate and we have replans left, the
  graph routes back to ``diagnose`` with the failure reason injected, and the
  second diagnosis is materially different — wider windows, different candidates.
  That retry is visible in the run timeline, which is the point.
* **Human checkpoint.** A real ``interrupt``: state is checkpointed to Postgres and
  execution resumes in a different HTTP request. Rejecting the plan ends the run.
* **Recovery.** Any node that raises routes to ``recover``, which records the
  failure and finishes in a reportable state. No traceback reaches the projector.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from kairos.agent.nodes import (
    node_decide,
    node_diagnose,
    node_engineer_features,
    node_evaluate,
    node_explain,
    node_human_checkpoint,
    node_narrate,
    node_profile_dataset,
    node_recover,
    node_report,
    node_schedule,
    node_train_candidates,
)
from kairos.agent.state import KairosState
from kairos.config import get_settings

log = logging.getLogger(__name__)

#: PROJECT_BRIEF.md §7: at most two replans, then proceed with what we have and
#: say so. An agent that loops forever looking for a better plan is a hung demo.
MAX_REPLANS = 2

_graph = None
_checkpointer_kind = "none"


def _guard(name: str, fn: Callable[[KairosState], dict[str, Any]]):
    """Wrap a node so a raised exception becomes routable state, not a stack trace."""

    def wrapped(state: KairosState) -> dict[str, Any]:
        if state.get("error"):
            return {}          # already failing; let the graph drain to recover
        try:
            return fn(state)
        except Exception as exc:  # noqa: BLE001
            log.exception("node %s failed", name)
            return {"error": f"{name}: {type(exc).__name__}: {exc}"}

    wrapped.__name__ = f"guarded_{name}"
    return wrapped


# ----------------------------------------------------------------- conditions


def route_after_checkpoint(state: KairosState) -> str:
    if state.get("error"):
        return "recover"
    return "engineer_features" if state.get("approved") else "recover"


def route_after_evaluate(state: KairosState) -> str:
    """The branch that makes the agent an agent."""
    if state.get("error"):
        return "recover"
    if state.get("quality_gate_passed"):
        return "decide"
    if state.get("replan_count", 0) < MAX_REPLANS:
        return "replan"
    return "decide"      # out of retries: proceed, and the findings say why


def route_if_error(next_node: str) -> Callable[[KairosState], str]:
    def router(state: KairosState) -> str:
        return "recover" if state.get("error") else next_node

    return router


def node_bump_replan(state: KairosState) -> dict[str, Any]:
    """Increment the replan counter between evaluate and the second diagnosis."""
    return {
        "replan_count": state.get("replan_count", 0) + 1,
        "quality_gate_passed": False,
    }


# --------------------------------------------------------------- construction


def build_checkpointer():
    """Postgres if it will have us, MemorySaver if not. Never fail the app."""
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
    except Exception as exc:  # noqa: BLE001
        log.warning("Postgres checkpointer unavailable (%s); using MemorySaver", exc)
        _checkpointer_kind = "memory"
        return MemorySaver()


def build_graph(checkpointer: Any = None):
    builder = StateGraph(KairosState)

    builder.add_node("profile_dataset", _guard("profile_dataset", node_profile_dataset))
    builder.add_node("diagnose", _guard("diagnose", node_diagnose))
    builder.add_node("human_checkpoint", node_human_checkpoint)   # must not swallow interrupt
    builder.add_node("engineer_features", _guard("engineer_features", node_engineer_features))
    builder.add_node("train_candidates", _guard("train_candidates", node_train_candidates))
    builder.add_node("evaluate", _guard("evaluate", node_evaluate))
    builder.add_node("bump_replan", node_bump_replan)
    builder.add_node("decide", _guard("decide", node_decide))
    builder.add_node("explain", _guard("explain", node_explain))
    builder.add_node("schedule", _guard("schedule", node_schedule))
    builder.add_node("narrate", _guard("narrate", node_narrate))
    builder.add_node("report", _guard("report", node_report))
    builder.add_node("recover", node_recover)

    builder.add_edge(START, "profile_dataset")
    builder.add_conditional_edges(
        "profile_dataset", route_if_error("diagnose"),
        {"diagnose": "diagnose", "recover": "recover"},
    )
    builder.add_conditional_edges(
        "diagnose", route_if_error("human_checkpoint"),
        {"human_checkpoint": "human_checkpoint", "recover": "recover"},
    )
    builder.add_conditional_edges(
        "human_checkpoint", route_after_checkpoint,
        {"engineer_features": "engineer_features", "recover": "recover"},
    )
    builder.add_conditional_edges(
        "engineer_features", route_if_error("train_candidates"),
        {"train_candidates": "train_candidates", "recover": "recover"},
    )
    builder.add_conditional_edges(
        "train_candidates", route_if_error("evaluate"),
        {"evaluate": "evaluate", "recover": "recover"},
    )
    # The replan loop.
    builder.add_conditional_edges(
        "evaluate", route_after_evaluate,
        {"decide": "decide", "replan": "bump_replan", "recover": "recover"},
    )
    builder.add_edge("bump_replan", "diagnose")

    builder.add_conditional_edges(
        "decide", route_if_error("explain"),
        {"explain": "explain", "recover": "recover"},
    )
    builder.add_edge("explain", "schedule")
    builder.add_edge("schedule", "narrate")
    builder.add_edge("narrate", "report")
    builder.add_edge("report", END)
    builder.add_edge("recover", END)

    return builder.compile(checkpointer=checkpointer or build_checkpointer())


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def checkpointer_kind() -> str:
    get_graph()
    return _checkpointer_kind
