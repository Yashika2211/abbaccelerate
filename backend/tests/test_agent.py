"""Agent graph tests — PROJECT_BRIEF.md §7.

The claim under test is that the routing is real: the graph branches on what it
observes, retries with a materially different plan, honours a rejection, and
never lets an exception escape as a traceback.

These run against MemorySaver on synthetic data, so they are fast and do not need
Postgres. The Postgres checkpointer is exercised by the live end-to-end run.
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from kairos.agent.graph import (
    MAX_REPLANS,
    _guard,
    node_bump_replan,
    route_after_checkpoint,
    route_after_evaluate,
    route_if_error,
)
from kairos.agent.nodes.diagnose import DiagnosisPlan, _rules_plan
from kairos.agent.state import KairosState, initial_state


# ------------------------------------------------------------------- routing


def test_a_passing_gate_goes_straight_to_decide() -> None:
    assert route_after_evaluate({"quality_gate_passed": True}) == "decide"


def test_a_failing_gate_triggers_a_replan_while_retries_remain() -> None:
    for attempt in range(MAX_REPLANS):
        state = {"quality_gate_passed": False, "replan_count": attempt}
        assert route_after_evaluate(state) == "replan"


def test_the_agent_stops_retrying_rather_than_looping_forever() -> None:
    """An agent that keeps searching for a better plan is a hung demo."""
    state = {"quality_gate_passed": False, "replan_count": MAX_REPLANS}
    assert route_after_evaluate(state) == "decide"


def test_any_error_routes_to_recover_from_every_node() -> None:
    for target in ("diagnose", "train_candidates", "decide"):
        assert route_if_error(target)({"error": "boom"}) == "recover"
        assert route_if_error(target)({}) == target


def test_a_rejected_plan_halts_the_run() -> None:
    assert route_after_checkpoint({"approved": False}) == "recover"
    assert route_after_checkpoint({"approved": True}) == "engineer_features"


def test_bump_replan_increments_and_clears_the_gate() -> None:
    out = node_bump_replan({"replan_count": 1, "quality_gate_passed": True})
    assert out == {"replan_count": 2, "quality_gate_passed": False}


# --------------------------------------------------------------------- guard


def test_the_guard_turns_an_exception_into_routable_state() -> None:
    """No traceback reaches the projector (brief §11)."""

    def explodes(state: KairosState) -> dict[str, Any]:
        raise RuntimeError("sensor bus offline")

    result = _guard("train_candidates", explodes)({"run_id": "r"})
    assert "error" in result
    assert "sensor bus offline" in result["error"]
    assert "train_candidates" in result["error"]


def test_the_guard_short_circuits_once_the_run_is_already_failing() -> None:
    called = False

    def should_not_run(state: KairosState) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    assert _guard("decide", should_not_run)({"error": "earlier failure"}) == {}
    assert not called


# ------------------------------------------------------------- replan content


def test_a_replan_is_materially_different_not_a_repeat() -> None:
    """Retrying with an identical plan would be theatre, not agency."""
    profile = {
        "proposed_task_type": "rul_regression",
        "proposed_target": "rul",
        "recommended_drops": ["sensor_1"],
        "findings": [],
    }
    first = _rules_plan(profile, replan_count=0, replan_reason=None)
    second = _rules_plan(profile, replan_count=1, replan_reason="PR-AUC below gate")
    assert first.window_sizes != second.window_sizes
    assert max(second.window_sizes) > max(first.window_sizes)
    assert "replan" in second.reasoning.lower()


def test_the_replan_reason_is_carried_into_the_new_plan() -> None:
    plan = _rules_plan(
        {"proposed_task_type": "rul_regression", "proposed_target": "rul"},
        replan_count=1,
        replan_reason="RMSE worse than predict-the-mean.",
    )
    assert "RMSE worse than predict-the-mean." in plan.reasoning


def test_the_rules_plan_never_keeps_a_convicted_column() -> None:
    plan = _rules_plan(
        {
            "proposed_task_type": "binary_classification",
            "proposed_target": "machine_failure",
            "recommended_drops": ["twf", "hdf", "udi"],
        },
        replan_count=0,
        replan_reason=None,
    )
    assert {"twf", "hdf", "udi"} <= set(plan.drop_columns)


def test_diagnosis_schema_rejects_an_invented_task_type() -> None:
    """Structured output is validated, so a hallucinated task cannot reach the graph."""
    with pytest.raises(Exception):
        DiagnosisPlan.model_validate({
            "task_type": "telepathy",
            "target": "rul",
            "feature_strategy": "none",
        })


# ------------------------------------------------------- interrupt mechanics


def _interrupting_graph():
    """A miniature of the real graph: plan -> interrupt -> act."""

    def plan(state: KairosState) -> dict[str, Any]:
        return {"plan": {"windows": [5, 10, 20]}}

    def checkpoint(state: KairosState) -> dict[str, Any]:
        decision = interrupt({"plan": state["plan"]})
        return {
            "approved": bool(decision.get("approved")),
            "plan": decision.get("plan") or state["plan"],
        }

    def act(state: KairosState) -> dict[str, Any]:
        return {"findings": [f"ran with {state['plan']}"]}

    builder = StateGraph(KairosState)
    builder.add_node("plan", plan)
    builder.add_node("checkpoint", checkpoint)
    builder.add_node("act", act)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "checkpoint")
    builder.add_conditional_edges(
        "checkpoint",
        lambda s: "act" if s.get("approved") else END,
        {"act": "act", END: END},
    )
    builder.add_edge("act", END)
    return builder.compile(checkpointer=MemorySaver())


def test_the_graph_genuinely_parks_at_the_checkpoint() -> None:
    graph = _interrupting_graph()
    config = {"configurable": {"thread_id": "t1"}}
    graph.invoke(initial_state("t1", "d1"), config=config)
    snapshot = graph.get_state(config)
    assert snapshot.next == ("checkpoint",), "execution must stop, not fall through"


def test_resuming_carries_the_operators_edit_into_the_run() -> None:
    graph = _interrupting_graph()
    config = {"configurable": {"thread_id": "t2"}}
    graph.invoke(initial_state("t2", "d1"), config=config)
    graph.invoke(
        Command(resume={"approved": True, "plan": {"windows": [30, 50]}}), config=config
    )
    final = graph.get_state(config).values
    assert final["plan"] == {"windows": [30, 50]}
    assert "ran with {'windows': [30, 50]}" in final["findings"]


def test_a_rejection_ends_the_run_without_acting() -> None:
    graph = _interrupting_graph()
    config = {"configurable": {"thread_id": "t3"}}
    graph.invoke(initial_state("t3", "d1"), config=config)
    graph.invoke(Command(resume={"approved": False}), config=config)
    final = graph.get_state(config).values
    assert final.get("approved") is False
    assert not final.get("findings")


def test_state_survives_between_two_separate_invocations() -> None:
    """The HITL guarantee: resume happens in a different request entirely."""
    graph = _interrupting_graph()
    config = {"configurable": {"thread_id": "t4"}}
    graph.invoke(initial_state("t4", "dataset-xyz"), config=config)
    parked = graph.get_state(config).values
    assert parked["dataset_id"] == "dataset-xyz"
    graph.invoke(Command(resume={"approved": True}), config=config)
    assert graph.get_state(config).values["dataset_id"] == "dataset-xyz"
