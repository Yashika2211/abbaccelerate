"""Agent state — PROJECT_BRIEF.md §7.

One TypedDict threaded through every node. Kept flat and JSON-serialisable so the
Postgres checkpointer can park it mid-run and hand it back to a different HTTP
request after a human approves the plan.

Large objects (fitted models, prediction arrays) deliberately do NOT live here.
They sit in a process-local cache keyed by run_id, because checkpointing a
LightGBM booster into Postgres on every state transition would make the run
timeline crawl. The state carries the run_id; the cache carries the weight.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

TaskType = Literal["rul_regression", "binary_classification"]


class KairosState(TypedDict, total=False):
    run_id: str
    dataset_id: str
    profile: dict[str, Any] | None
    task_type: TaskType | None
    plan: dict[str, Any] | None
    trials: list[dict[str, Any]]
    best: dict[str, Any] | None
    cost_analysis: dict[str, Any] | None
    explanations: dict[str, Any] | None
    decisions: list[dict[str, Any]]
    findings: list[str]
    replan_count: int
    awaiting_approval: bool
    approved: bool
    quality_gate_passed: bool
    replan_reason: str | None
    error: str | None
    llm_used: bool


def initial_state(run_id: str, dataset_id: str) -> KairosState:
    return KairosState(
        run_id=run_id,
        dataset_id=dataset_id,
        profile=None,
        task_type=None,
        plan=None,
        trials=[],
        best=None,
        cost_analysis=None,
        explanations=None,
        decisions=[],
        findings=[],
        replan_count=0,
        awaiting_approval=False,
        approved=False,
        quality_gate_passed=False,
        replan_reason=None,
        error=None,
        llm_used=False,
    )
