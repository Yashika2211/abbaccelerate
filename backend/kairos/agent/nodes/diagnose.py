"""Node 2: diagnose — LLM reads the profile and proposes a plan, rules if it can't.

Structured output validated by Pydantic. The rules path is not a stub: it produces
a complete, defensible plan from the profile alone, so a run with no API key is
fully functional and simply less articulate.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from kairos.agent import llm
from kairos.agent.runtime import artifacts, step
from kairos.agent.state import KairosState
from kairos.ml.registry import candidates_for

DEFAULT_WINDOWS = [5, 10, 20]
REPLAN_WINDOWS = [10, 30, 50]


class DiagnosisPlan(BaseModel):
    task_type: Literal["rul_regression", "binary_classification"]
    target: str
    drop_columns: list[str] = Field(default_factory=list)
    feature_strategy: Literal["timeseries_windows", "physics", "none"]
    window_sizes: list[int] = Field(default_factory=list)
    candidate_models: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    reasoning: str = ""


def _rules_plan(profile: dict[str, Any], replan_count: int, replan_reason: str | None) -> DiagnosisPlan:
    """Deterministic plan straight from the profile. The no-LLM path."""
    task_type = profile.get("proposed_task_type") or "binary_classification"
    target = profile.get("proposed_target") or ""
    drops = list(profile.get("recommended_drops", []))

    if task_type == "rul_regression":
        strategy: Any = "timeseries_windows"
        windows = REPLAN_WINDOWS if replan_count else DEFAULT_WINDOWS
    else:
        strategy = "physics"
        windows = []

    findings = list(profile.get("findings", []))
    reasoning = (
        f"Deterministic plan from the profile: {task_type} on {target!r}, dropping "
        f"{len(drops)} column(s) the profiler convicted or found dead."
    )
    if replan_count:
        reasoning += (
            f" This is replan #{replan_count}. Previous attempt: {replan_reason or 'underperformed'}. "
            f"Widening rolling windows to {windows} so slow degradation has room to show up."
        )
    return DiagnosisPlan(
        task_type=task_type,
        target=target,
        drop_columns=drops,
        feature_strategy=strategy,
        window_sizes=windows,
        candidate_models=[c.key for c in candidates_for(task_type)],
        findings=findings,
        reasoning=reasoning,
    )


def node_diagnose(state: KairosState) -> dict[str, Any]:
    run_id = state["run_id"]
    profile = state.get("profile") or {}
    replan_count = state.get("replan_count", 0)
    replan_reason = state.get("replan_reason")

    with step(run_id, "diagnose", inputs=profile.get("name")) as box:
        allowed = [c.key for c in candidates_for(
            profile.get("proposed_task_type") or "binary_classification"
        )]
        payload: dict[str, Any] = {
            "profile": profile,
            "allowed_candidate_models": allowed,
            "allowed_feature_strategies": ["timeseries_windows", "physics", "none"],
        }
        if replan_count:
            payload["previous_attempt"] = {
                "attempt": replan_count,
                "failure_reason": replan_reason,
                "instruction": "Change something material. Do not repeat the previous plan.",
            }

        outcome = llm.structured(llm.load_prompt("diagnose"), payload, DiagnosisPlan)
        plan = outcome.value or _rules_plan(profile, replan_count, replan_reason)

        # Guard rails: the LLM proposes, the profiler's convictions still stand.
        must_drop = set(profile.get("recommended_drops", []))
        plan.drop_columns = sorted(set(plan.drop_columns) | must_drop)
        plan.candidate_models = [m for m in plan.candidate_models if m in allowed] or allowed
        if plan.feature_strategy == "timeseries_windows" and not plan.window_sizes:
            plan.window_sizes = REPLAN_WINDOWS if replan_count else DEFAULT_WINDOWS

        artifacts(run_id).extra["plan"] = plan

        box["summary"] = (
            f"Plan: {plan.task_type} on {plan.target!r}, "
            f"{plan.feature_strategy} features, "
            f"{len(plan.candidate_models)} candidates, "
            f"dropping {len(plan.drop_columns)} columns."
        )
        box["reasoning"] = f"{plan.reasoning} [{outcome.note}]"
        box["payload"] = {"plan": plan.model_dump(), "llm_note": outcome.note}

        findings = [*state.get("findings", [])]
        if replan_count:
            findings.append(
                f"Replan #{replan_count}: {replan_reason} "
                f"Retrying with {plan.feature_strategy} features"
                + (f" over windows {plan.window_sizes}." if plan.window_sizes else ".")
            )
        return {
            "plan": plan.model_dump(),
            "task_type": plan.task_type,
            "findings": findings,
            "llm_used": bool(state.get("llm_used")) or outcome.used_llm,
            "awaiting_approval": True,
        }
