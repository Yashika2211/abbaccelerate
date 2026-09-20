"""Rebuild the decision surface from cached predictions.

One function serves three callers that all ask the same question — "given these
economics, what should we do?" — and none of which may retrain:

* the chart endpoints, after an API restart has emptied the in-process cache;
* the what-if simulator, on every slider drag (PROJECT_BRIEF.md §2.4, <300ms);
* ``make seed``, replaying a run baked before the process existed.

Because nothing here touches a model, changing the economics is a recomputation,
not a retrain. That is the entire reason the simulator can feel live.
"""

from __future__ import annotations

from typing import Any

from kairos.decision.cost import CostConfig
from kairos.decision.simulate import (
    ClassifierPredictions,
    Leaderboard,
    RulPredictions,
    simulate_classification,
    simulate_rul,
)


def _failure_cycles(frame) -> dict[Any, float]:
    return {int(u): float(g.cycle.max() + g.rul.min()) for u, g in frame.groupby("unit_id")}


def task_type_of(store: Any) -> str | None:
    """Infer the task from what was cached, without needing the run row."""
    for trial in store.models.values():
        if trial.task_type:
            return trial.task_type
    return None


def leaderboard_for(store: Any, cfg: CostConfig | None = None) -> Leaderboard | None:
    """Rank the cached candidates under ``cfg``. None when there is nothing cached."""
    split = store.split
    config = cfg or store.cost_config
    if split is None or config is None or not store.models:
        return None

    trials = [t for t in store.models.values() if t.val_prediction is not None]
    if not trials:
        return None

    task_type = task_type_of(store) or "binary_classification"
    target = split.target

    if task_type == "binary_classification":
        if not target or target not in split.val.columns:
            return None
        models = [
            ClassifierPredictions(
                model_id=t.model_id,
                model_name=t.model_name,
                val_y=split.val[target].to_numpy(),
                val_prob=t.val_prediction,
                test_y=split.test[target].to_numpy(),
                test_prob=t.test_prediction,
                metrics=t.metrics,
            )
            for t in trials
        ]
        return simulate_classification(models, config)

    models = [
        RulPredictions(
            model_id=t.model_id,
            model_name=t.model_name,
            val_units=split.val.unit_id.to_numpy(),
            val_cycles=split.val.cycle.to_numpy(),
            val_pred=t.val_prediction,
            val_failure=_failure_cycles(split.val),
            test_units=split.test.unit_id.to_numpy(),
            test_cycles=split.test.cycle.to_numpy(),
            test_pred=t.test_prediction,
            test_failure=_failure_cycles(split.test),
            metrics=t.metrics,
        )
        for t in trials
    ]
    return simulate_rul(models, config)


def ensure_leaderboard(store: Any, cfg: CostConfig | None = None) -> Leaderboard | None:
    """Return the cached leaderboard, rebuilding it once if the cache was cold."""
    if store.leaderboard is not None and cfg is None:
        return store.leaderboard
    board = leaderboard_for(store, cfg)
    if board is not None and cfg is None:
        store.leaderboard = board
    return board
