"""Threshold optimisation tests — PROJECT_BRIEF.md §2.1.

The claim these tests defend: Kairos never uses 0.5, and the threshold it does use
is the argmin of expected cost, chosen on validation and frozen for test.
"""

from __future__ import annotations

import numpy as np
import pytest

from kairos.decision.cost import CostConfig, expected_cost
from kairos.decision.policy import (
    apply_threshold,
    optimal_threshold,
    sweep_thresholds,
)
from tests.test_cost import PLANT


def _separable(n: int = 400, seed: int = 7):
    """A well-behaved classifier: positives score high, negatives low, with overlap."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.15).astype(int)
    p = np.clip(rng.normal(np.where(y == 1, 0.72, 0.28), 0.18), 0.001, 0.999)
    return y, p


def test_sweep_costs_match_the_cost_engine_row_by_row() -> None:
    """The fast cumsum sweep must agree with the naive per-threshold computation."""
    y, p = _separable()
    grid = np.linspace(0.05, 0.95, 19)
    for point in sweep_thresholds(y, p, PLANT, grid=grid):
        naive = expected_cost(y, (p >= point.threshold).astype(int), PLANT)
        assert point.total_cost == pytest.approx(naive)


def test_confusion_counts_are_internally_consistent_at_every_threshold() -> None:
    y, p = _separable()
    for point in sweep_thresholds(y, p, PLANT):
        assert point.tp + point.fn == int(y.sum())
        assert point.tn + point.fp == int((1 - y).sum())


def test_threshold_zero_alerts_on_everything_and_one_alerts_on_almost_nothing() -> None:
    y, p = _separable()
    curve = {round(pt.threshold, 6): pt for pt in sweep_thresholds(y, p, PLANT)}
    assert curve[0.0].alerts == y.size          # p >= 0 is always true
    assert curve[1.0].alerts == 0               # no probability reaches 1.0 here


def test_the_chosen_threshold_beats_the_naive_half() -> None:
    """The entire §2.1 argument in one assertion."""
    y, p = _separable()
    decision = optimal_threshold(y, p, PLANT)
    assert decision.point.total_cost <= decision.naive_point.total_cost
    assert decision.savings_vs_naive >= 0


def test_high_consequence_economics_push_the_threshold_below_half() -> None:
    """When a miss costs 89x a false alarm, you alert far earlier than 0.5."""
    y, p = _separable()
    decision = optimal_threshold(y, p, PLANT)
    assert PLANT.consequence_ratio > 40
    assert decision.threshold < 0.5
    assert not decision.degenerate


def test_cheap_failures_push_the_threshold_above_half() -> None:
    cheap = PLANT.with_changes(
        c_unplanned_repair=140_000, unplanned_downtime_hours=1, c_inspection=90_000
    )
    y, p = _separable()
    assert optimal_threshold(y, p, cheap).threshold > 0.5


def test_optimum_is_the_true_argmin_of_the_curve() -> None:
    y, p = _separable()
    decision = optimal_threshold(y, p, PLANT)
    assert decision.point.total_cost == pytest.approx(min(c.total_cost for c in decision.curve))


def test_ties_break_toward_the_middle_of_the_plateau_not_its_edge() -> None:
    """A threshold on a cliff is one relabelled row from behaving differently."""
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.1, 0.9, 0.9])          # any tau in (0.1, 0.9] is perfect
    decision = optimal_threshold(y, p, PLANT)
    assert 0.1 < decision.threshold <= 0.9
    assert 0.3 < decision.threshold < 0.8, "should sit mid-plateau, not on either cliff"


def test_free_inspection_is_reported_as_degenerate_not_as_a_result() -> None:
    """PROJECT_BRIEF.md §7: honesty as a feature.

    Free inspections make false alarms costless, so every threshold below the
    lowest true-positive score is equally optimal. The policy is degenerate even
    though the threshold it reports looks perfectly ordinary — which is exactly
    why degeneracy is diagnosed from behaviour rather than from the number.
    """
    y, p = _separable()
    decision = optimal_threshold(y, p, PLANT.with_changes(c_inspection=0.0))
    assert decision.degenerate
    assert decision.point.fn == 0, "when misses are the only cost, never miss"
    # The operating point is only weakly constrained: a large share of the sweep
    # ties for optimal, which is the quantity the UI shows as "threshold is not
    # what is driving this decision".
    assert decision.plateau_fraction > 0.25
    assert "inspecting everything" in decision.degenerate_reason


def test_acting_no_cheaper_than_failing_is_reported_as_degenerate() -> None:
    pointless = CostConfig(
        c_unplanned_repair=100_000,
        c_planned_repair=100_000,
        c_inspection=60_000,
        downtime_rate_per_hour=0,
        unplanned_downtime_hours=0,
        planned_downtime_hours=0,
    )
    y, p = _separable()
    decision = optimal_threshold(y, p, pointless)
    assert decision.degenerate
    assert "run-to-failure" in decision.degenerate_reason


def test_frozen_threshold_transfers_to_a_held_out_split() -> None:
    """Choose on validation, report on test. Never re-optimise on test."""
    y_val, p_val = _separable(seed=1)
    y_test, p_test = _separable(seed=2)
    decision = optimal_threshold(y_val, p_val, PLANT)

    frozen = apply_threshold(p_test, decision.threshold)
    test_cost = expected_cost(y_test, frozen, PLANT)
    naive_cost = expected_cost(y_test, apply_threshold(p_test, 0.5), PLANT)
    assert test_cost <= naive_cost
    assert decision.chosen_on == "validation"


def test_uncalibrated_scores_outside_zero_one_are_rejected() -> None:
    with pytest.raises(ValueError, match="calibrate"):
        sweep_thresholds([0, 1], [0.4, 1.7], PLANT)


def test_empty_input_raises_rather_than_returning_a_meaningless_threshold() -> None:
    with pytest.raises(ValueError, match="empty"):
        optimal_threshold([], [], PLANT)


def test_sweep_is_fast_enough_for_the_live_simulator() -> None:
    """PROJECT_BRIEF.md §2.4 budgets 300 ms for the whole recompute; the sweep is
    the expensive part of it."""
    import time

    y, p = _separable(n=20_000, seed=3)
    start = time.perf_counter()
    optimal_threshold(y, p, PLANT)
    assert (time.perf_counter() - start) < 0.25
