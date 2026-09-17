"""Cost engine tests — PROJECT_BRIEF.md §3, Phase 1 gate.

Every headline number in Kairos comes out of kairos.decision.cost, so this file
is written before a single model is trained. A wrong cost matrix does not show up
as a crash; it shows up as a confident, wrong recommendation on a projector.
"""

from __future__ import annotations

import numpy as np
import pytest

from kairos.decision.cost import (
    CostConfig,
    annualize,
    classification_cost_matrix,
    confusion_counts,
    cost_breakdown,
    expected_cost,
    rul_lead_time_sweep,
    rul_policy_cost,
)

# A realistic Indian plant, matching the demo script in PROJECT_BRIEF.md §12:
# unplanned failure ~Rs 8L, planned repair Rs 1.2L, downtime Rs 45k/hour.
PLANT = CostConfig(
    c_unplanned_repair=800_000,
    c_planned_repair=120_000,
    c_inspection=15_000,
    downtime_rate_per_hour=45_000,
    unplanned_downtime_hours=12,
    planned_downtime_hours=3,
    c_secondary_damage=0,
    part_lead_time_cycles=5,
    value_per_remaining_cycle=2_000,
)


# --------------------------------------------------------------- cost matrix


def test_cost_matrix_follows_the_brief_exactly() -> None:
    m = classification_cost_matrix(PLANT)
    assert m.tn == 0
    assert m.fp == 15_000
    assert m.tp == 120_000 + 3 * 45_000          # 255,000
    assert m.fn == 800_000 + 12 * 45_000         # 1,340,000


def test_missing_a_failure_dominates_every_other_outcome() -> None:
    """The ordering FN > TP > FP > TN=0 is the whole economic argument."""
    m = classification_cost_matrix(PLANT)
    assert m.fn > m.tp > m.fp > m.tn == 0


def test_consequence_ratio_is_the_number_that_explains_the_product() -> None:
    assert PLANT.consequence_ratio == pytest.approx(1_340_000 / 15_000)
    assert PLANT.consequence_ratio > 40  # missing one failure is worth 89 false alarms here


def test_free_inspection_gives_an_infinite_consequence_ratio() -> None:
    assert PLANT.with_changes(c_inspection=0).consequence_ratio == float("inf")


def test_secondary_damage_lands_only_on_false_negatives() -> None:
    cfg = PLANT.with_changes(c_secondary_damage=500_000)
    m = classification_cost_matrix(cfg)
    assert m.fn == classification_cost_matrix(PLANT).fn + 500_000
    assert m.tp == classification_cost_matrix(PLANT).tp


# ------------------------------------------------------------ expected cost


def test_confusion_counts_are_correct() -> None:
    counts = confusion_counts([1, 0, 1, 0, 1], [1, 0, 0, 1, 1])
    assert counts == {"tp": 2, "tn": 1, "fp": 1, "fn": 1}


def test_perfect_predictions_still_cost_money() -> None:
    """A perfectly predicted failure is a planned repair, not a free lunch."""
    y = [1, 0, 1, 0]
    assert expected_cost(y, y, PLANT) == 2 * classification_cost_matrix(PLANT).tp


def test_run_to_failure_costs_one_unplanned_event_per_positive() -> None:
    y_true = [1, 0, 1, 0, 0]
    never_act = [0, 0, 0, 0, 0]
    assert expected_cost(y_true, never_act, PLANT) == 2 * 1_340_000


def test_always_inspect_costs_an_inspection_per_negative() -> None:
    y_true = [1, 0, 1, 0, 0]
    always_act = [1, 1, 1, 1, 1]
    expected = 2 * 255_000 + 3 * 15_000
    assert expected_cost(y_true, always_act, PLANT) == expected


def test_mismatched_lengths_are_rejected_loudly() -> None:
    with pytest.raises(ValueError, match="same length"):
        expected_cost([1, 0, 1], [1, 0], PLANT)


def test_breakdown_contributions_reconstruct_the_total() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 500)
    y_pred = rng.integers(0, 2, 500)
    b = cost_breakdown(y_true, y_pred, PLANT)
    assert sum(b["contributions"].values()) == pytest.approx(b["total_cost"])
    assert b["total_cost"] == pytest.approx(expected_cost(y_true, y_pred, PLANT))


def test_annualization_states_its_own_denominator() -> None:
    """10,000 observations == 1 asset-year, so 10,000 rows annualise 1:1."""
    y_true = np.zeros(10_000, dtype=int)
    y_pred = np.zeros(10_000, dtype=int)
    y_true[:100] = 1  # 100 genuine failures, all missed
    b = cost_breakdown(y_true, y_pred, PLANT)
    assert b["exposure_asset_years"] == pytest.approx(1.0)
    assert b["cost_per_asset_year"] == pytest.approx(b["total_cost"])
    assert "asset-year" in b["annualization_note"]


def test_empty_exposure_returns_zero_instead_of_dividing_by_zero() -> None:
    """An empty fold renders an empty state; it does not crash the demo."""
    assert annualize(1_000.0, 0.0) == 0.0


# ------------------------------------------------------------ config guards


def test_negative_costs_are_rejected() -> None:
    with pytest.raises(ValueError, match="c_inspection must be >= 0"):
        PLANT.with_changes(c_inspection=-1)


def test_config_is_frozen_so_a_simulation_cannot_corrupt_the_baseline() -> None:
    variant = PLANT.with_changes(c_unplanned_repair=2_000_000)
    assert PLANT.c_unplanned_repair == 800_000       # original untouched
    assert variant.c_unplanned_repair == 2_000_000
    with pytest.raises(Exception):
        PLANT.c_inspection = 1  # type: ignore[misc]


# --------------------------------------------------------------------- RUL


def _perfect_fleet(n_units: int = 3, life: int = 100):
    """Units that fail at `life`, with a model that predicts RUL exactly right."""
    unit_ids, cycles, pred = [], [], []
    for u in range(n_units):
        for t in range(1, life + 1):
            unit_ids.append(f"u{u}")
            cycles.append(t)
            pred.append(life - t)
    failure = {f"u{u}": float(life) for u in range(n_units)}
    return unit_ids, cycles, pred, failure


def test_alerting_in_time_buys_a_planned_repair_and_wastes_the_rest_of_the_life() -> None:
    units, cycles, pred, failure = _perfect_fleet(n_units=1)
    # L=20: alert at cycle 80, +5 cycles lead time -> act at 85, 15 cycles of life binned.
    r = rul_policy_cost(units, cycles, pred, failure, lead_time=20, cfg=PLANT)
    assert r.caught == 1 and r.missed == 0
    assert r.wasted_life_cycles == pytest.approx(15.0)
    assert r.total_cost == pytest.approx(PLANT.cost_planned_event + 15 * 2_000)


def test_part_lead_time_can_turn_a_correct_alert_into_an_unplanned_failure() -> None:
    """The trap the brief calls out: an alert that lands too late is worth nothing."""
    units, cycles, pred, failure = _perfect_fleet(n_units=1)
    # L=3 alerts at cycle 97; parts take 5 cycles; the engine fails at 100 first.
    r = rul_policy_cost(units, cycles, pred, failure, lead_time=3, cfg=PLANT)
    assert r.caught == 0 and r.missed == 1 and r.never_alerted == 0
    assert r.total_cost == pytest.approx(PLANT.cost_unplanned_event)


def test_a_silent_model_is_charged_the_full_unplanned_cost() -> None:
    units, cycles, pred, failure = _perfect_fleet(n_units=2)
    pred = [999.0] * len(pred)  # model never drops below any sane lead time
    r = rul_policy_cost(units, cycles, pred, failure, lead_time=30, cfg=PLANT)
    assert r.never_alerted == 2
    assert r.total_cost == pytest.approx(2 * PLANT.cost_unplanned_event)


def test_lead_time_curve_is_u_shaped_with_an_interior_optimum() -> None:
    """The hero chart. With perfect predictions the optimum is exactly one cycle
    more than the part lead time: early enough to get the part, not a cycle earlier."""
    units, cycles, pred, failure = _perfect_fleet(n_units=5)
    sweep = rul_lead_time_sweep(units, cycles, pred, failure, PLANT, lead_times=range(0, 41))
    costs = [r.total_cost for r in sweep]
    best = min(range(len(costs)), key=costs.__getitem__)

    assert best == PLANT.part_lead_time_cycles + 1 == 6
    assert 0 < best < len(costs) - 1, "optimum must be interior, not at a grid edge"
    assert costs[0] > costs[best], "alerting too late costs unplanned failures"
    assert costs[-1] > costs[best], "alerting too early throws away good life"


def test_missing_ground_truth_for_a_unit_is_an_error_not_a_silent_zero() -> None:
    units, cycles, pred, failure = _perfect_fleet(n_units=2)
    del failure["u1"]
    with pytest.raises(KeyError, match="u1"):
        rul_policy_cost(units, cycles, pred, failure, lead_time=20, cfg=PLANT)


# ------------------------------------------------------ degenerate economics


def test_when_failure_is_no_worse_than_a_repair_the_model_is_worthless() -> None:
    """PROJECT_BRIEF.md §7: honesty as a feature. If an unplanned failure costs the
    same as a planned one, acting early only burns inspections and good life."""
    flat = CostConfig(
        c_unplanned_repair=100_000,
        c_planned_repair=100_000,
        c_inspection=15_000,
        downtime_rate_per_hour=0,
        unplanned_downtime_hours=0,
        planned_downtime_hours=0,
    )
    y_true = [1, 0, 1, 0, 0, 0]
    never_act = [0] * 6
    always_act = [1] * 6
    assert expected_cost(y_true, never_act, flat) < expected_cost(y_true, always_act, flat)


def test_when_inspection_is_free_acting_on_everything_is_optimal() -> None:
    free = PLANT.with_changes(c_inspection=0.0)
    y_true = [1, 0, 1, 0, 0, 0]
    always_act = [1] * 6
    never_act = [0] * 6
    assert expected_cost(y_true, always_act, free) < expected_cost(y_true, never_act, free)


def test_zero_value_per_remaining_cycle_removes_the_u_and_flattens_the_curve() -> None:
    """Without a scrap-value penalty there is no cost to alerting early, so the
    curve stops being U-shaped. The simulator must not assume an interior optimum."""
    cfg = PLANT.with_changes(value_per_remaining_cycle=0.0)
    units, cycles, pred, failure = _perfect_fleet(n_units=3)
    sweep = rul_lead_time_sweep(units, cycles, pred, failure, cfg, lead_times=range(6, 41))
    costs = {round(r.total_cost, 6) for r in sweep}
    assert len(costs) == 1, "with no scrap penalty every sufficient lead time costs the same"
