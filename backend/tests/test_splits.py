"""Leakage guards — PROJECT_BRIEF.md §6, Traps 1, 2, 4, 5. The Phase 2 gate.

These tests are adversarial on purpose. Each one asks "what would a fraudulent
result look like here?" and then asserts it cannot happen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kairos.data.loaders import (
    AI4I_ID_COLUMNS,
    AI4I_LEAKAGE_COLUMNS,
    RUL_CAP,
    add_cmapss_rul_labels,
    add_cmapss_test_rul_labels,
    load_ai4i,
    load_cmapss,
)
from kairos.data.splits import (
    LeakageError,
    Split,
    group_split,
    group_train_val_test,
    make_splits,
    stratified_train_val_test,
    temporal_train_val_test,
)


@pytest.fixture(scope="module")
def cmapss():
    return load_cmapss()


@pytest.fixture(scope="module")
def ai4i():
    return load_ai4i()


# ------------------------------------------------------- Trap 1: split leakage


def test_no_engine_appears_in_two_splits(cmapss) -> None:
    """The headline guard. A random row split would fail this instantly."""
    split = group_train_val_test(cmapss.frame, "unit_id", seed=0)
    train_units = set(split.train.unit_id)
    val_units = set(split.val.unit_id)
    test_units = set(split.test.unit_id)

    assert train_units & val_units == set()
    assert train_units & test_units == set()
    assert val_units & test_units == set()
    assert train_units | val_units | test_units == set(cmapss.frame.unit_id.unique())


def test_every_engine_lands_somewhere_and_keeps_its_whole_trajectory(cmapss) -> None:
    """A split that truncates trajectories would silently change the RUL problem."""
    split = group_train_val_test(cmapss.frame, "unit_id", seed=3)
    original = cmapss.frame.groupby("unit_id").size()
    for part in (split.train, split.val, split.test):
        for unit, n in part.groupby("unit_id").size().items():
            assert n == original[unit], f"engine {unit} lost rows across the split"


def test_a_random_row_split_is_detected_as_leakage(cmapss) -> None:
    """Prove the guard actually fires — the naive split must be rejected."""
    frame = cmapss.frame
    shuffled = frame.sample(frac=1.0, random_state=0)
    n = len(shuffled) // 3
    naive = Split(
        train=shuffled.iloc[:n],
        val=shuffled.iloc[n : 2 * n],
        test=shuffled.iloc[2 * n :],
        strategy="naive_row_split",
        group_col="unit_id",
    )
    with pytest.raises(LeakageError, match="appear in both"):
        naive.assert_no_group_leakage()


def test_group_split_refuses_a_single_group() -> None:
    frame = pd.DataFrame({"unit_id": [1] * 20, "x": range(20)})
    with pytest.raises(ValueError, match="at least 2 groups"):
        group_split(frame, "unit_id")


def test_policy_test_set_holds_complete_trajectories_not_the_truncated_holdout(cmapss) -> None:
    """Two questions, two test sets.

    The published C-MAPSS test set stops recording before failure, so a lead-time
    policy evaluated on it eats an unplanned failure for every engine whose data
    ran out — 81 of 100 at a 25-cycle lead time. Cost must be simulated on complete
    run-to-failure trajectories; the truncated set is kept only for accuracy
    metrics comparable to published results.
    """
    split = make_splits(cmapss)
    assert split.strategy == "group_shuffle_complete_trajectories"

    # Every policy-test engine runs to failure: its last cycle has zero life left.
    last = split.test.sort_values("cycle").groupby("unit_id").tail(1)
    assert (last.rul == 0).all()

    # The published holdout is carried, unused for cost, and is genuinely truncated.
    assert split.reference_holdout is not None
    ref_last = split.reference_holdout.sort_values("cycle").groupby("unit_id").tail(1)
    assert (ref_last.rul > 0).all()

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert set(getattr(split, a).unit_id) & set(getattr(split, b).unit_id) == set()


def test_the_truncation_artefact_is_real_and_measured(cmapss) -> None:
    """Documents the number that justifies the split design above."""
    holdout = cmapss.holdout
    final_rul = holdout.groupby("unit_id").rul.min()
    never_alert = int((final_rul > 25).sum())
    assert never_alert >= 70, (
        "if this drops, the truncation artefact has changed and the split design "
        "should be revisited"
    )


# ----------------------------------------------------------- Trap 2: RUL labels


def test_rul_is_capped_piecewise_linearly(cmapss) -> None:
    assert cmapss.frame.rul.max() == RUL_CAP
    assert cmapss.frame.rul.min() == 0


def test_training_rul_hits_zero_at_the_final_cycle(cmapss) -> None:
    """Training trajectories run to failure, so the last cycle has zero life left."""
    last = cmapss.frame.sort_values("cycle").groupby("unit_id").tail(1)
    assert (last.rul == 0).all()


def test_test_rul_does_not_hit_zero_because_trajectories_are_truncated(cmapss) -> None:
    """The single most-fumbled detail in C-MAPSS. If these labels reach 0 at the
    last recorded cycle, the loader has mistaken truncation for failure."""
    last = cmapss.holdout.sort_values("cycle").groupby("unit_id").tail(1)
    assert (last.rul > 0).all(), "test labels must include the unobserved remaining life"


def test_test_labels_combine_the_separate_ground_truth_file() -> None:
    """Explicit arithmetic check on a hand-built case."""
    frame = pd.DataFrame({"unit_id": [1, 1, 1], "cycle": [1, 2, 3]})
    labelled = add_cmapss_test_rul_labels(frame, pd.Series({1: 10.0}))
    # Last recorded cycle is 3 with 10 cycles of life left, so cycle 1 has 12.
    assert list(labelled.rul) == [12.0, 11.0, 10.0]


def test_missing_ground_truth_for_a_test_unit_is_an_error() -> None:
    frame = pd.DataFrame({"unit_id": [1, 2], "cycle": [1, 1]})
    with pytest.raises(ValueError, match="no final RUL"):
        add_cmapss_test_rul_labels(frame, pd.Series({1: 10.0}))


def test_uncapped_rul_would_exceed_the_cap() -> None:
    """Sanity check that the cap is doing work rather than being decorative."""
    frame = pd.DataFrame({"unit_id": [1] * 300, "cycle": range(1, 301)})
    assert add_cmapss_rul_labels(frame, cap=125).rul.max() == 125
    assert add_cmapss_rul_labels(frame, cap=10_000).rul.max() == 299


# --------------------------------------------------------- Trap 4: target leak


def test_ai4i_ships_the_leakage_columns_so_the_profiler_can_catch_them(ai4i) -> None:
    """They are loaded on purpose. Hiding them would make the agent finding fake."""
    for col in AI4I_LEAKAGE_COLUMNS:
        assert col in ai4i.frame.columns
    assert set(ai4i.leakage_candidates) == set(AI4I_LEAKAGE_COLUMNS)


def test_the_leakage_columns_really_do_compose_the_target(ai4i) -> None:
    """Justifies calling them leakage rather than merely correlated features.

    Machine failure is defined as the OR of the failure-mode flags, so a model
    keeping them scores near-perfectly while learning nothing.
    """
    flags = ai4i.frame[["twf", "hdf", "pwf", "osf"]].max(axis=1)
    agreement = (flags == ai4i.frame.machine_failure).mean()
    assert agreement > 0.99, f"expected the flags to reconstruct the target, got {agreement:.4f}"


def test_feature_columns_exclude_identifiers_and_the_target(ai4i) -> None:
    features = ai4i.feature_columns(exclude=ai4i.leakage_candidates)
    for banned in [*AI4I_ID_COLUMNS, *AI4I_LEAKAGE_COLUMNS, "machine_failure"]:
        assert banned not in features
    assert "torque_nm" in features and "tool_wear_min" in features


# ------------------------------------------------------------ Trap 5: imbalance


def test_stratified_split_preserves_the_positive_rate(ai4i) -> None:
    """At 3.4% positives an unstratified split can hand a fold almost none."""
    split = stratified_train_val_test(ai4i.frame, "machine_failure", seed=0)
    overall = ai4i.frame.machine_failure.mean()
    for part in (split.train, split.val, split.test):
        assert part.machine_failure.mean() == pytest.approx(overall, abs=0.01)
        assert part.machine_failure.sum() > 0, "every split must contain positives"


def test_splits_partition_the_rows_exactly(ai4i) -> None:
    split = stratified_train_val_test(ai4i.frame, "machine_failure", seed=1)
    assert sum(split.sizes().values()) == len(ai4i.frame)
    indices = set(split.train.index) | set(split.val.index) | set(split.test.index)
    assert len(indices) == len(ai4i.frame), "a row must land in exactly one split"


def test_a_single_class_target_is_rejected() -> None:
    frame = pd.DataFrame({"y": [0] * 50, "x": range(50)})
    with pytest.raises(ValueError, match="single class"):
        stratified_train_val_test(frame, "y")


# ------------------------------------------------------------------- temporal


def test_temporal_split_never_trains_on_the_future() -> None:
    frame = pd.DataFrame({"t": np.arange(500), "x": np.random.default_rng(0).random(500)})
    split = temporal_train_val_test(frame, "t")
    assert split.train.t.max() < split.val.t.min() < split.test.t.min()


def test_temporal_split_refuses_a_frame_too_small_to_divide() -> None:
    with pytest.raises(ValueError, match="not enough rows"):
        temporal_train_val_test(pd.DataFrame({"t": [1, 2]}), "t")


# --------------------------------------------------------------- reproducibility


def test_splits_are_deterministic_for_a_given_seed(cmapss) -> None:
    """A demo that reshuffles between runs cannot be reasoned about on stage."""
    a = group_train_val_test(cmapss.frame, "unit_id", seed=7)
    b = group_train_val_test(cmapss.frame, "unit_id", seed=7)
    assert set(a.test.unit_id) == set(b.test.unit_id)


def test_different_seeds_really_do_produce_different_splits(cmapss) -> None:
    a = group_train_val_test(cmapss.frame, "unit_id", seed=7)
    b = group_train_val_test(cmapss.frame, "unit_id", seed=8)
    assert set(a.test.unit_id) != set(b.test.unit_id)
