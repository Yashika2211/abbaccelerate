"""Feature engineering and profiler tests — PROJECT_BRIEF.md §6.1–§6.3."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kairos.data.features import (
    drop_columns,
    engineer_ai4i_features,
    engineer_timeseries_features,
)
from kairos.data.loaders import load_ai4i, load_cmapss
from kairos.data.profiler import detect_leakage, profile_dataset


@pytest.fixture(scope="module")
def cmapss():
    return load_cmapss()


@pytest.fixture(scope="module")
def ai4i():
    return load_ai4i()


# ------------------------------------------------- per-unit windows, not global


def test_rolling_windows_never_cross_a_unit_boundary() -> None:
    """The trap in one test: a global window would blend unit 1 into unit 2.

    Unit 1 sits at 100 throughout, unit 2 at 0. A correctly grouped rolling mean
    gives unit 2's first row exactly 0. A global window gives it something near 80.
    """
    frame = pd.DataFrame({
        "unit_id": [1] * 5 + [2] * 5,
        "cycle": list(range(1, 6)) * 2,
        "sensor_1": [100.0] * 5 + [0.0] * 5,
    })
    result = engineer_timeseries_features(frame, "unit_id", "cycle", ["sensor_1"], windows=(5,))
    unit2 = result.frame[result.frame.unit_id == 2].sort_values("cycle")

    assert unit2.sensor_1_roll5_mean.iloc[0] == 0.0, "unit 1 leaked into unit 2's window"
    assert (unit2.sensor_1_roll5_mean == 0.0).all()
    assert (result.frame[result.frame.unit_id == 1].sensor_1_roll5_mean == 100.0).all()


def test_first_difference_resets_at_each_unit() -> None:
    frame = pd.DataFrame({
        "unit_id": [1, 1, 2, 2],
        "cycle": [1, 2, 1, 2],
        "sensor_1": [10.0, 12.0, 500.0, 503.0],
    })
    out = engineer_timeseries_features(frame, "unit_id", "cycle", ["sensor_1"], windows=(2,)).frame
    diffs = out.sort_values(["unit_id", "cycle"]).sensor_1_diff1.tolist()
    # Unit 2's first row must be 0 (filled), not 488 — which is what a global diff gives.
    assert diffs == [0.0, 2.0, 0.0, 3.0]


def test_cycles_elapsed_counts_within_a_unit() -> None:
    frame = pd.DataFrame({
        "unit_id": [1, 1, 1, 2, 2],
        "cycle": [1, 2, 3, 1, 2],
        "sensor_1": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    out = engineer_timeseries_features(frame, "unit_id", "cycle", ["sensor_1"], windows=(2,)).frame
    assert out.sort_values(["unit_id", "cycle"]).cycles_elapsed.tolist() == [1, 2, 3, 1, 2]


def test_delta_from_start_is_zero_on_each_units_first_row() -> None:
    frame = pd.DataFrame({
        "unit_id": [1, 1, 2, 2],
        "cycle": [1, 2, 1, 2],
        "sensor_1": [10.0, 15.0, 100.0, 90.0],
    })
    out = engineer_timeseries_features(frame, "unit_id", "cycle", ["sensor_1"], windows=(2,)).frame
    out = out.sort_values(["unit_id", "cycle"])
    assert out.sensor_1_delta_from_start.tolist() == [0.0, 5.0, 0.0, -10.0]


def test_trend_slope_has_the_right_sign_and_magnitude() -> None:
    """A sensor rising by exactly 2 per cycle must produce a slope of 2."""
    frame = pd.DataFrame({
        "unit_id": [1] * 10,
        "cycle": range(1, 11),
        "sensor_1": [float(2 * i) for i in range(10)],
    })
    out = engineer_timeseries_features(
        frame, "unit_id", "cycle", ["sensor_1"], windows=(5,), add_trend=True
    ).frame
    assert out.sensor_1_trend5.iloc[-1] == pytest.approx(2.0)


def test_no_nan_survives_feature_engineering(cmapss) -> None:
    """A NaN reaching the model is a crash on stage."""
    sensors = [f"sensor_{i}" for i in (2, 3, 4, 7, 11)]
    subset = cmapss.frame[cmapss.frame.unit_id <= 5]
    out = engineer_timeseries_features(subset, "unit_id", "cycle", sensors).frame
    assert not out.isna().any().any()


def test_engineering_preserves_row_count(cmapss) -> None:
    subset = cmapss.frame[cmapss.frame.unit_id <= 10]
    out = engineer_timeseries_features(subset, "unit_id", "cycle", ["sensor_2", "sensor_11"])
    assert len(out.frame) == len(subset)


def test_missing_group_column_is_rejected() -> None:
    with pytest.raises(KeyError, match="group column"):
        engineer_timeseries_features(pd.DataFrame({"a": [1]}), "unit_id", "cycle", ["a"])


# ----------------------------------------------------------- AI4I physics


def test_power_matches_the_textbook_formula(ai4i) -> None:
    out = engineer_ai4i_features(ai4i.frame).frame
    expected = ai4i.frame.torque_nm * ai4i.frame.rotational_speed_rpm * 2 * np.pi / 60.0
    pd.testing.assert_series_equal(out.power_w, expected, check_names=False)


def test_physics_features_are_all_added(ai4i) -> None:
    result = engineer_ai4i_features(ai4i.frame)
    assert set(result.added_columns) == {"power_w", "temp_delta_k", "wear_x_torque", "type_ordinal"}


def test_product_quality_is_treated_as_ordinal_not_nominal(ai4i) -> None:
    out = engineer_ai4i_features(ai4i.frame).frame
    assert set(out.type_ordinal.unique()) <= {0, 1, 2}


def test_engineered_features_do_not_leak(ai4i) -> None:
    """Physics features must be predictive, not a restatement of the answer."""
    out = engineer_ai4i_features(ai4i.frame).frame
    warnings = detect_leakage(
        out, "machine_failure", "binary_classification",
        exclude=["udi", "product_id", "twf", "hdf", "pwf", "osf", "rnf"],
    )
    assert [w.column for w in warnings] == [], f"engineered features flagged: {warnings}"


# --------------------------------------------------------------- profiler


def test_profiler_convicts_the_ai4i_leakage_columns(ai4i) -> None:
    convicted = {w.column for w in profile_dataset(ai4i).leakage_warnings}
    assert {"twf", "hdf", "pwf", "osf"} <= convicted


def test_profiler_does_not_accuse_genuine_sensors(ai4i) -> None:
    """The false-positive guard. Mutual information fails this test; lift passes it."""
    convicted = {w.column for w in profile_dataset(ai4i).leakage_warnings}
    for honest in ("torque_nm", "rotational_speed_rpm", "tool_wear_min", "air_temp_k"):
        assert honest not in convicted, f"{honest} is a real sensor, not a leak"


def test_rnf_is_separated_as_provenance_not_evidence(ai4i) -> None:
    """RNF is a random-failure flag that often does not set the target, so no
    statistical test can convict it. It must be dropped, but honestly labelled."""
    profile = profile_dataset(ai4i)
    assert "rnf" in profile.documented_leakage_undetected
    assert "rnf" not in {w.column for w in profile.leakage_warnings}
    assert "rnf" in profile.recommended_drops


def test_profiler_finds_every_dead_cmapss_column(cmapss) -> None:
    """The brief names 7 dead sensors; op_setting_3 is dead too, so 8."""
    constant = set(profile_dataset(cmapss).constant_columns)
    for i in (1, 5, 6, 10, 16, 18, 19):
        assert f"sensor_{i}" in constant, f"sensor_{i} should be detected as dead"
    assert "op_setting_3" in constant
    assert len(constant) == 8


def test_profiler_catches_the_near_constant_sensor(cmapss) -> None:
    """sensor_6 holds two distinct values but 98% of rows share one. A pure
    standard-deviation test misses it; dominance catches it."""
    assert cmapss.frame.sensor_6.nunique() == 2
    assert "sensor_6" in profile_dataset(cmapss).constant_columns


def test_profiler_proposes_the_right_task_for_each_dataset(cmapss, ai4i) -> None:
    assert profile_dataset(cmapss).proposed_task_type == "rul_regression"
    assert profile_dataset(ai4i).proposed_task_type == "binary_classification"


def test_profiler_reports_class_balance_and_warns_about_accuracy(ai4i) -> None:
    profile = profile_dataset(ai4i)
    assert profile.class_balance["1"] == pytest.approx(0.0339, abs=0.001)
    assert any("Accuracy is meaningless" in f for f in profile.findings)


def test_profiler_identifies_group_and_time_columns(cmapss) -> None:
    profile = profile_dataset(cmapss)
    assert "unit_id" in profile.candidate_group_columns
    assert "cycle" in profile.candidate_time_columns


def test_drop_columns_reports_exactly_what_it_removed(ai4i) -> None:
    result = drop_columns(ai4i.frame, ["twf", "hdf", "does_not_exist"])
    assert result.dropped_columns == ["twf", "hdf"]
    assert "twf" not in result.frame.columns
