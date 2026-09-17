"""Training core tests — PROJECT_BRIEF.md §10 Phase 3.

Fast by construction: these test the training *machinery* — budgets, guards,
calibration wiring, the accuracy ban — on small synthetic frames. Whether the
models are any good is answered by scripts/run_pipeline.py against real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kairos.data.splits import Split
from kairos.ml.calibrate import (
    brier_score,
    calibrate_probabilities,
    choose_method,
    expected_calibration_error,
    reliability_bins,
)
from kairos.ml.evaluate import (
    assert_no_accuracy,
    beats_baseline,
    classification_metrics,
    nasa_score,
    recall_at_precision,
    regression_metrics,
)
from kairos.ml.registry import candidates_for, get_candidate, scale_pos_weight_for
from kairos.ml.train import train_all, train_candidate


def _classification_split(n: int = 900, seed: int = 0) -> Split:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.15).astype(int)
    frame = pd.DataFrame({
        "f1": rng.normal(np.where(y == 1, 2.0, 0.0), 1.0),
        "f2": rng.normal(np.where(y == 1, -1.0, 0.0), 1.0),
        "noise": rng.normal(0, 1, n),
        "y": y,
    })
    a, b = n // 3, 2 * n // 3
    return Split(train=frame.iloc[:a], val=frame.iloc[a:b], test=frame.iloc[b:],
                 strategy="test_fixture", target="y")


def _regression_split(n_units: int = 30, life: int = 60, seed: int = 0) -> Split:
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        for c in range(1, life + 1):
            wear = c / life
            rows.append({
                "unit_id": u, "cycle": c,
                "s1": 100 - 40 * wear + rng.normal(0, 1),
                "s2": 50 + 20 * wear + rng.normal(0, 1),
                "rul": float(life - c),
            })
    frame = pd.DataFrame(rows)
    train = frame[frame.unit_id < 18]
    val = frame[(frame.unit_id >= 18) & (frame.unit_id < 24)]
    test = frame[frame.unit_id >= 24]
    return Split(train=train, val=val, test=test, strategy="test_fixture",
                 group_col="unit_id", target="rul")


# ------------------------------------------------------------------ registry


def test_every_task_offers_a_baseline_and_both_boosters() -> None:
    for task in ("binary_classification", "rul_regression"):
        families = {c.family for c in candidates_for(task)}
        assert "baseline" in families and "boosting" in families
        keys = {c.key for c in candidates_for(task)}
        assert {"lightgbm", "xgboost"} <= keys


def test_scale_pos_weight_matches_the_class_ratio() -> None:
    y = np.array([0] * 966 + [1] * 34)
    assert scale_pos_weight_for(y) == pytest.approx(966 / 34)


def test_scale_pos_weight_is_safe_when_there_are_no_positives() -> None:
    assert scale_pos_weight_for(np.zeros(50)) == 1.0


def test_unknown_candidate_raises() -> None:
    with pytest.raises(KeyError):
        get_candidate("binary_classification", "not_a_model")


# ------------------------------------------------------------------- metrics


def test_accuracy_is_never_reported() -> None:
    """PROJECT_BRIEF.md §11, enforced rather than remembered."""
    y = np.array([0, 1, 0, 1, 0, 0, 1, 0])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.1, 0.7, 0.4])
    assert_no_accuracy(classification_metrics(y, p, (p >= 0.5).astype(int)))
    assert_no_accuracy(regression_metrics(np.array([1.0, 2.0]), np.array([1.1, 2.1])))


def test_assert_no_accuracy_actually_fires() -> None:
    with pytest.raises(AssertionError, match="accuracy"):
        assert_no_accuracy({"accuracy": 0.97})


def test_classification_metrics_carry_pr_auc_and_recall_at_precision() -> None:
    y = np.array([0] * 90 + [1] * 10)
    p = np.concatenate([np.linspace(0.0, 0.5, 90), np.linspace(0.6, 0.99, 10)])
    m = classification_metrics(y, p)
    assert m["pr_auc"] > 0.9 and m["roc_auc"] > 0.9
    assert m["recall_at_p50"] == pytest.approx(1.0)


def test_recall_at_precision_is_zero_when_the_floor_is_unreachable() -> None:
    y = np.array([0] * 99 + [1])
    p = np.full(100, 0.5)  # no ranking information at all
    assert recall_at_precision(y, p, target_precision=0.9) == 0.0


def test_nasa_score_punishes_late_predictions_harder_than_early_ones() -> None:
    """The asymmetry is the point: late means the engine already broke."""
    truth = np.array([50.0])
    early = nasa_score(truth, np.array([40.0]))   # predicts failure sooner than reality
    late = nasa_score(truth, np.array([60.0]))    # predicts failure later than reality
    assert late > early


def test_regression_metrics_report_direction_of_error() -> None:
    truth = np.array([10.0, 20.0, 30.0])
    optimistic = np.array([15.0, 25.0, 35.0])     # always predicts more life than exists
    m = regression_metrics(truth, optimistic)
    assert m["mean_bias"] > 0 and m["late_fraction"] == 1.0


def test_quality_gate_rejects_a_model_that_cannot_beat_the_mean() -> None:
    assert not beats_baseline({"rmse": 40.0}, {"rmse": 35.0}, "rul_regression")
    assert beats_baseline({"rmse": 12.0}, {"rmse": 35.0}, "rul_regression")
    assert not beats_baseline({"pr_auc": 0.4}, {}, "binary_classification")
    assert beats_baseline({"pr_auc": 0.8}, {}, "binary_classification")


# --------------------------------------------------------------- calibration


def test_isotonic_needs_data_and_platt_does_not() -> None:
    plenty = np.array([1] * 500 + [0] * 500)
    scarce = np.array([1] * 20 + [0] * 5000)
    assert choose_method(plenty) == "isotonic"
    assert choose_method(scarce) == "sigmoid"


def test_calibration_improves_a_deliberately_overconfident_model() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.3).astype(int)
    honest = np.where(y == 1, rng.beta(6, 4, 4000), rng.beta(4, 6, 4000))
    overconfident = np.clip((honest - 0.5) * 3.0 + 0.5, 0.001, 0.999)

    _, report, _ = calibrate_probabilities(y, overconfident)
    assert report.brier_after < report.brier_before
    assert report.ece_after < report.ece_before


def test_reliability_bins_omit_empty_bins_rather_than_plotting_zero() -> None:
    y = np.array([0, 1, 0, 1])
    p = np.array([0.05, 0.95, 0.05, 0.95])
    bins = reliability_bins(y, p, n_bins=10)
    assert len(bins) == 2
    assert all(b["count"] > 0 for b in bins)


def test_ece_is_zero_for_a_perfectly_calibrated_model() -> None:
    p = np.concatenate([np.full(1000, 0.25), np.full(1000, 0.75)])
    y = np.concatenate([
        np.array([1] * 250 + [0] * 750),
        np.array([1] * 750 + [0] * 250),
    ])
    assert expected_calibration_error(y, p, n_bins=4) < 0.01


def test_calibration_failure_returns_raw_probabilities_rather_than_raising() -> None:
    """A calibration problem must never take down a run."""
    y = np.array([1, 1, 1, 1])  # single class: nothing to calibrate against
    p = np.array([0.6, 0.7, 0.8, 0.9])
    out, report, _ = calibrate_probabilities(y, p)
    assert len(out) == 4
    assert report.method in {"none", "sigmoid", "isotonic"}


# ------------------------------------------------------------------ training


def test_a_trained_classifier_caches_predictions_for_both_splits() -> None:
    split = _classification_split()
    trial = train_candidate(
        get_candidate("binary_classification", "logistic"),
        split, ["f1", "f2", "noise"], "y", "binary_classification", run_cv=False,
    )
    assert trial.ok
    assert len(trial.val_prediction) == len(split.val)
    assert len(trial.test_prediction) == len(split.test)
    assert trial.calibration is not None


def test_predictions_are_probabilities_so_thresholds_mean_something() -> None:
    split = _classification_split()
    trial = train_candidate(
        get_candidate("binary_classification", "lightgbm"),
        split, ["f1", "f2", "noise"], "y", "binary_classification", run_cv=False,
    )
    assert trial.val_prediction.min() >= 0.0 and trial.val_prediction.max() <= 1.0


def test_a_broken_candidate_is_recorded_as_a_finding_not_a_crash() -> None:
    """One dead model must never take down a run in front of a judge."""
    split = _classification_split()
    trial = train_candidate(
        get_candidate("binary_classification", "logistic"),
        split, ["column_that_does_not_exist"], "y", "binary_classification", run_cv=False,
    )
    assert not trial.ok
    assert trial.error


def test_group_cv_runs_on_grouped_regression_data() -> None:
    split = _regression_split()
    trial = train_candidate(
        get_candidate("rul_regression", "ridge"),
        split, ["s1", "s2", "cycle"], "rul", "rul_regression", group_col="unit_id",
    )
    assert trial.ok
    assert any(k.startswith("cv_") for k in trial.cv_metrics)


def test_a_real_model_beats_the_mean_baseline_on_learnable_data() -> None:
    split = _regression_split()
    report = train_all(
        split, ["s1", "s2", "cycle"], "rul", "rul_regression",
        group_col="unit_id", budget_seconds=60, only=["baseline_mean", "ridge"], run_cv=False,
    )
    by_key = {t.candidate_key: t for t in report.successful}
    assert by_key["ridge"].metrics["rmse"] < by_key["baseline_mean"].metrics["rmse"]


def test_budget_exhaustion_still_leaves_a_usable_leaderboard() -> None:
    """Cheapest families run first precisely so this is true."""
    split = _classification_split()
    report = train_all(
        split, ["f1", "f2"], "y", "binary_classification",
        budget_seconds=0.0001, run_cv=False,
    )
    assert report.budget_exhausted
    assert any("budget" in f.lower() for f in report.findings)


def test_training_order_puts_baselines_before_boosters() -> None:
    split = _classification_split()
    report = train_all(split, ["f1", "f2"], "y", "binary_classification",
                       budget_seconds=120, run_cv=False)
    families = [t.family for t in report.trials]
    assert families.index("baseline") < families.index("boosting")


def test_on_trial_callback_fires_per_candidate_for_sse_streaming() -> None:
    seen: list[str] = []
    split = _classification_split()
    train_all(
        split, ["f1", "f2"], "y", "binary_classification", budget_seconds=120,
        only=["baseline_majority", "logistic"], run_cv=False,
        on_trial=lambda t: seen.append(t.candidate_key),
    )
    assert seen == ["baseline_majority", "logistic"]
