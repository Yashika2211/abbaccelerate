"""Leaderboard and what-if simulator tests — PROJECT_BRIEF.md §2.3, §2.4."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from kairos.decision.simulate import (
    ClassifierPredictions,
    simulate_classification,
)
from tests.test_cost import PLANT


def _model(model_id: str, name: str, y: np.ndarray, p: np.ndarray, **metrics) -> ClassifierPredictions:
    """Same arrays for validation and test keeps these tests about the ranking logic."""
    return ClassifierPredictions(
        model_id=model_id,
        model_name=name,
        val_y=y,
        val_prob=p,
        test_y=y,
        test_prob=p,
        metrics=metrics,
    )


def _two_models_with_a_real_inversion() -> tuple[ClassifierPredictions, ClassifierPredictions]:
    """Construct the demo's headline honestly, from a mechanism that really happens.

    "Precise" ranks better on average but writes off a handful of assets entirely.
    "Recall-first" catches every failure at the price of extra inspections. When a
    miss costs 89x a false alarm, abandoning five engines is not a trade you can
    afford — no matter what the AUC column says.
    """
    rng = np.random.default_rng(42)
    n_pos, n_neg = 100, 900
    y = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])

    # Higher AUC: 95 positives ranked at the top, 5 written off completely.
    precise = np.concatenate([
        np.full(95, 0.90),
        np.full(5, 0.00),
        rng.uniform(0.0, 0.80, n_neg),
    ])

    # Lower AUC: every positive flagged, but 0.5 sits inside the negative spread.
    recall_first = np.concatenate([
        np.full(n_pos, 0.50),
        rng.uniform(0.0, 0.60, n_neg),
    ])

    auc_precise = roc_auc_score(y, precise)
    auc_recall = roc_auc_score(y, recall_first)
    assert auc_precise > auc_recall, "setup invariant: the precise model must win on AUC"

    return (
        _model("m_precise", "Precise (XGBoost-like)", y, precise, roc_auc=auc_precise),
        _model("m_recall", "Recall-first (LightGBM-like)", y, recall_first, roc_auc=auc_recall),
    )


def test_leaderboard_is_sorted_by_cost_not_by_auc() -> None:
    precise, recall = _two_models_with_a_real_inversion()
    board = simulate_classification([precise, recall], PLANT)
    costs = [r.total_cost for r in board.rows]
    assert costs == sorted(costs), "primary sort key must be expected cost (brief §2.3)"


def test_the_best_auc_model_is_not_always_the_best_cost_model() -> None:
    """The money shot. Detected, not manufactured."""
    precise, recall = _two_models_with_a_real_inversion()
    board = simulate_classification([precise, recall], PLANT)

    inversion = board.inversion("roc_auc")
    assert inversion is not None, "this fixture is built to invert; if it stops, the fixture broke"
    assert inversion["metric_winner"] == "Precise (XGBoost-like)"
    assert inversion["cost_winner"] == "Recall-first (LightGBM-like)"
    assert inversion["annual_cost_penalty"] > 0, "the AUC winner must be the more expensive one"


def test_inversion_stays_silent_when_metric_and_economics_agree() -> None:
    """Kairos never invents a disagreement to have something dramatic to say."""
    rng = np.random.default_rng(3)
    y = (rng.random(800) < 0.2).astype(int)
    strong = np.clip(rng.normal(np.where(y == 1, 0.85, 0.15), 0.10), 0.001, 0.999)
    weak = np.clip(rng.normal(np.where(y == 1, 0.60, 0.40), 0.25), 0.001, 0.999)

    board = simulate_classification(
        [
            _model("strong", "Strong", y, strong, roc_auc=roc_auc_score(y, strong)),
            _model("weak", "Weak", y, weak, roc_auc=roc_auc_score(y, weak)),
        ],
        PLANT,
    )
    assert board.best.model_id == "strong"
    assert board.inversion("roc_auc") is None


def test_cost_regret_is_zero_for_the_winner_and_positive_for_the_rest() -> None:
    precise, recall = _two_models_with_a_real_inversion()
    board = simulate_classification([precise, recall], PLANT)
    assert board.rows[0].cost_regret == 0.0
    assert all(r.cost_regret > 0 for r in board.rows[1:])
    assert board.rows[1].cost_regret == pytest.approx(
        board.rows[1].total_cost - board.rows[0].total_cost
    )


def test_changing_the_economics_can_reorder_the_leaderboard() -> None:
    """The entire premise of the what-if simulator: the ranking is a function of
    the plant's costs, not a property of the models."""
    precise, recall = _two_models_with_a_real_inversion()

    high_consequence = simulate_classification([precise, recall], PLANT)
    # Make inspections punishing and failures survivable: the trade-off inverts.
    cheap_failure = PLANT.with_changes(
        c_unplanned_repair=150_000,
        unplanned_downtime_hours=1,
        c_inspection=120_000,
    )
    low_consequence = simulate_classification([precise, recall], cheap_failure)

    assert high_consequence.best.model_id == "m_recall"
    assert low_consequence.best.model_id == "m_precise"


def test_every_row_carries_its_own_operating_point() -> None:
    precise, recall = _two_models_with_a_real_inversion()
    board = simulate_classification([precise, recall], PLANT)
    for row in board.rows:
        assert row.operating_point_kind == "threshold"
        assert 0.0 <= row.operating_point <= 1.0


def test_baselines_are_attached_for_the_winning_model() -> None:
    precise, recall = _two_models_with_a_real_inversion()
    board = simulate_classification([precise, recall], PLANT)
    assert board.baselines is not None
    assert board.baselines.kairos.key == "kairos"
    assert board.baselines.verdict()


def test_simulator_recomputes_inside_the_300ms_budget() -> None:
    """PROJECT_BRIEF.md §2.4: sliders must feel live, so four models over a
    realistic test set have to re-rank in under 300 ms with no retraining."""
    rng = np.random.default_rng(5)
    n = 20_000
    y = (rng.random(n) < 0.034).astype(int)  # AI4I's real 3.4% positive rate
    models = [
        _model(f"m{i}", f"Model {i}", y,
               np.clip(rng.normal(np.where(y == 1, 0.7 - i * 0.05, 0.3), 0.2), 0.001, 0.999),
               roc_auc=0.8)
        for i in range(4)
    ]
    board = simulate_classification(models, PLANT)
    assert board.elapsed_ms < 300, f"took {board.elapsed_ms:.0f}ms, budget is 300ms"


def test_empty_model_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="no models"):
        simulate_classification([], PLANT)
