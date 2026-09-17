"""Metrics — PROJECT_BRIEF.md §6.2 Trap 5, §11.

Accuracy does not appear in this file, and there is a test asserting it never
appears in any metric payload. On AI4I a model that predicts "never fails" scores
96.6% accuracy while catching zero failures; reporting that number at all invites
someone to quote it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)

#: Precision floor used for the "recall at fixed precision" operating summary.
TARGET_PRECISION = 0.5


def recall_at_precision(
    y_true: np.ndarray, y_prob: np.ndarray, target_precision: float = TARGET_PRECISION
) -> float:
    """Best recall achievable while holding precision at or above the floor.

    More useful than F1 on imbalanced data because it answers a question a plant
    actually asks: "if I accept that half my callouts find nothing, how many real
    failures do I catch?"
    """
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    if y.sum() == 0 or y.sum() == y.size:
        return 0.0
    precision, recall, _ = precision_recall_curve(y, p)
    viable = recall[precision >= target_precision]
    return float(viable.max()) if viable.size else 0.0


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray | None = None
) -> dict[str, float]:
    """Threshold-free ranking quality plus calibration quality. No accuracy."""
    y = np.asarray(y_true).astype(int).ravel()
    p = np.clip(np.asarray(y_prob, dtype=float).ravel(), 1e-9, 1 - 1e-9)

    metrics: dict[str, float] = {}
    if y.sum() > 0 and y.sum() < y.size:
        metrics["roc_auc"] = float(roc_auc_score(y, p))
        metrics["pr_auc"] = float(average_precision_score(y, p))
        metrics["recall_at_p50"] = recall_at_precision(y, p, 0.5)
        metrics["recall_at_p80"] = recall_at_precision(y, p, 0.8)
        metrics["log_loss"] = float(log_loss(y, p, labels=[0, 1]))
    metrics["brier"] = float(np.mean((p - y) ** 2))
    metrics["positive_rate"] = float(y.mean())
    metrics["n_rows"] = float(y.size)

    if y_pred is not None:
        pred = np.asarray(y_pred).astype(int).ravel()
        tp = float(np.sum((y == 1) & (pred == 1)))
        fp = float(np.sum((y == 0) & (pred == 1)))
        fn = float(np.sum((y == 1) & (pred == 0)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = (
            2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        )
    return metrics


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """The C-MAPSS asymmetric scoring function from the original PHM08 challenge.

    Late predictions are punished harder than early ones (exp(d/10) versus
    exp(-d/13)), because in the real problem a late alert means the engine is
    already broken. Lower is better, and unlike RMSE it encodes which direction of
    error actually costs money — which is the same argument Kairos makes in rupees.
    """
    true = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    d = pred - true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, near_failure_cycles: int = 30
) -> dict[str, float]:
    """RMSE/MAE/R² plus the two RUL-specific measures that matter operationally."""
    true = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "mae": float(mean_absolute_error(true, pred)),
        "r2": float(r2_score(true, pred)) if true.std() > 0 else 0.0,
        "nasa_score": nasa_score(true, pred),
        "n_rows": float(true.size),
    }

    # Accuracy far from failure is nearly free; accuracy near failure is the product.
    near = true <= near_failure_cycles
    if near.any():
        metrics[f"rmse_last_{near_failure_cycles}"] = float(
            np.sqrt(mean_squared_error(true[near], pred[near]))
        )
    # Which way does the model err? Negative bias means it predicts failure early,
    # which is the survivable direction.
    metrics["mean_bias"] = float(np.mean(pred - true))
    metrics["late_fraction"] = float(np.mean(pred > true))
    return metrics


def beats_baseline(
    metrics: dict[str, float], baseline_metrics: dict[str, float], task_type: str
) -> bool:
    """Quality gate from PROJECT_BRIEF.md §7's conditional edge."""
    if task_type == "binary_classification":
        return metrics.get("pr_auc", 0.0) >= 0.6
    return metrics.get("rmse", float("inf")) < baseline_metrics.get("rmse", float("inf"))


def assert_no_accuracy(metrics: dict[str, Any]) -> None:
    """Guard for §11. Called by tests; cheap enough to call anywhere."""
    offenders = [k for k in metrics if "accuracy" in k.lower()]
    if offenders:
        raise AssertionError(f"accuracy must never be reported on imbalanced data: {offenders}")
