"""Probability calibration — PROJECT_BRIEF.md §2.2.

Cost-optimal thresholding is only valid if probabilities mean what they say. A
model that outputs 0.71 for events that happen 40% of the time will place its
threshold in the wrong place, and the resulting rupee figure will be confidently
wrong — which is worse than no figure.

Isotonic on enough data, Platt (sigmoid) when there is not. The switch is made on
the count of the *minority* class, because that is what actually constrains the
fit: 10,000 rows with 30 positives cannot support isotonic regression no matter
how large the frame looks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

#: Minority-class count below which isotonic overfits and Platt is the safer choice.
ISOTONIC_MIN_POSITIVES = 200


@dataclass
class CalibrationReport:
    """Reliability-diagram data plus the scalar scores that summarise it."""

    method: str
    brier_before: float
    brier_after: float
    ece_before: float
    ece_after: float
    bins: list[dict[str, float]]
    n_calibration_rows: int
    n_calibration_positives: int

    @property
    def improved(self) -> bool:
        return self.brier_after <= self.brier_before

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "brier_before": self.brier_before,
            "brier_after": self.brier_after,
            "ece_before": self.ece_before,
            "ece_after": self.ece_after,
            "improved": self.improved,
            "bins": self.bins,
            "n_calibration_rows": self.n_calibration_rows,
            "n_calibration_positives": self.n_calibration_positives,
        }


def choose_method(y: np.ndarray) -> str:
    """Isotonic needs data; Platt needs only a shape. Decide on minority count."""
    y = np.asarray(y).ravel()
    minority = int(min((y == 1).sum(), (y == 0).sum()))
    return "isotonic" if minority >= ISOTONIC_MIN_POSITIVES else "sigmoid"


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error of the probabilities themselves."""
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Average gap between predicted confidence and observed frequency.

    This is the number that answers "why should I trust 0.71?" — an ECE of 0.02
    means that when the model says 71%, it is right about 69-73% of the time.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    if y.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        total += (mask.sum() / y.size) * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def reliability_bins(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> list[dict[str, float]]:
    """Reliability-diagram points. Empty bins are omitted, never plotted as zero."""
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict[str, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        out.append({
            "bin_lower": float(lo),
            "bin_upper": float(hi),
            "mean_predicted": float(p[mask].mean()),
            "observed_frequency": float(y[mask].mean()),
            "count": int(mask.sum()),
        })
    return out


def calibrate_probabilities(
    y_val: np.ndarray,
    p_val: np.ndarray,
    p_target: np.ndarray | None = None,
    method: str | None = None,
) -> tuple[np.ndarray, CalibrationReport, Any]:
    """Fit a calibrator on validation predictions and apply it.

    Returns the calibrated probabilities for ``p_target`` (or for ``p_val`` itself
    when no target is given), a report for the reliability diagram, and the fitted
    calibrator so it can be reused at inference time.

    Fitting on validation and reporting on test is what keeps this honest: a
    calibrator fitted and scored on the same rows always looks perfect.
    """
    y = np.asarray(y_val, dtype=float).ravel()
    p = np.asarray(p_val, dtype=float).ravel()
    target = p if p_target is None else np.asarray(p_target, dtype=float).ravel()

    chosen = method or choose_method(y)
    before_brier = brier_score(y, p)
    before_ece = expected_calibration_error(y, p)

    try:
        if chosen == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            calibrator.fit(p, y)
            calibrated = np.clip(calibrator.predict(target), 0.0, 1.0)
            in_sample = np.clip(calibrator.predict(p), 0.0, 1.0)
        else:
            from sklearn.linear_model import LogisticRegression

            calibrator = LogisticRegression(max_iter=1000)
            calibrator.fit(p.reshape(-1, 1), y.astype(int))
            calibrated = calibrator.predict_proba(target.reshape(-1, 1))[:, 1]
            in_sample = calibrator.predict_proba(p.reshape(-1, 1))[:, 1]
    except Exception as exc:  # noqa: BLE001 - an uncalibrated model beats no model
        log.warning("calibration failed (%s); returning raw probabilities", exc)
        return target, CalibrationReport(
            method="none",
            brier_before=before_brier,
            brier_after=before_brier,
            ece_before=before_ece,
            ece_after=before_ece,
            bins=reliability_bins(y, p),
            n_calibration_rows=int(y.size),
            n_calibration_positives=int((y == 1).sum()),
        ), None

    report = CalibrationReport(
        method=chosen,
        brier_before=before_brier,
        brier_after=brier_score(y, in_sample),
        ece_before=before_ece,
        ece_after=expected_calibration_error(y, in_sample),
        bins=reliability_bins(y, in_sample),
        n_calibration_rows=int(y.size),
        n_calibration_positives=int((y == 1).sum()),
    )
    return calibrated, report, calibrator
