"""Training core — PROJECT_BRIEF.md §10 Phase 3.

Trains the candidate zoo under a wall-clock budget, with group-aware
cross-validation, calibration, and predictions cached for the Decision Layer.

Two properties this module is built around:

* **Nothing here knows about money.** Training produces probabilities and RUL
  estimates; the cost engine decides what to do with them. Keeping that boundary
  clean is what lets the what-if simulator re-decide everything without retraining.
* **A failed candidate is a finding, not a crash.** One model blowing up must not
  take down a run in front of a judge, so every trial is individually guarded and
  its error recorded.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

from kairos.data.splits import Split
from kairos.ml.calibrate import CalibrationReport, calibrate_probabilities
from kairos.ml.evaluate import classification_metrics, regression_metrics
from kairos.ml.registry import Candidate, candidates_for, scale_pos_weight_for

log = logging.getLogger(__name__)


@dataclass
class TrialResult:
    """One trained candidate, with its predictions cached for the Decision Layer."""

    model_id: str
    candidate_key: str
    model_name: str
    task_type: str
    family: str
    metrics: dict[str, float] = field(default_factory=dict)
    cv_metrics: dict[str, float] = field(default_factory=dict)
    fit_seconds: float = 0.0
    feature_names: list[str] = field(default_factory=list)
    supports_shap: bool = False
    model: Any = None
    calibrator: Any = None
    calibration: CalibrationReport | None = None
    val_prediction: np.ndarray | None = None
    test_prediction: np.ndarray | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.model is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "candidate_key": self.candidate_key,
            "model_name": self.model_name,
            "task_type": self.task_type,
            "family": self.family,
            "metrics": self.metrics,
            "cv_metrics": self.cv_metrics,
            "fit_seconds": self.fit_seconds,
            "n_features": len(self.feature_names),
            "supports_shap": self.supports_shap,
            "calibration": self.calibration.as_dict() if self.calibration else None,
            "error": self.error,
            "ok": self.ok,
        }


@dataclass
class TrainingReport:
    trials: list[TrialResult]
    task_type: str
    feature_names: list[str]
    elapsed_seconds: float
    budget_seconds: float
    budget_exhausted: bool
    findings: list[str] = field(default_factory=list)

    @property
    def successful(self) -> list[TrialResult]:
        return [t for t in self.trials if t.ok]

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "n_features": len(self.feature_names),
            "elapsed_seconds": self.elapsed_seconds,
            "budget_seconds": self.budget_seconds,
            "budget_exhausted": self.budget_exhausted,
            "trials": [t.as_dict() for t in self.trials],
            "findings": self.findings,
        }


def _predict_scores(model: Any, X: pd.DataFrame, task_type: str) -> np.ndarray:
    """Probabilities for classification, point estimates for regression."""
    if task_type == "binary_classification":
        if hasattr(model, "predict_proba"):
            return np.asarray(model.predict_proba(X))[:, 1]
        raw = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-raw))  # squash to [0,1] so thresholds mean something
    return np.asarray(model.predict(X), dtype=float)


def cross_validate(
    candidate: Candidate,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray | None,
    task_type: str,
    n_splits: int = 3,
    seed: int = 0,
    build_kwargs: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Group-aware CV. Falls back to stratified only when there are no groups.

    ``GroupKFold`` is what stops a fold boundary from cutting an engine in half,
    which would report a score the model cannot reproduce on an engine it has
    never seen.
    """
    build_kwargs = build_kwargs or {}
    n_samples = len(X)
    if n_samples < n_splits * 2:
        return {}

    if groups is not None and len(np.unique(groups)) >= n_splits:
        splitter: Any = GroupKFold(n_splits=n_splits)
        folds = splitter.split(X, y, groups=groups)
    elif task_type == "binary_classification" and len(np.unique(y)) > 1:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = splitter.split(X, y)
    else:
        return {}

    scores: list[dict[str, float]] = []
    for train_idx, test_idx in folds:
        try:
            model = candidate.build(**build_kwargs)
            model.fit(X.iloc[train_idx], y[train_idx])
            preds = _predict_scores(model, X.iloc[test_idx], task_type)
            if task_type == "binary_classification":
                scores.append(classification_metrics(y[test_idx], preds))
            else:
                scores.append(regression_metrics(y[test_idx], preds))
        except Exception as exc:  # noqa: BLE001 - a bad fold is a finding, not a crash
            log.warning("CV fold failed for %s: %s", candidate.key, exc)

    if not scores:
        return {}
    keys = set().union(*(s.keys() for s in scores))
    return {f"cv_{k}": float(np.mean([s[k] for s in scores if k in s])) for k in keys}


def train_candidate(
    candidate: Candidate,
    split: Split,
    feature_names: Sequence[str],
    target: str,
    task_type: str,
    group_col: str | None = None,
    seed: int = 0,
    run_cv: bool = True,
) -> TrialResult:
    """Fit one candidate, calibrate it, and cache its predictions."""
    result = TrialResult(
        model_id=f"{candidate.key}-{uuid.uuid4().hex[:8]}",
        candidate_key=candidate.key,
        model_name=candidate.name,
        task_type=task_type,
        family=candidate.family,
        feature_names=list(feature_names),
        supports_shap=candidate.supports_shap,
    )
    started = time.perf_counter()
    try:
        X_train = split.train[list(feature_names)]
        y_train = split.train[target].to_numpy()
        X_val = split.val[list(feature_names)]
        y_val = split.val[target].to_numpy()
        X_test = split.test[list(feature_names)]
        y_test = split.test[target].to_numpy()

        build_kwargs: dict[str, Any] = {"seed": seed}
        if task_type == "binary_classification":
            build_kwargs["scale_pos_weight"] = scale_pos_weight_for(y_train)

        if run_cv:
            groups = (
                split.train[group_col].to_numpy()
                if group_col and group_col in split.train.columns
                else None
            )
            result.cv_metrics = cross_validate(
                candidate, X_train, y_train, groups, task_type,
                seed=seed, build_kwargs=build_kwargs,
            )

        model = candidate.build(**build_kwargs)
        model.fit(X_train, y_train)
        result.model = model

        val_scores = _predict_scores(model, X_val, task_type)
        test_scores = _predict_scores(model, X_test, task_type)

        if task_type == "binary_classification":
            # Calibrate on validation, then apply the SAME mapping to both splits so
            # the threshold chosen on validation means the same thing on test.
            calibrated_val, report, calibrator = calibrate_probabilities(y_val, val_scores)
            calibrated_test, _, _ = calibrate_probabilities(
                y_val, val_scores, p_target=test_scores, method=report.method
            )
            result.calibration = report
            result.calibrator = calibrator
            result.val_prediction = calibrated_val
            result.test_prediction = calibrated_test
            result.metrics = classification_metrics(y_test, calibrated_test)
        else:
            result.val_prediction = val_scores
            result.test_prediction = test_scores
            result.metrics = regression_metrics(y_test, test_scores)

    except Exception as exc:  # noqa: BLE001 - one dead candidate must not kill a run
        log.exception("candidate %s failed", candidate.key)
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        result.fit_seconds = time.perf_counter() - started
    return result


def train_all(
    split: Split,
    feature_names: Sequence[str],
    target: str,
    task_type: str,
    group_col: str | None = None,
    budget_seconds: float = 120.0,
    seed: int = 0,
    only: Sequence[str] | None = None,
    on_trial: Callable[[TrialResult], None] | None = None,
    run_cv: bool = True,
) -> TrainingReport:
    """Train the zoo under a wall-clock budget, cheapest families first.

    Ordering matters: baselines and linear models finish in seconds, so even a
    budget that runs out leaves a usable leaderboard rather than nothing. The
    budget is checked between candidates, not inside a fit — killing a
    half-trained booster would leave an unusable artefact.
    """
    zoo = candidates_for(task_type)  # type: ignore[arg-type]
    if only:
        wanted = set(only)
        zoo = [c for c in zoo if c.key in wanted]

    # Ordered by measured cost, not by intuition. On C-MAPSS with 257 features a
    # random forest takes 87s while LightGBM takes 13s, so putting bagging before
    # boosting burned the entire budget before reaching the two models the whole
    # comparison is about. Cheapest first, for real.
    order = {"baseline": 0, "linear": 1, "boosting": 2, "bagging": 3}
    zoo = sorted(zoo, key=lambda c: order.get(c.family, 99))

    started = time.perf_counter()
    trials: list[TrialResult] = []
    findings: list[str] = []
    exhausted = False

    for candidate in zoo:
        elapsed = time.perf_counter() - started
        if elapsed >= budget_seconds:
            exhausted = True
            findings.append(
                f"Time budget of {budget_seconds:.0f}s exhausted after {len(trials)} "
                f"candidate(s); {candidate.name} and any later candidates were skipped."
            )
            break
        trial = train_candidate(
            candidate, split, feature_names, target, task_type,
            group_col=group_col, seed=seed, run_cv=run_cv,
        )
        trials.append(trial)
        if trial.error:
            findings.append(f"{candidate.name} failed to train: {trial.error}")
        if on_trial:
            on_trial(trial)

    report = TrainingReport(
        trials=trials,
        task_type=task_type,
        feature_names=list(feature_names),
        elapsed_seconds=time.perf_counter() - started,
        budget_seconds=budget_seconds,
        budget_exhausted=exhausted,
        findings=findings,
    )
    if not report.successful:
        report.findings.append("No candidate trained successfully.")
    return report
