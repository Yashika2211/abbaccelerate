"""Candidate model zoo — PROJECT_BRIEF.md §10 Phase 3.

Four candidates per task, chosen so the leaderboard has something to say rather
than to maximise the count:

* a **baseline** that is deliberately dumb, so "did the model learn anything?" is
  answerable without squinting,
* a **linear/shallow** model that is fast and hard to overfit,
* **LightGBM** and **XGBoost**, which is where the cost-versus-AUC disagreement
  that drives the whole demo tends to appear.

Imbalance is handled with class weights rather than SMOTE. At a 3.4% positive rate
SMOTE mostly manufactures plausible-looking noise between real positives, and it
would have to be defended on stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np

TaskType = Literal["rul_regression", "binary_classification"]


@dataclass(frozen=True)
class Candidate:
    """One entry in the zoo. ``build`` is deferred so nothing is constructed until
    a plan is approved and the class balance is known."""

    key: str
    name: str
    task_type: TaskType
    build: Callable[..., Any]
    family: str
    supports_shap: bool = False        # TreeExplainer only; KernelExplainer is too slow
    notes: str = ""
    params: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------- classification


def _build_majority_baseline(**_: Any):
    from sklearn.dummy import DummyClassifier

    return DummyClassifier(strategy="prior")


def _build_logistic(scale_pos_weight: float = 1.0, **_: Any):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced" if scale_pos_weight > 1 else None,
            n_jobs=None,
        )),
    ])


def _build_random_forest(scale_pos_weight: float = 1.0, seed: int = 0, **_: Any):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced_subsample" if scale_pos_weight > 1 else None,
        random_state=seed,
        n_jobs=-1,
    )


def _build_lightgbm_clf(scale_pos_weight: float = 1.0, seed: int = 0, **_: Any):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def _build_xgboost_clf(scale_pos_weight: float = 1.0, seed: int = 0, **_: Any):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
        eval_metric="aucpr",          # never optimise accuracy on 3.4% positives
        tree_method="hist",
    )


# ----------------------------------------------------------------- regression


def _build_mean_baseline(**_: Any):
    from sklearn.dummy import DummyRegressor

    return DummyRegressor(strategy="mean")


def _build_ridge(**_: Any):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))])


def _build_random_forest_reg(seed: int = 0, **_: Any):
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=200, min_samples_leaf=5, random_state=seed, n_jobs=-1
    )


def _build_lightgbm_reg(seed: int = 0, **_: Any):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def _build_xgboost_reg(seed: int = 0, **_: Any):
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )


def build_quantile_lightgbm(alpha: float, seed: int = 0):
    """Risk-aware RUL — PROJECT_BRIEF.md §3.4.

    Deciding on the pessimistic (0.1) quantile makes the system err toward early
    intervention on high-consequence assets. The cost engine then picks whichever
    quantile is actually cheapest, which is a second decision variable entirely
    independent of the threshold.
    """
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


CLASSIFIERS: tuple[Candidate, ...] = (
    Candidate(
        key="baseline_majority",
        name="Baseline (predict prior)",
        task_type="binary_classification",
        build=_build_majority_baseline,
        family="baseline",
        notes="Predicts the base rate for every row. Exists so 'did we learn anything' "
              "is answerable, and so the quality gate has a floor to compare against.",
    ),
    Candidate(
        key="logistic",
        name="Logistic regression",
        task_type="binary_classification",
        build=_build_logistic,
        family="linear",
        notes="Scaled, class-weighted. Fast, hard to overfit, and a genuine contender "
              "on tabular sensor data.",
    ),
    Candidate(
        key="random_forest",
        name="Random forest",
        task_type="binary_classification",
        build=_build_random_forest,
        family="bagging",
        supports_shap=True,
        notes="Balanced subsampling rather than SMOTE.",
    ),
    Candidate(
        key="lightgbm",
        name="LightGBM",
        task_type="binary_classification",
        build=_build_lightgbm_clf,
        family="boosting",
        supports_shap=True,
        notes="scale_pos_weight set from the observed class ratio.",
    ),
    Candidate(
        key="xgboost",
        name="XGBoost",
        task_type="binary_classification",
        build=_build_xgboost_clf,
        family="boosting",
        supports_shap=True,
        notes="Evaluated on aucpr, never accuracy.",
    ),
)

REGRESSORS: tuple[Candidate, ...] = (
    Candidate(
        key="baseline_mean",
        name="Baseline (predict mean RUL)",
        task_type="rul_regression",
        build=_build_mean_baseline,
        family="baseline",
        notes="The quality gate in brief §7: a model that cannot beat this is not a model.",
    ),
    Candidate(
        key="ridge",
        name="Ridge regression",
        task_type="rul_regression",
        build=_build_ridge,
        family="linear",
    ),
    Candidate(
        key="random_forest",
        name="Random forest",
        task_type="rul_regression",
        build=_build_random_forest_reg,
        family="bagging",
        supports_shap=True,
    ),
    Candidate(
        key="lightgbm",
        name="LightGBM",
        task_type="rul_regression",
        build=_build_lightgbm_reg,
        family="boosting",
        supports_shap=True,
    ),
    Candidate(
        key="xgboost",
        name="XGBoost",
        task_type="rul_regression",
        build=_build_xgboost_reg,
        family="boosting",
        supports_shap=True,
    ),
)


def candidates_for(task_type: TaskType, exclude_baseline: bool = False) -> list[Candidate]:
    zoo = CLASSIFIERS if task_type == "binary_classification" else REGRESSORS
    return [c for c in zoo if not (exclude_baseline and c.family == "baseline")]


def get_candidate(task_type: TaskType, key: str) -> Candidate:
    for candidate in candidates_for(task_type):
        if candidate.key == key:
            return candidate
    raise KeyError(f"no candidate {key!r} for task {task_type!r}")


def scale_pos_weight_for(y: np.ndarray) -> float:
    """negatives / positives, the standard imbalance correction for boosted trees."""
    y = np.asarray(y).ravel()
    positives = float((y == 1).sum())
    negatives = float((y == 0).sum())
    if positives == 0:
        return 1.0
    return max(negatives / positives, 1.0)
