"""Cost-ranked leaderboard and the what-if simulator — PROJECT_BRIEF.md §2.3, §2.4.

Given predictions cached at training time, recompute the *entire* decision surface
for a new set of plant economics: every model's optimal operating point, the
leaderboard ordering, cost regret, and which model a plant should actually deploy.

No model is touched. That is the whole trick, and it is why dragging a slider can
reorder the leaderboard in milliseconds instead of minutes.

The headline this exists to produce:

    "XGBoost has 0.02 higher AUC. It also costs Rs 4.2 lakh more per year, because
     on this asset a missed failure costs 40x a false alarm and XGBoost trades
     recall for precision. Kairos picks LightGBM."

:attr:`Leaderboard.inversion` detects exactly that situation. It is reported when
it happens and stays quiet when it does not — it is never manufactured.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from kairos.decision.baselines import BaselineComparison, compare_classification, compare_rul
from kairos.decision.cost import CostConfig, annualize, classification_exposure_asset_years, expected_cost, rul_policy_cost
from kairos.decision.policy import apply_threshold, optimal_lead_time, optimal_threshold


@dataclass(frozen=True)
class ClassifierPredictions:
    """Cached predictions for one candidate. Calibrated probabilities only.

    Validation is what the threshold is chosen on; test is what gets reported.
    Keeping both here is what enforces that discipline at the type level.
    """

    model_id: str
    model_name: str
    val_y: np.ndarray
    val_prob: np.ndarray
    test_y: np.ndarray
    test_prob: np.ndarray
    metrics: dict[str, float] = field(default_factory=dict)  # cost-independent, computed once


@dataclass(frozen=True)
class RulPredictions:
    """Cached RUL predictions for one candidate, per split."""

    model_id: str
    model_name: str
    val_units: np.ndarray
    val_cycles: np.ndarray
    val_pred: np.ndarray
    val_failure: dict[Any, float]
    test_units: np.ndarray
    test_cycles: np.ndarray
    test_pred: np.ndarray
    test_failure: dict[Any, float]
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LeaderboardRow:
    model_id: str
    model_name: str
    operating_point: float          # threshold, or lead time in cycles
    operating_point_kind: str       # "threshold" | "lead_time"
    total_cost: float
    cost_per_asset_year: float
    cost_regret: float              # this model's cost minus the cheapest model's
    metrics: dict[str, float]
    degenerate: bool
    degenerate_reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "operating_point": self.operating_point,
            "operating_point_kind": self.operating_point_kind,
            "total_cost": self.total_cost,
            "cost_per_asset_year": self.cost_per_asset_year,
            "cost_regret": self.cost_regret,
            "metrics": self.metrics,
            "degenerate": self.degenerate,
            "degenerate_reason": self.degenerate_reason,
        }


@dataclass(frozen=True)
class Leaderboard:
    """Rows sorted by expected cost — PROJECT_BRIEF.md §2.3. Cost is the sort key."""

    rows: list[LeaderboardRow]
    currency: str
    elapsed_ms: float
    baselines: BaselineComparison | None = None

    @property
    def best(self) -> LeaderboardRow:
        return self.rows[0]

    def best_by_metric(self, metric: str, higher_is_better: bool = True) -> LeaderboardRow | None:
        """The model a conventional AutoML leaderboard would have crowned."""
        scored = [r for r in self.rows if metric in r.metrics]
        if not scored:
            return None
        return (max if higher_is_better else min)(scored, key=lambda r: r.metrics[metric])

    def inversion(self, metric: str, higher_is_better: bool = True) -> dict[str, Any] | None:
        """The money shot: report when the best-`metric` model is NOT the best-cost model.

        Returns ``None`` when the two agree. Nothing here manufactures a disagreement;
        if the metrics and the economics happen to point the same way, Kairos says so
        by staying silent.
        """
        champion = self.best_by_metric(metric, higher_is_better)
        if champion is None or champion.model_id == self.best.model_id:
            return None
        return {
            "metric": metric,
            "metric_winner": champion.model_name,
            "metric_winner_value": champion.metrics[metric],
            "cost_winner": self.best.model_name,
            "cost_winner_value": self.best.metrics.get(metric),
            "metric_gap": abs(champion.metrics[metric] - self.best.metrics.get(metric, 0.0)),
            "annual_cost_penalty": champion.cost_per_asset_year - self.best.cost_per_asset_year,
            "currency": self.currency,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": [r.as_dict() for r in self.rows],
            "currency": self.currency,
            "elapsed_ms": self.elapsed_ms,
            "baselines": self.baselines.as_dict() if self.baselines else None,
        }


def simulate_classification(
    models: Sequence[ClassifierPredictions],
    cfg: CostConfig,
    with_baselines: bool = True,
) -> Leaderboard:
    """Rank candidates by expected cost under `cfg`. Recomputed, never retrained."""
    if not models:
        raise ValueError("cannot build a leaderboard with no models")
    started = time.perf_counter()

    rows: list[LeaderboardRow] = []
    for m in models:
        decision = optimal_threshold(m.val_y, m.val_prob, cfg)   # chosen on validation
        test_pred = apply_threshold(m.test_prob, decision.threshold)  # frozen onto test
        total = expected_cost(m.test_y, test_pred, cfg)
        exposure = classification_exposure_asset_years(len(m.test_y), cfg)
        rows.append(
            LeaderboardRow(
                model_id=m.model_id,
                model_name=m.model_name,
                operating_point=decision.threshold,
                operating_point_kind="threshold",
                total_cost=total,
                cost_per_asset_year=annualize(total, exposure),
                cost_regret=0.0,
                metrics=dict(m.metrics),
                degenerate=decision.degenerate,
                degenerate_reason=decision.degenerate_reason,
            )
        )

    rows.sort(key=lambda r: r.total_cost)
    cheapest = rows[0].total_cost
    rows = [
        LeaderboardRow(**{**r.__dict__, "cost_regret": r.total_cost - cheapest}) for r in rows
    ]

    baselines = None
    if with_baselines:
        winner = next(m for m in models if m.model_id == rows[0].model_id)
        baselines = compare_classification(
            winner.test_y, winner.test_prob, rows[0].operating_point, cfg
        )

    return Leaderboard(
        rows=rows,
        currency=cfg.currency,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        baselines=baselines,
    )


def simulate_rul(
    models: Sequence[RulPredictions],
    cfg: CostConfig,
    with_baselines: bool = True,
    lead_times: Sequence[int] | None = None,
) -> Leaderboard:
    """Rank RUL candidates by fleet cost at each model's own optimal lead time."""
    if not models:
        raise ValueError("cannot build a leaderboard with no models")
    started = time.perf_counter()

    rows: list[LeaderboardRow] = []
    for m in models:
        decision = optimal_lead_time(
            m.val_units, m.val_cycles, m.val_pred, m.val_failure, cfg, lead_times=lead_times
        )
        test = rul_policy_cost(
            m.test_units, m.test_cycles, m.test_pred, m.test_failure, decision.lead_time, cfg
        )
        rows.append(
            LeaderboardRow(
                model_id=m.model_id,
                model_name=m.model_name,
                operating_point=float(decision.lead_time),
                operating_point_kind="lead_time",
                total_cost=test.total_cost,
                cost_per_asset_year=test.cost_per_asset_year,
                cost_regret=0.0,
                metrics=dict(m.metrics),
                degenerate=decision.degenerate,
                degenerate_reason=decision.degenerate_reason,
            )
        )

    rows.sort(key=lambda r: r.total_cost)
    cheapest = rows[0].total_cost
    rows = [LeaderboardRow(**{**r.__dict__, "cost_regret": r.total_cost - cheapest}) for r in rows]

    baselines = None
    if with_baselines:
        winner = next(m for m in models if m.model_id == rows[0].model_id)
        baselines = compare_rul(
            winner.test_units,
            winner.test_cycles,
            winner.test_pred,
            winner.test_failure,
            int(rows[0].operating_point),
            cfg,
        )

    return Leaderboard(
        rows=rows,
        currency=cfg.currency,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        baselines=baselines,
    )
