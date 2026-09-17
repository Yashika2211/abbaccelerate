"""Honest baselines — PROJECT_BRIEF.md §2.5.

Every savings claim Kairos makes is stated against named alternatives computed on
the same test set:

1. **Run-to-failure** — never intervene.
2. **Fixed-interval preventive** — intervene every *k* periods, with *k* itself
   optimised. This is what real plants actually do, and it is a strong baseline.
   It is deliberately not straw-manned: *k* is tuned to the same economics Kairos
   is tuned to, and the interval policy pays no spare-part lead-time penalty
   because scheduled work orders their parts in advance. Both choices make the
   baseline harder to beat, which is the honest direction to err in.
3. **Model at threshold 0.5** — the naive ML approach.
4. **Kairos** — the model at its cost-optimal operating point.

If Kairos loses, :func:`compare` says so. A tool that admits when ML is not worth
the trouble is worth more than one that always finds a way to win.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from kairos.decision.cost import (
    CostConfig,
    annualize,
    classification_exposure_asset_years,
    expected_cost,
    rul_policy_cost,
)
from kairos.decision.policy import apply_threshold

RUN_TO_FAILURE = "run_to_failure"
FIXED_INTERVAL = "fixed_interval"
MODEL_AT_HALF = "model_at_0.5"
KAIROS = "kairos"


@dataclass(frozen=True)
class Baseline:
    key: str
    name: str
    description: str
    total_cost: float
    cost_per_asset_year: float
    detail: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "total_cost": self.total_cost,
            "cost_per_asset_year": self.cost_per_asset_year,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class BaselineComparison:
    baselines: list[Baseline]
    currency: str

    @property
    def kairos(self) -> Baseline:
        return next(b for b in self.baselines if b.key == KAIROS)

    @property
    def best(self) -> Baseline:
        return min(self.baselines, key=lambda b: b.total_cost)

    @property
    def best_alternative(self) -> Baseline:
        return min((b for b in self.baselines if b.key != KAIROS), key=lambda b: b.total_cost)

    @property
    def kairos_wins(self) -> bool:
        return self.kairos.total_cost <= self.best_alternative.total_cost

    @property
    def savings_vs_best_alternative(self) -> float:
        """Positive when Kairos wins, negative when it loses. Both get displayed."""
        return self.best_alternative.total_cost - self.kairos.total_cost

    @property
    def savings_per_asset_year(self) -> float:
        return self.best_alternative.cost_per_asset_year - self.kairos.cost_per_asset_year

    def verdict(self) -> str:
        """One sentence, suitable for putting on a slide without a lawyer present."""
        if self.kairos_wins:
            return (
                f"Kairos is the cheapest policy tested, saving {self.currency} "
                f"{self.savings_per_asset_year:,.0f} per asset per year against the next best "
                f"alternative ({self.best_alternative.name})."
            )
        return (
            f"Kairos does NOT win on this configuration. {self.best_alternative.name} is cheaper "
            f"by {self.currency} {-self.savings_per_asset_year:,.0f} per asset per year. At these "
            f"economics the model is not worth deploying."
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "baselines": [b.as_dict() for b in self.baselines],
            "best": self.best.key,
            "kairos_wins": self.kairos_wins,
            "savings_vs_best_alternative": self.savings_vs_best_alternative,
            "savings_per_asset_year": self.savings_per_asset_year,
            "verdict": self.verdict(),
            "currency": self.currency,
        }


# ------------------------------------------------------------- classification


def _fixed_interval_mask(n: int, k: int) -> np.ndarray:
    """Intervene on every k-th observation: periodic preventive maintenance."""
    idx = np.arange(n)
    return ((idx + 1) % k == 0).astype(int)


def optimise_fixed_interval(
    y_true: Sequence[int] | np.ndarray, cfg: CostConfig, max_k: int | None = None
) -> tuple[int, float]:
    """Best periodic-maintenance interval and its cost. Tuned to the same economics."""
    y = np.asarray(y_true).astype(int).ravel()
    n = y.size
    if n == 0:
        return 1, 0.0
    ceiling = n if max_k is None else min(max_k, n)
    best_k, best_cost = 1, float("inf")
    for k in range(1, ceiling + 1):
        cost = expected_cost(y, _fixed_interval_mask(n, k), cfg)
        if cost < best_cost:
            best_k, best_cost = k, cost
    # k = n+1 means "never intervene", which run_to_failure already covers.
    return best_k, best_cost


def compare_classification(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    kairos_threshold: float,
    cfg: CostConfig,
    max_k: int | None = 500,
) -> BaselineComparison:
    """All four policies on one test set. PROJECT_BRIEF.md §2.5."""
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError("y_true and y_prob must be the same length")
    if y.size == 0:
        raise ValueError("cannot compare baselines on an empty set")

    exposure = classification_exposure_asset_years(y.size, cfg)

    def make(key: str, name: str, desc: str, pred: np.ndarray, detail: dict[str, Any]) -> Baseline:
        total = expected_cost(y, pred, cfg)
        return Baseline(key, name, desc, total, annualize(total, exposure), detail)

    never = np.zeros_like(y)
    best_k, _ = optimise_fixed_interval(y, cfg, max_k=max_k)
    interval = _fixed_interval_mask(y.size, best_k)
    half = apply_threshold(p, 0.5)
    tuned = apply_threshold(p, kairos_threshold)

    baselines = [
        make(RUN_TO_FAILURE, "Run to failure", "Never intervene; absorb every failure.",
             never, {}),
        make(FIXED_INTERVAL, f"Fixed interval (every {best_k})",
             f"Preventive maintenance every {best_k} cycles, with the interval itself "
             f"optimised against these same costs.",
             interval, {"k": best_k}),
        make(MODEL_AT_HALF, "Model at 0.5",
             "The same model, using the default 0.5 probability cut-off.",
             half, {"threshold": 0.5}),
        make(KAIROS, "Kairos",
             f"The same model at its cost-optimal threshold ({kairos_threshold:.3f}).",
             tuned, {"threshold": kairos_threshold}),
    ]
    return BaselineComparison(baselines=baselines, currency=cfg.currency)


# ----------------------------------------------------------------------- RUL


def optimise_interval_cycles(
    failure_cycle: dict[Any, float], cfg: CostConfig, max_k: int | None = None
) -> tuple[int, float]:
    """Best fixed servicing interval in cycles, for a run-to-failure fleet.

    Each unit is serviced at cycle *k*. Scheduled work orders its parts ahead, so
    no lead-time penalty applies — this makes the baseline stronger, on purpose.
    """
    if not failure_cycle:
        return 1, 0.0
    lives = np.array(list(failure_cycle.values()), dtype=float)
    ceiling = int(lives.max()) if max_k is None else max_k
    best_k, best_cost = 1, float("inf")
    for k in range(1, max(ceiling, 1) + 1):
        in_time = lives > k
        cost = float(
            np.sum(
                np.where(
                    in_time,
                    cfg.cost_planned_event + (lives - k) * cfg.value_per_remaining_cycle,
                    cfg.cost_unplanned_event,
                )
            )
        )
        if cost < best_cost:
            best_k, best_cost = k, cost
    return best_k, best_cost


def compare_rul(
    unit_ids: Sequence[Any] | np.ndarray,
    cycles: Sequence[float] | np.ndarray,
    pred_rul: Sequence[float] | np.ndarray,
    failure_cycle: dict[Any, float],
    kairos_lead_time: int,
    cfg: CostConfig,
    naive_lead_time: int = 30,
) -> BaselineComparison:
    """All four policies on one fleet. The RUL analogue of §2.5.

    ``naive_lead_time`` stands in for "what the maintenance manual says" — a fixed
    rule of thumb applied regardless of what the model or the economics imply.
    """
    lives = np.array(list(failure_cycle.values()), dtype=float)
    fleet = len(failure_cycle)
    if fleet == 0:
        raise ValueError("cannot compare baselines on an empty fleet")
    exposure = float(lives.sum()) / cfg.cycles_per_year

    run_cost = fleet * cfg.cost_unplanned_event
    best_k, interval_cost = optimise_interval_cycles(failure_cycle, cfg)
    naive = rul_policy_cost(unit_ids, cycles, pred_rul, failure_cycle, naive_lead_time, cfg)
    tuned = rul_policy_cost(unit_ids, cycles, pred_rul, failure_cycle, kairos_lead_time, cfg)

    baselines = [
        Baseline(RUN_TO_FAILURE, "Run to failure", "Never intervene; absorb every failure.",
                 run_cost, annualize(run_cost, exposure), {"fleet": fleet}),
        Baseline(FIXED_INTERVAL, f"Fixed interval (every {best_k} cycles)",
                 f"Service every unit at cycle {best_k}, with the interval optimised against "
                 f"these same costs and no spare-part delay.",
                 interval_cost, annualize(interval_cost, exposure), {"k": best_k}),
        Baseline(MODEL_AT_HALF, f"Manual rule (alert at RUL {naive_lead_time})",
                 f"The same model, but acting on the fixed {naive_lead_time}-cycle rule of thumb "
                 f"rather than an optimised lead time.",
                 naive.total_cost, naive.cost_per_asset_year, naive.as_dict()),
        Baseline(KAIROS, "Kairos", f"The same model at its cost-optimal lead time "
                 f"({kairos_lead_time} cycles).",
                 tuned.total_cost, tuned.cost_per_asset_year, tuned.as_dict()),
    ]
    return BaselineComparison(baselines=baselines, currency=cfg.currency)
