"""Decision-variable optimisation — PROJECT_BRIEF.md §2.1 and §3.3.

Two decision variables, one idea: never accept a default.

* Classification: sweep the probability threshold, pick the argmin of expected
  cost on **validation**, then freeze it and report on **test**. Never 0.5, never
  max-F1 — both answer a question nobody at a plant is asking.
* RUL: sweep the lead time, same discipline.

Pure functions over cached predictions, so the what-if simulator can re-run the
whole thing on new economics in milliseconds without touching a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from kairos.decision.cost import (
    CostConfig,
    RulPolicyResult,
    annualize,
    classification_cost_matrix,
    classification_exposure_asset_years,
    rul_lead_time_sweep,
)

#: Thresholds this close to the ends mean the economics, not the model, are deciding.
DEGENERATE_LOW = 0.02
DEGENERATE_HIGH = 0.98

#: Resolution of the plotted cost curve. Data-derived thresholds are added on top,
#: so the argmin is exact even though the curve is drawn on an even grid.
CURVE_POINTS = 501


@dataclass(frozen=True)
class ThresholdPoint:
    threshold: float
    total_cost: float
    cost_per_asset_year: float
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def alerts(self) -> int:
        return self.tp + self.fp

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "total_cost": self.total_cost,
            "cost_per_asset_year": self.cost_per_asset_year,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "alerts": self.alerts,
        }


@dataclass(frozen=True)
class ThresholdDecision:
    """The chosen operating point, plus everything needed to defend it on stage."""

    threshold: float
    chosen_on: str                    # which split the argmin was taken on
    point: ThresholdPoint             # performance at the chosen threshold, on `chosen_on`
    curve: list[ThresholdPoint]       # the plotted cost-vs-threshold curve
    naive_point: ThresholdPoint       # what threshold 0.5 would have cost
    degenerate: bool
    degenerate_reason: str | None
    plateau_fraction: float = 0.0     # share of the sweep sitting at minimum cost

    @property
    def savings_vs_naive(self) -> float:
        return self.naive_point.total_cost - self.point.total_cost

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "chosen_on": self.chosen_on,
            "point": self.point.as_dict(),
            "naive_point": self.naive_point.as_dict(),
            "savings_vs_naive": self.savings_vs_naive,
            "degenerate": self.degenerate,
            "degenerate_reason": self.degenerate_reason,
            "plateau_fraction": self.plateau_fraction,
            "curve": [p.as_dict() for p in self.curve],
        }


def _confusion_at_thresholds(
    y_true: np.ndarray, y_prob: np.ndarray, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised tp/fp/tn/fn for every threshold at once.

    Sorting once and reading off a cumulative sum turns an O(grid x rows) sweep
    into O(rows log rows + grid), which is what keeps the what-if simulator under
    the 300 ms budget in PROJECT_BRIEF.md §2.4.

    A row is predicted positive when ``prob >= threshold``.
    """
    n = y_true.size
    ascending = np.sort(y_prob, kind="mergesort")
    order = np.argsort(-y_prob, kind="mergesort")
    cum_tp = np.concatenate(([0], np.cumsum(y_true[order])))

    # How many rows have prob >= tau, for each tau in the grid.
    k = n - np.searchsorted(ascending, grid, side="left")
    tp = cum_tp[k]
    fp = k - tp
    positives = int(y_true.sum())
    fn = positives - tp
    tn = (n - positives) - fp
    return tp, fp, tn, fn


def sweep_thresholds(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    cfg: CostConfig,
    grid: Sequence[float] | None = None,
) -> list[ThresholdPoint]:
    """Expected cost at every candidate threshold. PROJECT_BRIEF.md §2.1."""
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"y_true and y_prob must be the same length, got {y.shape} and {p.shape}")
    if y.size == 0:
        return []
    if np.any((p < 0) | (p > 1)):
        raise ValueError("y_prob must lie in [0, 1]; calibrate before optimising a threshold")

    if grid is None:
        # Even grid for a smooth chart, plus the data's own probabilities so the
        # argmin is exact rather than the nearest grid point.
        taus = np.union1d(np.linspace(0.0, 1.0, CURVE_POINTS), np.unique(p))
    else:
        taus = np.unique(np.asarray(grid, dtype=float))

    tp, fp, tn, fn = _confusion_at_thresholds(y, p, taus)
    m = classification_cost_matrix(cfg)
    totals = tp * m.tp + fp * m.fp + tn * m.tn + fn * m.fn
    exposure = classification_exposure_asset_years(y.size, cfg)

    return [
        ThresholdPoint(
            threshold=float(t),
            total_cost=float(c),
            cost_per_asset_year=annualize(float(c), exposure),
            tp=int(a),
            fp=int(b),
            tn=int(d),
            fn=int(e),
        )
        for t, c, a, b, d, e in zip(taus, totals, tp, fp, tn, fn)
    ]


def _point_at(points: list[ThresholdPoint], threshold: float) -> ThresholdPoint:
    """The swept point closest to `threshold` — used to price the naive 0.5 default."""
    return min(points, key=lambda p: abs(p.threshold - threshold))


def optimal_threshold(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    cfg: CostConfig,
    chosen_on: str = "validation",
) -> ThresholdDecision:
    """Pick the cost-minimising threshold, and say so honestly when it is degenerate.

    Ties are broken toward the **middle** of the tied run rather than its edge. A
    threshold sitting on a cliff is one relabelled row away from behaving
    differently; the centre of a plateau is the operating point you can actually
    hand to a plant.
    """
    curve = sweep_thresholds(y_true, y_prob, cfg)
    if not curve:
        raise ValueError("cannot optimise a threshold on an empty set")

    costs = np.array([p.total_cost for p in curve])
    tied = np.flatnonzero(costs == costs.min())
    best = curve[int(tied[len(tied) // 2])]

    degenerate, reason = _diagnose_degeneracy(best, cfg, curve)
    return ThresholdDecision(
        threshold=best.threshold,
        chosen_on=chosen_on,
        point=best,
        curve=curve,
        naive_point=_point_at(curve, 0.5),
        degenerate=degenerate,
        degenerate_reason=reason,
        plateau_fraction=_plateau_fraction(curve),
    )


def _diagnose_degeneracy(
    best: ThresholdPoint, cfg: CostConfig, curve: list[ThresholdPoint]
) -> tuple[bool, str | None]:
    """PROJECT_BRIEF.md §7: when the economics decide, say the model is irrelevant.

    Diagnosed by asking whether the optimised policy actually beats the two trivial
    policies available to any plant without a model — never act, and always act.
    If it merely ties one of them, the model is decoration, and Kairos says so.

    Judging this from the threshold's numeric value does not work: because ties
    break toward the middle of a plateau, a thoroughly degenerate policy can carry
    a perfectly moderate-looking threshold. Behaviour is the honest signal.
    """
    m = classification_cost_matrix(cfg)
    positives = best.tp + best.fn
    negatives = best.tn + best.fp

    never_act = positives * m.fn
    always_act = positives * m.tp + negatives * m.fp

    if best.total_cost >= never_act:
        return True, (
            f"The cost-optimal policy saves nothing against run-to-failure "
            f"({cfg.currency} {never_act:,.0f} either way). Acting is never cheaper than "
            f"failing at these costs, so the model changes nothing. Revisit the cost of "
            f"unplanned downtime before trusting this recommendation."
        )
    if best.total_cost >= always_act:
        return True, (
            f"The cost-optimal policy saves nothing against inspecting everything "
            f"({cfg.currency} {always_act:,.0f} either way). At a consequence ratio of "
            f"{cfg.consequence_ratio:,.0f}:1, a blanket inspection policy is already optimal "
            f"and the model changes nothing."
        )
    if best.threshold <= DEGENERATE_LOW:
        return True, (
            f"Cost-optimal threshold is {best.threshold:.3f} — effectively alert-on-everything. "
            f"At a consequence ratio of {cfg.consequence_ratio:,.0f}:1 the economics, not the "
            f"model, are making this decision."
        )
    if best.threshold >= DEGENERATE_HIGH:
        return True, (
            f"Cost-optimal threshold is {best.threshold:.3f} — effectively never-alert. Acting "
            f"is barely cheaper than failing at these costs."
        )
    spread = max(p.total_cost for p in curve) - min(p.total_cost for p in curve)
    if spread <= 0:
        return True, (
            "Expected cost is flat across every threshold: these economics are indifferent "
            "to the model's output."
        )
    return False, None


def _plateau_fraction(curve: list[ThresholdPoint]) -> float:
    """Share of swept thresholds that sit exactly at the minimum cost.

    A wide plateau is good news for robustness and bad news for the claim that the
    model is driving the decision. The UI shows both readings.
    """
    if not curve:
        return 0.0
    costs = np.array([p.total_cost for p in curve])
    return float(np.sum(costs == costs.min()) / costs.size)


def apply_threshold(
    y_prob: Sequence[float] | np.ndarray, threshold: float
) -> np.ndarray:
    """Freeze a validation-chosen threshold onto another split. Positive when p >= tau."""
    return (np.asarray(y_prob, dtype=float).ravel() >= threshold).astype(int)


# ------------------------------------------------------------------ RUL policy


@dataclass(frozen=True)
class LeadTimeDecision:
    """The chosen intervention lead time, plus the U-curve that justifies it."""

    lead_time: int
    chosen_on: str
    point: RulPolicyResult
    curve: list[RulPolicyResult]
    degenerate: bool
    degenerate_reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "lead_time": self.lead_time,
            "chosen_on": self.chosen_on,
            "point": self.point.as_dict(),
            "degenerate": self.degenerate,
            "degenerate_reason": self.degenerate_reason,
            "curve": [p.as_dict() for p in self.curve],
        }


def optimal_lead_time(
    unit_ids: Sequence[Any] | np.ndarray,
    cycles: Sequence[float] | np.ndarray,
    pred_rul: Sequence[float] | np.ndarray,
    failure_cycle: dict[Any, float],
    cfg: CostConfig,
    lead_times: Sequence[int] | None = None,
    chosen_on: str = "validation",
) -> LeadTimeDecision:
    """Pick the cost-minimising lead time. PROJECT_BRIEF.md §3.3.

    The curve this sweeps is the hero chart: too late means unplanned failures,
    too early means throwing away good component life, and the bottom of the U is
    the recommendation. Ties break toward the *earlier* lead time here — unlike
    the classification threshold, the two directions are not symmetric. Being a
    cycle early costs a known sliver of component life; being a cycle late risks
    the whole unplanned-failure bill.
    """
    grid = list(range(0, 61)) if lead_times is None else list(lead_times)
    if not grid:
        raise ValueError("cannot optimise a lead time over an empty grid")

    curve = rul_lead_time_sweep(unit_ids, cycles, pred_rul, failure_cycle, cfg, lead_times=grid)
    costs = np.array([r.total_cost for r in curve])
    tied = np.flatnonzero(costs == costs.min())
    best = curve[int(tied[-1])]  # latest index == longest tied lead time == most margin

    degenerate, reason = _diagnose_lead_time_degeneracy(best, curve, cfg)
    return LeadTimeDecision(
        lead_time=best.lead_time,
        chosen_on=chosen_on,
        point=best,
        curve=curve,
        degenerate=degenerate,
        degenerate_reason=reason,
    )


def _diagnose_lead_time_degeneracy(
    best: RulPolicyResult, curve: list[RulPolicyResult], cfg: CostConfig
) -> tuple[bool, str | None]:
    """Say so when the optimum is pinned to a grid edge or saves nothing."""
    fleet = best.caught + best.missed
    never_act = fleet * cfg.cost_unplanned_event

    if best.total_cost >= never_act:
        return True, (
            f"The optimal lead-time policy saves nothing against run-to-failure "
            f"({cfg.currency} {never_act:,.0f} either way). At these costs, intervening early "
            f"is not worth it on this fleet."
        )
    if best.lead_time == curve[-1].lead_time:
        return True, (
            f"The optimum sits at the top of the swept range ({best.lead_time} cycles), so the "
            f"true optimum may be higher. Widen the lead-time grid before acting on this."
        )
    if best.lead_time == curve[0].lead_time and len(curve) > 1:
        return True, (
            f"The optimum sits at the bottom of the swept range ({best.lead_time} cycles): "
            f"intervening as late as possible is best, which usually means the scrap value of "
            f"remaining life outweighs the failure risk at these costs."
        )
    return False, None
