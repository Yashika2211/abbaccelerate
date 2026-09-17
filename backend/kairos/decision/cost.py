"""Cost model — PROJECT_BRIEF.md §3.

Pure functions. No ML imports, no I/O, no global state. Everything here is
unit-tested before a single model is trained, because every headline number in
Kairos is produced by this file and a wrong cost matrix is worse than no product.

Two decision problems live here:

* **Classification** — the decision variable is a probability threshold.
* **RUL regression** — the decision variable is a lead time ``L`` in cycles:
  intervene at the first cycle where predicted RUL falls to ``L`` or below.

Both are swept exhaustively rather than solved analytically. The grids are small,
the evaluation is vectorised, and an exhaustive sweep is something you can defend
to a judge in one sentence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

import numpy as np

# Cost fields that are physically impossible to be negative.
_NON_NEGATIVE = (
    "c_unplanned_repair",
    "c_planned_repair",
    "c_inspection",
    "downtime_rate_per_hour",
    "unplanned_downtime_hours",
    "planned_downtime_hours",
    "c_secondary_damage",
    "part_lead_time_cycles",
    "value_per_remaining_cycle",
)


@dataclass(frozen=True)
class CostConfig:
    """Plant economics. One object drives thresholds, lead times, ranking and scheduling.

    Frozen so a simulation can never mutate the run's baseline config by accident;
    use :meth:`with_changes` to derive a variant for the what-if simulator.
    """

    c_unplanned_repair: float           # emergency parts + labour
    c_planned_repair: float             # scheduled parts + labour
    c_inspection: float                 # technician dispatched, nothing found
    downtime_rate_per_hour: float       # lost production per hour of stoppage
    unplanned_downtime_hours: float     # typically 3-10x planned
    planned_downtime_hours: float
    c_secondary_damage: float = 0.0     # collateral from a catastrophic failure
    part_lead_time_cycles: int = 0      # an alert is useless if it lands too late
    value_per_remaining_cycle: float = 0.0   # cost of scrapping good remaining life
    technician_capacity_per_day: int = 999   # scheduling constraint
    currency: str = "INR"

    # --- annualisation -----------------------------------------------------
    # Neither dataset carries a calendar, so rupees-per-year needs an explicit
    # assumption. These are surfaced in the UI next to every annual figure;
    # a number whose denominator is invisible is not a number.
    cycles_per_year: float = 500.0            # RUL: cycles per asset per year
    observations_per_asset_year: float = 10_000.0  # classification: rows per asset-year

    def __post_init__(self) -> None:
        for field_name in _NON_NEGATIVE:
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be >= 0, got {value}")
        if self.technician_capacity_per_day < 0:
            raise ValueError("technician_capacity_per_day must be >= 0")
        if self.cycles_per_year <= 0 or self.observations_per_asset_year <= 0:
            raise ValueError("annualisation rates must be > 0")

    def with_changes(self, **changes: Any) -> CostConfig:
        """Derive a variant. This is the entry point for the what-if simulator."""
        return replace(self, **changes)

    # -- derived unit costs -------------------------------------------------

    @property
    def cost_planned_event(self) -> float:
        """Total cost of a repair we saw coming."""
        return self.c_planned_repair + self.planned_downtime_hours * self.downtime_rate_per_hour

    @property
    def cost_unplanned_event(self) -> float:
        """Total cost of a repair we did not see coming."""
        return (
            self.c_unplanned_repair
            + self.unplanned_downtime_hours * self.downtime_rate_per_hour
            + self.c_secondary_damage
        )

    @property
    def consequence_ratio(self) -> float:
        """How many false alarms one missed failure is worth.

        This single number explains the entire product: when it is large, recall
        is worth more than precision, and the best-AUC model stops being the best
        model. Infinite when inspection is free.
        """
        if self.c_inspection <= 0:
            return float("inf")
        return self.cost_unplanned_event / self.c_inspection


@dataclass(frozen=True)
class ClassificationCosts:
    """The per-outcome cost matrix, in currency units."""

    tn: float
    fp: float
    tp: float
    fn: float

    def as_dict(self) -> dict[str, float]:
        return {"tn": self.tn, "fp": self.fp, "tp": self.tp, "fn": self.fn}


def classification_cost_matrix(cfg: CostConfig) -> ClassificationCosts:
    """PROJECT_BRIEF.md §3.2.

    TN -> 0                     nothing happened, nobody was sent
    FP -> inspection            technician dispatched, nothing found
    TP -> planned repair        caught it, fixed it on our schedule
    FN -> unplanned failure     missed it; this is the number that dominates
    """
    return ClassificationCosts(
        tn=0.0,
        fp=cfg.c_inspection,
        tp=cfg.cost_planned_event,
        fn=cfg.cost_unplanned_event,
    )


def confusion_counts(y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray) -> dict[str, int]:
    """Counts of tn/fp/tp/fn. Written by hand so this module stays dependency-free."""
    t = np.asarray(y_true).astype(bool).ravel()
    p = np.asarray(y_pred).astype(bool).ravel()
    if t.shape != p.shape:
        raise ValueError(f"y_true and y_pred must be the same length, got {t.shape} and {p.shape}")
    return {
        "tn": int(np.sum(~t & ~p)),
        "fp": int(np.sum(~t & p)),
        "fn": int(np.sum(t & ~p)),
        "tp": int(np.sum(t & p)),
    }


def expected_cost(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    cfg: CostConfig,
) -> float:
    """Total expected cost of a set of hard decisions. PROJECT_BRIEF.md §3.2."""
    counts = confusion_counts(y_true, y_pred)
    m = classification_cost_matrix(cfg)
    return (
        counts["tn"] * m.tn
        + counts["fp"] * m.fp
        + counts["tp"] * m.tp
        + counts["fn"] * m.fn
    )


def annualize(total_cost: float, exposure_asset_years: float) -> float:
    """Scale an observed cost to cost per asset per year.

    ``exposure_asset_years`` is how much asset-time the observed cost covers.
    Zero exposure yields 0.0 rather than an exception: an empty test fold must
    render an empty state, not crash the demo.
    """
    if exposure_asset_years <= 0:
        return 0.0
    return total_cost / exposure_asset_years


def classification_exposure_asset_years(n_rows: int, cfg: CostConfig) -> float:
    """Asset-years represented by ``n_rows`` observations, per the stated assumption."""
    return n_rows / cfg.observations_per_asset_year


def cost_breakdown(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    cfg: CostConfig,
) -> dict[str, Any]:
    """Everything the UI needs to show *why* a number is what it is."""
    counts = confusion_counts(y_true, y_pred)
    m = classification_cost_matrix(cfg)
    contributions = {
        "tn": counts["tn"] * m.tn,
        "fp": counts["fp"] * m.fp,
        "tp": counts["tp"] * m.tp,
        "fn": counts["fn"] * m.fn,
    }
    total = sum(contributions.values())
    n_rows = len(np.asarray(y_true).ravel())
    exposure = classification_exposure_asset_years(n_rows, cfg)
    return {
        "counts": counts,
        "unit_costs": m.as_dict(),
        "contributions": contributions,
        "total_cost": total,
        "cost_per_asset_year": annualize(total, exposure),
        "exposure_asset_years": exposure,
        "n_rows": n_rows,
        "currency": cfg.currency,
        "annualization_note": (
            f"{cfg.observations_per_asset_year:,.0f} observations assumed to represent "
            f"one asset-year."
        ),
    }


# --------------------------------------------------------------------------- RUL


@dataclass(frozen=True)
class RulPolicyResult:
    """Outcome of running a fixed-lead-time policy over a fleet."""

    lead_time: int
    total_cost: float
    cost_per_asset_year: float
    caught: int            # intervened before failure
    missed: int            # alerted too late, or never alerted
    never_alerted: int     # subset of missed: the model stayed silent to the end
    wasted_life_cycles: float
    exposure_asset_years: float
    currency: str = "INR"

    def as_dict(self) -> dict[str, Any]:
        return {
            "lead_time": self.lead_time,
            "total_cost": self.total_cost,
            "cost_per_asset_year": self.cost_per_asset_year,
            "caught": self.caught,
            "missed": self.missed,
            "never_alerted": self.never_alerted,
            "wasted_life_cycles": self.wasted_life_cycles,
            "exposure_asset_years": self.exposure_asset_years,
            "currency": self.currency,
        }


def rul_policy_cost(
    unit_ids: Sequence[Any] | np.ndarray,
    cycles: Sequence[float] | np.ndarray,
    pred_rul: Sequence[float] | np.ndarray,
    failure_cycle: dict[Any, float],
    lead_time: int,
    cfg: CostConfig,
) -> RulPolicyResult:
    """Cost of the policy "intervene at the first cycle where predicted RUL <= L".

    PROJECT_BRIEF.md §3.3. For each unit with true failure cycle ``T``::

        t_alert  = min{ t : pred_RUL(t) <= L }, or infinity if never
        t_action = t_alert + part_lead_time_cycles

        t_action < T  ->  planned repair + downtime + (T - t_action) * value_per_remaining_cycle
        otherwise     ->  unplanned failure

    The ``(T - t_action)`` term is what makes the cost-vs-lead-time curve U-shaped:
    alert too late and you eat an unplanned failure, alert too early and you throw
    away good component life. The bottom of that U is the product.
    """
    units = np.asarray(unit_ids)
    t = np.asarray(cycles, dtype=float)
    rul = np.asarray(pred_rul, dtype=float)
    if not (units.shape == t.shape == rul.shape):
        raise ValueError("unit_ids, cycles and pred_rul must all be the same length")

    total = 0.0
    caught = missed = never_alerted = 0
    wasted_life = 0.0
    total_cycles = 0.0

    for unit in dict.fromkeys(units.tolist()):  # stable order, no numpy unique sort surprises
        if unit not in failure_cycle:
            raise KeyError(f"no true failure cycle supplied for unit {unit!r}")
        T = float(failure_cycle[unit])
        mask = units == unit
        unit_cycles = t[mask]
        unit_rul = rul[mask]
        order = np.argsort(unit_cycles, kind="stable")
        unit_cycles, unit_rul = unit_cycles[order], unit_rul[order]
        total_cycles += T

        alerted = np.flatnonzero(unit_rul <= lead_time)
        if alerted.size == 0:
            total += cfg.cost_unplanned_event
            missed += 1
            never_alerted += 1
            continue

        t_action = float(unit_cycles[alerted[0]]) + cfg.part_lead_time_cycles
        if t_action < T:
            remaining = T - t_action
            total += cfg.cost_planned_event + remaining * cfg.value_per_remaining_cycle
            wasted_life += remaining
            caught += 1
        else:
            total += cfg.cost_unplanned_event
            missed += 1

    exposure = total_cycles / cfg.cycles_per_year
    return RulPolicyResult(
        lead_time=lead_time,
        total_cost=total,
        cost_per_asset_year=annualize(total, exposure),
        caught=caught,
        missed=missed,
        never_alerted=never_alerted,
        wasted_life_cycles=wasted_life,
        exposure_asset_years=exposure,
        currency=cfg.currency,
    )


def rul_lead_time_sweep(
    unit_ids: Sequence[Any] | np.ndarray,
    cycles: Sequence[float] | np.ndarray,
    pred_rul: Sequence[float] | np.ndarray,
    failure_cycle: dict[Any, float],
    cfg: CostConfig,
    lead_times: Iterable[int] | None = None,
) -> list[RulPolicyResult]:
    """Evaluate the policy across a grid of lead times. The hero chart's data."""
    grid = list(range(0, 61)) if lead_times is None else list(lead_times)
    return [
        rul_policy_cost(unit_ids, cycles, pred_rul, failure_cycle, L, cfg) for L in grid
    ]
