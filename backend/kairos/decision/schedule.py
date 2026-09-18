"""Work-order scheduling — PROJECT_BRIEF.md §3.5.

Greedy, not an LP. Sort by expected cost avoided per day of slack, fill each day
to the technician limit, and flag whatever cannot be reached in time. Greedy is
correct enough here, takes twenty minutes, and is explainable on a slide — an LP
is none of those things.

The brief's ranking key divides by ``days_until_deadline``, which is zero or
negative for assets already past due. Those are not ranked at all: they go into a
separate "cannot be serviced in time" bucket and consume no capacity, because
dispatching a technician to an engine whose part cannot arrive before failure
spends money to change nothing. Surfacing them is the point — that is the slide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from kairos.decision.cost import CostConfig


@dataclass
class WorkOrder:
    asset_id: str
    asset_label: str
    risk: float                       # probability, or predicted RUL for regression
    deadline: float                   # cycle or day index by which action must start
    days_until_deadline: float
    expected_cost_if_acted: float
    expected_cost_if_ignored: float
    recommended_action: str
    top_drivers: list[dict[str, Any]] = field(default_factory=list)
    scheduled_day: int | None = None
    serviceable: bool = True
    reason: str | None = None
    currency: str = "INR"

    @property
    def cost_avoided(self) -> float:
        return max(self.expected_cost_if_ignored - self.expected_cost_if_acted, 0.0)

    @property
    def priority(self) -> float:
        """Cost avoided per day of slack. Tighter deadlines rank higher."""
        return self.cost_avoided / max(self.days_until_deadline, 0.5)

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "asset_label": self.asset_label,
            "risk": self.risk,
            "deadline": self.deadline,
            "days_until_deadline": self.days_until_deadline,
            "expected_cost_if_acted": self.expected_cost_if_acted,
            "expected_cost_if_ignored": self.expected_cost_if_ignored,
            "cost_avoided": self.cost_avoided,
            "priority": self.priority,
            "recommended_action": self.recommended_action,
            "act_by": f"cycle {self.deadline:.0f}",
            "top_drivers": self.top_drivers,
            "scheduled_day": self.scheduled_day,
            "serviceable": self.serviceable,
            "reason": self.reason,
            "currency": self.currency,
        }


@dataclass
class Schedule:
    orders: list[WorkOrder]
    capacity_per_day: int
    horizon_days: int

    @property
    def serviceable(self) -> list[WorkOrder]:
        return [o for o in self.orders if o.serviceable]

    @property
    def unserviceable(self) -> list[WorkOrder]:
        return [o for o in self.orders if not o.serviceable]

    @property
    def cost_avoided(self) -> float:
        return sum(o.cost_avoided for o in self.serviceable)

    @property
    def cost_conceded(self) -> float:
        """What we lose on assets we cannot reach in time. Shown, never hidden."""
        return sum(o.cost_avoided for o in self.unserviceable)

    def summary(self) -> dict[str, Any]:
        return {
            "total": len(self.orders),
            "scheduled": len(self.serviceable),
            "unserviceable": len(self.unserviceable),
            "capacity_per_day": self.capacity_per_day,
            "cost_avoided": self.cost_avoided,
            "cost_conceded": self.cost_conceded,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "orders": [o.as_dict() for o in self.orders],
        }


def schedule_work_orders(
    orders: Sequence[WorkOrder],
    cfg: CostConfig,
    horizon_days: int | None = None,
) -> Schedule:
    """Greedily fill each day with the highest-priority orders that can still be met.

    An order is unserviceable when its deadline has already passed once the spare
    part's lead time is accounted for, or when no day inside its remaining slack
    still has a free technician.
    """
    capacity = max(int(cfg.technician_capacity_per_day), 0)
    pending = list(orders)
    horizon = horizon_days or (
        int(max((o.days_until_deadline for o in pending), default=0)) + 1
    )

    for order in pending:
        if order.days_until_deadline <= 0:
            order.serviceable = False
            order.scheduled_day = None
            order.reason = (
                f"Deadline already passed once the {cfg.part_lead_time_cycles}-cycle spare-part "
                f"lead time is counted. This asset cannot be saved by scheduling; it needs an "
                f"expedited part or an accepted failure."
            )

    ranked = sorted(
        (o for o in pending if o.serviceable), key=lambda o: -o.priority
    )
    used: dict[int, int] = {}
    if capacity == 0:
        for order in ranked:
            order.serviceable = False
            order.reason = "No technician capacity configured."
    else:
        for order in ranked:
            latest = int(min(order.days_until_deadline, horizon))
            for day in range(1, latest + 1):
                if used.get(day, 0) < capacity:
                    used[day] = used.get(day, 0) + 1
                    order.scheduled_day = day
                    break
            else:
                order.serviceable = False
                order.reason = (
                    f"Every day before its deadline (day {latest}) is already full at "
                    f"{capacity} technician(s) per day, and higher-value assets took those "
                    f"slots."
                )

    ordered = sorted(
        pending,
        key=lambda o: (not o.serviceable, o.scheduled_day or 0, -o.priority),
    )
    return Schedule(orders=ordered, capacity_per_day=capacity, horizon_days=horizon)
