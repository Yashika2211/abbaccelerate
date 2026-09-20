"""Node 7b: schedule — turn the champion's predictions into a work order queue."""

from __future__ import annotations

from typing import Any

import numpy as np

from kairos.agent.runtime import artifacts, step
from kairos.agent.state import KairosState
from kairos.decision.schedule import WorkOrder, schedule_work_orders

MAX_ORDERS = 25


def _rank(contributions, features: list[str], top_n: int) -> list[dict[str, Any]]:
    order = np.argsort(-np.abs(contributions))[:top_n]
    return [
        {
            "feature": features[i],
            "shap_value": float(contributions[i]),
            "direction": "increases risk" if contributions[i] > 0 else "reduces risk",
        }
        for i in order
    ]


def _drivers_for(store, row_index: Any, top_n: int = 3) -> list[dict[str, Any]]:
    """Per-asset SHAP drivers.

    The global explanation samples 500 rows, and the riskiest assets are usually
    not among them. Rather than return nothing — which would leave work orders
    citing no cause at all — the retained explainer is evaluated on that single
    row. One row of TreeSHAP is sub-millisecond, so this costs nothing.
    """
    features = store.feature_names
    values = store.extra.get("shap_values")
    sample = store.extra.get("shap_sample")

    if values is not None and sample is not None:
        positions = {idx: i for i, idx in enumerate(sample.index)}
        pos = positions.get(row_index)
        if pos is not None:
            return _rank(values[pos], features, top_n)

    explainer = store.extra.get("shap_explainer")
    if explainer is None or store.split is None:
        return []
    try:
        from kairos.agent.nodes.explain import shap_for

        row = store.split.test.loc[[row_index], features]
        single = shap_for(explainer, row)
        return _rank(single[0], features, top_n) if single is not None else []
    except Exception:  # noqa: BLE001 - a missing driver list is not worth a failed run
        return []


def node_schedule(state: KairosState) -> dict[str, Any]:
    run_id = state["run_id"]
    store = artifacts(run_id)
    board = store.leaderboard
    cfg = store.cost_config
    task_type = state.get("task_type") or "binary_classification"

    with step(run_id, "schedule", inputs={"capacity": cfg.technician_capacity_per_day if cfg else None}) as box:
        if board is None or store.split is None or cfg is None:
            box["summary"] = "No decisions to schedule."
            return {"decisions": []}

        champion = board.best
        trial = store.models.get(champion.model_id)
        if trial is None:
            box["summary"] = "Champion model unavailable; nothing scheduled."
            return {"decisions": []}

        test = store.split.test
        orders: list[WorkOrder] = []

        if task_type == "binary_classification":
            probabilities = trial.test_prediction
            threshold = champion.operating_point
            flagged = np.flatnonzero(probabilities >= threshold)
            # Highest risk first; the scheduler re-ranks by cost avoided per day.
            flagged = flagged[np.argsort(-probabilities[flagged])][:MAX_ORDERS]
            for pos in flagged:
                idx = test.index[pos]
                # No time column exists on this dataset, so urgency is derived from
                # risk: the more likely the failure, the less slack we assume.
                slack_days = float(max(1, round((1 - probabilities[pos]) * 10)))
                orders.append(WorkOrder(
                    asset_id=str(idx),
                    asset_label=f"Unit {idx}",
                    risk=float(probabilities[pos]),
                    deadline=slack_days,
                    deadline_unit="day",
                    days_until_deadline=slack_days,
                    expected_cost_if_acted=cfg.cost_planned_event,
                    expected_cost_if_ignored=float(probabilities[pos]) * cfg.cost_unplanned_event,
                    recommended_action="Inspect and replace if degraded",
                    top_drivers=_drivers_for(store, idx),
                    currency=cfg.currency,
                ))
        else:
            lead_time = champion.operating_point
            predictions = trial.test_prediction
            frame = test.assign(_pred=predictions)
            for unit, group in frame.groupby("unit_id"):
                group = group.sort_values("cycle")
                alerted = group[group._pred <= lead_time]
                if alerted.empty:
                    continue
                first = alerted.iloc[0]
                failure_cycle = float(group.cycle.max() + group.rul.min())
                action_cycle = float(first.cycle) + cfg.part_lead_time_cycles
                slack = failure_cycle - action_cycle
                orders.append(WorkOrder(
                    asset_id=str(int(unit)),
                    asset_label=f"Engine {int(unit)}",
                    risk=float(first._pred),
                    deadline=failure_cycle,
                    deadline_unit="cycle",
                    days_until_deadline=slack,
                    expected_cost_if_acted=cfg.cost_planned_event,
                    expected_cost_if_ignored=cfg.cost_unplanned_event,
                    recommended_action="Schedule overhaul",
                    top_drivers=_drivers_for(store, first.name),
                    currency=cfg.currency,
                ))
            orders = sorted(orders, key=lambda o: o.days_until_deadline)[:MAX_ORDERS]

        schedule = schedule_work_orders(orders, cfg)
        store.extra["schedule"] = schedule

        summary = schedule.summary()
        findings = [*state.get("findings", [])]
        if summary["unserviceable"]:
            findings.append(
                f"{summary['unserviceable']} asset(s) cannot be serviced in time at "
                f"{cfg.technician_capacity_per_day} technician(s) per day — "
                f"{cfg.currency} {schedule.cost_conceded:,.0f} of avoidable cost is conceded. "
                f"Kairos shows these rather than quietly dropping them."
            )

        box["summary"] = (
            f"{summary['scheduled']} work order(s) scheduled, "
            f"{summary['unserviceable']} unserviceable, "
            f"{cfg.currency} {schedule.cost_avoided:,.0f} of cost avoided."
        )
        box["reasoning"] = findings[-1] if summary["unserviceable"] else (
            "Every flagged asset fits inside the available technician capacity."
        )
        box["payload"] = schedule.as_dict()

        return {
            "decisions": [o.as_dict() for o in schedule.orders],
            "findings": findings,
        }
