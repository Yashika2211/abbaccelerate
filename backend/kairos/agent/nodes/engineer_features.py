"""Node 4: engineer_features — executes the approved plan."""

from __future__ import annotations

from typing import Any

from kairos.agent.runtime import artifacts, step
from kairos.agent.state import KairosState
from kairos.data.features import (
    drop_columns,
    engineer_ai4i_features,
    engineer_timeseries_features,
)
from kairos.data.splits import make_splits


def node_engineer_features(state: KairosState) -> dict[str, Any]:
    run_id = state["run_id"]
    plan = state.get("plan") or {}
    store = artifacts(run_id)
    dataset = store.dataset
    if dataset is None:
        raise ValueError("no dataset loaded; profile_dataset must run first")

    with step(run_id, "engineer_features", inputs=plan) as box:
        drops = [c for c in plan.get("drop_columns", []) if c != plan.get("target")]
        notes: list[str] = []
        added: list[str] = []

        def prepare(frame):
            nonlocal added, notes
            trimmed = drop_columns(frame, drops)
            if plan.get("feature_strategy") == "timeseries_windows":
                sensors = [
                    c for c in trimmed.frame.columns
                    if c.startswith(("sensor_", "op_setting_"))
                ]
                result = engineer_timeseries_features(
                    trimmed.frame,
                    dataset.group_col,
                    dataset.time_col,
                    sensors,
                    windows=tuple(plan.get("window_sizes") or (5, 10, 20)),
                )
            elif plan.get("feature_strategy") == "physics":
                result = engineer_ai4i_features(trimmed.frame)
            else:
                result = trimmed
            added = result.added_columns
            notes = [*trimmed.notes, *result.notes]
            return result.frame

        dataset.frame = prepare(dataset.frame)
        if dataset.holdout is not None:
            dataset.holdout = prepare(dataset.holdout)

        target = plan.get("target") or dataset.target
        excluded = {target, dataset.group_col, dataset.time_col, "type"}
        features = [
            c for c in dataset.frame.columns
            if c not in excluded and dataset.frame[c].dtype.kind in "ifb"
        ]

        split = make_splits(dataset)
        store.split = split
        store.feature_names = features

        box["summary"] = (
            f"Dropped {len(drops)}, added {len(added)}, "
            f"{len(features)} features into a {split.strategy} split "
            f"({split.sizes()})."
        )
        box["reasoning"] = " ".join(notes[:3])
        box["payload"] = {
            "dropped": drops,
            "added_count": len(added),
            "n_features": len(features),
            "split": split.summary(),
        }
        return {
            "findings": [*state.get("findings", []), *split.notes[:1]],
        }
