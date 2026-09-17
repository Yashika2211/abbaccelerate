"""Feature engineering — PROJECT_BRIEF.md §6.1, §6.2.

The trap this module exists to avoid: a rolling window computed over the whole
frame bleeds the end of engine 7 into the start of engine 8. Every window, diff
and trend here is computed **grouped by unit**, and there is a test that builds
a frame where a global window would be visibly wrong and asserts it is not.

Nothing here runs until a human approves the plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

DEFAULT_WINDOWS = (5, 10, 20)


@dataclass
class FeatureResult:
    frame: pd.DataFrame
    added_columns: list[str]
    dropped_columns: list[str]
    recipe: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "rows": len(self.frame),
            "added": len(self.added_columns),
            "dropped": len(self.dropped_columns),
            "added_columns": self.added_columns,
            "dropped_columns": self.dropped_columns,
            "recipe": self.recipe,
            "notes": self.notes,
        }


def _rolling_trend(values: pd.Series, window: int) -> pd.Series:
    """Slope of a least-squares line over a trailing window.

    Degradation shows up as a change in *direction* before it shows up as a change
    in level, so the slope is often the earliest warning available.

    The regressor is rebuilt per call because the leading rows of every unit have
    partial windows: with ``min_periods=2`` pandas hands this function 2 points
    before it hands it ``window`` points, and a fixed-length x vector would not
    align. The slope over those early rows is a genuine slope over the points that
    exist, not a padded approximation.
    """

    def slope(window_values: np.ndarray) -> float:
        n = window_values.size
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=float)
        x_centred = x - x.mean()
        denominator = float((x_centred**2).sum())
        if denominator == 0.0:
            return 0.0
        return float(np.dot(window_values - window_values.mean(), x_centred) / denominator)

    return values.rolling(window, min_periods=2).apply(slope, raw=True)


def engineer_timeseries_features(
    frame: pd.DataFrame,
    group_col: str,
    time_col: str,
    sensor_columns: Sequence[str],
    windows: Sequence[int] = DEFAULT_WINDOWS,
    add_trend: bool = True,
) -> FeatureResult:
    """Per-unit rolling statistics, first differences, trend slopes and cycle count.

    Every operation is grouped by ``group_col``. A global rolling window would let
    the final cycles of one engine contaminate the first cycles of the next, which
    is leakage that produces a better validation score and a worse model.
    """
    if group_col not in frame.columns:
        raise KeyError(f"group column {group_col!r} not in frame")
    if time_col not in frame.columns:
        raise KeyError(f"time column {time_col!r} not in frame")

    out = frame.sort_values([group_col, time_col]).reset_index(drop=True)
    sensors = [c for c in sensor_columns if c in out.columns]
    grouped = out.groupby(group_col, sort=False)
    added: list[str] = []
    new_columns: dict[str, pd.Series] = {}

    for sensor in sensors:
        series = grouped[sensor]
        for window in windows:
            roll = series.rolling(window, min_periods=1)
            for stat in ("mean", "std", "min", "max"):
                name = f"{sensor}_roll{window}_{stat}"
                new_columns[name] = getattr(roll, stat)().reset_index(level=0, drop=True)
                added.append(name)
        new_columns[f"{sensor}_diff1"] = series.diff().reset_index(level=0, drop=True)
        added.append(f"{sensor}_diff1")
        # Deviation from the unit's own starting condition: normalises away the
        # fact that no two engines leave the factory identical.
        new_columns[f"{sensor}_delta_from_start"] = (
            out[sensor] - grouped[sensor].transform("first")
        )
        added.append(f"{sensor}_delta_from_start")

    if add_trend:
        longest = max(windows)
        for sensor in sensors:
            name = f"{sensor}_trend{longest}"
            new_columns[name] = (
                grouped[sensor]
                .apply(lambda s: _rolling_trend(s, longest))
                .reset_index(level=0, drop=True)
            )
            added.append(name)

    new_columns["cycles_elapsed"] = grouped.cumcount() + 1
    added.append("cycles_elapsed")

    out = pd.concat([out, pd.DataFrame(new_columns, index=out.index)], axis=1)
    # Rolling std is undefined on a unit's first row; zero is the honest value for
    # "no variation observed yet".
    out[added] = out[added].fillna(0.0)

    return FeatureResult(
        frame=out,
        added_columns=added,
        dropped_columns=[],
        recipe={
            "group_col": group_col,
            "time_col": time_col,
            "windows": list(windows),
            "sensors": sensors,
            "trend": add_trend,
        },
        notes=[
            f"All windows computed grouped by {group_col!r}; no engine contaminates another.",
            f"Rolling mean/std/min/max over windows {list(windows)}, plus first difference, "
            f"deviation from each unit's own baseline, and a least-squares trend slope.",
        ],
    )


def engineer_ai4i_features(frame: pd.DataFrame) -> FeatureResult:
    """Physics-derived features for AI4I — PROJECT_BRIEF.md §6.2.

    These are not arbitrary interactions. Each corresponds to a documented failure
    mechanism in the dataset, which is what makes them defensible in a work order:

    * ``power`` — the machine's mechanical power draw. Power failure (PWF) is
      defined directly on this quantity.
    * ``temp_delta`` — process minus air temperature. Heat dissipation failure
      (HDF) occurs when this gap is too small to shed heat.
    * ``wear_x_torque`` — the overstrain (OSF) mechanism: a worn tool under load.
    """
    out = frame.copy()
    added: list[str] = []

    if {"torque_nm", "rotational_speed_rpm"} <= set(out.columns):
        out["power_w"] = out["torque_nm"] * out["rotational_speed_rpm"] * 2 * np.pi / 60.0
        added.append("power_w")
    if {"process_temp_k", "air_temp_k"} <= set(out.columns):
        out["temp_delta_k"] = out["process_temp_k"] - out["air_temp_k"]
        added.append("temp_delta_k")
    if {"tool_wear_min", "torque_nm"} <= set(out.columns):
        out["wear_x_torque"] = out["tool_wear_min"] * out["torque_nm"]
        added.append("wear_x_torque")
    if "type" in out.columns:
        # L/M/H product quality is ordinal, not nominal.
        out["type_ordinal"] = out["type"].map({"L": 0, "M": 1, "H": 2}).fillna(-1).astype(int)
        added.append("type_ordinal")

    return FeatureResult(
        frame=out,
        added_columns=added,
        dropped_columns=[],
        recipe={"physics_features": added},
        notes=[
            "power = torque x speed x 2pi/60, the quantity the PWF failure mode is defined on.",
            "temp_delta = process - air temperature, which drives the HDF mechanism.",
            "wear_x_torque captures the OSF overstrain mechanism: a worn tool under load.",
        ],
    )


def drop_columns(frame: pd.DataFrame, columns: Sequence[str]) -> FeatureResult:
    """Drop the profiler's recommended columns, reporting exactly what went."""
    present = [c for c in columns if c in frame.columns]
    return FeatureResult(
        frame=frame.drop(columns=present),
        added_columns=[],
        dropped_columns=present,
        recipe={"dropped": present},
        notes=[f"Dropped {len(present)} column(s): {', '.join(present) or 'none'}."],
    )
