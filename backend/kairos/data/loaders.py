"""Dataset loading — PROJECT_BRIEF.md §6.

Three sources, in preference order: the full download in ``data/raw``, the small
vendored slice in ``data/samples``, and a physics-based synthetic generator. The
demo must survive a dead venue network, so every loader degrades without asking.

Two things this module deliberately does NOT do:

* It does not drop the AI4I leakage columns. They are loaded, flagged as
  *candidates*, and left for the profiler to catch. Silently dropping them would
  make the agent's "I found target leakage" finding a scripted lie; catching them
  for real is the whole point.
* It does not engineer features. That is ``features.py``, and it happens after a
  human approves the plan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TaskType = Literal["rul_regression", "binary_classification"]

#: Standard piecewise-linear RUL cap for C-MAPSS (brief §6.1, Trap 2). True RUL is
#: only linear near end of life; uncapped labels ask the model to predict an
#: unknowable number during the healthy phase.
RUL_CAP = 125

CMAPSS_COLUMNS = (
    ["unit_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

#: AI4I columns that are components of the target, not predictors of it (Trap 4).
AI4I_LEAKAGE_COLUMNS = ["twf", "hdf", "pwf", "osf", "rnf"]
#: AI4I columns that identify a row rather than describe it.
AI4I_ID_COLUMNS = ["udi", "product_id"]

AI4I_RENAMES = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}


def project_root() -> Path:
    """Repo root, whether running from the host venv or inside the container."""
    for candidate in (Path("/app").parent, Path(__file__).resolve().parents[3]):
        if (candidate / "data").is_dir():
            return candidate
    return Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    # The container bind-mounts ./data at /data; the host venv sees ./data.
    return Path("/data") if Path("/data").is_dir() else project_root() / "data"


@dataclass
class Dataset:
    """A loaded dataset plus the provenance needed to defend any number from it."""

    name: str
    task_type: TaskType
    frame: pd.DataFrame
    target: str
    source: Literal["full", "sample", "synthetic"]
    citation: str
    group_col: str | None = None
    time_col: str | None = None
    holdout: pd.DataFrame | None = None       # C-MAPSS ships its own truncated test set
    leakage_candidates: list[str] = field(default_factory=list)
    id_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.frame)

    @property
    def n_groups(self) -> int | None:
        return int(self.frame[self.group_col].nunique()) if self.group_col else None

    def feature_columns(self, exclude: list[str] | None = None) -> list[str]:
        """Columns usable as features once identifiers, target and leaks are removed."""
        banned = set(exclude or []) | {self.target} | set(self.id_columns)
        if self.group_col:
            banned.add(self.group_col)
        return [c for c in self.frame.columns if c not in banned]

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task_type": self.task_type,
            "rows": self.n_rows,
            "columns": len(self.frame.columns),
            "target": self.target,
            "group_col": self.group_col,
            "time_col": self.time_col,
            "groups": self.n_groups,
            "source": self.source,
            "citation": self.citation,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------- C-MAPSS


def _read_cmapss_table(path: Path) -> pd.DataFrame:
    """Whitespace-delimited, with trailing separators that create phantom columns."""
    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    frame = frame.dropna(axis=1, how="all")
    if frame.shape[1] != len(CMAPSS_COLUMNS):
        raise ValueError(
            f"{path.name}: expected {len(CMAPSS_COLUMNS)} columns, found {frame.shape[1]}"
        )
    frame.columns = CMAPSS_COLUMNS
    return frame


def add_cmapss_rul_labels(frame: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Piecewise-linear RUL for a *training* trajectory that runs to failure.

    The last cycle recorded for a unit IS its failure, so RUL(t) = T - t.
    """
    out = frame.copy()
    failure_cycle = out.groupby("unit_id")["cycle"].transform("max")
    out["rul"] = (failure_cycle - out["cycle"]).clip(upper=cap)
    return out


def add_cmapss_test_rul_labels(
    frame: pd.DataFrame, final_rul: pd.Series, cap: int = RUL_CAP
) -> pd.DataFrame:
    """Piecewise-linear RUL for the *test* set, which is truncated before failure.

    This is the single most-fumbled detail in C-MAPSS (brief §6.1). Test
    trajectories stop at some arbitrary point short of failure, and the remaining
    life at that final recorded cycle lives in a separate RUL_FD00x.txt file. So::

        RUL(t) = (last_recorded_cycle - t) + final_rul_from_file

    Treating the last recorded cycle as the failure — the way the training labels
    work — understates every test label and produces a model that looks superb in
    validation and is useless in the field.
    """
    out = frame.copy()
    last_cycle = out.groupby("unit_id")["cycle"].transform("max")
    remaining_at_end = out["unit_id"].map(final_rul)
    if remaining_at_end.isna().any():
        missing = sorted(out.loc[remaining_at_end.isna(), "unit_id"].unique())
        raise ValueError(f"no final RUL supplied for units {missing}")
    out["rul"] = (last_cycle - out["cycle"] + remaining_at_end).clip(upper=cap)
    return out


def load_cmapss(subset: str = "FD001", prefer: str = "auto") -> Dataset:
    """NASA C-MAPSS turbofan degradation -> RUL regression."""
    root = data_dir()
    full = root / "raw" / "cmapss"
    sample = root / "samples"

    use_full = (full / f"train_{subset}.txt").exists() and prefer in {"auto", "full"}
    if use_full:
        train = _read_cmapss_table(full / f"train_{subset}.txt")
        test = _read_cmapss_table(full / f"test_{subset}.txt")
        final_rul = pd.read_csv(full / f"RUL_{subset}.txt", header=None)[0]
        source: Any = "full"
    elif (sample / "cmapss_fd001_train_sample.txt").exists() and prefer in {"auto", "sample"}:
        if subset != "FD001":
            log.warning("only FD001 is vendored offline; falling back to it for %s", subset)
            subset = "FD001"
        train = _read_cmapss_table(sample / "cmapss_fd001_train_sample.txt")
        test = _read_cmapss_table(sample / "cmapss_fd001_test_sample.txt")
        final_rul = pd.read_csv(sample / "cmapss_fd001_rul_sample.txt", header=None)[0]
        source = "sample"
    else:
        log.warning("no C-MAPSS data found; generating synthetic turbofan fleet")
        return synthetic_turbofan()

    # The RUL file is positional: its i-th line is the i-th unit in sorted order.
    units = sorted(test["unit_id"].unique())
    if len(final_rul) != len(units):
        raise ValueError(f"RUL file has {len(final_rul)} rows for {len(units)} test units")
    final_rul_by_unit = pd.Series(final_rul.to_numpy(), index=units)

    return Dataset(
        name=f"C-MAPSS {subset}",
        task_type="rul_regression",
        frame=add_cmapss_rul_labels(train),
        holdout=add_cmapss_test_rul_labels(test, final_rul_by_unit),
        target="rul",
        group_col="unit_id",
        time_col="cycle",
        source=source,
        citation=(
            "A. Saxena and K. Goebel (2008), 'Turbofan Engine Degradation Simulation "
            "Data Set', NASA Prognostics Data Repository, NASA Ames Research Center."
        ),
        notes=[
            f"RUL labels capped piecewise-linearly at {RUL_CAP} cycles.",
            "Test trajectories are truncated before failure; their labels combine the "
            f"separate RUL_{subset}.txt ground truth with the cycles still recorded.",
            f"Loaded from the {source} source.",
        ],
    )


# ------------------------------------------------------------------------ AI4I


def load_ai4i(prefer: str = "auto") -> Dataset:
    """UCI AI4I 2020 predictive maintenance -> binary classification."""
    root = data_dir()
    full = root / "raw" / "ai4i" / "ai4i2020.csv"
    sample = root / "samples" / "ai4i2020_sample.csv"

    if full.exists() and prefer in {"auto", "full"}:
        path, source = full, "full"
    elif sample.exists() and prefer in {"auto", "sample"}:
        path, source = sample, "sample"
    else:
        log.warning("no AI4I data found; generating synthetic machine log")
        return synthetic_machine_log()

    # utf-8-sig: the UCI file ships with a BOM that otherwise corrupts the first header.
    frame = pd.read_csv(path, encoding="utf-8-sig").rename(columns=AI4I_RENAMES)

    return Dataset(
        name="AI4I 2020",
        task_type="binary_classification",
        frame=frame,
        target="machine_failure",
        group_col=None,
        time_col=None,
        source=source,  # type: ignore[arg-type]
        citation=(
            "S. Matzka (2020), 'AI4I 2020 Predictive Maintenance Dataset', "
            "UCI Machine Learning Repository. DOI 10.24432/C5HS5C."
        ),
        leakage_candidates=list(AI4I_LEAKAGE_COLUMNS),
        id_columns=list(AI4I_ID_COLUMNS),
        notes=[
            "TWF/HDF/PWF/OSF/RNF are failure-mode flags that COMPOSE the target. They are "
            "loaded deliberately so the profiler can catch them rather than being hidden.",
            f"Positive rate {frame['machine_failure'].mean():.4f}; never report accuracy.",
            f"Loaded from the {source} source.",
        ],
    )


# ------------------------------------------------------------------- synthetic


def synthetic_turbofan(
    n_units: int = 60, seed: int = 0, cap: int = RUL_CAP
) -> Dataset:
    """Physics-flavoured degradation fleet, for when there is no data at all.

    Each unit accumulates wear on an exponential path with unit-specific rate and
    noise, and fails when health crosses zero. Sensors are monotone functions of
    health plus noise, and three are deliberately dead so the profiler has
    something real to find even in fallback mode.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_units + 1):
        life = int(rng.integers(130, 330))
        rate = rng.uniform(0.9, 1.1)
        for cycle in range(1, life + 1):
            wear = (cycle / life) ** (1.6 * rate)
            health = 1.0 - wear
            row = {
                "unit_id": unit,
                "cycle": cycle,
                "op_setting_1": rng.normal(0, 0.002),
                "op_setting_2": rng.normal(0, 0.0003),
                "op_setting_3": 100.0,                        # dead by construction
                "sensor_1": 518.67,                           # dead by construction
                "sensor_2": 641.0 + 8.0 * wear + rng.normal(0, 0.5),
                "sensor_3": 1580.0 + 40.0 * wear + rng.normal(0, 6.0),
                "sensor_4": 1400.0 + 60.0 * wear + rng.normal(0, 9.0),
                "sensor_5": 14.62,                            # dead by construction
                "sensor_7": 554.0 - 12.0 * wear + rng.normal(0, 0.9),
                "sensor_9": 9050.0 + 90.0 * wear + rng.normal(0, 22.0),
                "sensor_11": 47.0 + 3.0 * wear + rng.normal(0, 0.27),
                "sensor_12": 521.0 - 9.0 * wear + rng.normal(0, 0.74),
                "sensor_14": 8140.0 - 70.0 * wear + rng.normal(0, 19.0),
                "sensor_15": 8.42 + 0.35 * wear + rng.normal(0, 0.04),
                "sensor_20": 39.0 - 1.6 * wear + rng.normal(0, 0.18),
                "sensor_21": 23.4 - 1.0 * wear + rng.normal(0, 0.11),
                "health": health,
            }
            rows.append(row)
    frame = pd.DataFrame(rows).drop(columns=["health"])
    return Dataset(
        name="Synthetic turbofan fleet",
        task_type="rul_regression",
        frame=add_cmapss_rul_labels(frame, cap=cap),
        target="rul",
        group_col="unit_id",
        time_col="cycle",
        source="synthetic",
        citation="Generated locally by kairos.data.loaders.synthetic_turbofan.",
        notes=[
            "SYNTHETIC DATA — generated because no real dataset was available.",
            "Every figure derived from this run is illustrative, not evidential.",
        ],
    )


def synthetic_machine_log(n_rows: int = 10_000, seed: int = 0) -> Dataset:
    """Synthetic milling-machine log with a physically motivated failure rule."""
    rng = np.random.default_rng(seed)
    air = rng.normal(300.0, 2.0, n_rows)
    process = air + rng.normal(10.0, 1.0, n_rows)
    speed = rng.normal(1538.0, 179.0, n_rows).clip(1168, 2886)
    torque = rng.normal(40.0, 10.0, n_rows).clip(3.8, 76.6)
    wear = rng.uniform(0, 253, n_rows)
    power = torque * speed * 2 * np.pi / 60.0

    # Overstrain, heat dissipation and power failure modes, roughly after the
    # rules described in the AI4I documentation.
    osf = (wear * torque) > 11_000
    hdf = ((process - air) < 8.6) & (speed < 1380)
    pwf = (power < 3_500) | (power > 9_000)
    failure = (osf | hdf | pwf | (rng.random(n_rows) < 0.001)).astype(int)

    frame = pd.DataFrame({
        "udi": np.arange(1, n_rows + 1),
        "product_id": [f"S{i}" for i in range(n_rows)],
        "type": rng.choice(["L", "M", "H"], n_rows, p=[0.6, 0.3, 0.1]),
        "air_temp_k": air,
        "process_temp_k": process,
        "rotational_speed_rpm": speed,
        "torque_nm": torque,
        "tool_wear_min": wear,
        "machine_failure": failure,
        "osf": osf.astype(int),
        "hdf": hdf.astype(int),
        "pwf": pwf.astype(int),
        "twf": (rng.random(n_rows) < 0.005).astype(int),
        "rnf": (rng.random(n_rows) < 0.001).astype(int),
    })
    return Dataset(
        name="Synthetic machine log",
        task_type="binary_classification",
        frame=frame,
        target="machine_failure",
        source="synthetic",
        citation="Generated locally by kairos.data.loaders.synthetic_machine_log.",
        leakage_candidates=list(AI4I_LEAKAGE_COLUMNS),
        id_columns=list(AI4I_ID_COLUMNS),
        notes=[
            "SYNTHETIC DATA — generated because no real dataset was available.",
            "Every figure derived from this run is illustrative, not evidential.",
        ],
    )


# ----------------------------------------------------------------- generic CSV


def load_csv(path: str | Path, target: str | None = None) -> Dataset:
    """Load an arbitrary uploaded CSV. Task type is proposed later, by the profiler."""
    path = Path(path)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if frame.empty:
        raise ValueError(f"{path.name} contains no rows")
    return Dataset(
        name=path.stem,
        task_type="binary_classification",   # provisional; diagnose() revises this
        frame=frame,
        target=target or "",
        source="full",
        citation=f"User upload: {path.name}",
        notes=["Uploaded dataset; task type and target are proposed by the profiler."],
    )
