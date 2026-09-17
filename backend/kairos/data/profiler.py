"""Statistical profiling and leakage detection — PROJECT_BRIEF.md §6.3, §7 node 1.

Deterministic. No LLM. The agent's ``diagnose`` node reads this output and writes
prose about it, but every fact here is computed.

The profiler's job is to notice things before anyone asks: dead sensors, columns
that are really row identifiers, a target that one feature reconstructs almost
perfectly, a class balance that makes accuracy meaningless. Surfacing its own
leakage finding in front of a judge is worth more than a clean-looking model.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from kairos.data.loaders import Dataset

log = logging.getLogger(__name__)

#: A column whose standard deviation is this small carries no usable signal.
CONSTANT_STD = 1e-9
#: Unique-value share above which a column is behaving like a row identifier.
IDENTIFIER_UNIQUE_RATIO = 0.95
#: Share of missing values above which a column is more hole than data.
HIGH_MISSING = 0.5
#: Share of a column held by its single most common value, above which the column
#: is effectively constant. C-MAPSS sensor_6 sits at 0.98 with two distinct values.
NEAR_CONSTANT_DOMINANCE = 0.98
#: P(target | column fires) above which a binary flag is a component of the target.
LEAKAGE_IMPLICATION = 0.90
#: ...and how many times the base rate that has to be, so a common target does not
#: make every column look guilty.
LEAKAGE_LIFT = 5.0
#: Single-feature ROC AUC above which one column alone all but resolves the target.
LEAKAGE_SOLO_AUC = 0.99

_TIME_NAME = re.compile(r"(cycle|time|date|timestamp|step|hour|day|seq)", re.IGNORECASE)
_GROUP_NAME = re.compile(r"(unit|engine|asset|machine|device|serial|group|id)$", re.IGNORECASE)


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    missing_fraction: float
    n_unique: int
    is_constant: bool
    is_binary: bool
    is_identifier: bool
    is_numeric: bool
    mean: float | None = None
    std: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    top_values: dict[str, int] = field(default_factory=dict)
    drift_score: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "missing_fraction": self.missing_fraction,
            "n_unique": self.n_unique,
            "is_constant": self.is_constant,
            "is_binary": self.is_binary,
            "is_identifier": self.is_identifier,
            "is_numeric": self.is_numeric,
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
            "top_values": self.top_values,
            "drift_score": self.drift_score,
        }


@dataclass
class LeakageWarning:
    column: str
    severity: str            # "certain" | "suspected"
    evidence: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "severity": self.severity,
            "evidence": self.evidence,
            "score": self.score,
        }


@dataclass
class DatasetProfile:
    name: str
    n_rows: int
    n_columns: int
    duplicate_rows: int
    columns: list[ColumnProfile]
    constant_columns: list[str]
    identifier_columns: list[str]
    high_missing_columns: list[str]
    candidate_group_columns: list[str]
    candidate_time_columns: list[str]
    proposed_task_type: str | None
    proposed_target: str | None
    class_balance: dict[str, float] | None
    leakage_warnings: list[LeakageWarning]
    #: Columns the dataset's documentation marks as leaky that the detector did NOT
    #: independently flag. Dropped on provenance, and labelled as such.
    documented_leakage_undetected: list[str]
    most_informative_features: list[dict[str, float]]
    recommended_drops: list[str]
    findings: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "duplicate_rows": self.duplicate_rows,
            "columns": [c.as_dict() for c in self.columns],
            "constant_columns": self.constant_columns,
            "identifier_columns": self.identifier_columns,
            "high_missing_columns": self.high_missing_columns,
            "candidate_group_columns": self.candidate_group_columns,
            "candidate_time_columns": self.candidate_time_columns,
            "proposed_task_type": self.proposed_task_type,
            "proposed_target": self.proposed_target,
            "class_balance": self.class_balance,
            "leakage_warnings": [w.as_dict() for w in self.leakage_warnings],
            "documented_leakage_undetected": self.documented_leakage_undetected,
            "most_informative_features": self.most_informative_features,
            "recommended_drops": self.recommended_drops,
            "findings": self.findings,
        }


def _profile_column(series: pd.Series, drift_score: float | None) -> ColumnProfile:
    n = len(series)
    non_null = series.dropna()
    is_numeric = pd.api.types.is_numeric_dtype(series)
    n_unique = int(non_null.nunique())

    std = float(non_null.std()) if is_numeric and len(non_null) > 1 else None
    # "Constant" has to mean near-constant, not exactly constant: C-MAPSS sensor_6
    # holds two distinct values but 98% of rows share one of them, and it is just
    # as useless as a sensor that never moves.
    dominance = float(non_null.value_counts(normalize=True).iloc[0]) if len(non_null) else 1.0
    is_constant = (
        n_unique <= 1
        or (std is not None and std < CONSTANT_STD)
        or dominance >= NEAR_CONSTANT_DOMINANCE
    )

    return ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        missing_fraction=float(series.isna().mean()),
        n_unique=n_unique,
        is_constant=is_constant,
        is_binary=n_unique == 2,
        is_identifier=n > 0 and (n_unique / n) >= IDENTIFIER_UNIQUE_RATIO,
        is_numeric=is_numeric,
        mean=float(non_null.mean()) if is_numeric and len(non_null) else None,
        std=std,
        minimum=float(non_null.min()) if is_numeric and len(non_null) else None,
        maximum=float(non_null.max()) if is_numeric and len(non_null) else None,
        top_values={} if is_numeric else {
            str(k): int(v) for k, v in non_null.value_counts().head(5).items()
        },
        drift_score=drift_score,
    )


def _drift_scores(frame: pd.DataFrame, time_col: str | None) -> dict[str, float]:
    """Standardised mean shift between the first and second half of the data.

    Ordered by time when a time column exists, otherwise by row order. This is a
    cheap proxy, and it is labelled as one in the UI — it flags columns worth
    watching, it does not claim a distribution has drifted.
    """
    ordered = frame.sort_values(time_col) if time_col and time_col in frame else frame
    half = len(ordered) // 2
    if half < 10:
        return {}
    first, second = ordered.iloc[:half], ordered.iloc[half:]
    scores: dict[str, float] = {}
    for col in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        a, b = first[col].dropna(), second[col].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        pooled = float(np.sqrt((a.var() + b.var()) / 2.0))
        if pooled < CONSTANT_STD:
            continue
        scores[col] = float(abs(a.mean() - b.mean()) / pooled)
    return scores


def detect_leakage(
    frame: pd.DataFrame,
    target: str,
    task_type: str,
    exclude: list[str] | None = None,
    sample_size: int = 5_000,
    seed: int = 0,
) -> list[LeakageWarning]:
    """Flag features that are components of the target rather than predictors of it.

    **Mutual information does not work for this**, despite being the obvious choice.
    Measured on AI4I 2020: ``torque_nm`` — a genuine sensor reading — scores 0.317
    of the target's entropy, *higher* than ``hdf`` at 0.304, which is literally one
    of the OR-terms defining the target. Any MI threshold either flags torque as a
    leak or lets hdf through. MI measures how informative a column is, and the most
    informative honest feature outscores a weak dishonest one.

    What does separate them cleanly is **lift**: P(target | column fires) divided by
    the base rate. On the same dataset every leakage flag scores 1.000 against a
    0.034 base rate — a lift of 29x — while every genuine sensor sits exactly at the
    base rate, lift 1.0. No overlap, no threshold tuning.

    For continuous columns the equivalent question is whether one column alone very
    nearly resolves the target, which is a single-feature ROC AUC.

    Mutual information is still computed, but reported as "most informative
    features" rather than dressed up as a leakage verdict.
    """
    exclude = set(exclude or []) | {target}
    candidates = [
        c for c in frame.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(frame[c])
    ]
    if not candidates:
        return []

    work = frame[[*candidates, target]].dropna()
    if len(work) > sample_size:
        work = work.sample(sample_size, random_state=seed)
    if work.empty or work[target].nunique() < 2:
        return []

    y = work[target]
    warnings: list[LeakageWarning] = []

    if task_type != "binary_classification" or y.nunique() != 2:
        return _detect_regression_leakage(work, candidates, target)

    y_bool = y.astype(bool)
    base_rate = float(y_bool.mean())
    if base_rate <= 0:
        return []

    for col in candidates:
        series = work[col]

        # -- binary flag: does firing this column imply the target? -------------
        if series.nunique() == 2:
            fires = series.astype(bool)
            if not fires.any():
                continue
            implication = float(y_bool[fires].mean())
            lift = implication / base_rate
            if implication >= LEAKAGE_IMPLICATION and lift >= LEAKAGE_LIFT:
                warnings.append(
                    LeakageWarning(
                        column=col,
                        severity="certain",
                        evidence=(
                            f"Whenever {col} fires, the target is set {implication:.1%} of the "
                            f"time against a {base_rate:.1%} base rate — a lift of {lift:.0f}x. "
                            f"This column is a component of the target, not a predictor of it."
                        ),
                        score=float(lift),
                    )
                )
            continue

        # -- continuous column: does it alone all but resolve the target? -------
        try:
            from sklearn.metrics import roc_auc_score

            auc = float(roc_auc_score(y_bool, series))
            auc = max(auc, 1.0 - auc)  # a perfectly inverted column leaks just as hard
        except Exception:  # noqa: BLE001 - profiling never breaks a run
            continue
        if auc >= LEAKAGE_SOLO_AUC:
            warnings.append(
                LeakageWarning(
                    column=col,
                    severity="certain",
                    evidence=(
                        f"{col} alone separates the target with ROC AUC {auc:.4f}. A single "
                        f"honest sensor does not do that; this column encodes the answer."
                    ),
                    score=auc,
                )
            )

    return sorted(warnings, key=lambda w: (-w.score, w.column))


def _detect_regression_leakage(
    work: pd.DataFrame, candidates: list[str], target: str
) -> list[LeakageWarning]:
    """For a continuous target, a near-perfect linear relationship is the tell."""
    warnings: list[LeakageWarning] = []
    y = work[target]
    if float(y.std()) == 0:
        return []
    for col in candidates:
        series = work[col]
        if float(series.std()) == 0:
            continue
        r = float(np.corrcoef(series, y)[0, 1])
        if abs(r) >= 0.995:
            warnings.append(
                LeakageWarning(
                    column=col,
                    severity="certain",
                    evidence=(
                        f"{col} correlates with the target at r={r:.4f}. That is not a sensor "
                        f"reading, it is the answer in different units."
                    ),
                    score=abs(r),
                )
            )
    return sorted(warnings, key=lambda w: (-w.score, w.column))


def most_informative_features(
    frame: pd.DataFrame,
    target: str,
    task_type: str,
    exclude: list[str] | None = None,
    top_n: int = 8,
    sample_size: int = 5_000,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Mutual information ranking. Informativeness, explicitly NOT a leakage verdict."""
    exclude = set(exclude or []) | {target}
    candidates = [
        c for c in frame.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(frame[c])
    ]
    if not candidates:
        return []
    work = frame[[*candidates, target]].dropna()
    if len(work) > sample_size:
        work = work.sample(sample_size, random_state=seed)
    if work.empty:
        return []
    try:
        if task_type == "binary_classification":
            mi = mutual_info_classif(work[candidates], work[target], random_state=seed)
        else:
            mi = mutual_info_regression(work[candidates], work[target], random_state=seed)
    except Exception as exc:  # noqa: BLE001
        log.warning("mutual information failed: %s", exc)
        return []
    ranked = sorted(zip(candidates, (float(m) for m in mi)), key=lambda t: -t[1])
    return [{"column": c, "mutual_information": m} for c, m in ranked[:top_n]]


def propose_task(frame: pd.DataFrame, target: str | None) -> tuple[str | None, str | None]:
    """Propose (task_type, target) from the data's shape alone."""
    if target and target in frame.columns:
        y = frame[target]
        if y.nunique() == 2:
            return "binary_classification", target
        return "rul_regression", target

    if "rul" in frame.columns:
        return "rul_regression", "rul"
    for name in ("machine_failure", "failure", "target", "label"):
        if name in frame.columns and frame[name].nunique() == 2:
            return "binary_classification", name
    return None, None


def profile_dataset(dataset: Dataset, seed: int = 0) -> DatasetProfile:
    """Full statistical profile plus the findings a human should see first."""
    frame = dataset.frame
    drift = _drift_scores(frame, dataset.time_col)
    columns = [_profile_column(frame[c], drift.get(c)) for c in frame.columns]

    constant = [c.name for c in columns if c.is_constant]
    identifiers = [
        c.name for c in columns
        if c.is_identifier and c.name not in {dataset.target, dataset.group_col}
    ]
    high_missing = [c.name for c in columns if c.missing_fraction > HIGH_MISSING]

    group_candidates = [
        c.name for c in columns
        if _GROUP_NAME.search(c.name) and 1 < c.n_unique < len(frame) * 0.5
    ]
    time_candidates = [c.name for c in columns if _TIME_NAME.search(c.name) and c.is_numeric]

    task_type, target = propose_task(frame, dataset.target or None)

    class_balance = None
    if task_type == "binary_classification" and target:
        counts = frame[target].value_counts(normalize=True)
        class_balance = {str(k): float(v) for k, v in counts.items()}

    excluded = [*identifiers, *dataset.id_columns]
    leakage = (
        detect_leakage(frame, target, task_type, exclude=excluded, seed=seed)
        if target and task_type
        else []
    )
    detected = {w.column for w in leakage}
    # Columns the dataset documentation calls leaky that our statistics did not
    # independently convict. We still drop them, but on provenance, and we say so.
    documented_undetected = sorted(
        c for c in dataset.leakage_candidates if c in frame.columns and c not in detected
    )
    informative = (
        most_informative_features(
            frame, target, task_type,
            exclude=[*excluded, *detected, *documented_undetected], seed=seed,
        )
        if target and task_type
        else []
    )

    drops = sorted(
        set(constant) | set(identifiers) | set(dataset.id_columns)
        | {w.column for w in leakage if w.severity == "certain"}
        | set(documented_undetected)
    )

    findings = _build_findings(
        dataset, frame, constant, identifiers, high_missing, leakage,
        documented_undetected, informative, class_balance, task_type, drift,
    )

    return DatasetProfile(
        name=dataset.name,
        n_rows=len(frame),
        n_columns=len(frame.columns),
        duplicate_rows=int(frame.duplicated().sum()),
        columns=columns,
        constant_columns=constant,
        identifier_columns=sorted(set(identifiers) | set(dataset.id_columns)),
        high_missing_columns=high_missing,
        candidate_group_columns=group_candidates,
        candidate_time_columns=time_candidates,
        proposed_task_type=task_type,
        proposed_target=target,
        class_balance=class_balance,
        leakage_warnings=leakage,
        documented_leakage_undetected=documented_undetected,
        most_informative_features=informative,
        recommended_drops=drops,
        findings=findings,
    )


def _build_findings(
    dataset: Dataset,
    frame: pd.DataFrame,
    constant: list[str],
    identifiers: list[str],
    high_missing: list[str],
    leakage: list[LeakageWarning],
    documented_undetected: list[str],
    informative: list[dict[str, float]],
    class_balance: dict[str, float] | None,
    task_type: str | None,
    drift: dict[str, float],
) -> list[str]:
    """Human-readable observations. This is what the run timeline shows."""
    out: list[str] = []

    if constant:
        out.append(
            f"{len(constant)} column(s) carry no signal at all and will be dropped: "
            f"{', '.join(constant)}. On C-MAPSS FD001 these are the sensors that never "
            f"move under a single operating condition."
        )
    if identifiers:
        out.append(
            f"{len(identifiers)} column(s) look like row identifiers rather than "
            f"measurements: {', '.join(identifiers)}. Keeping them invites memorisation."
        )
    for w in leakage:
        if w.severity == "certain":
            out.append(f"TARGET LEAKAGE in {w.column}: {w.evidence}")
    if documented_undetected:
        out.append(
            f"{len(documented_undetected)} column(s) are documented as failure-mode flags but "
            f"were NOT convicted by the lift test: {', '.join(documented_undetected)}. They are "
            f"dropped on the dataset's provenance rather than on our statistics — on AI4I, RNF "
            f"is a random-failure flag that does not always set the target, so no statistical "
            f"test would catch it."
        )
    if informative:
        top = ", ".join(f"{f['column']}" for f in informative[:3])
        out.append(
            f"Most informative surviving features by mutual information: {top}. This ranks "
            f"usefulness, not leakage — mutual information cannot tell the two apart."
        )
    if class_balance:
        minority = min(class_balance.values())
        out.append(
            f"Class balance is {minority:.2%} positive. Accuracy is meaningless at this rate — "
            f"a model predicting 'never fails' would score {1 - minority:.2%}. Kairos reports "
            f"PR-AUC, recall at fixed precision, and expected cost instead."
        )
    if high_missing:
        out.append(f"{len(high_missing)} column(s) are more than half missing: {', '.join(high_missing)}.")
    dupes = int(frame.duplicated().sum())
    if dupes:
        out.append(f"{dupes} exactly duplicated row(s) found.")
    drifty = sorted(((v, k) for k, v in drift.items() if v > 1.0), reverse=True)[:3]
    if drifty:
        out.append(
            "Columns whose mean shifts most between the first and second half of the data: "
            + ", ".join(f"{k} ({v:.1f} sd)" for v, k in drifty)
            + ". On a run-to-failure dataset this is degradation, which is signal, not drift."
        )
    if dataset.source == "synthetic":
        out.append("SYNTHETIC DATA: every figure from this run is illustrative, not evidential.")
    if task_type:
        out.append(f"Proposed task: {task_type} on target {dataset.target or 'unknown'!r}.")
    return out
