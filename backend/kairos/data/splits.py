"""Leakage-safe splitting — PROJECT_BRIEF.md §6, Trap 1.

The failure mode this module exists to prevent: splitting C-MAPSS rows at random
puts cycle 40 of engine 7 in train and cycle 41 of engine 7 in test. The model
then "predicts" a trajectory it has already memorised, reports an R² near 0.99,
and is worthless in the field. Every split here is group-aware by construction,
and :meth:`Split.assert_no_group_leakage` is called on the way out rather than
being left as a thing someone might remember to check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, train_test_split

from kairos.data.loaders import Dataset


class LeakageError(AssertionError):
    """Raised when a split would let the same asset appear on both sides."""


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    strategy: str
    group_col: str | None = None
    target: str | None = None
    #: The dataset's own published test set, when it has one. Used ONLY for accuracy
    #: metrics comparable to the literature — never for policy simulation. See
    #: make_splits() for why those are different questions.
    reference_holdout: pd.DataFrame | None = None
    notes: list[str] = field(default_factory=list)

    def assert_no_group_leakage(self) -> None:
        """Hard guarantee that no group straddles two splits."""
        if not self.group_col:
            return
        parts = {"train": self.train, "val": self.val, "test": self.test}
        seen: dict[str, set[Any]] = {
            name: set(frame[self.group_col].unique()) for name, frame in parts.items()
        }
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = seen[a] & seen[b]
            if overlap:
                raise LeakageError(
                    f"{len(overlap)} group(s) appear in both {a} and {b}: "
                    f"{sorted(overlap)[:5]}{'...' if len(overlap) > 5 else ''}"
                )

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def group_counts(self) -> dict[str, int] | None:
        if not self.group_col:
            return None
        return {
            name: int(frame[self.group_col].nunique())
            for name, frame in (("train", self.train), ("val", self.val), ("test", self.test))
        }

    def summary(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "group_col": self.group_col,
            "rows": self.sizes(),
            "groups": self.group_counts(),
            "notes": list(self.notes),
        }


def group_split(
    frame: pd.DataFrame,
    group_col: str,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split so that every row of a group lands on the same side."""
    if group_col not in frame.columns:
        raise KeyError(f"group column {group_col!r} is not in the frame")
    n_groups = frame[group_col].nunique()
    if n_groups < 2:
        raise ValueError(f"need at least 2 groups to split, found {n_groups}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    left_idx, right_idx = next(splitter.split(frame, groups=frame[group_col]))
    return frame.iloc[left_idx].copy(), frame.iloc[right_idx].copy()


def group_train_val_test(
    frame: pd.DataFrame,
    group_col: str,
    val_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 42,
) -> Split:
    """Three-way group split. Validation is where the threshold gets chosen."""
    remainder, test = group_split(frame, group_col, test_size=test_size, seed=seed)
    # Re-base the validation fraction against what is left after removing test.
    val_fraction = val_size / (1.0 - test_size)
    train, val = group_split(remainder, group_col, test_size=val_fraction, seed=seed + 1)

    split = Split(
        train=train,
        val=val,
        test=test,
        strategy="group_shuffle",
        group_col=group_col,
        notes=[
            f"Split by {group_col!r} with GroupShuffleSplit: no asset appears in two splits.",
            "A random row split here would leak a trajectory across the boundary and "
            "inflate R² toward 0.99.",
        ],
    )
    split.assert_no_group_leakage()
    return split


def stratified_train_val_test(
    frame: pd.DataFrame,
    target: str,
    val_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 42,
) -> Split:
    """Stratified split preserving the positive rate — essential at 3.4% positives."""
    y = frame[target]
    if y.nunique() < 2:
        raise ValueError(f"target {target!r} has a single class; nothing to stratify")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    rest_idx, test_idx = next(sss.split(frame, y))
    rest, test = frame.iloc[rest_idx].copy(), frame.iloc[test_idx].copy()

    val_fraction = val_size / (1.0 - test_size)
    train, val = train_test_split(
        rest, test_size=val_fraction, random_state=seed + 1, stratify=rest[target]
    )

    return Split(
        train=train.copy(),
        val=val.copy(),
        test=test.copy(),
        strategy="stratified_shuffle",
        target=target,
        notes=[
            f"Stratified on {target!r} so every split carries the same positive rate.",
            "At a 3.4% positive rate an unstratified split can hand a fold almost no "
            "positives, which makes PR-AUC and expected cost meaningless.",
        ],
    )


def temporal_train_val_test(
    frame: pd.DataFrame,
    time_col: str,
    val_size: float = 0.2,
    test_size: float = 0.2,
) -> Split:
    """Chronological split: train on the past, test on the future. No shuffling."""
    ordered = frame.sort_values(time_col).reset_index(drop=True)
    n = len(ordered)
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    n_train = n - n_val - n_test
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("not enough rows for a three-way temporal split")

    return Split(
        train=ordered.iloc[:n_train].copy(),
        val=ordered.iloc[n_train : n_train + n_val].copy(),
        test=ordered.iloc[n_train + n_val :].copy(),
        strategy="temporal",
        notes=[
            f"Ordered by {time_col!r}: train precedes validation, which precedes test.",
            "No shuffling, so the model is never asked to predict its own past.",
        ],
    )


def make_splits(
    dataset: Dataset,
    val_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 42,
) -> Split:
    """Choose the safe strategy for this dataset and apply it.

    C-MAPSS ships its own held-out test set, so that is used as test rather than
    carving one out — using the official holdout is what makes results comparable
    to published numbers.
    """
    frame = dataset.frame

    if dataset.group_col and dataset.holdout is not None:
        # Two different questions need two different test sets.
        #
        # C-MAPSS's published test set is TRUNCATED: each engine's recording stops
        # at an arbitrary point before failure. That is fine for "how accurate is
        # the RUL estimate at this moment", which is what the literature reports.
        #
        # It is invalid for "what would this maintenance policy have cost", because
        # a lead-time policy can only alert on cycles that were recorded. Measured
        # on FD001: 81 of 100 test engines stop before their RUL ever reaches 25, so
        # a policy evaluated there eats 81 unplanned failures regardless of model
        # quality. That is an artefact of the recording window, not a property of
        # the model, and it makes fixed-interval maintenance look 3x better than it is.
        #
        # So: policy cost is simulated on held-out engines with COMPLETE
        # run-to-failure trajectories, and the published holdout is carried
        # alongside for literature-comparable accuracy only.
        split = group_train_val_test(
            frame, dataset.group_col, val_size=val_size, test_size=test_size, seed=seed
        )
        split.target = dataset.target
        split.strategy = "group_shuffle_complete_trajectories"
        split.reference_holdout = dataset.holdout.copy()
        split.notes = [
            f"Train/validation/test split by {dataset.group_col!r}; no engine appears twice.",
            "Test engines have COMPLETE run-to-failure trajectories, which is the only "
            "valid basis for simulating an intervention policy.",
            "The dataset's published (truncated) test set is retained separately for "
            "accuracy metrics comparable to published C-MAPSS results. It is not used "
            "for cost, because 81% of its engines stop recording before any sane alert "
            "threshold is reached.",
        ]
        return split

    if dataset.group_col:
        split = group_train_val_test(
            frame, dataset.group_col, val_size=val_size, test_size=test_size, seed=seed
        )
        split.target = dataset.target
        return split

    if dataset.task_type == "binary_classification" and dataset.target in frame.columns:
        return stratified_train_val_test(
            frame, dataset.target, val_size=val_size, test_size=test_size, seed=seed
        )

    if dataset.time_col:
        return temporal_train_val_test(
            frame, dataset.time_col, val_size=val_size, test_size=test_size
        )

    raise ValueError(f"no safe split strategy for dataset {dataset.name!r}")
