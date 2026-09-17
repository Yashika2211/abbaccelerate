#!/usr/bin/env python3
"""End-to-end pipeline from the CLI — PROJECT_BRIEF.md §10 Phase 3 gate.

    python scripts/run_pipeline.py --dataset ai4i
    python scripts/run_pipeline.py --dataset cmapss --budget 180

Profiles, drops what the profiler convicts, engineers features, splits without
leakage, trains the zoo, and ranks the result by expected annual cost rather than
by AUC. No agent, no API, no UI — this is the pipeline proving it works on its own.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from kairos.data.features import (  # noqa: E402
    drop_columns,
    engineer_ai4i_features,
    engineer_timeseries_features,
)
from kairos.data.loaders import load_ai4i, load_cmapss  # noqa: E402
from kairos.data.profiler import profile_dataset  # noqa: E402
from kairos.data.splits import make_splits  # noqa: E402
from kairos.decision.presets import MILLING_PLANT, TURBOFAN_PLANT, describe  # noqa: E402
from kairos.decision.simulate import (  # noqa: E402
    ClassifierPredictions,
    RulPredictions,
    simulate_classification,
    simulate_rul,
)
from kairos.ml.train import train_all  # noqa: E402

RUPEE = "₹"


def rule(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def lakh(amount: float) -> str:
    """Indian numbering: a plant manager reads lakhs, not millions."""
    return f"{RUPEE}{amount / 100_000:,.2f}L"


def run_ai4i(budget: float, seed: int) -> int:
    dataset = load_ai4i()
    rule(f"DATASET  {dataset.name}  ({dataset.source} source)")
    print(f"{dataset.n_rows:,} rows x {len(dataset.frame.columns)} columns")

    rule("PLANT ECONOMICS")
    print("  " + str(describe(
        MILLING_PLANT if dataset.task_type == "binary_classification" else TURBOFAN_PLANT
    )["summary"]))

    profile = profile_dataset(dataset, seed=seed)
    rule("PROFILER FINDINGS")
    for finding in profile.findings:
        print(f"  - {finding}")

    trimmed = drop_columns(dataset.frame, profile.recommended_drops)
    engineered = engineer_ai4i_features(trimmed.frame)
    print(f"\ndropped {len(trimmed.dropped_columns)}: {', '.join(trimmed.dropped_columns)}")
    print(f"added   {len(engineered.added_columns)}: {', '.join(engineered.added_columns)}")

    frame = engineered.frame
    features = [
        c for c in frame.columns
        if c != dataset.target and c != "type" and frame[c].dtype.kind in "ifb"
    ]

    dataset.frame = frame
    split = make_splits(dataset, seed=seed)
    rule("SPLIT")
    print(f"  strategy={split.strategy}  rows={split.sizes()}")
    for note in split.notes:
        print(f"  - {note}")

    rule(f"TRAINING  (budget {budget:.0f}s, {len(features)} features)")
    started = time.perf_counter()
    report = train_all(
        split, features, dataset.target, "binary_classification",
        budget_seconds=budget, seed=seed,
    )
    for trial in report.trials:
        status = "ok " if trial.ok else "FAIL"
        pr = trial.metrics.get("pr_auc")
        auc = trial.metrics.get("roc_auc")
        cal = trial.calibration.method if trial.calibration else "-"
        print(
            f"  [{status}] {trial.model_name:28s} {trial.fit_seconds:6.1f}s  "
            f"PR-AUC={pr:.4f}  ROC-AUC={auc:.4f}  calib={cal}"
            if trial.ok and pr is not None
            else f"  [{status}] {trial.model_name:28s} {trial.error or ''}"
        )
    print(f"  total {time.perf_counter() - started:.1f}s")

    models = [
        ClassifierPredictions(
            model_id=t.model_id,
            model_name=t.model_name,
            val_y=split.val[dataset.target].to_numpy(),
            val_prob=t.val_prediction,
            test_y=split.test[dataset.target].to_numpy(),
            test_prob=t.test_prediction,
            metrics=t.metrics,
        )
        for t in report.successful
    ]
    board = simulate_classification(models, MILLING_PLANT)
    print_board(board, "threshold")
    return 0


def run_cmapss(budget: float, seed: int) -> int:
    dataset = load_cmapss()
    rule(f"DATASET  {dataset.name}  ({dataset.source} source)")
    print(f"{dataset.n_rows:,} rows, {dataset.n_groups} engines")

    rule("PLANT ECONOMICS")
    print("  " + str(describe(
        MILLING_PLANT if dataset.task_type == "binary_classification" else TURBOFAN_PLANT
    )["summary"]))

    profile = profile_dataset(dataset, seed=seed)
    rule("PROFILER FINDINGS")
    for finding in profile.findings:
        print(f"  - {finding}")

    sensors = [
        c for c in dataset.frame.columns
        if c.startswith(("sensor_", "op_setting_")) and c not in profile.constant_columns
    ]
    print(f"\nlive sensors after dropping {len(profile.constant_columns)} dead: {len(sensors)}")

    def prepare(frame):
        trimmed = drop_columns(frame, profile.constant_columns)
        return engineer_timeseries_features(
            trimmed.frame, "unit_id", "cycle", sensors, windows=(5, 10, 20)
        ).frame

    dataset.frame = prepare(dataset.frame)
    dataset.holdout = prepare(dataset.holdout)  # kept for accuracy metrics only

    features = [
        c for c in dataset.frame.columns
        if c not in {"rul", "unit_id"} and dataset.frame[c].dtype.kind in "if"
    ]
    split = make_splits(dataset, seed=seed)
    rule("SPLIT")
    print(f"  strategy={split.strategy}  rows={split.sizes()}  engines={split.group_counts()}")
    if split.reference_holdout is not None:
        print(f"  published holdout retained for accuracy only: "
              f"{len(split.reference_holdout):,} rows, "
              f"{split.reference_holdout.unit_id.nunique()} engines")
    for note in split.notes:
        print(f"  - {note}")

    rule(f"TRAINING  (budget {budget:.0f}s, {len(features)} features)")
    started = time.perf_counter()
    report = train_all(
        split, features, "rul", "rul_regression",
        group_col="unit_id", budget_seconds=budget, seed=seed,
    )
    for trial in report.trials:
        if trial.ok:
            print(
                f"  [ok ] {trial.model_name:28s} {trial.fit_seconds:6.1f}s  "
                f"RMSE={trial.metrics['rmse']:7.3f}  "
                f"RMSE@last30={trial.metrics.get('rmse_last_30', float('nan')):7.3f}  "
                f"late={trial.metrics['late_fraction']:.1%}"
            )
        else:
            print(f"  [FAIL] {trial.model_name:28s} {trial.error}")
    print(f"  total {time.perf_counter() - started:.1f}s")

    def failure_cycles(frame):
        return {int(u): float(g.cycle.max() + g.rul.min()) for u, g in frame.groupby("unit_id")}

    models = [
        RulPredictions(
            model_id=t.model_id,
            model_name=t.model_name,
            val_units=split.val.unit_id.to_numpy(),
            val_cycles=split.val.cycle.to_numpy(),
            val_pred=t.val_prediction,
            val_failure=failure_cycles(split.val),
            test_units=split.test.unit_id.to_numpy(),
            test_cycles=split.test.cycle.to_numpy(),
            test_pred=t.test_prediction,
            test_failure=failure_cycles(split.test),
            metrics=t.metrics,
        )
        for t in report.successful
    ]
    board = simulate_rul(models, TURBOFAN_PLANT)
    print_board(board, "lead_time")
    return 0


def print_board(board, kind: str) -> None:
    rule("LEADERBOARD  (ranked by expected annual cost, NOT by AUC/RMSE)")
    label = "thresh" if kind == "threshold" else "lead"
    unit = "cost/machine-yr" if kind == "threshold" else "cost/engine-yr"
    secondary = "PR-AUC" if kind == "threshold" else "RMSE"
    print(f"  {'#':<3}{'model':<30}{label:>8}{unit:>16}{'regret':>14}{secondary:>10}")
    print("  " + "-" * 79)
    for i, row in enumerate(board.rows, 1):
        metric = row.metrics.get("pr_auc" if kind == "threshold" else "rmse", float("nan"))
        print(
            f"  {i:<3}{row.model_name:<30}{row.operating_point:>8.3f}"
            f"{lakh(row.cost_per_asset_year):>16}{lakh(row.cost_regret):>14}{metric:>10.4f}"
        )
        if row.degenerate:
            print(f"      ! {row.degenerate_reason}")

    inversion = board.inversion("pr_auc" if kind == "threshold" else "rmse",
                                higher_is_better=(kind == "threshold"))
    if inversion:
        rule("COST / METRIC INVERSION")
        print(f"  Best {inversion['metric']}: {inversion['metric_winner']}")
        print(f"  Best cost      : {inversion['cost_winner']}")
        print(f"  Choosing on {inversion['metric']} would cost an extra "
              f"{lakh(inversion['annual_cost_penalty'])} per asset per year.")
    else:
        print("\n  No inversion: the best-metric model is also the cheapest. "
              "Kairos reports that rather than inventing a disagreement.")

    if board.baselines:
        rule("HONEST BASELINES")
        for b in board.baselines.baselines:
            marker = " <- Kairos" if b.key == "kairos" else ""
            print(f"  {b.name:<42}{lakh(b.cost_per_asset_year):>16}{marker}")
        print(f"\n  {board.baselines.verdict()}")
    print(f"\n  leaderboard computed in {board.elapsed_ms:.0f}ms")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["ai4i", "cmapss"], default="ai4i")
    parser.add_argument("--budget", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return run_ai4i(args.budget, args.seed) if args.dataset == "ai4i" else run_cmapss(
        args.budget, args.seed
    )


if __name__ == "__main__":
    raise SystemExit(main())
