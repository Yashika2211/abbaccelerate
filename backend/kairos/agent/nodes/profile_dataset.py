"""Node 1: profile_dataset — deterministic, no LLM."""

from __future__ import annotations

from typing import Any

from sqlmodel import Session

from kairos.agent.runtime import artifacts, step
from kairos.agent.state import KairosState
from kairos.data.loaders import Dataset as LoadedDataset
from kairos.data.loaders import load_ai4i, load_cmapss, load_csv
from kairos.data.profiler import profile_dataset as compute_profile
from kairos.db.models import Dataset as DatasetRow
from kairos.db.session import engine

BUILTIN_LOADERS = {
    "cmapss": load_cmapss,
    "cmapss_fd001": load_cmapss,
    "ai4i": load_ai4i,
}


def _load(row: DatasetRow) -> LoadedDataset:
    if row.kind in BUILTIN_LOADERS:
        return BUILTIN_LOADERS[row.kind]()
    if row.path:
        return load_csv(row.path, target=row.target)
    raise ValueError(f"cannot load dataset {row.id}: unknown kind {row.kind!r}")


def node_profile_dataset(state: KairosState) -> dict[str, Any]:
    run_id = state["run_id"]
    with step(run_id, "profile_dataset", inputs=state["dataset_id"]) as box:
        with Session(engine) as session:
            row = session.get(DatasetRow, state["dataset_id"])
            if row is None:
                raise ValueError(f"dataset {state['dataset_id']} not found")
            dataset = _load(row)

            profile = compute_profile(dataset)
            row.profile = profile.as_dict()
            row.task_type = profile.proposed_task_type
            row.target = profile.proposed_target
            row.group_col = dataset.group_col
            row.time_col = dataset.time_col
            row.n_rows = dataset.n_rows
            row.n_columns = len(dataset.frame.columns)
            row.source = dataset.source
            row.citation = dataset.citation
            session.add(row)
            session.commit()

        artifacts(run_id).dataset = dataset

        box["summary"] = (
            f"Profiled {dataset.name}: {dataset.n_rows:,} rows, "
            f"{len(dataset.frame.columns)} columns, "
            f"{len(profile.constant_columns)} dead, "
            f"{len(profile.leakage_warnings)} leakage convictions."
        )
        box["reasoning"] = " ".join(profile.findings[:3])
        box["payload"] = {
            "profile": profile.as_dict(),
            "source": dataset.source,
            "citation": dataset.citation,
        }
        return {
            "profile": profile.as_dict(),
            "findings": [*state.get("findings", []), *profile.findings],
        }
