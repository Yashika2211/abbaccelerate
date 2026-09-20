"""Dataset endpoints — select a builtin or upload a CSV."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from sqlmodel import Session, select

from kairos.api.schemas import DatasetSelect
from kairos.data.loaders import data_dir, load_ai4i, load_cmapss, load_csv
from kairos.data.profiler import profile_dataset
from kairos.db.models import Dataset
from kairos.db.session import engine

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

BUILTINS = {
    "cmapss": ("NASA C-MAPSS FD001", load_cmapss),
    "ai4i": ("AI4I 2020", load_ai4i),
}


def _register(name: str, kind: str, loaded, path: str | None = None) -> Dataset:
    profile = profile_dataset(loaded)
    row = Dataset(
        name=name,
        kind=kind,
        task_type=profile.proposed_task_type,
        target=profile.proposed_target,
        group_col=loaded.group_col,
        time_col=loaded.time_col,
        n_rows=loaded.n_rows,
        n_columns=len(loaded.frame.columns),
        source=loaded.source,
        citation=loaded.citation,
        path=path,
        profile=profile.as_dict(),
    )
    with Session(engine) as session:
        session.add(row)
        session.commit()
        session.refresh(row)
        # Detach with values loaded: callers read attributes after the session closes.
        session.expunge(row)
    return row


@router.get("/builtin")
def list_builtin() -> list[dict[str, str]]:
    return [
        {"key": key, "name": name, "description": desc}
        for key, (name, desc) in {
            "cmapss": ("NASA C-MAPSS FD001", "100 turbofan engines run to failure -> RUL regression"),
            "ai4i": ("AI4I 2020", "10,000 machine cycles, 3.4% failures -> binary classification"),
        }.items()
    ]


@router.post("")
def create_dataset(body: DatasetSelect) -> dict[str, Any]:
    if not body.builtin:
        raise HTTPException(400, "specify a builtin dataset or use /upload")
    name, loader = BUILTINS[body.builtin]
    row = _register(body.name or name, body.builtin, loader())
    return {"dataset_id": row.id, "name": row.name, "rows": row.n_rows}


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "only .csv uploads are supported")
    uploads = data_dir() / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    destination = uploads / file.filename
    with destination.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    try:
        loaded = load_csv(destination)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"could not read CSV: {exc}") from exc
    row = _register(Path(file.filename).stem, "upload", loaded, path=str(destination))
    return {"dataset_id": row.id, "name": row.name, "rows": row.n_rows}


@router.get("")
def list_datasets() -> list[dict[str, Any]]:
    with Session(engine) as session:
        rows = session.exec(select(Dataset).order_by(Dataset.created_at.desc())).all()
    return [
        {
            "dataset_id": r.id, "name": r.name, "kind": r.kind, "rows": r.n_rows,
            "task_type": r.task_type, "target": r.target, "source": r.source,
        }
        for r in rows
    ]


@router.get("/{dataset_id}/profile")
def get_profile(dataset_id: str) -> dict[str, Any]:
    with Session(engine) as session:
        row = session.get(Dataset, dataset_id)
    if row is None:
        raise HTTPException(404, "dataset not found")
    return {
        "dataset_id": row.id,
        "name": row.name,
        "citation": row.citation,
        "source": row.source,
        "profile": row.profile,
    }
