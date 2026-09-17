"""SQLModel tables — PROJECT_BRIEF.md §7.

The audit trail is the point. Every node writes a row recording what it saw, what
it decided and why, and the UI renders those rows as a live timeline. An agent
whose reasoning you cannot inspect is indistinguishable from a script with an LLM
sticker on it, and that is exactly what a judge will probe.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Column, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Dataset(SQLModel, table=True):
    __tablename__ = "datasets"

    id: str = Field(default_factory=_uuid, primary_key=True)
    name: str
    kind: str                                   # builtin key or "upload"
    task_type: str | None = None
    target: str | None = None
    group_col: str | None = None
    time_col: str | None = None
    n_rows: int = 0
    n_columns: int = 0
    source: str = "full"
    citation: str = ""
    path: str | None = None                     # uploads only
    profile: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    created_at: datetime = Field(default_factory=_now)


class Run(SQLModel, table=True):
    __tablename__ = "runs"

    id: str = Field(default_factory=_uuid, primary_key=True)
    dataset_id: str = Field(foreign_key="datasets.id", index=True)
    status: str = "pending"    # pending|running|awaiting_approval|complete|failed
    task_type: str | None = None
    time_budget_s: float = 120.0
    cost_config: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    plan: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    leaderboard: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    cost_analysis: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    baselines: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    impact: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    findings: list[str] | None = Field(default=None, sa_column=Column(JSONB))
    replan_count: int = 0
    error: str | None = None
    llm_used: bool = False
    created_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None


class Trial(SQLModel, table=True):
    __tablename__ = "trials"

    id: str = Field(default_factory=_uuid, primary_key=True)
    run_id: str = Field(foreign_key="runs.id", index=True)
    model_id: str = Field(index=True)
    candidate_key: str
    model_name: str
    family: str
    attempt: int = 1                            # which replan round produced this
    metrics: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    cv_metrics: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    calibration: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    operating_point: float | None = None
    total_cost: float | None = None
    cost_per_asset_year: float | None = None
    cost_regret: float | None = None
    fit_seconds: float = 0.0
    is_champion: bool = False
    error: str | None = None
    created_at: datetime = Field(default_factory=_now)


class AuditEvent(SQLModel, table=True):
    """One row per node execution. This is the 'show your work' panel."""

    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_run_seq", "run_id", "sequence"),)

    id: str = Field(default_factory=_uuid, primary_key=True)
    run_id: str = Field(foreign_key="runs.id", index=True)
    sequence: int = 0
    node: str
    status: str                                 # started|done|failed|skipped
    summary: str = ""
    reasoning: str | None = None
    inputs_hash: str | None = None              # so a rerun is provably the same inputs
    payload: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    duration_ms: float | None = None
    created_at: datetime = Field(default_factory=_now)


class Deployment(SQLModel, table=True):
    """A promoted champion, servable from /api/predict."""

    __tablename__ = "deployments"

    id: str = Field(default_factory=_uuid, primary_key=True)
    run_id: str = Field(foreign_key="runs.id", index=True)
    model_id: str
    alias: str = Field(index=True)
    task_type: str
    operating_point: float
    feature_names: list[str] | None = Field(default=None, sa_column=Column(JSONB))
    cost_config: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    artifact_path: str | None = None
    created_at: datetime = Field(default_factory=_now)
