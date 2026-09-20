"""Per-run artefact cache and audit recording.

Two jobs the agent state cannot do itself:

* **Heavy artefacts.** Fitted models, engineered frames and prediction arrays live
  here, keyed by run_id, because LangGraph checkpoints its state into Postgres on
  every transition and a booster does not belong in a JSONB column.
* **The audit trail.** Every node opens a step, and the step records duration and
  outcome whether it succeeds or throws, then publishes the same event to SSE so
  the browser timeline and the database never disagree.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from sqlmodel import Session, select

from kairos.db.models import AuditEvent, Run
from kairos.db.session import engine
from kairos.events import bus

log = logging.getLogger(__name__)


@dataclass
class RunArtifacts:
    """Everything too heavy to checkpoint, held for the life of the process."""

    dataset: Any = None
    split: Any = None
    feature_names: list[str] = field(default_factory=list)
    trials: list[Any] = field(default_factory=list)
    leaderboard: Any = None
    models: dict[str, Any] = field(default_factory=dict)
    cost_config: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


_ARTIFACTS: dict[str, RunArtifacts] = {}


def artifacts(run_id: str, rehydrate: bool = True) -> RunArtifacts:
    """The cache entry for a run, rebuilt from disk on a cold miss.

    Without the rehydrate step, restarting the API turns every finished run's
    cost curve and SHAP ranking into an empty state — and ``make seed`` would
    produce a demo that evaporates on the next boot.
    """
    store = _ARTIFACTS.get(run_id)
    if store is not None:
        return store
    store = RunArtifacts()
    _ARTIFACTS[run_id] = store
    if rehydrate:
        from kairos.agent import persistence

        persistence.load(run_id, store)
    return store


def clear_artifacts(run_id: str) -> None:
    _ARTIFACTS.pop(run_id, None)


def hash_inputs(payload: Any) -> str:
    """Stable hash so a rerun is provably over the same inputs."""
    try:
        blob = json.dumps(payload, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001
        blob = repr(payload)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _next_sequence(session: Session, run_id: str) -> int:
    rows = session.exec(
        select(AuditEvent).where(AuditEvent.run_id == run_id)
    ).all()
    return len(rows) + 1


def record(
    run_id: str,
    node: str,
    status: str,
    summary: str = "",
    reasoning: str | None = None,
    payload: dict[str, Any] | None = None,
    duration_ms: float | None = None,
    inputs_hash: str | None = None,
) -> None:
    """Write one audit row and publish the matching SSE event.

    Persistence failures are logged, never raised: losing a timeline row must not
    take down a demo.
    """
    event = {
        "type": "node",
        "run_id": run_id,
        "node": node,
        "status": status,
        "summary": summary,
        "reasoning": reasoning,
        "duration_ms": duration_ms,
        "payload": payload or {},
    }
    try:
        with Session(engine) as session:
            session.add(AuditEvent(
                run_id=run_id,
                sequence=_next_sequence(session, run_id),
                node=node,
                status=status,
                summary=summary,
                reasoning=reasoning,
                payload=payload,
                duration_ms=duration_ms,
                inputs_hash=inputs_hash,
            ))
            session.commit()
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to persist audit event %s/%s: %s", node, status, exc)
    bus.publish(run_id, event)


@contextmanager
def step(run_id: str, node: str, inputs: Any = None) -> Iterator[dict[str, Any]]:
    """Bracket a node: emit started, then done or failed with a duration.

    Yields a mutable dict the node fills in with its summary, reasoning and
    payload. Whatever is in that dict when the block exits is what gets recorded,
    so a node that throws halfway still reports what it had learned.
    """
    started = time.perf_counter()
    inputs_hash = hash_inputs(inputs) if inputs is not None else None
    record(run_id, node, "started", inputs_hash=inputs_hash)
    box: dict[str, Any] = {"summary": "", "reasoning": None, "payload": {}}
    try:
        yield box
    except Exception as exc:  # noqa: BLE001 - recorded, then re-raised for the graph
        record(
            run_id, node, "failed",
            summary=box.get("summary") or f"{type(exc).__name__}: {exc}",
            reasoning=box.get("reasoning"),
            payload=box.get("payload"),
            duration_ms=(time.perf_counter() - started) * 1000.0,
            inputs_hash=inputs_hash,
        )
        raise
    else:
        record(
            run_id, node, "done",
            summary=box.get("summary", ""),
            reasoning=box.get("reasoning"),
            payload=box.get("payload"),
            duration_ms=(time.perf_counter() - started) * 1000.0,
            inputs_hash=inputs_hash,
        )


def set_run_status(run_id: str, status: str, **fields: Any) -> None:
    try:
        with Session(engine) as session:
            run = session.get(Run, run_id)
            if run is None:
                return
            run.status = status
            for key, value in fields.items():
                setattr(run, key, value)
            session.add(run)
            session.commit()
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to update run %s status: %s", run_id, exc)
    bus.publish(run_id, {"type": "run_status", "run_id": run_id, "status": status})
