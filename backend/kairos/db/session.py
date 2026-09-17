"""SQLModel engine and session plumbing."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine

from kairos.config import get_settings

log = logging.getLogger(__name__)

_settings = get_settings()

engine = create_engine(
    _settings.database_url,
    pool_pre_ping=True,   # demo laptops sleep; stale connections must not surface as 500s
    pool_size=5,
    max_overflow=10,
    echo=False,
)


def init_db() -> None:
    """Create tables for every imported SQLModel. Safe to call repeatedly."""
    SQLModel.metadata.create_all(engine)


def ping() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001 - health check reports, never raises
        log.warning("postgres ping failed: %s", exc)
        return False


def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session
