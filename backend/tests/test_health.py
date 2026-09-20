"""Phase 0 smoke tests: the app boots, answers, and never leaks a traceback."""

from __future__ import annotations

from fastapi.testclient import TestClient

from kairos.main import app


def test_root_advertises_service() -> None:
    with TestClient(app) as client:
        body = client.get("/").json()
    assert body["service"] == "kairos"


def test_health_reports_every_component() -> None:
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert {"postgres", "mlflow", "llm", "checkpointer"} <= set(body["components"])


def test_health_never_500s_even_if_postgres_is_gone(monkeypatch) -> None:
    """A dead database degrades the report; it does not break the endpoint."""
    monkeypatch.setattr("kairos.db.session.ping", lambda: False)
    monkeypatch.setattr("kairos.main.db.ping", lambda: False)
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["components"]["postgres"] == "down"


def test_sse_frames_are_well_formed() -> None:
    from kairos.events import sse_format

    frame = sse_format({"type": "tick", "n": 1})
    assert frame.startswith("event: tick\ndata: ")
    assert frame.endswith("\n\n")
