"""Kairos FastAPI application."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from kairos import __version__
from kairos.config import get_settings
from kairos.db import session as db
from kairos.events import bind_loop, sse_format
from kairos.spike.hitl_spike import router as spike_router

settings = get_settings()
logging.basicConfig(
    level=settings.kairos_log_level,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
)
log = logging.getLogger("kairos")

STARTED_AT = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    bind_loop(asyncio.get_running_loop())
    for attempt in range(1, 11):  # postgres wins the compose race sometimes, not always
        if db.ping():
            db.init_db()
            log.info("database ready")
            break
        log.warning("waiting for postgres (%s/10)", attempt)
        await asyncio.sleep(1.5)
    else:
        log.error("database unreachable at startup; /api/health will report degraded")
    yield
    log.info("shutting down")


app = FastAPI(
    title="Kairos API",
    version=__version__,
    description="Agentic AutoML for predictive maintenance that outputs decisions, not probabilities.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(spike_router)


@app.middleware("http")
async def no_naked_tracebacks(request, call_next):
    """PROJECT_BRIEF.md §11: no unhandled exception ever reaches the demo path."""
    try:
        return await call_next(request)
    except Exception as exc:  # noqa: BLE001
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "detail": str(exc), "path": request.url.path},
        )


def _mlflow_ok() -> bool:
    """MLflow is a soft dependency; this only colours a dot in the UI."""
    try:
        r = httpx.get(f"{settings.mlflow_tracking_uri}/health", timeout=2.0)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


@app.get("/api/health")
def health() -> dict[str, Any]:
    postgres_ok = db.ping()
    return {
        "status": "ok" if postgres_ok else "degraded",
        "version": __version__,
        "env": settings.kairos_env,
        "uptime_s": round(time.time() - STARTED_AT, 1),
        "components": {
            "postgres": "up" if postgres_ok else "down",
            "mlflow": "up" if _mlflow_ok() else "down",
            "llm": "configured" if settings.llm_enabled else "fallback_templates",
        },
    }


@app.get("/api/events/demo")
async def demo_stream() -> StreamingResponse:
    """Phase 0 SSE smoke test: a counter the frontend renders to prove the pipe is live."""

    async def gen() -> AsyncIterator[str]:
        for i in range(1, 1_000):
            yield sse_format({"type": "tick", "n": i, "source": "api"})
            await asyncio.sleep(1.0)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "kairos", "version": __version__, "docs": "/docs", "health": "/api/health"}
