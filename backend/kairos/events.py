"""In-memory pub/sub for streaming agent progress to the browser over SSE.

The agent graph runs in a worker thread (FastAPI ``BackgroundTasks``), while SSE
consumers live on the event loop. Every publish therefore hops threads via
``call_soon_threadsafe``. Late subscribers get the backlog replayed so a browser
that connects mid-run still sees the whole timeline — which matters, because the
run timeline IS the demo.

No Redis. Single process, single tenant, by design (PROJECT_BRIEF.md §4).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any

log = logging.getLogger(__name__)

_loop: asyncio.AbstractEventLoop | None = None

#: Terminal marker pushed into every subscriber queue when a channel closes.
_SENTINEL = object()

HEARTBEAT_SECONDS = 15.0


def bind_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Record the serving event loop. Called once from the app lifespan."""
    global _loop
    _loop = loop


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue]] = defaultdict(set)
        self._history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._closed: set[str] = set()
        self._lock = threading.Lock()

    # -- producer side (may be called from any thread) -----------------------

    def publish(self, channel: str, event: dict[str, Any]) -> None:
        event = {"ts": time.time(), **event}
        with self._lock:
            self._history[channel].append(event)
            targets = list(self._subscribers[channel])
        for queue in targets:
            self._deliver(queue, event)

    def close(self, channel: str) -> None:
        with self._lock:
            self._closed.add(channel)
            targets = list(self._subscribers[channel])
        for queue in targets:
            self._deliver(queue, _SENTINEL)

    def _deliver(self, queue: asyncio.Queue, item: Any) -> None:
        if _loop is not None and _loop.is_running():
            _loop.call_soon_threadsafe(queue.put_nowait, item)
        else:  # pragma: no cover - only in sync tests
            queue.put_nowait(item)

    # -- consumer side -------------------------------------------------------

    def history(self, channel: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history[channel])

    def is_closed(self, channel: str) -> bool:
        with self._lock:
            return channel in self._closed

    async def stream(self, channel: str) -> AsyncIterator[dict[str, Any]]:
        """Yield the channel backlog, then live events, until the channel closes."""
        queue: asyncio.Queue = asyncio.Queue()
        with self._lock:
            backlog = list(self._history[channel])
            already_closed = channel in self._closed
            self._subscribers[channel].add(queue)
        try:
            for event in backlog:
                yield event
            if already_closed:
                return
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    return
                yield item
        finally:
            with self._lock:
                self._subscribers[channel].discard(queue)


bus = EventBus()


def sse_format(event: dict[str, Any]) -> str:
    """Encode one Server-Sent Event frame. ``event.type`` becomes the SSE event name."""
    name = event.get("type", "message")
    payload = json.dumps(event, default=str)
    return f"event: {name}\ndata: {payload}\n\n"


async def sse_source(channel: str) -> AsyncIterator[str]:
    """SSE body generator with heartbeats, so idle connections survive proxies.

    The pending ``__anext__`` future is carried across heartbeat iterations. Starting
    a second one while the first is still in flight would raise "anext() already
    running" on the underlying async generator.
    """
    agen = bus.stream(channel)
    pending: asyncio.Future | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(agen.__anext__())
            try:
                event = await asyncio.wait_for(asyncio.shield(pending), HEARTBEAT_SECONDS)
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
                continue
            except StopAsyncIteration:
                pending = None
                yield sse_format({"type": "stream_end", "channel": channel})
                return
            pending = None
            yield sse_format(event)
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
        await agen.aclose()
