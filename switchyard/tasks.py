"""Fire-and-forget work that must still be tracked (cache writes after a response is sent).

A bare ``asyncio.create_task`` can be garbage-collected mid-flight and loses exceptions. This
keeps strong references, logs failures, and on shutdown lets in-flight work finish for a
bounded time.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)


class BackgroundTasks:
    def __init__(self, max_pending: int = 10_000) -> None:
        self._tasks: set[asyncio.Task[Any]] = set()
        self._max_pending = max_pending

    def spawn(self, coro: Coroutine[Any, Any, Any], *, name: str) -> None:
        if len(self._tasks) >= self._max_pending:
            # Shedding a cache write under extreme load is fine; running out of memory is not.
            coro.close()
            logger.warning("background task dropped: queue full", extra={"task": name})
            return
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._done)

    def _done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error(
                "background task failed", extra={"task": task.get_name()},
                exc_info=task.exception(),
            )  # fmt: skip

    @property
    def pending(self) -> int:
        return len(self._tasks)

    async def drain(self, timeout_s: float = 5.0) -> None:
        if not self._tasks:
            return
        _, still_running = await asyncio.wait(set(self._tasks), timeout=timeout_s)
        for task in still_running:
            task.cancel()
