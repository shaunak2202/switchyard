"""Per-provider circuit breaker: CLOSED → OPEN → HALF_OPEN → CLOSED.

* **CLOSED**: calls flow. Outcomes go into a sliding window of the last ``window_size`` calls.
  Once the window holds at least ``min_calls`` outcomes and the failure rate reaches
  ``failure_rate_threshold``, the breaker opens.
* **OPEN**: calls are rejected immediately (the router fails over without paying a timeout)
  until ``open_s`` has elapsed.
* **HALF_OPEN**: up to ``half_open_max_calls`` trial calls are let through. Any failure re-opens
  the breaker; ``half_open_max_calls`` consecutive successes close it and clear the window.

A failure *rate* over a window, rather than N consecutive failures, is used so that a provider
failing 40% of calls under heavy traffic still trips; with a consecutive count, one lucky success
in every few calls would keep it closed forever.

The breaker is used from a single event loop and never awaits, so it needs no locks. With
several worker processes each keeps its own breaker; see ADR-010.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

logger = logging.getLogger(__name__)


class BreakerState(IntEnum):
    # Integer values are exported as a Prometheus gauge.
    CLOSED = 0
    HALF_OPEN = 1
    OPEN = 2


@dataclass(frozen=True, slots=True)
class BreakerSettings:
    window_size: int = 20
    min_calls: int = 10
    failure_rate_threshold: float = 0.5
    open_s: float = 10.0
    half_open_max_calls: int = 1

    def __post_init__(self) -> None:
        if not 1 <= self.min_calls <= self.window_size:
            raise ValueError("need 1 <= min_calls <= window_size")
        if not 0 < self.failure_rate_threshold <= 1:
            raise ValueError("failure_rate_threshold must be in (0, 1]")


StateListener = Callable[[str, BreakerState, BreakerState], None]


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        settings: BreakerSettings | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        on_state_change: StateListener | None = None,
    ) -> None:
        self.name = name
        self.settings = settings or BreakerSettings()
        self._clock = clock
        self._on_state_change = on_state_change
        self._state = BreakerState.CLOSED
        self._window: deque[bool] = deque(maxlen=self.settings.window_size)  # True = failure
        self._opened_at = 0.0
        self._half_open_in_flight = 0
        self._half_open_successes = 0

    # -- queries ---------------------------------------------------------------------------

    @property
    def state(self) -> BreakerState:
        self._maybe_half_open()
        return self._state

    def failure_rate(self) -> float:
        return sum(self._window) / len(self._window) if self._window else 0.0

    def retry_after_s(self) -> float:
        """Time until an OPEN breaker will admit a trial call."""
        if self._state is not BreakerState.OPEN:
            return 0.0
        return max(self._opened_at + self.settings.open_s - self._clock(), 0.0)

    # -- call protocol ---------------------------------------------------------------------

    def allow(self) -> bool:
        """Ask to make a call. Every ``True`` must be matched by exactly one
        ``record_success``, ``record_failure`` or ``release``."""
        state = self.state
        if state is BreakerState.CLOSED:
            return True
        if state is BreakerState.OPEN:
            return False
        if self._half_open_in_flight < self.settings.half_open_max_calls:
            self._half_open_in_flight += 1
            return True
        return False

    def record_success(self) -> None:
        if self._state is BreakerState.HALF_OPEN:
            self._half_open_in_flight = max(self._half_open_in_flight - 1, 0)
            self._half_open_successes += 1
            if self._half_open_successes >= self.settings.half_open_max_calls:
                self._transition(BreakerState.CLOSED)
            return
        self._window.append(False)

    def record_failure(self) -> None:
        if self._state is BreakerState.HALF_OPEN:
            self._half_open_in_flight = max(self._half_open_in_flight - 1, 0)
            self._transition(BreakerState.OPEN)
            return
        if self._state is BreakerState.OPEN:
            return  # a straggler that started before we opened
        self._window.append(True)
        if (
            len(self._window) >= self.settings.min_calls
            and self.failure_rate() >= self.settings.failure_rate_threshold
        ):
            self._transition(BreakerState.OPEN)

    def release(self) -> None:
        """The call ended without telling us anything about provider health
        (client disconnected, caller's own bad request)."""
        if self._state is BreakerState.HALF_OPEN:
            self._half_open_in_flight = max(self._half_open_in_flight - 1, 0)

    # -- internals -------------------------------------------------------------------------

    def _maybe_half_open(self) -> None:
        if (
            self._state is BreakerState.OPEN
            and self._clock() - self._opened_at >= self.settings.open_s
        ):
            self._transition(BreakerState.HALF_OPEN)

    def _transition(self, new: BreakerState) -> None:
        old = self._state
        if old is new:
            return
        self._state = new
        if new is BreakerState.OPEN:
            self._opened_at = self._clock()
        if new is BreakerState.HALF_OPEN:
            self._half_open_in_flight = 0
            self._half_open_successes = 0
        if new is BreakerState.CLOSED:
            self._window.clear()
        logger.warning(
            "circuit breaker state change",
            extra={
                "provider": self.name,
                "from_state": old.name,
                "to_state": new.name,
                "failure_rate": round(self.failure_rate(), 3),
            },
        )
        if self._on_state_change is not None:
            self._on_state_change(self.name, old, new)
