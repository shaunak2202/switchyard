from __future__ import annotations

import pytest

from switchyard.reliability.breaker import BreakerSettings, BreakerState, CircuitBreaker


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def make(clock: FakeClock, **kwargs: float) -> CircuitBreaker:
    defaults = {
        "window_size": 10,
        "min_calls": 4,
        "failure_rate_threshold": 0.5,
        "open_s": 5.0,
        "half_open_max_calls": 2,
    }
    settings = BreakerSettings(**(defaults | kwargs))  # type: ignore[arg-type]
    return CircuitBreaker("p", settings, clock=clock)


def call(breaker: CircuitBreaker, ok: bool) -> None:
    assert breaker.allow()
    if ok:
        breaker.record_success()
    else:
        breaker.record_failure()


def test_does_not_trip_before_min_calls(clock: FakeClock) -> None:
    breaker = make(clock)
    for _ in range(3):
        call(breaker, ok=False)
    assert breaker.state is BreakerState.CLOSED


def test_trips_when_failure_rate_reaches_threshold(clock: FakeClock) -> None:
    breaker = make(clock)
    for ok in (True, False, True):
        call(breaker, ok)
    assert breaker.state is BreakerState.CLOSED
    call(breaker, ok=False)  # 2 of 4 = 50%
    assert breaker.state is BreakerState.OPEN
    assert not breaker.allow()


def test_intermittent_failures_still_trip(clock: FakeClock) -> None:
    """A consecutive-failure counter would never trip on this pattern."""
    breaker = make(clock, window_size=10, min_calls=10, failure_rate_threshold=0.5)
    for i in range(10):
        call(breaker, ok=i % 2 == 0)
    assert breaker.state is BreakerState.OPEN


def test_sliding_window_forgets_old_failures(clock: FakeClock) -> None:
    breaker = make(clock, window_size=4, min_calls=4, failure_rate_threshold=0.75)
    for ok in (False, False, True, True):
        call(breaker, ok)
    for _ in range(4):
        call(breaker, ok=True)
    assert breaker.failure_rate() == 0.0
    call(breaker, ok=False)
    call(breaker, ok=False)
    assert breaker.state is BreakerState.CLOSED  # 2/4 < 75%


def test_open_to_half_open_after_cooldown(clock: FakeClock) -> None:
    breaker = make(clock)
    for _ in range(4):
        call(breaker, ok=False)
    assert breaker.retry_after_s() == pytest.approx(5.0)
    clock.advance(4.9)
    assert breaker.state is BreakerState.OPEN
    assert breaker.retry_after_s() == pytest.approx(0.1)
    clock.advance(0.1)
    assert breaker.state is BreakerState.HALF_OPEN
    assert breaker.retry_after_s() == 0.0


def _half_open(clock: FakeClock, **kwargs: float) -> CircuitBreaker:
    breaker = make(clock, **kwargs)
    for _ in range(4):
        call(breaker, ok=False)
    clock.advance(5)
    assert breaker.state is BreakerState.HALF_OPEN
    return breaker


def test_half_open_admits_limited_trial_calls(clock: FakeClock) -> None:
    breaker = _half_open(clock)
    assert breaker.allow()
    assert breaker.allow()
    assert not breaker.allow()  # half_open_max_calls = 2


def test_half_open_closes_after_enough_successes_and_clears_window(clock: FakeClock) -> None:
    breaker = _half_open(clock)
    assert breaker.allow() and breaker.allow()
    breaker.record_success()
    assert breaker.state is BreakerState.HALF_OPEN
    breaker.record_success()
    assert breaker.state is BreakerState.CLOSED
    assert breaker.failure_rate() == 0.0
    call(breaker, ok=False)
    assert breaker.state is BreakerState.CLOSED  # history was cleared


def test_half_open_failure_reopens_with_fresh_cooldown(clock: FakeClock) -> None:
    breaker = _half_open(clock)
    assert breaker.allow()
    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    clock.advance(4.9)
    assert breaker.state is BreakerState.OPEN
    clock.advance(0.1)
    assert breaker.state is BreakerState.HALF_OPEN


def test_release_returns_half_open_permit(clock: FakeClock) -> None:
    breaker = _half_open(clock, half_open_max_calls=1)
    assert breaker.allow()
    assert not breaker.allow()
    breaker.release()  # e.g. the client disconnected
    assert breaker.allow()


def test_stragglers_while_open_are_ignored(clock: FakeClock) -> None:
    breaker = make(clock)
    for _ in range(4):
        call(breaker, ok=False)
    breaker.record_failure()
    clock.advance(5)
    assert breaker.state is BreakerState.HALF_OPEN


def test_state_change_listener(clock: FakeClock) -> None:
    changes: list[tuple[BreakerState, BreakerState]] = []
    breaker = CircuitBreaker(
        "p",
        BreakerSettings(window_size=2, min_calls=2, open_s=1, half_open_max_calls=1),
        clock=clock,
        on_state_change=lambda _, old, new: changes.append((old, new)),
    )
    call(breaker, ok=False)
    call(breaker, ok=False)
    clock.advance(1)
    call(breaker, ok=True)
    assert changes == [
        (BreakerState.CLOSED, BreakerState.OPEN),
        (BreakerState.OPEN, BreakerState.HALF_OPEN),
        (BreakerState.HALF_OPEN, BreakerState.CLOSED),
    ]


@pytest.mark.parametrize(
    "kwargs",
    [{"min_calls": 0}, {"min_calls": 30, "window_size": 20}, {"failure_rate_threshold": 0}],
)
def test_invalid_settings(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        BreakerSettings(**kwargs)  # type: ignore[arg-type]
