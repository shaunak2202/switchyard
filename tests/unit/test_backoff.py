from __future__ import annotations

import random

import pytest

from switchyard.reliability.backoff import RetryPolicy


def policy(**kwargs: float) -> RetryPolicy:
    defaults = {"max_attempts": 5, "base_delay_s": 0.1, "max_delay_s": 1.0}
    return RetryPolicy(rng=random.Random(42), **(defaults | kwargs))  # type: ignore[arg-type]


def test_ceiling_grows_exponentially_then_caps() -> None:
    p = policy()
    assert [p.ceiling(n) for n in range(1, 7)] == pytest.approx([0.1, 0.2, 0.4, 0.8, 1.0, 1.0])


def test_full_jitter_stays_within_window_and_actually_varies() -> None:
    p = policy()
    for retry in range(1, 6):
        samples = [p.delay(retry) for _ in range(500)]
        assert all(s is not None and 0 <= s <= p.ceiling(retry) for s in samples)
        # Full jitter spreads retries across the whole window rather than bunching them.
        assert len({round(s or 0, 6) for s in samples}) > 400
        assert max(s or 0 for s in samples) > 0.8 * p.ceiling(retry)
        assert min(s or 0 for s in samples) < 0.2 * p.ceiling(retry)


def test_retry_after_is_a_floor() -> None:
    p = policy()
    for _ in range(100):
        delay = p.delay(1, retry_after_s=0.5)
        assert delay is not None
        assert delay >= 0.5


def test_retry_after_beyond_cap_means_give_up_on_this_provider() -> None:
    assert policy().delay(1, retry_after_s=30) is None


@pytest.mark.parametrize(
    "kwargs",
    [{"max_attempts": 0}, {"base_delay_s": -1}, {"base_delay_s": 2.0, "max_delay_s": 1.0}],
)
def test_invalid_policies(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        policy(**kwargs)
