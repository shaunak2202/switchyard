"""Exponential backoff with full jitter.

Full jitter (``uniform(0, min(cap, base * 2**n))``) rather than "equal" or no jitter: when a
provider blips, every in-flight request fails at roughly the same moment, and deterministic
backoff makes them all retry at the same moment too. Spreading retries over the whole window
minimises the synchronised retry spike (see the AWS Architecture Blog, "Exponential Backoff
And Jitter").
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 2
    """Attempts per provider, including the first. 1 disables retries."""
    base_delay_s: float = 0.05
    max_delay_s: float = 1.0
    multiplier: float = 2.0
    rng: random.Random = field(default_factory=random.Random, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay_s < 0 or self.max_delay_s < self.base_delay_s:
            raise ValueError("need 0 <= base_delay_s <= max_delay_s")

    def ceiling(self, retry_number: int) -> float:
        """Upper bound of the jitter window before retry ``retry_number`` (1-based)."""
        exponent = max(retry_number - 1, 0)
        return min(self.max_delay_s, self.base_delay_s * self.multiplier**exponent)

    def delay(self, retry_number: int, retry_after_s: float | None = None) -> float | None:
        """Seconds to sleep before retry ``retry_number``, or ``None`` to not retry here.

        A provider's ``Retry-After`` is a floor: retrying sooner is pointless. If it asks for
        longer than ``max_delay_s`` we give up on this provider and let the caller fail over,
        rather than holding the client's request open.
        """
        jittered = self.rng.uniform(0, self.ceiling(retry_number))
        if retry_after_s is None:
            return jittered
        if retry_after_s > self.max_delay_s:
            return None
        return max(jittered, retry_after_s)
