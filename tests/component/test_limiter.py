"""Token-bucket limiter against a real Redis, including the concurrency proof.

The atomicity claim is tested the only way that means anything: many clients on separate
connection pools (standing in for separate gateway processes) racing on one key, asserting that
*exactly* the limit is admitted. A deliberately non-atomic implementation runs the same race as
a control, to show the test would catch an over-admitting limiter.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest
from redis.asyncio import BlockingConnectionPool, Redis

from switchyard.auth.keys import KeyStore, hash_api_key
from switchyard.ratelimit.limiter import AdmitStatus, RateLimiter
from tests.conftest import REDIS_PREFIX, REDIS_URL

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def redis() -> AsyncIterator[Redis]:
    client = Redis(connection_pool=BlockingConnectionPool.from_url(REDIS_URL, max_connections=64))
    try:
        await client.ping()
    except Exception as exc:  # pragma: no cover - environment problem, not a test failure
        pytest.fail(f"Redis is required at {REDIS_URL}: {exc!r} (run `make up` or `make redis`)")
    yield client
    await client.aclose()


@pytest.fixture
def store(redis: Redis) -> KeyStore:
    return KeyStore(redis, REDIS_PREFIX)


@pytest.fixture
def limiter(redis: Redis, store: KeyStore) -> RateLimiter:
    return RateLimiter(redis, store)


async def test_unknown_and_disabled_keys(store: KeyStore, limiter: RateLimiter) -> None:
    assert (await limiter.admit("sk-sy-nope", 1)).status is AdmitStatus.UNKNOWN_KEY
    key, record = await store.create("k", rpm=10, tpm=1000)
    assert (await limiter.admit(key, 1)).status is AdmitStatus.ADMITTED
    assert await store.set_disabled(record.key_id)
    assert (await limiter.admit(key, 1)).status is AdmitStatus.DISABLED
    assert await store.set_disabled(record.key_id, False)
    assert (await limiter.admit(key, 1)).status is AdmitStatus.ADMITTED


async def test_plaintext_key_is_never_stored(redis: Redis, store: KeyStore) -> None:
    key, _ = await store.create("secret", rpm=1, tpm=1)
    async for name in redis.scan_iter(f"{REDIS_PREFIX}:*"):
        assert key.encode() not in name
        if await redis.type(name) == b"hash":
            values = await redis.hvals(name)
            assert all(key.encode() not in v for v in values)
    assert await redis.exists(store.meta_key(hash_api_key(key)))


async def test_requests_bucket_limits_and_reports_retry_after(
    store: KeyStore, limiter: RateLimiter
) -> None:
    key, _ = await store.create("rpm", rpm=6, tpm=1_000_000)
    decisions = [await limiter.admit(key, 1) for _ in range(7)]
    assert [d.status for d in decisions[:6]] == [AdmitStatus.ADMITTED] * 6
    assert [d.remaining_requests for d in decisions[:6]] == [5, 4, 3, 2, 1, 0]
    limited = decisions[6]
    assert limited.status is AdmitStatus.LIMITED
    # 6/min refills one request every 10s.
    assert 9_000 < limited.retry_after_ms <= 10_000
    assert limited.headers()["retry-after"] == "10"


async def test_tokens_bucket_and_all_or_nothing_charging(
    store: KeyStore, limiter: RateLimiter
) -> None:
    key, _ = await store.create("tpm", rpm=1000, tpm=1000)
    assert (await limiter.admit(key, 600)).remaining_tokens == 400
    denied = await limiter.admit(key, 600)
    assert denied.status is AdmitStatus.LIMITED
    # The denied request charged neither bucket.
    assert denied.remaining_requests == 999
    ok = await limiter.admit(key, 400)
    assert ok.status is AdmitStatus.ADMITTED
    assert ok.remaining_requests == 998


async def test_request_larger_than_limit_is_rejected_outright(
    store: KeyStore, limiter: RateLimiter
) -> None:
    key, _ = await store.create("small", rpm=10, tpm=100)
    assert (await limiter.admit(key, 101)).status is AdmitStatus.TOO_LARGE


async def test_continuous_refill(store: KeyStore, limiter: RateLimiter) -> None:
    key, _ = await store.create("refill", rpm=1_000_000, tpm=1200)  # 20 tokens per second
    assert (await limiter.admit(key, 1200)).admitted  # drain in one call
    assert (await limiter.admit(key, 1)).status is AdmitStatus.LIMITED
    await asyncio.sleep(0.25)  # ~5 tokens' worth
    admitted = 0
    while (await limiter.admit(key, 1)).admitted:
        admitted += 1
    assert 3 <= admitted <= 7


async def test_reconcile_refunds_and_charges_debt(store: KeyStore, limiter: RateLimiter) -> None:
    key, _ = await store.create("recon", rpm=1000, tpm=1000)
    decision = await limiter.admit(key, 500)
    assert decision.remaining_tokens == 500
    assert await limiter.reconcile(decision, 100) >= 900  # refunded 400
    over = await limiter.admit(key, 800)
    assert await limiter.reconcile(over, 2000) < 0  # used far more than estimated: in debt
    assert (await limiter.admit(key, 1)).status is AdmitStatus.LIMITED


async def test_limit_changes_apply_to_the_next_request(
    store: KeyStore, limiter: RateLimiter
) -> None:
    """Raising a limit changes the refill rate at once but does not grant a fresh burst."""
    key, record = await store.create("change", rpm=1, tpm=1000)
    assert (await limiter.admit(key, 1)).admitted
    before = await limiter.admit(key, 1)
    assert not before.admitted
    assert before.retry_after_ms > 55_000  # 1/min
    await store.update_limits(record.key_id, rpm=100, tpm=1000)
    after = await limiter.admit(key, 1)
    assert after.limit_requests == 100
    assert after.retry_after_ms <= 600  # now refills at 100/min
    await asyncio.sleep(after.retry_after_ms / 1000 + 0.05)
    assert (await limiter.admit(key, 1)).admitted


# -- the concurrency proof --------------------------------------------------------------------

PROCESSES = 8
REQUESTS = 400


async def _race(store: KeyStore, api_key: str, cost: int) -> list[AdmitStatus]:
    """REQUESTS concurrent admits spread over PROCESSES independent connection pools."""
    clients = [
        Redis(connection_pool=BlockingConnectionPool.from_url(REDIS_URL, max_connections=32))
        for _ in range(PROCESSES)
    ]
    limiters = [RateLimiter(c, KeyStore(c, store.prefix)) for c in clients]
    start = asyncio.Event()

    async def one(i: int) -> AdmitStatus:
        await start.wait()
        return (await limiters[i % PROCESSES].admit(api_key, cost)).status

    try:
        tasks = [asyncio.create_task(one(i)) for i in range(REQUESTS)]
        await asyncio.sleep(0.05)
        start.set()
        return await asyncio.gather(*tasks)
    finally:
        for c in clients:
            await c.aclose()


async def test_concurrent_requests_admit_exactly_the_rpm_limit(store: KeyStore) -> None:
    limit = 25
    key, _ = await store.create("race-rpm", rpm=limit, tpm=1_000_000)
    started = time.monotonic()
    statuses = await _race(store, key, cost=1)
    # Refill during the race must be < 1 request for "exactly" to be meaningful.
    assert time.monotonic() - started < 60 / limit
    assert statuses.count(AdmitStatus.ADMITTED) == limit
    assert statuses.count(AdmitStatus.LIMITED) == REQUESTS - limit


async def test_concurrent_requests_admit_exactly_the_tpm_limit(store: KeyStore) -> None:
    key, _ = await store.create("race-tpm", rpm=1_000_000, tpm=2000)
    started = time.monotonic()
    statuses = await _race(store, key, cost=100)
    assert time.monotonic() - started < 60 * 100 / 2000  # refill < one request's cost
    assert statuses.count(AdmitStatus.ADMITTED) == 20


async def test_control_non_atomic_limiter_over_admits(redis: Redis) -> None:
    """GET-then-SET across two round trips: the classic race the Lua script exists to avoid."""
    limit = 25
    bucket = f"{REDIS_PREFIX}:naive:{time.monotonic_ns()}"
    await redis.set(bucket, limit)
    start = asyncio.Event()

    async def naive_admit() -> bool:
        await start.wait()
        remaining = int(await redis.get(bucket) or 0)
        if remaining < 1:
            return False
        await redis.set(bucket, remaining - 1)
        return True

    tasks = [asyncio.create_task(naive_admit()) for _ in range(REQUESTS)]
    await asyncio.sleep(0.05)
    start.set()
    admitted = sum(await asyncio.gather(*tasks))
    assert admitted > limit, "expected the naive limiter to over-admit under concurrency"
