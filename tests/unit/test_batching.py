from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest

from switchyard.cache.embeddings import MicroBatcher, Vector
from switchyard.cache.normalize import CacheDirective
from switchyard.cache.store import ResponseCache, SemanticCache
from switchyard.schemas import ChatCompletionRequest


async def test_concurrent_calls_share_one_batch() -> None:
    sizes: list[int] = []

    def double(items: list[int]) -> list[int]:
        sizes.append(len(items))
        return [2 * i for i in items]

    batcher: MicroBatcher[int, int] = MicroBatcher(double, name="t")
    results = await asyncio.gather(*(batcher.submit(i) for i in range(10)))
    assert results == [2 * i for i in range(10)]
    assert sizes == [10]
    batcher.close()


async def test_arrivals_during_a_batch_form_the_next_batch() -> None:
    sizes: list[int] = []
    release = threading.Event()

    def slow(items: list[int]) -> list[int]:
        sizes.append(len(items))
        if len(sizes) == 1:
            release.wait(2)  # hold the first batch until the second wave has queued
        return items

    batcher: MicroBatcher[int, int] = MicroBatcher(slow, name="t")
    first = asyncio.create_task(batcher.submit(0))
    await asyncio.sleep(0.01)
    second = [asyncio.create_task(batcher.submit(i)) for i in range(1, 6)]
    await asyncio.sleep(0.01)
    assert batcher.depth == 6
    release.set()
    await asyncio.gather(first, *second)
    assert sizes == [1, 5]
    batcher.close()


async def test_batches_are_capped() -> None:
    sizes: list[int] = []

    def ident(items: list[int]) -> list[int]:
        sizes.append(len(items))
        return items

    batcher: MicroBatcher[int, int] = MicroBatcher(ident, name="t", max_batch=4)
    await asyncio.gather(*(batcher.submit(i) for i in range(10)))
    assert sizes == [4, 4, 2]
    batcher.close()


async def test_errors_reach_every_caller_in_the_batch() -> None:
    def boom(items: list[int]) -> list[int]:
        raise RuntimeError("model crashed")

    batcher: MicroBatcher[int, int] = MicroBatcher(boom, name="t")
    results = await asyncio.gather(*(batcher.submit(i) for i in range(3)), return_exceptions=True)
    assert all(isinstance(r, RuntimeError) for r in results)
    batcher.close()


async def test_abandoned_calls_are_not_computed() -> None:
    seen: list[list[int]] = []
    gate = threading.Event()

    def record(items: list[int]) -> list[int]:
        seen.append(items)
        if len(seen) == 1:
            gate.wait(2)
        return items

    batcher: MicroBatcher[int, int] = MicroBatcher(record, name="t")
    blocker = asyncio.create_task(batcher.submit(0))
    await asyncio.sleep(0.01)
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await batcher.submit(99)  # gives up while the first batch runs
    kept = asyncio.create_task(batcher.submit(1))
    gate.set()
    await asyncio.gather(blocker, kept)
    assert seen == [[0], [1]]
    batcher.close()


# -- degradation in the cache ------------------------------------------------------------------


class SlowEmbedder:
    name = "slow"
    dim = 4

    def __init__(self, delay_s: float, depth: int = 0) -> None:
        self.delay_s = delay_s
        self.queue_depth = depth
        self.calls = 0

    async def embed(self, text: str) -> Vector:
        self.calls += 1
        await asyncio.sleep(self.delay_s)
        return np.full(4, 0.5, dtype=np.float32)


REQUEST = ChatCompletionRequest.model_validate(
    {"model": "m", "temperature": 0, "messages": [{"role": "user", "content": "hello there"}]}
)


def _cache(embedder: SlowEmbedder, budget_s: float = 0.05) -> ResponseCache:
    semantic = SemanticCache(
        None,  # type: ignore[arg-type]  # never reached: the embedding stage gives up first
        "x",
        embedder,
        candidate_threshold=0.8,
        ttl_s=60,
    )
    return ResponseCache(None, semantic, semantic_budget_s=budget_s, semantic_max_queue=8)


async def test_semantic_lookup_gives_up_at_its_budget() -> None:
    embedder = SlowEmbedder(delay_s=1.0)
    started = time.perf_counter()
    lookup = await _cache(embedder).lookup(REQUEST, CacheDirective())
    assert time.perf_counter() - started < 0.2
    assert lookup.eligible and lookup.hit is None
    assert embedder.calls == 1


async def test_semantic_lookup_is_skipped_when_the_model_is_saturated() -> None:
    embedder = SlowEmbedder(delay_s=0.0, depth=8)
    lookup = await _cache(embedder).lookup(REQUEST, CacheDirective())
    assert lookup.hit is None
    assert embedder.calls == 0
