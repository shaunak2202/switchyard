"""Models behind the semantic cache.

* An ``Embedder`` (bi-encoder) turns a prompt into a vector. It is cheap (one forward pass per
  prompt) and finds *candidates* by vector search, but its similarity cannot tell "5 km to
  miles" from "5 miles to km".
* A ``Verifier`` (cross-encoder) reads the new prompt and the cached prompt *together* and
  scores whether they ask the same thing. It is more expensive (one pass per pair), so it only
  runs when a candidate exists.

The two-stage design and its thresholds come from ``scripts/eval_semantic_threshold.py``
(ADR-016). Both models run through a ``MicroBatcher`` (ADR-021).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

import anyio
import numpy as np
import numpy.typing as npt

from switchyard import metrics

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer

logger = logging.getLogger(__name__)

Vector = npt.NDArray[np.float32]
T = TypeVar("T")
R = TypeVar("R")


class MicroBatcher(Generic[T, R]):
    """Runs a batch function on one dedicated thread, grouping concurrent calls.

    CPU inference cost is dominated by per-call overhead at these input sizes: encoding 32
    prompts in one forward pass costs a small multiple of encoding one. Under load, callers
    that arrive while a batch is running queue up and form the next batch, so batch size adapts
    to load with no fixed delay. The only deliberate wait is ``max_wait_s`` when the batcher is
    idle, to give a burst a chance to form.

    One thread, one batch at a time: torch already uses several cores inside one batch, and
    concurrent batches would only contend for them.
    """

    def __init__(
        self,
        fn: Callable[[list[T]], list[R]],
        *,
        name: str,
        max_batch: int = 64,
        max_wait_s: float = 0.002,
    ) -> None:
        self._name = name
        self._fn = fn
        self._max_batch = max_batch
        self._max_wait_s = max_wait_s
        self._pending: list[tuple[T, asyncio.Future[R]]] = []
        self._in_batch = 0
        self._wakeup: asyncio.Event | None = None
        self._worker: asyncio.Task[None] | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)

    @property
    def depth(self) -> int:
        """Items waiting or being processed: the queueing delay a new call would face."""
        return len(self._pending) + self._in_batch

    async def submit(self, item: T) -> R:
        loop = asyncio.get_running_loop()
        if self._worker is None or self._worker.done():
            self._wakeup = asyncio.Event()
            self._worker = loop.create_task(self._run(), name="micro-batcher")
        future: asyncio.Future[R] = loop.create_future()
        self._pending.append((item, future))
        assert self._wakeup is not None
        self._wakeup.set()
        return await future

    async def _run(self) -> None:
        assert self._wakeup is not None
        loop = asyncio.get_running_loop()
        while True:
            await self._wakeup.wait()
            self._wakeup.clear()
            if len(self._pending) < self._max_batch:
                await asyncio.sleep(self._max_wait_s)
            while self._pending:
                batch = self._pending[: self._max_batch]
                del self._pending[: self._max_batch]
                # Callers that gave up (their lookup budget ran out) don't need an answer.
                batch = [(item, fut) for item, fut in batch if not fut.done()]
                if not batch:
                    continue
                self._in_batch = len(batch)
                metrics.BATCH_SIZE.labels(self._name).observe(len(batch))
                try:
                    results = await loop.run_in_executor(
                        self._executor, self._fn, [item for item, _ in batch]
                    )
                except Exception as exc:
                    for _, fut in batch:
                        if not fut.done():
                            fut.set_exception(exc)
                else:
                    for (_, fut), result in zip(batch, results, strict=True):
                        if not fut.done():
                            fut.set_result(result)
                finally:
                    self._in_batch = 0

    def close(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)


class Embedder(Protocol):
    name: str
    dim: int

    @property
    def queue_depth(self) -> int:
        """Prompts waiting for or in the current batch."""
        ...

    async def embed(self, text: str) -> Vector:
        """Unit-length float32 vector, so cosine similarity is a dot product."""
        ...


class Verifier(Protocol):
    name: str

    @property
    def queue_depth(self) -> int: ...

    async def score(self, query: str, candidate: str) -> float:
        """Probability-like score in [0, 1] that both prompts ask the same thing."""
        ...


class SentenceTransformerEmbedder:
    """A local sentence-transformers bi-encoder on CPU, micro-batched."""

    def __init__(self, model: SentenceTransformer, name: str) -> None:
        self._model = model
        self.name = name
        dim = model.get_sentence_embedding_dimension()
        if dim is None:
            raise ValueError(f"model {name} has no fixed embedding dimension")
        self.dim = int(dim)
        self._batcher: MicroBatcher[str, Vector] = MicroBatcher(self._encode, name="embed")

    @property
    def queue_depth(self) -> int:
        return self._batcher.depth

    @classmethod
    async def load(cls, name: str) -> SentenceTransformerEmbedder:
        from sentence_transformers import SentenceTransformer

        model = await anyio.to_thread.run_sync(lambda: SentenceTransformer(name, device="cpu"))
        embedder = cls(model, name)
        await embedder.embed("warm up")  # first call pays one-off initialisation cost
        return embedder

    def _encode(self, texts: list[str]) -> list[Vector]:
        vectors = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=len(texts)
        )
        return [np.asarray(v, dtype=np.float32) for v in vectors]

    async def embed(self, text: str) -> Vector:
        return await self._batcher.submit(text)

    def close(self) -> None:
        self._batcher.close()


class CrossEncoderVerifier:
    """A local sentence-transformers cross-encoder on CPU, micro-batched."""

    def __init__(self, model: CrossEncoder, name: str) -> None:
        self._model = model
        self.name = name
        self._batcher: MicroBatcher[tuple[str, str], float] = MicroBatcher(
            self._predict, name="verify"
        )

    @property
    def queue_depth(self) -> int:
        return self._batcher.depth

    @classmethod
    async def load(cls, name: str) -> CrossEncoderVerifier:
        from sentence_transformers import CrossEncoder

        model = await anyio.to_thread.run_sync(lambda: CrossEncoder(name, device="cpu"))
        verifier = cls(model, name)
        await verifier.score("warm up", "warm up")
        return verifier

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        # Same call as the offline evaluation (which also scored in batches), so the chosen
        # threshold means the same thing.
        scores = self._model.predict(pairs, batch_size=len(pairs), show_progress_bar=False)
        return [float(s) for s in scores]

    async def score(self, query: str, candidate: str) -> float:
        return await self._batcher.submit((query, candidate))

    def close(self) -> None:
        self._batcher.close()


class HashingEmbedder:
    """Deterministic bag-of-words embedder for tests: shared words ⇒ high cosine similarity."""

    name = "hashing"
    queue_depth = 0

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    async def embed(self, text: str) -> Vector:
        vector = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            bucket = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=4).digest())
            vector[bucket % self.dim] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector


class ExactTextVerifier:
    """Test verifier: accepts a candidate only if the prompts match ignoring case and
    punctuation."""

    name = "exact-text"
    queue_depth = 0

    async def score(self, query: str, candidate: str) -> float:
        def canon(text: str) -> list[str]:
            return re.findall(r"\w+", text.lower())

        return 1.0 if canon(query) == canon(candidate) else 0.0
