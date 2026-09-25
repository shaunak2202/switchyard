"""The real embedding + verifier models, at the thresholds the gateway ships with.

Skipped when the optional ``semantic`` extra is not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sentence_transformers")

from sentence_transformers import CrossEncoder, SentenceTransformer

from switchyard.cache.embeddings import CrossEncoderVerifier, SentenceTransformerEmbedder
from switchyard.config import SemanticCacheConfig

DEFAULTS = SemanticCacheConfig()
Models = tuple[SentenceTransformerEmbedder, CrossEncoderVerifier]


@pytest.fixture(scope="module")
def models() -> Models:
    assert DEFAULTS.verifier_model is not None
    embedder = SentenceTransformer(DEFAULTS.embedding_model, device="cpu")
    verifier = CrossEncoder(DEFAULTS.verifier_model, device="cpu")
    return (
        SentenceTransformerEmbedder(embedder, DEFAULTS.embedding_model),
        CrossEncoderVerifier(verifier, DEFAULTS.verifier_model),
    )


async def _similarity(models: Models, a: str, b: str) -> float:
    embedder, _ = models
    return float((await embedder.embed(a)) @ (await embedder.embed(b)))


async def _accepted(models: Models, a: str, b: str) -> bool:
    """The gateway's decision: candidate by embedding, then confirmed by the verifier."""
    if await _similarity(models, a, b) < DEFAULTS.candidate_threshold:
        return False
    return await models[1].score(a, b) >= DEFAULTS.verifier_threshold


@pytest.mark.parametrize(
    ("a", "b"),
    [
        # The highest-similarity hard negatives in data/semantic_pairs.jsonl.
        (
            "How do I convert a string to an integer in Python?",
            "How do I convert an integer to a string in Python?",
        ),
        ("Convert 5 kilometers to miles.", "Convert 5 miles to kilometers."),
        (
            "What foods should I eat to lower cholesterol?",
            "What foods should I avoid to lower cholesterol?",
        ),
        ("Summarize the plot of Hamlet.", "Summarize the plot of Hamlet in one sentence."),
        ("Give me 3 tips for better sleep.", "Give me 10 tips for better sleep."),
    ],
)
async def test_hard_negatives_are_rejected(models: Models, a: str, b: str) -> None:
    # The embedding alone rates these as close matches; that is the problem being solved.
    assert await _similarity(models, a, b) >= DEFAULTS.candidate_threshold
    assert not await _accepted(models, a, b)


async def test_trivial_rewording_is_accepted(models: Models) -> None:
    assert await _accepted(
        models, "What is the capital of France?", "what is the capital of france"
    )
