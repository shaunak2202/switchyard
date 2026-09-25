"""Application factory. Run with ``uvicorn --factory switchyard.main:create_app``."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from redis.asyncio import BlockingConnectionPool, Redis

from switchyard import __version__, metrics
from switchyard.api import chat, health
from switchyard.api.guard import Guard
from switchyard.auth.keys import KeyStore
from switchyard.cache.embeddings import (
    CrossEncoderVerifier,
    Embedder,
    SentenceTransformerEmbedder,
    Verifier,
)
from switchyard.cache.store import ExactCache, ResponseCache, SemanticCache
from switchyard.config import GatewayConfig, load_config
from switchyard.errors import GatewayError, InvalidRequestError
from switchyard.logs import configure_logging
from switchyard.middleware import RequestContextMiddleware
from switchyard.providers import TransportFactory, build_providers
from switchyard.ratelimit.limiter import RateLimiter
from switchyard.reliability.backoff import RetryPolicy
from switchyard.reliability.breaker import BreakerSettings, BreakerState, CircuitBreaker
from switchyard.router import Router
from switchyard.service import ChatService
from switchyard.tasks import BackgroundTasks

logger = logging.getLogger(__name__)


def create_app(
    config: GatewayConfig | None = None,
    *,
    transport_factory: TransportFactory | None = None,
    embedder: Embedder | None = None,
    verifier: Verifier | None = None,
) -> FastAPI:
    config = config or load_config()
    configure_logging(config.logging.level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # A *blocking* pool: redis-py's default async pool raises MaxConnectionsError the
        # moment it is exhausted, which would turn a traffic burst into 503s. This one queues
        # callers for up to socket_timeout_s instead.
        redis = Redis(
            connection_pool=BlockingConnectionPool.from_url(
                config.redis.url,
                max_connections=config.redis.max_connections,
                timeout=config.redis.socket_timeout_s,
                socket_timeout=config.redis.socket_timeout_s,
                socket_connect_timeout=config.redis.socket_timeout_s,
            )
        )
        keys = KeyStore(redis, config.redis.key_prefix)
        app.state.redis = redis
        app.state.guard = Guard(
            RateLimiter(redis, keys),
            keys,
            enabled=config.auth.enabled,
            default_max_tokens=config.auth.default_max_tokens,
        )
        if not config.auth.enabled:
            logger.warning("auth is DISABLED: every request is anonymous and unlimited")
        app.state.tasks = BackgroundTasks()
        app.state.cache = await _build_cache(config, redis, embedder, verifier)
        providers = build_providers(config, transport_factory)
        router = Router(config, providers)
        rel = config.reliability
        breaker_settings = BreakerSettings(**rel.circuit_breaker.model_dump())
        breakers = {
            name: CircuitBreaker(name, breaker_settings, on_state_change=_export_breaker_state)
            for name in providers
        }
        for name in providers:
            metrics.CIRCUIT_STATE.labels(name).set(BreakerState.CLOSED)
        app.state.config = config
        app.state.providers = providers
        app.state.breakers = breakers
        app.state.chat_service = ChatService(
            router,
            breakers,
            retry=RetryPolicy(**rel.retry.model_dump()),
            request_timeout_s=rel.request_timeout_s,
        )
        logger.info(
            "gateway started",
            extra={"providers": sorted(providers), "models": router.models()},
        )
        try:
            yield
        finally:
            await app.state.tasks.drain()
            for provider in providers.values():
                await provider.aclose()
            await redis.aclose()

    app = FastAPI(title="Switchyard", version=__version__, lifespan=lifespan)
    app.add_middleware(RequestContextMiddleware)
    app.include_router(chat.router)
    app.include_router(health.router)
    _install_error_handlers(app)
    return app


def _export_breaker_state(provider: str, _old: BreakerState, new: BreakerState) -> None:
    metrics.CIRCUIT_STATE.labels(provider).set(new)
    metrics.CIRCUIT_TRANSITIONS.labels(provider, new.name.lower()).inc()


async def _build_cache(
    config: GatewayConfig,
    redis: Redis,
    embedder: Embedder | None,
    verifier: Verifier | None,
) -> ResponseCache:
    cfg = config.cache
    prefix = config.redis.key_prefix
    exact = ExactCache(redis, prefix, cfg.exact.ttl_s) if cfg.exact.enabled else None
    semantic = None
    if cfg.semantic.enabled:
        try:
            if embedder is None:
                embedder = await SentenceTransformerEmbedder.load(cfg.semantic.embedding_model)
            if verifier is None and cfg.semantic.verifier_model is not None:
                verifier = await CrossEncoderVerifier.load(cfg.semantic.verifier_model)
            if verifier is None:
                logger.warning(
                    "semantic cache running WITHOUT a verifier: see ADR-016 for the measured "
                    "false-hit rate of embedding-only matching"
                )
            semantic = SemanticCache(
                redis,
                prefix,
                embedder,
                candidate_threshold=cfg.semantic.candidate_threshold,
                verifier=verifier,
                verifier_threshold=cfg.semantic.verifier_threshold,
                ttl_s=cfg.semantic.ttl_s,
            )
            await semantic.ensure_index()
        except Exception:
            # Missing optional deps, no model, or a Redis without the query engine: run
            # without the semantic tier instead of refusing to start.
            logger.exception("semantic cache disabled: failed to initialise")
            semantic = None
    logger.info(
        "cache configured",
        extra={
            "exact": exact is not None,
            "semantic": semantic is not None,
            "candidate_threshold": cfg.semantic.candidate_threshold if semantic else None,
            "verifier": semantic.verifier.name if semantic and semantic.verifier else None,
            "verifier_threshold": cfg.semantic.verifier_threshold if semantic else None,
        },
    )
    return ResponseCache(exact, semantic, max_semantic_chars=cfg.semantic.max_prompt_chars)


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(GatewayError)
    async def gateway_error(_: Request, exc: GatewayError) -> JSONResponse:
        return JSONResponse(exc.to_body(), status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        loc = [str(part) for part in first.get("loc", ()) if part != "body"]
        error = InvalidRequestError(
            f"{'.'.join(loc) or 'body'}: {first.get('msg', 'invalid request')}",
            param=".".join(loc) or None,
        )
        return JSONResponse(error.to_body(), status_code=error.status_code)

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error")
        error = GatewayError("internal gateway error", code="internal_error")
        return JSONResponse(error.to_body(), status_code=500)
