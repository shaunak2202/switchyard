"""Application factory. Run with ``uvicorn --factory switchyard.main:create_app``."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from switchyard import __version__
from switchyard.api import chat, health
from switchyard.config import GatewayConfig, load_config
from switchyard.errors import GatewayError, InvalidRequestError
from switchyard.logs import configure_logging
from switchyard.middleware import RequestContextMiddleware
from switchyard.providers import TransportFactory, build_providers
from switchyard.reliability.backoff import RetryPolicy
from switchyard.reliability.breaker import BreakerSettings, CircuitBreaker
from switchyard.router import Router
from switchyard.service import ChatService

logger = logging.getLogger(__name__)


def create_app(
    config: GatewayConfig | None = None,
    *,
    transport_factory: TransportFactory | None = None,
) -> FastAPI:
    config = config or load_config()
    configure_logging(config.logging.level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        providers = build_providers(config, transport_factory)
        router = Router(config, providers)
        rel = config.reliability
        breaker_settings = BreakerSettings(**rel.circuit_breaker.model_dump())
        breakers = {name: CircuitBreaker(name, breaker_settings) for name in providers}
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
            for provider in providers.values():
                await provider.aclose()

    app = FastAPI(title="Switchyard", version=__version__, lifespan=lifespan)
    app.add_middleware(RequestContextMiddleware)
    app.include_router(chat.router)
    app.include_router(health.router)
    _install_error_handlers(app)
    return app


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
