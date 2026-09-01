"""FastAPI application for the LAN inference gateway."""

from __future__ import annotations

import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .adapters import adapter_for
from .config import ConfigurationError, GatewaySettings
from .models import ChatCompletionRequest
from .routing import BackendRegistry, UnknownModelError


def create_app(
    settings: GatewaySettings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    resolved_settings = settings or GatewaySettings.from_env()
    registry = BackendRegistry(resolved_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(resolved_settings.request_timeout_seconds),
            transport=transport,
        )
        yield
        await app.state.http_client.aclose()

    app = FastAPI(
        title="LAN Inference Gateway",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.registry = registry

    @app.middleware("http")
    async def authenticate_lan_clients(request: Request, call_next: Any) -> Response:
        if request.url.path.startswith("/v1/"):
            try:
                expected_api_key = resolved_settings.client_api_key()
            except ConfigurationError:
                return openai_error("Gateway authentication is misconfigured.", 500, "server_error")

            if expected_api_key is not None:
                authorization = request.headers.get("authorization", "")
                supplied_api_key = authorization.removeprefix("Bearer ")
                if not hmac.compare_digest(supplied_api_key, expected_api_key):
                    return openai_error("Invalid API key.", 401, "invalid_api_key")

        return await call_next(request)

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        return {
            "status": "ok",
            "backends": [backend.name for backend in registry.backends],
        }

    @app.get("/readyz")
    async def readyz(request: Request) -> Response:
        client: httpx.AsyncClient = request.app.state.http_client
        results = await _backend_health(client, registry)
        ready = bool(results) and all(result["status"] == "ready" for result in results)
        return JSONResponse(
            {"status": "ready" if ready else "degraded", "backends": results},
            status_code=200 if ready else 503,
        )

    @app.get("/v1/models")
    async def list_models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [
                {"id": model, "object": "model", "owned_by": backend.name}
                for model, backend in registry.models()
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: ChatCompletionRequest, request: Request) -> Response:
        try:
            backend = registry.resolve(payload.model)
            adapter = adapter_for(backend.adapter)
            headers = adapter.request_headers(backend)
        except UnknownModelError:
            return openai_error(f"No backend is configured for model {payload.model!r}.", 404, "model_not_found")
        except ConfigurationError:
            return openai_error("Backend configuration is invalid.", 500, "server_error")

        client: httpx.AsyncClient = request.app.state.http_client
        request_body = payload.model_dump(mode="json", exclude_none=True)
        upstream_url = adapter.chat_completions_url(backend)

        if payload.stream:
            return await _stream_upstream(client, upstream_url, request_body, headers)
        return await _request_upstream(client, upstream_url, request_body, headers)

    return app


async def _request_upstream(
    client: httpx.AsyncClient,
    url: str,
    request_body: dict[str, object],
    headers: dict[str, str],
) -> Response:
    try:
        upstream = await client.post(url, json=request_body, headers=headers)
    except httpx.HTTPError:
        return openai_error("Configured backend is unavailable.", 502, "backend_unavailable")

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
        headers=_forwarded_headers(upstream),
    )


async def _stream_upstream(
    client: httpx.AsyncClient,
    url: str,
    request_body: dict[str, object],
    headers: dict[str, str],
) -> Response:
    upstream_request = client.build_request("POST", url, json=request_body, headers=headers)
    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError:
        return openai_error("Configured backend is unavailable.", 502, "backend_unavailable")

    if upstream.is_error:
        content = await upstream.aread()
        await upstream.aclose()
        return Response(
            content=content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
            headers=_forwarded_headers(upstream),
        )

    async def body() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        body(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "text/event-stream"),
        headers=_forwarded_headers(upstream),
    )


async def _backend_health(
    client: httpx.AsyncClient,
    registry: BackendRegistry,
) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for backend in registry.backends:
        try:
            adapter = adapter_for(backend.adapter)
            health_url = adapter.health_url(backend)
            headers = adapter.request_headers(backend)
        except ConfigurationError:
            results.append({"name": backend.name, "status": "misconfigured"})
            continue

        if health_url is None:
            results.append({"name": backend.name, "status": "not_configured"})
            continue

        try:
            response = await client.get(health_url, headers=headers, timeout=5.0)
        except httpx.HTTPError:
            results.append({"name": backend.name, "status": "unavailable"})
            continue

        results.append(
            {"name": backend.name, "status": "ready" if response.is_success else "unavailable"}
        )
    return results


def _forwarded_headers(response: httpx.Response) -> dict[str, str]:
    return {
        name: value
        for name in ("cache-control", "x-request-id")
        if (value := response.headers.get(name)) is not None
    }


def openai_error(message: str, status_code: int, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": code, "code": code}},
    )


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run("lan_inference_gateway.app:app", host="0.0.0.0", port=8088)
