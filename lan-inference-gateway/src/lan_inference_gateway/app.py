"""FastAPI application for the LAN inference gateway."""

from __future__ import annotations

import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from re import fullmatch
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .adapters import adapter_for
from .admission import AdmissionCancelled, AdmissionExpired, AdmissionQueueFull, ColdAdmissionController
from .config import BackendConfig, ConfigurationError, GatewaySettings, PoolConfig
from .models import ChatCompletionRequest
from .routing import (
    POOL_AFFINITY_HITS,
    POOL_AFFINITY_MISSES,
    POOL_COLD_FAILOVER,
    POOL_ROUTE_DECISIONS,
    POOL_SATURATION_REJECTIONS,
    BackendRegistry,
    RouteSelection,
    UnknownModelError,
)


_MAX_SESSION_LENGTH = 128
_SESSION_PATTERN = r"[A-Za-z0-9._~-]+"


@dataclass(frozen=True, slots=True)
class _PoolSaturated:
    backend: BackendConfig
    failover: bool


def create_app(
    settings: GatewaySettings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    resolved_settings = settings or GatewaySettings.from_env()
    registry = BackendRegistry(resolved_settings)
    admission_controllers = {
        backend.name: ColdAdmissionController(backend.name, backend.cold_admission)
        for backend in resolved_settings.backends
        if backend.cold_admission is not None
    }

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
    app.state.admission_controllers = admission_controllers

    @app.middleware("http")
    async def authenticate_lan_clients(request: Request, call_next: Any) -> Response:
        request_id = uuid4().hex
        request.state.gateway_request_id = request_id
        if request.url.path.startswith("/v1/") or request.url.path == "/readyz":
            try:
                expected_api_key = resolved_settings.client_api_key()
            except ConfigurationError:
                response = openai_error("Gateway authentication is misconfigured.", 500, "server_error")
                response.headers["X-Gateway-Request-ID"] = request_id
                return response

            if expected_api_key is not None:
                authorization = request.headers.get("authorization", "")
                supplied_api_key = authorization.removeprefix("Bearer ")
                if not hmac.compare_digest(
                    supplied_api_key.encode("utf-8"),
                    expected_api_key.encode("utf-8"),
                ):
                    response = openai_error("Invalid API key.", 401, "invalid_api_key")
                    response.headers["X-Gateway-Request-ID"] = request_id
                    return response

        response = await call_next(request)
        response.headers["X-Gateway-Request-ID"] = request_id
        return response

    @app.get("/metrics")
    async def metrics() -> Response:
        admission_metrics = "".join(controller.metrics.render() for controller in admission_controllers.values())
        return Response(
            content=registry.metrics.render() + admission_metrics,
            media_type="text/plain; version=0.0.4",
        )

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
        pools = _pool_health(registry, results)
        ready = (
            bool(results)
            and all(result["status"] == "ready" for result in results)
            and all(pool["status"] == "ready" for pool in pools)
        )
        return JSONResponse(
            {"status": "ready" if ready else "degraded", "backends": results, "pools": pools},
            status_code=200 if ready else 503,
        )

    @app.get("/v1/models")
    async def list_models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [
                {"id": model, "object": "model", "owned_by": owner}
                for model, owner in registry.public_models()
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: ChatCompletionRequest, request: Request) -> Response:
        client: httpx.AsyncClient = request.app.state.http_client
        pool = registry.pool_for(payload.model)
        release: Any = None

        if pool is not None:
            session = _session_header(request, pool.session_headers)
            if session is not None and not _valid_session(session):
                return openai_error("Invalid inference session header.", 400, "invalid_session")

            selected = await _select_pool(client, registry, pool, session)
            if isinstance(selected, _PoolSaturated):
                response_headers = _route_headers(selected.backend, selected.failover)
                response_headers["Retry-After"] = "1"
                return openai_error(
                    "The selected inference replica is saturated.",
                    503,
                    "pool_saturated",
                    headers=response_headers,
                )
            if selected is None:
                return openai_error(
                    "No healthy inference replica is available.",
                    503,
                    "pool_unavailable",
                    headers={"Retry-After": "1"},
                )

            backend = selected.backend
            pool_route = selected
            released = False

            def release_pool_slot() -> None:
                nonlocal released
                if not released:
                    released = True
                    registry.release(pool, backend.name)

            release = release_pool_slot
        else:
            try:
                backend = registry.resolve(payload.model)
                pool_route = RouteSelection(backend=backend)
            except UnknownModelError:
                return openai_error(
                    f"No backend is configured for model {payload.model!r}.",
                    404,
                    "model_not_found",
                )

        try:
            adapter = adapter_for(backend.adapter)
            headers = adapter.request_headers(backend)
        except ConfigurationError:
            if release is not None:
                release()
            return openai_error("Backend configuration is invalid.", 500, "server_error")

        request_body = payload.model_dump(mode="json", exclude_none=True)
        upstream_url = adapter.chat_completions_url(backend)
        upstream_timeout = (
            backend.cold_admission.upstream_timeout_seconds
            if backend.cold_admission is not None
            else resolved_settings.request_timeout_seconds
        )
        response_headers = _route_headers(backend, pool_route.failover)
        admission_release: Any = None
        admission_controller = admission_controllers.get(backend.name)
        if admission_controller is not None:
            try:
                admission_lease = await admission_controller.acquire(client, request, request_body)
            except AdmissionQueueFull:
                if release is not None:
                    release()
                return openai_error(
                    "The long-request admission queue is full.",
                    503,
                    "admission_queue_full",
                    headers={"Retry-After": "1"},
                )
            except AdmissionExpired:
                if release is not None:
                    release()
                return openai_error(
                    "The long-request admission deadline expired.",
                    504,
                    "admission_queue_deadline_exceeded",
                )
            except AdmissionCancelled:
                if release is not None:
                    release()
                return openai_error("The long request was cancelled before dispatch.", 499, "request_cancelled")
            if admission_lease is not None:
                admission_release = admission_lease.release
                response_headers.update(
                    {
                        "X-Inference-Admission": "c2",
                        "X-Inference-Gateway-Admission-Wait-Ms": str(round(admission_lease.wait_seconds * 1000, 3)),
                    }
                )

        def release_all() -> None:
            if admission_release is not None:
                admission_release()
            if release is not None:
                release()

        if payload.stream:
            return await _stream_upstream(
                client,
                upstream_url,
                request_body,
                headers,
                response_headers,
                release_all,
                upstream_timeout,
            )
        try:
            return await _request_upstream(
                client,
                upstream_url,
                request_body,
                headers,
                response_headers,
                upstream_timeout,
            )
        finally:
            release_all()

    return app


async def _request_upstream(
    client: httpx.AsyncClient,
    url: str,
    request_body: dict[str, object],
    headers: dict[str, str],
    response_headers: dict[str, str] | None = None,
    timeout: float | None = None,
) -> Response:
    try:
        upstream = await client.post(url, json=request_body, headers=headers, timeout=timeout)
    except httpx.HTTPError:
        return openai_error("Configured backend is unavailable.", 502, "backend_unavailable")

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
        headers=_response_headers(upstream, response_headers),
    )


async def _stream_upstream(
    client: httpx.AsyncClient,
    url: str,
    request_body: dict[str, object],
    headers: dict[str, str],
    response_headers: dict[str, str] | None = None,
    release: Any = None,
    timeout: float | None = None,
) -> Response:
    try:
        upstream_request = client.build_request(
            "POST",
            url,
            json=request_body,
            headers=headers,
            timeout=timeout,
        )
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError:
        if release is not None:
            release()
        return openai_error("Configured backend is unavailable.", 502, "backend_unavailable")
    except BaseException:
        if release is not None:
            release()
        raise

    if upstream.is_error:
        try:
            content = await upstream.aread()
        finally:
            try:
                await upstream.aclose()
            finally:
                if release is not None:
                    release()
        return Response(
            content=content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
            headers=_response_headers(upstream, response_headers),
        )

    async def body() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            try:
                await upstream.aclose()
            finally:
                if release is not None:
                    release()

    try:
        return StreamingResponse(
            body(),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "text/event-stream"),
            headers=_response_headers(upstream, response_headers),
        )
    except BaseException:
        try:
            await upstream.aclose()
        finally:
            if release is not None:
                release()
        raise


async def _backend_health(
    client: httpx.AsyncClient,
    registry: BackendRegistry,
) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for backend in registry.backends:
        status, healthy = await _probe_backend(client, backend)
        registry.mark_backend_health(backend.name, healthy)
        results.append({"name": backend.name, "status": status})
    return results


async def _select_pool(
    client: httpx.AsyncClient,
    registry: BackendRegistry,
    pool: PoolConfig,
    session: str | None,
) -> RouteSelection | _PoolSaturated | None:
    unhealthy_seen = False
    healthy_backend: BackendConfig | None = None
    saturated_backend: BackendConfig | None = None
    candidates = registry.candidates(pool, session)

    for replica_index, replica_name in enumerate(candidates):
        backend = registry.backend_for(replica_name)
        _status, healthy = await _probe_backend(client, backend)
        registry.mark_health(pool, replica_name, healthy)
        if not healthy:
            unhealthy_seen = True
            continue

        healthy_backend = backend
        if registry.try_acquire(pool, replica_name):
            if session is not None and not unhealthy_seen:
                registry.metrics.increment(POOL_AFFINITY_HITS, pool, replica_name, "preferred_healthy")
                route_reason = "affinity_hit"
            elif session is not None:
                registry.metrics.increment(POOL_AFFINITY_MISSES, pool, replica_name, "preferred_unhealthy")
                registry.metrics.increment(POOL_COLD_FAILOVER, pool, replica_name, "preferred_unhealthy")
                route_reason = "cold_failover"
            elif unhealthy_seen:
                registry.metrics.increment(POOL_COLD_FAILOVER, pool, replica_name, "preferred_unhealthy")
                route_reason = "cold_failover"
            else:
                route_reason = "least_inflight"
            registry.metrics.increment(POOL_ROUTE_DECISIONS, pool, replica_name, route_reason)
            return RouteSelection(
                backend=backend,
                pool=pool,
                failover=unhealthy_seen or (session is not None and replica_index > 0),
            )

        if session is not None:
            registry.metrics.increment(POOL_SATURATION_REJECTIONS, pool, replica_name, "pinned")
            return _PoolSaturated(backend, unhealthy_seen)
        if saturated_backend is None:
            saturated_backend = backend

    if saturated_backend is not None:
        registry.metrics.increment(
            POOL_SATURATION_REJECTIONS,
            pool,
            saturated_backend.name,
            "all_replicas_saturated",
        )
        return _PoolSaturated(saturated_backend, unhealthy_seen)
    if healthy_backend is not None:
        registry.metrics.increment(
            POOL_SATURATION_REJECTIONS,
            pool,
            healthy_backend.name,
            "all_replicas_saturated",
        )
        return _PoolSaturated(healthy_backend, unhealthy_seen)
    return None


async def _probe_backend(
    client: httpx.AsyncClient,
    backend: BackendConfig,
) -> tuple[str, bool]:
    try:
        adapter = adapter_for(backend.adapter)
        health_url = adapter.health_url(backend)
        headers = adapter.request_headers(backend)
    except ConfigurationError:
        return "misconfigured", False

    if health_url is None:
        return "not_configured", False

    try:
        response = await client.get(health_url, headers=headers, timeout=5.0)
    except httpx.HTTPError:
        return "unavailable", False

    return ("ready", True) if response.is_success else ("unavailable", False)


def _pool_health(
    registry: BackendRegistry,
    backend_results: list[dict[str, str]],
) -> list[dict[str, object]]:
    statuses = {result["name"]: result["status"] for result in backend_results}
    pools: list[dict[str, object]] = []
    for pool in registry.pools:
        replicas = []
        for replica in registry.pool_state(pool):
            replica = dict(replica)
            replica["status"] = statuses.get(str(replica["name"]), replica["status"])
            replicas.append(replica)
        ready = bool(replicas) and all(replica["status"] == "ready" for replica in replicas)
        pools.append(
            {
                "name": pool.name,
                "model": pool.model,
                "status": "ready" if ready else "unavailable",
                "max_inflight": pool.max_inflight,
                "replicas": replicas,
            }
        )
    return pools


def _valid_session(value: str) -> bool:
    return len(value) <= _MAX_SESSION_LENGTH and fullmatch(_SESSION_PATTERN, value) is not None


def _session_header(request: Request, names: tuple[str, ...]) -> str | None:
    for name in names:
        value = request.headers.get(name)
        if value is not None:
            return value
    return None


def _route_headers(backend: BackendConfig, failover: bool) -> dict[str, str]:
    headers = {"X-Inference-Replica": backend.name}
    if failover:
        headers["X-Inference-Failover"] = "true"
    return headers


def _response_headers(
    response: httpx.Response,
    route_headers: dict[str, str] | None = None,
) -> dict[str, str]:
    headers = _forwarded_headers(response)
    if route_headers is not None:
        headers.update(route_headers)
    return headers


def _forwarded_headers(response: httpx.Response) -> dict[str, str]:
    return {
        name: value
        for name in ("cache-control", "x-request-id")
        if (value := response.headers.get(name)) is not None
    }


def openai_error(
    message: str,
    status_code: int,
    code: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": code, "code": code}},
        headers=headers,
    )


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run("lan_inference_gateway.app:app", host="0.0.0.0", port=8088)
