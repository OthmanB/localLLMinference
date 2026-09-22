from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from lan_inference_gateway.app import create_app
from lan_inference_gateway.config import BackendConfig, ConfigurationError, GatewaySettings, PoolConfig


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_lists_models_and_routes_to_matching_backend() -> None:
    received: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        if request.url.path == "/health":
            return httpx.Response(200)
        return httpx.Response(
            200,
            json={"id": "chatcmpl-1", "object": "chat.completion", "model": "ornith"},
        )

    app = create_app(
        GatewaySettings(
            backends=(
                BackendConfig("llama", "http://llama.test", ("qwen",)),
                BackendConfig("vllm", "http://vllm.test", ("ornith",)),
            ),
        ),
        httpx.MockTransport(handler),
    )

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
            models = await client.get("/v1/models")
            completion = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "ornith",
                    "messages": [{"role": "user", "content": "hello"}],
                    "reasoning_effort": "xhigh",
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 20,
                    "presence_penalty": 0,
                },
            )
            unknown_model = await client.post(
                "/v1/chat/completions",
                json={"model": "not-configured", "messages": [{"role": "user", "content": "hello"}]},
            )

    assert models.status_code == 200
    assert [model["id"] for model in models.json()["data"]] == ["qwen", "ornith"]
    assert completion.status_code == 200
    assert completion.json()["model"] == "ornith"
    assert unknown_model.status_code == 404
    assert unknown_model.json()["error"]["type"] == "model_not_found"
    assert received[-1].url == httpx.URL("http://vllm.test/v1/chat/completions")
    assert json.loads(received[-1].content)["model"] == "ornith"
    assert json.loads(received[-1].content)["reasoning_effort"] == "xhigh"
    assert json.loads(received[-1].content)["temperature"] == 1.0
    assert json.loads(received[-1].content)["top_p"] == 0.95
    assert json.loads(received[-1].content)["top_k"] == 20
    assert json.loads(received[-1].content)["presence_penalty"] == 0


@pytest.mark.anyio
async def test_streaming_response_is_forwarded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n',
            headers={"content-type": "text/event-stream"},
        )

    app = create_app(
        GatewaySettings(backends=(BackendConfig("llama", "http://llama.test", ("qwen",)),)),
        httpx.MockTransport(handler),
    )

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content.endswith(b"data: [DONE]\n\n")


@pytest.mark.anyio
async def test_gateway_api_key_and_backend_readiness(monkeypatch) -> None:
    monkeypatch.setenv("UPSTREAM_KEY", "backend-secret")
    monkeypatch.setenv("GATEWAY_KEY", "gateway-secret")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer backend-secret"
        if request.url.path == "/health":
            return httpx.Response(200)
        return httpx.Response(200, json={"id": "chatcmpl-1"})

    app = create_app(
        GatewaySettings(
            backends=(
                BackendConfig(
                    "llama",
                    "http://llama.test",
                    ("qwen",),
                    api_key_env="UPSTREAM_KEY",
                ),
            ),
            client_api_key_env="GATEWAY_KEY",
        ),
        httpx.MockTransport(handler),
    )

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
            unauthorized = await client.get("/v1/models")
            non_ascii_unauthorized = await client.get(
                "/v1/models",
                headers=[(b"authorization", b"Bearer caf\xc3\xa9")],
            )
            readiness_unauthorized = await client.get("/readyz")
            readiness = await client.get(
                "/readyz",
                headers={"authorization": "Bearer gateway-secret"},
            )
            metrics = await client.get("/metrics")
            authorized = await client.get(
                "/v1/models",
                headers={"authorization": "Bearer gateway-secret"},
            )

    assert unauthorized.status_code == 401
    assert non_ascii_unauthorized.status_code == 401
    assert readiness_unauthorized.status_code == 401
    assert readiness.status_code == 200
    assert metrics.status_code == 200
    assert authorized.status_code == 200


def _pool_settings(max_inflight: int = 1) -> GatewaySettings:
    return GatewaySettings(
        backends=(
            BackendConfig("gpu1", "http://gpu1.test", ("model-gpu1",)),
            BackendConfig("gpu2", "http://gpu2.test", ("model-gpu2",)),
        ),
        pools=(
            PoolConfig("qwen-pool", "model-pooled", ("gpu1", "gpu2"), max_inflight=max_inflight),
        ),
    )


def _chat_payload(model: str = "model-pooled", stream: bool = False) -> dict[str, object]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": stream,
    }


@pytest.mark.anyio
async def test_pool_models_and_explicit_replica_routing() -> None:
    received: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        received.append(request.url.host or "")
        return httpx.Response(200, json={"id": "chatcmpl-1", "model": "ok"})

    app = create_app(_pool_settings(), httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            models = await client.get("/v1/models")
            gpu1 = await client.post("/v1/chat/completions", json=_chat_payload("model-gpu1"))
            gpu2 = await client.post("/v1/chat/completions", json=_chat_payload("model-gpu2"))
            pooled = await client.post("/v1/chat/completions", json=_chat_payload())

    assert [item["id"] for item in models.json()["data"]] == [
        "model-gpu1",
        "model-gpu2",
        "model-pooled",
    ]
    assert gpu1.headers["x-inference-replica"] == "gpu1"
    assert gpu2.headers["x-inference-replica"] == "gpu2"
    assert pooled.headers["x-inference-replica"] in {"gpu1", "gpu2"}
    assert received[:2] == ["gpu1.test", "gpu2.test"]


@pytest.mark.anyio
async def test_pooled_session_affinity_and_distribution() -> None:
    received: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        received.append(request.url.host or "")
        return httpx.Response(200, json={"id": "chatcmpl-1"})

    app = create_app(_pool_settings(), httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            repeated = [
                await client.post(
                    "/v1/chat/completions",
                    json=_chat_payload(),
                    headers={"X-Inference-Session": "stable-session"},
                )
                for _ in range(3)
            ]
            for index in range(24):
                await client.post(
                    "/v1/chat/completions",
                    json=_chat_payload(),
                    headers={"X-Inference-Session": f"session-{index}"},
                )

    assert len({response.headers["x-inference-replica"] for response in repeated}) == 1
    assert set(received) == {"gpu1.test", "gpu2.test"}


@pytest.mark.anyio
async def test_unkeyed_requests_use_least_inflight_replica() -> None:
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    received: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        received.append(request.url.host or "")
        if len(received) == 1:
            first_started.set()
            await release_first.wait()
        return httpx.Response(200, json={"id": "chatcmpl-1"})

    app = create_app(_pool_settings(), httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            first = asyncio.create_task(
                client.post("/v1/chat/completions", json=_chat_payload())
            )
            await first_started.wait()
            second = await client.post("/v1/chat/completions", json=_chat_payload())
            release_first.set()
            await first

    assert second.headers["x-inference-replica"] == "gpu2"
    assert received == ["gpu1.test", "gpu2.test"]


@pytest.mark.anyio
async def test_pinned_saturation_returns_retryable_503_without_spill() -> None:
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    posts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        posts.append(request.url.host or "")
        first_started.set()
        await release_first.wait()
        return httpx.Response(200, json={"id": "chatcmpl-1"})

    app = create_app(_pool_settings(), httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            first = asyncio.create_task(
                client.post(
                    "/v1/chat/completions",
                    json=_chat_payload(),
                    headers={"X-Inference-Session": "pinned"},
                )
            )
            await first_started.wait()
            second = await client.post(
                "/v1/chat/completions",
                json=_chat_payload(),
                headers={"X-Inference-Session": "pinned"},
            )
            release_first.set()
            await first

    assert second.status_code == 503
    assert second.headers["retry-after"] == "1"
    assert second.json()["error"]["code"] == "pool_saturated"
    assert len(posts) == 1


@pytest.mark.anyio
async def test_invalid_session_headers_are_rejected_without_upstream_dispatch() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200)

    app = create_app(_pool_settings(), httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            invalid = await client.post(
                "/v1/chat/completions",
                json=_chat_payload(),
                headers={"X-Inference-Session": "bad session"},
            )
            oversized = await client.post(
                "/v1/chat/completions",
                json=_chat_payload(),
                headers={"X-Inference-Session": "x" * 129},
            )
            empty = await client.post(
                "/v1/chat/completions",
                json=_chat_payload(),
                headers={"X-Inference-Session": ""},
            )

    assert [response.status_code for response in (invalid, oversized, empty)] == [400, 400, 400]
    assert all(response.json()["error"]["code"] == "invalid_session" for response in (invalid, oversized, empty))
    assert calls == 0


@pytest.mark.anyio
async def test_unhealthy_selected_replica_cold_fails_over_before_post() -> None:
    posts: list[str] = []
    health_calls: list[str] = []
    settings = _pool_settings()
    selected = create_app(settings).state.registry.candidates(settings.pools[0], "failover-session")[0]
    unhealthy_host = f"{selected}.test"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            health_calls.append(request.url.host or "")
            return httpx.Response(503 if request.url.host == unhealthy_host else 200)
        posts.append(request.url.host or "")
        return httpx.Response(200, json={"id": "chatcmpl-1"})

    app = create_app(settings, httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json=_chat_payload(),
                headers={"X-Inference-Session": "failover-session"},
            )

    healthy = "gpu2.test" if unhealthy_host == "gpu1.test" else "gpu1.test"
    assert response.status_code == 200
    assert response.headers["x-inference-replica"] == healthy.removesuffix(".test")
    assert response.headers["x-inference-failover"] == "true"
    assert posts == [healthy]
    assert unhealthy_host in health_calls


@pytest.mark.anyio
async def test_streaming_response_releases_pool_slot() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        return httpx.Response(
            200,
            content=b"data: {\"choices\":[]}\n\ndata: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

    app = create_app(_pool_settings(), httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json=_chat_payload(stream=True),
                headers={"X-Inference-Session": "stream"},
            )
            state = app.state.registry.pool_state(app.state.settings.pools[0])

    assert response.status_code == 200
    assert response.content.endswith(b"data: [DONE]\n\n")
    assert all(replica["inflight"] == 0 for replica in state)


@pytest.mark.anyio
async def test_models_and_readiness_report_pool_replica_state() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "gpu2.test" and request.url.path == "/health":
            return httpx.Response(503)
        if request.url.path == "/health":
            return httpx.Response(200)
        return httpx.Response(200, json={"id": "chatcmpl-1"})

    app = create_app(_pool_settings(), httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            models = await client.get("/v1/models")
            readiness = await client.get("/readyz")

    assert models.status_code == 200
    assert readiness.status_code == 503
    pool = readiness.json()["pools"][0]
    assert pool["model"] == "model-pooled"
    assert {replica["name"] for replica in pool["replicas"]} == {"gpu1", "gpu2"}
    assert next(replica for replica in pool["replicas"] if replica["name"] == "gpu2")["status"] == "unavailable"


@pytest.mark.anyio
async def test_metrics_are_public_bounded_and_count_pool_events(monkeypatch) -> None:
    monkeypatch.setenv("METRICS_GATEWAY_KEY", "gateway-secret")
    settings = GatewaySettings(
        backends=_pool_settings().backends,
        pools=_pool_settings().pools,
        client_api_key_env="METRICS_GATEWAY_KEY",
    )
    probe_app = create_app(settings)
    preferred = probe_app.state.registry.candidates(settings.pools[0], "metrics-session")[0]
    unhealthy_preferred = False
    hold_requests = False
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    auth_headers = {"authorization": "Bearer gateway-secret"}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            if unhealthy_preferred and request.url.host == f"{preferred}.test":
                return httpx.Response(503)
            return httpx.Response(200)
        if hold_requests:
            first_started.set()
            await release_first.wait()
        return httpx.Response(200, json={"id": "chatcmpl-1"})

    app = create_app(settings, httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            hit = await client.post(
                "/v1/chat/completions",
                json=_chat_payload(),
                headers={**auth_headers, "X-Inference-Session": "metrics-session"},
            )
            unhealthy_preferred = True
            fallback = await client.post(
                "/v1/chat/completions",
                json=_chat_payload(),
                headers={**auth_headers, "X-Inference-Session": "metrics-session"},
            )
            unhealthy_preferred = False
            hold_requests = True
            first = asyncio.create_task(
                client.post(
                    "/v1/chat/completions",
                    json=_chat_payload(),
                    headers={**auth_headers, "X-Inference-Session": "saturation-session"},
                )
            )
            await first_started.wait()
            saturated = await client.post(
                "/v1/chat/completions",
                json=_chat_payload(),
                headers={**auth_headers, "X-Inference-Session": "saturation-session"},
            )
            release_first.set()
            await first
            metrics = await client.get("/metrics")
            readiness = await client.get("/readyz")

    assert hit.status_code == 200
    assert fallback.status_code == 200
    assert saturated.status_code == 503
    assert readiness.status_code == 401
    assert metrics.status_code == 200
    assert metrics.headers["content-type"].startswith("text/plain; version=0.0.4")

    metric_names = (
        "ai_gateway_pool_route_decisions_total",
        "ai_gateway_pool_affinity_hits_total",
        "ai_gateway_pool_affinity_misses_total",
        "ai_gateway_pool_saturation_rejections_total",
        "ai_gateway_pool_cold_failover_total",
    )
    for metric_name in metric_names:
        assert metric_name in metrics.text
    samples = [line for line in metrics.text.splitlines() if line and not line.startswith("#")]
    for sample in samples:
        labels = sample.split("{", 1)[1].split("}", 1)[0]
        assert {label.split("=", 1)[0] for label in labels.split(",")} == {
            "pool",
            "replica",
            "reason",
        }
    assert 'reason="preferred_healthy"} 1' in metrics.text
    assert 'reason="preferred_unhealthy"} 1' in metrics.text
    assert 'reason="pinned"} 1' in metrics.text
    assert "metrics-session" not in metrics.text
    assert "saturation-session" not in metrics.text


@pytest.mark.anyio
async def test_tool_calls_forward_and_upstream_failure_is_not_retried() -> None:
    posts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        posts.append(request.url.host or "")
        if len(posts) == 1:
            body = json.loads(request.content)
            assert body["tools"][0]["function"]["name"] == "lookup"
            raise httpx.ConnectError("upstream closed", request=request)
        return httpx.Response(200, json={"id": "chatcmpl-1"})

    app = create_app(_pool_settings(), httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    **_chat_payload(),
                    "tools": [
                        {
                            "type": "function",
                            "function": {"name": "lookup", "parameters": {"type": "object"}},
                        }
                    ],
                },
                headers={"X-Inference-Session": "no-retry"},
            )

    assert response.status_code == 502
    assert len(posts) == 1


def test_pool_environment_configuration_is_strict(monkeypatch) -> None:
    monkeypatch.setenv(
        "LAN_INFERENCE_BACKENDS",
        json.dumps(
            [
                {
                    "name": "gpu1",
                    "base_url": "http://gpu1.test",
                    "models": ["model-gpu1"],
                }
            ]
        ),
    )
    monkeypatch.setenv(
        "LAN_INFERENCE_POOLS",
        json.dumps(
            [
                {
                    "name": "qwen-pool",
                    "model": "model-pooled",
                    "replicas": ["missing"],
                    "unexpected": True,
                }
            ]
        ),
    )

    with pytest.raises(ConfigurationError):
        GatewaySettings.from_env()


@pytest.mark.anyio
async def test_sequential_unkeyed_requests_alternate_replicas() -> None:
    received: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        received.append(request.url.host or "")
        return httpx.Response(200, json={"id": "chatcmpl-1"})

    app = create_app(_pool_settings(), httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            for _ in range(4):
                await client.post("/v1/chat/completions", json=_chat_payload())

    assert received == ["gpu1.test", "gpu2.test", "gpu1.test", "gpu2.test"]


@pytest.mark.anyio
async def test_configured_session_headers_are_honored() -> None:
    settings = GatewaySettings(
        backends=(
            BackendConfig("gpu1", "http://gpu1.test", ("model-gpu1",)),
            BackendConfig("gpu2", "http://gpu2.test", ("model-gpu2",)),
        ),
        pools=(
            PoolConfig(
                "qwen-pool",
                "model-pooled",
                ("gpu1", "gpu2"),
                session_headers=("X-Inference-Session", "X-Session-Id"),
                max_inflight=1,
            ),
        ),
    )
    received: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        received.append(request.url.host or "")
        return httpx.Response(200, json={"id": "chatcmpl-1"})

    app = create_app(settings, httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            repeated = [
                await client.post(
                    "/v1/chat/completions",
                    json=_chat_payload(),
                    headers={"X-Session-Id": "opencode-session"},
                )
                for _ in range(3)
            ]

    assert len({response.headers["x-inference-replica"] for response in repeated}) == 1


def test_pool_session_header_configuration(monkeypatch) -> None:
    monkeypatch.setenv(
        "LAN_INFERENCE_BACKENDS",
        json.dumps(
            [
                {"name": "gpu1", "base_url": "http://gpu1.test", "models": ["model-gpu1"]},
                {"name": "gpu2", "base_url": "http://gpu2.test", "models": ["model-gpu2"]},
            ]
        ),
    )

    monkeypatch.setenv(
        "LAN_INFERENCE_POOLS",
        json.dumps(
            [{"name": "pool", "model": "pooled", "replicas": ["gpu1", "gpu2"], "session_header": "X-Inference-Session"}]
        ),
    )
    assert GatewaySettings.from_env().pools[0].session_headers == ("X-Inference-Session",)

    monkeypatch.setenv(
        "LAN_INFERENCE_POOLS",
        json.dumps(
            [
                {
                    "name": "pool",
                    "model": "pooled",
                    "replicas": ["gpu1", "gpu2"],
                    "session_headers": ["X-Inference-Session", "X-Session-Id"],
                }
            ]
        ),
    )
    assert GatewaySettings.from_env().pools[0].session_headers == ("X-Inference-Session", "X-Session-Id")

    monkeypatch.setenv(
        "LAN_INFERENCE_POOLS",
        json.dumps(
            [
                {
                    "name": "pool",
                    "model": "pooled",
                    "replicas": ["gpu1", "gpu2"],
                    "session_header": "X-Inference-Session",
                    "session_headers": ["X-Session-Id"],
                }
            ]
        ),
    )
    with pytest.raises(ConfigurationError):
        GatewaySettings.from_env()


