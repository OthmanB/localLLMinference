from __future__ import annotations

import json

import httpx
import pytest

from lan_inference_gateway.app import create_app
from lan_inference_gateway.config import BackendConfig, GatewaySettings


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
            readiness = await client.get("/readyz")
            authorized = await client.get(
                "/v1/models",
                headers={"authorization": "Bearer gateway-secret"},
            )

    assert unauthorized.status_code == 401
    assert readiness.status_code == 200
    assert authorized.status_code == 200
