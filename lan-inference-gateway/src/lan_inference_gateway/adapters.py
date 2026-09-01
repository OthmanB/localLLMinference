"""Backend protocol adapters. New runtimes are added here without changing clients."""

from __future__ import annotations

from dataclasses import dataclass

from .config import BackendConfig, ConfigurationError


@dataclass(frozen=True, slots=True)
class OpenAICompatibleAdapter:
    """Adapter for runtimes exposing OpenAI-compatible HTTP endpoints."""

    def chat_completions_url(self, backend: BackendConfig) -> str:
        return f"{backend.base_url}/v1/chat/completions"

    def health_url(self, backend: BackendConfig) -> str | None:
        if backend.health_path is None:
            return None
        return f"{backend.base_url}{backend.health_path}"

    def request_headers(self, backend: BackendConfig) -> dict[str, str]:
        headers = {"accept": "application/json, text/event-stream"}
        api_key = backend.upstream_api_key()
        if api_key is not None:
            headers["authorization"] = f"Bearer {api_key}"
        return headers


_ADAPTERS = {"openai": OpenAICompatibleAdapter()}


def adapter_for(name: str) -> OpenAICompatibleAdapter:
    try:
        return _ADAPTERS[name]
    except KeyError as error:
        supported = ", ".join(sorted(_ADAPTERS))
        raise ConfigurationError(f"Unsupported adapter {name!r}; supported adapters: {supported}.") from error
