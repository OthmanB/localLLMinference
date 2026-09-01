"""Environment-backed configuration for the gateway."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.parse import urlsplit


class ConfigurationError(ValueError):
    """Raised when a gateway configuration value is invalid."""


@dataclass(frozen=True, slots=True)
class BackendConfig:
    name: str
    base_url: str
    models: tuple[str, ...]
    adapter: str = "openai"
    health_path: str | None = "/health"
    api_key_env: str | None = None

    @classmethod
    def from_mapping(cls, value: object) -> "BackendConfig":
        if not isinstance(value, dict):
            raise ConfigurationError("Each backend must be a JSON object.")

        name = value.get("name")
        base_url = value.get("base_url")
        models = value.get("models", [])
        adapter = value.get("adapter", "openai")
        health_path = value.get("health_path", "/health")
        api_key_env = value.get("api_key_env")

        if not isinstance(name, str) or not name:
            raise ConfigurationError("Each backend requires a non-empty name.")
        if not isinstance(base_url, str) or not _is_http_url(base_url):
            raise ConfigurationError(f"Backend {name!r} requires an absolute HTTP(S) base_url.")
        if not isinstance(models, list) or not all(isinstance(model, str) and model for model in models):
            raise ConfigurationError(f"Backend {name!r} models must be a list of non-empty strings.")
        if not isinstance(adapter, str) or not adapter:
            raise ConfigurationError(f"Backend {name!r} adapter must be a non-empty string.")
        if health_path is not None and (
            not isinstance(health_path, str) or not health_path.startswith("/")
        ):
            raise ConfigurationError(f"Backend {name!r} health_path must start with '/'.")
        if api_key_env is not None and (not isinstance(api_key_env, str) or not api_key_env):
            raise ConfigurationError(f"Backend {name!r} api_key_env must be a non-empty string.")

        return cls(
            name=name,
            base_url=base_url.rstrip("/"),
            models=tuple(models),
            adapter=adapter,
            health_path=health_path,
            api_key_env=api_key_env,
        )

    def upstream_api_key(self) -> str | None:
        if self.api_key_env is None:
            return None
        value = os.getenv(self.api_key_env)
        if not value:
            raise ConfigurationError(
                f"Backend {self.name!r} references unset environment variable {self.api_key_env!r}."
            )
        return value


@dataclass(frozen=True, slots=True)
class GatewaySettings:
    backends: tuple[BackendConfig, ...]
    default_backend: str | None = None
    request_timeout_seconds: float = 300.0
    client_api_key_env: str | None = None

    @classmethod
    def from_env(cls) -> "GatewaySettings":
        raw_backends = os.getenv("LAN_INFERENCE_BACKENDS", "[]")
        try:
            configured_backends = json.loads(raw_backends)
        except json.JSONDecodeError as error:
            raise ConfigurationError("LAN_INFERENCE_BACKENDS must be valid JSON.") from error

        if not isinstance(configured_backends, list):
            raise ConfigurationError("LAN_INFERENCE_BACKENDS must be a JSON array.")

        timeout = _float_env("LAN_INFERENCE_REQUEST_TIMEOUT_SECONDS", 300.0)
        settings = cls(
            backends=tuple(BackendConfig.from_mapping(item) for item in configured_backends),
            default_backend=os.getenv("LAN_INFERENCE_DEFAULT_BACKEND") or None,
            request_timeout_seconds=timeout,
            client_api_key_env=os.getenv("LAN_INFERENCE_CLIENT_API_KEY_ENV") or None,
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        names = [backend.name for backend in self.backends]
        if len(names) != len(set(names)):
            raise ConfigurationError("Backend names must be unique.")
        if self.default_backend is not None and self.default_backend not in names:
            raise ConfigurationError("LAN_INFERENCE_DEFAULT_BACKEND must name a configured backend.")
        if self.request_timeout_seconds <= 0:
            raise ConfigurationError("LAN_INFERENCE_REQUEST_TIMEOUT_SECONDS must be positive.")

    def client_api_key(self) -> str | None:
        if self.client_api_key_env is None:
            return None
        value = os.getenv(self.client_api_key_env)
        if not value:
            raise ConfigurationError(
                f"Client API-key environment variable {self.client_api_key_env!r} is unset."
            )
        return value


def _is_http_url(value: str) -> bool:
    parsed = urlsplit(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be a number.") from error
