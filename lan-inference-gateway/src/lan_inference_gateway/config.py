"""Environment-backed configuration for the gateway."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from re import fullmatch
from urllib.parse import urlsplit


class ConfigurationError(ValueError):
    """Raised when a gateway configuration value is invalid."""


@dataclass(frozen=True, slots=True)
class ColdAdmissionConfig:
    metrics_url: str
    target_model: str
    max_active_leases: int = 3
    dispatch_running_limit: int = 2
    cold_min_input_tokens: int = 100_000
    poll_interval_seconds: float = 1.0
    queue_depth: int = 3
    queue_timeout_seconds: float = 900.0
    upstream_timeout_seconds: float = 600.0

    @classmethod
    def from_mapping(cls, value: object, backend_name: str) -> "ColdAdmissionConfig":
        if not isinstance(value, dict):
            raise ConfigurationError(f"Backend {backend_name!r} cold_admission must be a JSON object.")
        allowed = {
            "metrics_url",
            "target_model",
            "max_active_leases",
            "dispatch_running_limit",
            "cold_min_input_tokens",
            "poll_interval_seconds",
            "queue_depth",
            "queue_timeout_seconds",
            "upstream_timeout_seconds",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ConfigurationError(
                f"Backend {backend_name!r} cold_admission has unknown fields: {', '.join(unknown)}."
            )

        metrics_url = value.get("metrics_url")
        target_model = value.get("target_model")
        if not isinstance(metrics_url, str) or not _is_loopback_http_url(metrics_url):
            raise ConfigurationError(
                f"Backend {backend_name!r} cold_admission metrics_url must be an absolute loopback HTTP(S) URL."
            )
        if not isinstance(target_model, str) or not target_model:
            raise ConfigurationError(f"Backend {backend_name!r} cold_admission target_model is required.")

        integers = {
            "max_active_leases": value.get("max_active_leases", 3),
            "dispatch_running_limit": value.get("dispatch_running_limit", 2),
            "cold_min_input_tokens": value.get("cold_min_input_tokens", 100_000),
            "queue_depth": value.get("queue_depth", 3),
        }
        for name, item in integers.items():
            if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                raise ConfigurationError(f"Backend {backend_name!r} cold_admission {name} must be positive.")

        poll_interval = value.get("poll_interval_seconds", 1.0)
        queue_timeout = value.get("queue_timeout_seconds", 900.0)
        upstream_timeout = value.get("upstream_timeout_seconds", 600.0)
        for name, item in {
            "poll_interval_seconds": poll_interval,
            "queue_timeout_seconds": queue_timeout,
            "upstream_timeout_seconds": upstream_timeout,
        }.items():
            if isinstance(item, bool) or not isinstance(item, (int, float)) or item <= 0:
                raise ConfigurationError(f"Backend {backend_name!r} cold_admission {name} must be positive.")

        if integers["dispatch_running_limit"] >= integers["max_active_leases"]:
            raise ConfigurationError(
                f"Backend {backend_name!r} cold_admission dispatch_running_limit must be below max_active_leases."
            )
        if integers["max_active_leases"] > 3:
            raise ConfigurationError(f"Backend {backend_name!r} cold_admission max_active_leases must not exceed three.")
        if integers["queue_depth"] > 3:
            raise ConfigurationError(f"Backend {backend_name!r} cold_admission queue_depth must not exceed three.")

        return cls(
            metrics_url=metrics_url,
            target_model=target_model,
            max_active_leases=integers["max_active_leases"],
            dispatch_running_limit=integers["dispatch_running_limit"],
            cold_min_input_tokens=integers["cold_min_input_tokens"],
            poll_interval_seconds=float(poll_interval),
            queue_depth=integers["queue_depth"],
            queue_timeout_seconds=float(queue_timeout),
            upstream_timeout_seconds=float(upstream_timeout),
        )


@dataclass(frozen=True, slots=True)
class BackendConfig:
    name: str
    base_url: str
    models: tuple[str, ...]
    adapter: str = "openai"
    health_path: str | None = "/health"
    api_key_env: str | None = None
    cold_admission: ColdAdmissionConfig | None = None

    @classmethod
    def from_mapping(cls, value: object) -> "BackendConfig":
        if not isinstance(value, dict):
            raise ConfigurationError("Each backend must be a JSON object.")

        allowed = {"name", "base_url", "models", "adapter", "health_path", "api_key_env", "cold_admission"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ConfigurationError(f"Backend contains unknown fields: {', '.join(unknown)}.")

        name = value.get("name")
        base_url = value.get("base_url")
        models = value.get("models", [])
        adapter = value.get("adapter", "openai")
        health_path = value.get("health_path", "/health")
        api_key_env = value.get("api_key_env")
        cold_admission = (
            None
            if value.get("cold_admission") is None
            else ColdAdmissionConfig.from_mapping(value["cold_admission"], name if isinstance(name, str) else "<unnamed>")
        )

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
            cold_admission=cold_admission,
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
class PoolConfig:
    name: str
    model: str
    replicas: tuple[str, ...]
    session_headers: tuple[str, ...] = ("X-Inference-Session",)
    max_inflight: int = 1

    @classmethod
    def from_mapping(cls, value: object) -> "PoolConfig":
        if not isinstance(value, dict):
            raise ConfigurationError("Each pool must be a JSON object.")

        allowed = {"name", "model", "replicas", "session_header", "session_headers", "max_inflight"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ConfigurationError(f"Pool contains unknown fields: {', '.join(unknown)}.")

        name = value.get("name")
        model = value.get("model")
        replicas = value.get("replicas")
        session_header = value.get("session_header")
        session_headers = value.get("session_headers")
        max_inflight = value.get("max_inflight", 1)

        if not isinstance(name, str) or not name:
            raise ConfigurationError("Each pool requires a non-empty name.")
        if not isinstance(model, str) or not model:
            raise ConfigurationError(f"Pool {name!r} requires a non-empty model.")
        if not isinstance(replicas, list) or not replicas or not all(
            isinstance(replica, str) and replica for replica in replicas
        ):
            raise ConfigurationError(f"Pool {name!r} replicas must be a non-empty list of names.")
        if len(replicas) != len(set(replicas)):
            raise ConfigurationError(f"Pool {name!r} must not list a replica more than once.")
        if session_header is not None and session_headers is not None:
            raise ConfigurationError(
                f"Pool {name!r} must set either session_header or session_headers, not both."
            )
        if session_headers is None:
            session_headers = [session_header] if session_header is not None else ["X-Inference-Session"]
        if not isinstance(session_headers, list) or not session_headers or not all(
            isinstance(header, str) and _is_http_header_name(header) for header in session_headers
        ):
            raise ConfigurationError(
                f"Pool {name!r} session_headers must be a non-empty list of HTTP header names."
            )
        if len(session_headers) != len({header.lower() for header in session_headers}):
            raise ConfigurationError(f"Pool {name!r} must not repeat a session header.")
        if isinstance(max_inflight, bool) or not isinstance(max_inflight, int) or max_inflight <= 0:
            raise ConfigurationError(f"Pool {name!r} max_inflight must be a positive integer.")

        return cls(
            name=name,
            model=model,
            replicas=tuple(replicas),
            session_headers=tuple(session_headers),
            max_inflight=max_inflight,
        )


@dataclass(frozen=True, slots=True)
class GatewaySettings:
    backends: tuple[BackendConfig, ...]
    default_backend: str | None = None
    request_timeout_seconds: float = 300.0
    client_api_key_env: str | None = None
    pools: tuple[PoolConfig, ...] = ()

    @classmethod
    def from_env(cls) -> "GatewaySettings":
        raw_backends = os.getenv("LAN_INFERENCE_BACKENDS", "[]")
        try:
            configured_backends = json.loads(raw_backends)
        except json.JSONDecodeError as error:
            raise ConfigurationError("LAN_INFERENCE_BACKENDS must be valid JSON.") from error

        if not isinstance(configured_backends, list):
            raise ConfigurationError("LAN_INFERENCE_BACKENDS must be a JSON array.")

        raw_pools = os.getenv("LAN_INFERENCE_POOLS", "[]")
        try:
            configured_pools = json.loads(raw_pools)
        except json.JSONDecodeError as error:
            raise ConfigurationError("LAN_INFERENCE_POOLS must be valid JSON.") from error

        if not isinstance(configured_pools, list):
            raise ConfigurationError("LAN_INFERENCE_POOLS must be a JSON array.")

        timeout = _float_env("LAN_INFERENCE_REQUEST_TIMEOUT_SECONDS", 300.0)
        settings = cls(
            backends=tuple(BackendConfig.from_mapping(item) for item in configured_backends),
            default_backend=os.getenv("LAN_INFERENCE_DEFAULT_BACKEND") or None,
            request_timeout_seconds=timeout,
            client_api_key_env=os.getenv("LAN_INFERENCE_CLIENT_API_KEY_ENV") or None,
            pools=tuple(PoolConfig.from_mapping(item) for item in configured_pools),
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

        pool_names = [pool.name for pool in self.pools]
        if len(pool_names) != len(set(pool_names)):
            raise ConfigurationError("Pool names must be unique.")
        pool_models = [pool.model for pool in self.pools]
        if len(pool_models) != len(set(pool_models)):
            raise ConfigurationError("Pool model IDs must be unique.")

        backend_names = set(names)
        for pool in self.pools:
            if not isinstance(pool.name, str) or not pool.name:
                raise ConfigurationError("Pool names and model IDs must be non-empty.")
            if not isinstance(pool.model, str) or not pool.model:
                raise ConfigurationError("Pool names and model IDs must be non-empty.")
            if not isinstance(pool.replicas, (list, tuple)) or not pool.replicas or not all(
                isinstance(replica, str) and replica for replica in pool.replicas
            ) or len(pool.replicas) != len(set(pool.replicas)):
                raise ConfigurationError(f"Pool {pool.name!r} replicas must be unique and non-empty.")
            if not isinstance(pool.session_headers, (list, tuple)) or not pool.session_headers or not all(
                isinstance(header, str) and _is_http_header_name(header)
                for header in pool.session_headers
            ):
                raise ConfigurationError(
                    f"Pool {pool.name!r} session_headers must be a non-empty list of HTTP header names."
                )
            if (
                isinstance(pool.max_inflight, bool)
                or not isinstance(pool.max_inflight, int)
                or pool.max_inflight <= 0
            ):
                raise ConfigurationError(f"Pool {pool.name!r} max_inflight must be positive.")
            unknown = sorted(set(pool.replicas) - backend_names)
            if unknown:
                raise ConfigurationError(
                    f"Pool {pool.name!r} references unknown replicas: {', '.join(unknown)}."
                )
            pooled_admission = [
                backend.name
                for backend in self.backends
                if backend.name in pool.replicas and backend.cold_admission is not None
            ]
            if pooled_admission:
                raise ConfigurationError(
                    f"Pool {pool.name!r} cannot reference cold_admission backends: {', '.join(sorted(pooled_admission))}."
                )
            model_owners = {
                backend.name
                for backend in self.backends
                if pool.model in backend.models
            }
            if not model_owners.issubset(pool.replicas):
                raise ConfigurationError(
                    f"Pool model {pool.model!r} is also configured on a non-replica backend."
                )

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


def _is_loopback_http_url(value: str) -> bool:
    parsed = urlsplit(value)
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        and bool(parsed.port or parsed.netloc)
    )


def _is_http_header_name(value: str) -> bool:
    return fullmatch(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+", value) is not None


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be a number.") from error
