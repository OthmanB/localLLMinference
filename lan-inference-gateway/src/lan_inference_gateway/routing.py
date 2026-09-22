"""Model-to-backend routing that keeps client requests runtime independent."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from threading import Lock

from .config import BackendConfig, ConfigurationError, GatewaySettings, PoolConfig


POOL_ROUTE_DECISIONS = "ai_gateway_pool_route_decisions_total"
POOL_AFFINITY_HITS = "ai_gateway_pool_affinity_hits_total"
POOL_AFFINITY_MISSES = "ai_gateway_pool_affinity_misses_total"
POOL_SATURATION_REJECTIONS = "ai_gateway_pool_saturation_rejections_total"
POOL_COLD_FAILOVER = "ai_gateway_pool_cold_failover_total"

_METRIC_REASONS = {
    POOL_ROUTE_DECISIONS: ("affinity_hit", "cold_failover", "least_inflight"),
    POOL_AFFINITY_HITS: ("preferred_healthy",),
    POOL_AFFINITY_MISSES: ("preferred_unhealthy",),
    POOL_SATURATION_REJECTIONS: ("all_replicas_saturated", "pinned"),
    POOL_COLD_FAILOVER: ("preferred_unhealthy",),
}


class UnknownModelError(LookupError):
    """Raised when no backend can serve a requested model."""


@dataclass(frozen=True, slots=True)
class RouteSelection:
    backend: BackendConfig
    pool: PoolConfig | None = None
    failover: bool = False


@dataclass(slots=True)
class _ReplicaState:
    inflight: int = 0
    healthy: bool | None = None


class PoolMetrics:
    """Bounded counters whose label values come only from gateway config."""

    def __init__(self, pools: tuple[PoolConfig, ...]) -> None:
        self._replicas = {
            (pool.name, replica)
            for pool in pools
            for replica in pool.replicas
        }
        self._values: dict[str, dict[tuple[str, str, str], int]] = {
            metric: {}
            for metric in _METRIC_REASONS
        }
        self._lock = Lock()

    def increment(self, metric: str, pool: PoolConfig, replica: str, reason: str) -> None:
        if (pool.name, replica) not in self._replicas:
            return
        if reason not in _METRIC_REASONS.get(metric, ()):
            return
        key = (pool.name, replica, reason)
        with self._lock:
            values = self._values[metric]
            values[key] = values.get(key, 0) + 1

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            for metric, reasons in _METRIC_REASONS.items():
                lines.append(f"# HELP {metric} LAN inference gateway pool routing counter.")
                lines.append(f"# TYPE {metric} counter")
                for pool, replica in sorted(self._replicas):
                    for reason in reasons:
                        value = self._values[metric].get((pool, replica, reason), 0)
                        labels = (
                            f'pool="{_escape_label(pool)}",'
                            f'replica="{_escape_label(replica)}",'
                            f'reason="{_escape_label(reason)}"'
                        )
                        lines.append(f"{metric}{{{labels}}} {value}")
        return "\n".join(lines) + "\n"


class BackendRegistry:
    def __init__(self, settings: GatewaySettings) -> None:
        settings.validate()
        self._backends = {backend.name: backend for backend in settings.backends}
        self._pools = {pool.model: pool for pool in settings.pools}
        self._models: dict[str, BackendConfig] = {}
        self._states: dict[str, dict[str, _ReplicaState]] = {
            pool.model: {replica: _ReplicaState() for replica in pool.replicas}
            for pool in settings.pools
        }
        self._round_robin: dict[str, int] = {pool.model: 0 for pool in settings.pools}
        self._state_lock = Lock()
        self._metrics = PoolMetrics(settings.pools)

        for backend in settings.backends:
            for model in backend.models:
                if model in self._pools:
                    continue
                if model in self._models:
                    previous = self._models[model].name
                    raise ConfigurationError(
                        f"Model {model!r} is configured for both {previous!r} and {backend.name!r}."
                    )
                self._models[model] = backend

        self._default = (
            self._backends[settings.default_backend]
            if settings.default_backend is not None
            else None
        )

    @property
    def backends(self) -> tuple[BackendConfig, ...]:
        return tuple(self._backends.values())

    @property
    def pools(self) -> tuple[PoolConfig, ...]:
        return tuple(self._pools.values())

    @property
    def metrics(self) -> PoolMetrics:
        return self._metrics

    def models(self) -> tuple[tuple[str, BackendConfig], ...]:
        return tuple(self._models.items())

    def public_models(self) -> tuple[tuple[str, str], ...]:
        singleton_models = tuple((model, backend.name) for model, backend in self._models.items())
        pool_models = tuple((pool.model, pool.name) for pool in self._pools.values())
        return singleton_models + pool_models

    def pool_for(self, model: str) -> PoolConfig | None:
        return self._pools.get(model)

    def backend_for(self, name: str) -> BackendConfig:
        return self._backends[name]

    def mark_backend_health(self, name: str, healthy: bool) -> None:
        with self._state_lock:
            for states in self._states.values():
                if name in states:
                    states[name].healthy = healthy

    def resolve(self, model: str) -> BackendConfig:
        if model in self._models:
            return self._models[model]
        if self._default is not None:
            return self._default
        raise UnknownModelError(model)

    def candidates(self, pool: PoolConfig, session: str | None) -> tuple[str, ...]:
        if session is not None:
            return tuple(
                sorted(
                    pool.replicas,
                    key=lambda replica: (
                        -_rendezvous_score(session, replica),
                        replica,
                    ),
                )
            )

        with self._state_lock:
            states = self._states[pool.model]
            ordered = sorted(
                pool.replicas,
                key=lambda replica: (states[replica].inflight, replica),
            )
            # Rotate among the least-inflight replicas so sequential unkeyed
            # requests spread across the pool instead of always preferring the
            # alphabetically-first replica.
            least_inflight = states[ordered[0]].inflight
            tied = [replica for replica in ordered if states[replica].inflight == least_inflight]
            rest = [replica for replica in ordered if states[replica].inflight != least_inflight]
            offset = self._round_robin[pool.model] % len(tied)
            self._round_robin[pool.model] += 1
            rotated = tied[offset:] + tied[:offset]
            return tuple(rotated + rest)

    def mark_health(self, pool: PoolConfig, replica: str, healthy: bool) -> None:
        with self._state_lock:
            self._states[pool.model][replica].healthy = healthy

    def try_acquire(self, pool: PoolConfig, replica: str) -> bool:
        with self._state_lock:
            state = self._states[pool.model][replica]
            if state.inflight >= pool.max_inflight:
                return False
            state.inflight += 1
            return True

    def release(self, pool: PoolConfig, replica: str) -> None:
        with self._state_lock:
            state = self._states[pool.model][replica]
            if state.inflight > 0:
                state.inflight -= 1

    def pool_state(self, pool: PoolConfig) -> list[dict[str, object]]:
        with self._state_lock:
            return [
                {
                    "name": replica,
                    "status": _health_status(self._states[pool.model][replica].healthy),
                    "inflight": self._states[pool.model][replica].inflight,
                    "max_inflight": pool.max_inflight,
                }
                for replica in pool.replicas
            ]


def _rendezvous_score(session: str, replica: str) -> int:
    digest = hashlib.sha256(f"{session}\0{replica}".encode("ascii")).digest()
    return int.from_bytes(digest, "big")


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _health_status(healthy: bool | None) -> str:
    if healthy is True:
        return "ready"
    if healthy is False:
        return "unavailable"
    return "unknown"
