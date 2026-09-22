"""Minimal C<=2 admission for a dedicated long-context backend."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import logging
import re
import time
from typing import Any

import httpx

from .config import ColdAdmissionConfig


EVENTS = (
    "enqueued",
    "dispatched_immediate",
    "dispatched_queued",
    "rejected_queue_full",
    "expired",
    "cancelled",
    "idle_probe_failed",
)
LOGGER = logging.getLogger(__name__)


class AdmissionQueueFull(Exception):
    """Raised when the bounded long-request queue is full."""


class AdmissionExpired(Exception):
    """Raised when a queued long request reaches its deadline."""


class AdmissionCancelled(Exception):
    """Raised when a queued client disconnects before dispatch."""


@dataclass(frozen=True, slots=True)
class SchedulerObservation:
    running: int
    waiting: int


@dataclass(frozen=True, slots=True)
class AdmissionLease:
    wait_seconds: float
    release: Any


@dataclass(slots=True)
class _Ticket:
    accepted_at: float


class AdmissionMetrics:
    """Bounded metrics with labels supplied only by the configured backend."""

    def __init__(self, backend_name: str, model: str) -> None:
        self.backend_name = backend_name
        self.model = model
        self.queue_depth = 0
        self.inflight = 0
        self.events = {event: 0 for event in EVENTS}
        self.wait_seconds = {"dispatched": 0.0, "expired": 0.0, "cancelled": 0.0}
        self.idle_status = "unknown"
        self.idle_probe_failures = 0

    def event(self, name: str, wait_seconds: float | None = None) -> None:
        self.events[name] += 1
        outcome = "dispatched" if name.startswith("dispatched") else name
        if wait_seconds is not None and outcome in self.wait_seconds:
            self.wait_seconds[outcome] += wait_seconds

    def transition(self, event: str, wait_seconds: float | None = None) -> None:
        LOGGER.info(
            "cold_admission_transition",
            extra={
                "admission_backend": self.backend_name,
                "admission_model": self.model,
                "admission_event": event,
                "admission_wait_seconds": wait_seconds,
            },
        )

    def render(self) -> str:
        labels = f'backend="{_escape(self.backend_name)}",model="{_escape(self.model)}"'
        lines = [
            "# HELP ai_gateway_cold_admission_queue_depth Queued long requests.",
            "# TYPE ai_gateway_cold_admission_queue_depth gauge",
            f"ai_gateway_cold_admission_queue_depth{{{labels}}} {self.queue_depth}",
            "# HELP ai_gateway_cold_admission_inflight Active long-request leases.",
            "# TYPE ai_gateway_cold_admission_inflight gauge",
            f'ai_gateway_cold_admission_inflight{{{labels},request_class="long"}} {self.inflight}',
            "# HELP ai_gateway_cold_admission_events_total Long-request admission events.",
            "# TYPE ai_gateway_cold_admission_events_total counter",
        ]
        for event in EVENTS:
            lines.append(
                f'ai_gateway_cold_admission_events_total{{{labels},event="{event}"}} {self.events[event]}'
            )
        lines.extend(
            [
                "# HELP ai_gateway_cold_admission_wait_seconds Long-request gateway wait by outcome.",
                "# TYPE ai_gateway_cold_admission_wait_seconds counter",
            ]
        )
        for outcome, value in self.wait_seconds.items():
            lines.append(
                f'ai_gateway_cold_admission_wait_seconds{{{labels},outcome="{outcome}"}} {value}'
            )
        lines.extend(
            [
                "# HELP ai_gateway_cold_admission_idle_observation_info Last scheduler observation status.",
                "# TYPE ai_gateway_cold_admission_idle_observation_info gauge",
                f'ai_gateway_cold_admission_idle_observation_info{{{labels},status="{self.idle_status}"}} 1',
                "# HELP ai_gateway_cold_admission_idle_probe_failures_total Scheduler probe failures.",
                "# TYPE ai_gateway_cold_admission_idle_probe_failures_total counter",
                f"ai_gateway_cold_admission_idle_probe_failures_total{{{labels}}} {self.idle_probe_failures}",
            ]
        )
        return "\n".join(lines) + "\n"


class ColdAdmissionController:
    def __init__(self, backend_name: str, config: ColdAdmissionConfig) -> None:
        self.config = config
        self.metrics = AdmissionMetrics(backend_name, config.target_model)
        self._queue: deque[_Ticket] = deque()
        self._leases = 0
        self._lock = asyncio.Lock()

    async def acquire(
        self,
        client: httpx.AsyncClient,
        request: Any,
        body: dict[str, object],
    ) -> AdmissionLease | None:
        if estimate_input_tokens(body) < self.config.cold_min_input_tokens:
            return None

        accepted_at = time.monotonic()
        ticket = _Ticket(accepted_at)
        async with self._lock:
            if len(self._queue) >= self.config.queue_depth:
                self.metrics.event("rejected_queue_full")
                raise AdmissionQueueFull
            self._queue.append(ticket)
            self.metrics.queue_depth = len(self._queue)
            self.metrics.event("enqueued")

        while True:
            now = time.monotonic()
            wait_seconds = now - accepted_at
            if now >= accepted_at + self.config.queue_timeout_seconds:
                await self._remove(ticket, "expired", wait_seconds)
                raise AdmissionExpired
            if await request.is_disconnected():
                await self._remove(ticket, "cancelled", wait_seconds)
                raise AdmissionCancelled

            async with self._lock:
                is_head = bool(self._queue) and self._queue[0] is ticket
                lease_available = self._leases < self.config.max_active_leases

            if is_head and lease_available:
                observation = await self._scheduler(client)
                if observation is not None:
                    async with self._lock:
                        is_head = bool(self._queue) and self._queue[0] is ticket
                        lease_available = self._leases < self.config.max_active_leases
                        if (
                            is_head
                            and lease_available
                            and observation.running <= self.config.dispatch_running_limit
                            and observation.waiting == 0
                        ):
                            self._queue.popleft()
                            self._leases += 1
                            self.metrics.queue_depth = len(self._queue)
                            self.metrics.inflight = self._leases
                            event = "dispatched_immediate" if wait_seconds < 0.01 else "dispatched_queued"
                            self.metrics.event(event)
                            self.metrics.wait_seconds["dispatched"] += wait_seconds
                            self.metrics.transition(event, wait_seconds)
                            released = False

                            def release() -> None:
                                nonlocal released
                                if not released:
                                    released = True
                                    self._release()

                            return AdmissionLease(wait_seconds, release)

            await asyncio.sleep(self.config.poll_interval_seconds)

    async def _remove(self, ticket: _Ticket, outcome: str, wait_seconds: float) -> None:
        async with self._lock:
            try:
                self._queue.remove(ticket)
            except ValueError:
                return
            self.metrics.queue_depth = len(self._queue)
            self.metrics.event(outcome)
            self.metrics.wait_seconds[outcome] += wait_seconds
            self.metrics.transition(outcome, wait_seconds)

    async def _scheduler(self, client: httpx.AsyncClient) -> SchedulerObservation | None:
        try:
            response = await client.get(self.config.metrics_url, timeout=min(5.0, self.config.poll_interval_seconds * 2))
            response.raise_for_status()
            observation = parse_scheduler_metrics(response.text, self.config.target_model)
        except (httpx.HTTPError, ValueError):
            self.metrics.idle_status = "unknown"
            self.metrics.idle_probe_failures += 1
            self.metrics.event("idle_probe_failed")
            return None
        self.metrics.idle_status = "idle" if observation.running <= self.config.dispatch_running_limit else "busy"
        return observation

    def _release(self) -> None:
        if self._leases <= 0:
            return
        self._leases -= 1
        self.metrics.inflight = self._leases


def estimate_input_tokens(body: dict[str, object]) -> int:
    """Use a conservative character estimate without assuming tokenizer identity."""

    def content_length(value: object) -> int:
        if isinstance(value, str):
            return len(value)
        if isinstance(value, list):
            return sum(content_length(item) for item in value)
        if isinstance(value, dict):
            return sum(content_length(item) for item in value.values())
        return 0

    return content_length(body.get("messages", [])) // 4


def parse_scheduler_metrics(payload: str, target_model: str) -> SchedulerObservation:
    values: dict[str, int] = {}
    for line in payload.splitlines():
        if not line or line.startswith("#"):
            continue
        sample, separator, raw_value = line.rpartition(" ")
        if not separator:
            continue
        match = re.match(r"^(vllm:num_requests_(?:running|waiting))\{([^}]*)\}$", sample)
        if match is None or f'model_name="{_escape(target_model)}"' not in match.group(2):
            continue
        try:
            values[match.group(1)] = int(float(raw_value))
        except ValueError:
            continue
    if set(values) != {"vllm:num_requests_running", "vllm:num_requests_waiting"}:
        raise ValueError("target vLLM running/waiting metrics are missing")
    return SchedulerObservation(values["vllm:num_requests_running"], values["vllm:num_requests_waiting"])


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
