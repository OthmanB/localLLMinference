#!/usr/bin/env python3
"""Expose model, NVIDIA, and host metrics for the local AI server."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.request

from ai_cost_accounting import CostAccounting, gpu_excess_power_watts, metric_key
from ai_cpu_power_profiler import cpu_utilization_percent, interpolate_power, read_proc_stat_cpu


DEFAULT_BACKENDS = {
    "qwen_atx_gpu1": {
        "url": "http://127.0.0.1:8080/metrics",
        "model": "qwen3.8-27b-atx-iq4xs-m-262144-gpu1",
        "gpus": ["1"],
    },
    "qwen_atx_gpu2": {
        "url": "http://127.0.0.1:8081/metrics",
        "model": "qwen3.8-27b-atx-iq4xs-m-262144-gpu2",
        "gpus": ["2"],
    },
    "muse_glimmer_30b_131k": {
        "url": "http://127.0.0.1:8082/metrics",
        "model": "muse-glimmer-30b-kquant17",
        "gpus": ["0"],
    },
}
METRIC_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{([^}]*)\})?\s+(.+)$")
MODEL_METRIC_DEFINITIONS = {
    "ai_model_decode_tokens_per_second": (
        "Decode throughput; the last nonzero sample is retained while the runtime reports zero.",
        "gauge",
    ),
    "ai_model_prefill_tokens_per_second": (
        "Prefill throughput; the last nonzero sample is retained while the runtime reports zero.",
        "gauge",
    ),
    "ai_model_requests_active": ("Currently active model requests.", "gauge"),
    "ai_model_requests_completed_total": ("Completed model requests since process start.", "counter"),
    "ai_model_prompt_tokens_total": ("Prompt tokens processed by the runtime since process start, excluding cached prompt tokens.", "counter"),
    "ai_model_cached_prompt_tokens_total": ("Prompt tokens reused from the runtime prefix cache since process start.", "counter"),
    "ai_model_completion_tokens_total": ("Completion tokens since process start.", "counter"),
    "ai_model_request_p95_seconds": ("Request p95 duration.", "gauge"),
    "ai_model_ttft_seconds": ("Mean time to first token.", "gauge"),
    "ai_model_vram_bytes": ("Model process GPU allocation.", "gauge"),
    "ai_model_context_tokens": ("Configured model context capacity.", "gauge"),
    "ai_model_kv_capacity_tokens": ("Allocated KV capacity.", "gauge"),
    "ai_model_kv_used_tokens": ("Currently used KV tokens.", "gauge"),
}
LLAMA_TO_MODEL_METRICS = {
    "llamacpp:predicted_tokens_seconds": "ai_model_decode_tokens_per_second",
    "llamacpp:prompt_tokens_seconds": "ai_model_prefill_tokens_per_second",
    "llamacpp:requests_processing": "ai_model_requests_active",
    "llamacpp:prompt_tokens_total": "ai_model_prompt_tokens_total",
    "llamacpp:prompt_tokens_cached_total": "ai_model_cached_prompt_tokens_total",
    "llamacpp:tokens_predicted_total": "ai_model_completion_tokens_total",
}
VLLM_TO_MODEL_COUNTERS = {
    "vllm:prompt_tokens_total": "ai_model_prompt_tokens_total",
    "vllm:prompt_tokens_cached_total": "ai_model_cached_prompt_tokens_total",
    "vllm:generation_tokens_total": "ai_model_completion_tokens_total",
}
VLLM_COLD_ADMISSION_METRICS = {
    "ai_gateway_cold_admission_queue_depth",
    "ai_gateway_cold_admission_inflight",
    "ai_gateway_cold_admission_events_total",
    "ai_gateway_cold_admission_wait_seconds",
    "ai_gateway_cold_admission_idle_observation_info",
    "ai_gateway_cold_admission_idle_probe_failures_total",
}
RETAINED_RATE_METRICS = (
    "ai_model_decode_tokens_per_second",
    "ai_model_prefill_tokens_per_second",
)
DEFAULT_CPU_PROFILER_URL = "http://127.0.0.1:9109/metrics"
BUCKET_LABEL = re.compile(r'bucket="([0-9]+)-([0-9]+)"')
CPU_METRIC_DEFINITIONS = {
    "ai_host_cpu_power_watts": (
        "CPU package power used by host accounting; source label reports rapl, interpolated, or linear.",
        "gauge",
    ),
    "ai_host_cpu_utilization_percent": ("CPU utilization from /proc/stat deltas.", "gauge"),
    "ai_host_cpu_power_source_info": ("Selected CPU power source for this scrape.", "gauge"),
    "ai_host_power_attribution_watts": (
        "Current host power split by attribution source (base, cpu, gpu).",
        "gauge",
    ),
}


class CpuProfilerSnapshot:
    def __init__(self) -> None:
        self.power_watts: float | None = None
        self.utilization_percent: float | None = None
        self.rapl_available = 0.0
        self.buckets: dict[str, float] = {}


def read_cpu_profiler(url: str) -> CpuProfilerSnapshot | None:
    """Read the CPU power profiler endpoint; return None when it is unreachable."""
    try:
        text = fetch_metrics(url)
    except OSError:
        return None
    snapshot = CpuProfilerSnapshot()
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        match = METRIC_LINE.match(line)
        if not match:
            continue
        name, _, labels, value = match.groups()
        numeric = _number(value)
        if name == "ai_cpu_power_watts":
            snapshot.power_watts = numeric
        elif name == "ai_cpu_utilization_percent":
            snapshot.utilization_percent = numeric
        elif name == "ai_cpu_profiler_rapl_available":
            snapshot.rapl_available = numeric
        elif name == "ai_cpu_power_bucket_median_watts":
            bucket_match = BUCKET_LABEL.search(labels or "")
            if bucket_match:
                snapshot.buckets[bucket_match.group(0)] = numeric
    return snapshot


def bucket_points(buckets: Mapping[str, float]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for label, median in buckets.items():
        match = BUCKET_LABEL.search(label)
        if not match:
            continue
        lo, hi = float(match.group(1)), float(match.group(2))
        points.append(((lo + hi) / 2.0, median))
    return points


def retain_nonzero_rate(
    rate_state: dict[str, float] | None,
    key: str,
    name: str,
    value: float,
) -> float:
    """Hold the last nonzero throughput sample so idle scrapes form a plateau."""
    if rate_state is None or name not in RETAINED_RATE_METRICS:
        return value
    state_key = metric_key(name, key)
    if value > 0:
        rate_state[state_key] = value
    else:
        value = rate_state.get(state_key, value)
    return value
ACCOUNTING_METRIC_DEFINITIONS = {
    "ai_gpu_energy_joules_total": ("Measured GPU energy integrated from timestamped actual power samples.", "counter"),
    "ai_model_active_gpu_energy_joules_total": ("Measured GPU energy during sampled active model requests.", "counter"),
    "ai_model_energy_covered_completion_tokens_total": ("Completion tokens observed during sampled active-GPU energy intervals.", "counter"),
    "ai_host_estimated_energy_joules_total": ("Estimated whole-host energy using configured baseline semantics and actual GPU draw.", "counter"),
    "ai_host_cpu_energy_joules_total": ("CPU package energy integrated from measured or curve-estimated CPU power.", "counter"),
    "ai_energy_integration_gap_seconds_total": ("Time excluded from full host-energy integration because samples were missing or too far apart.", "counter"),
    "ai_host_reboot_downtime_seconds_total": ("Unintegrated interval following a confirmed host reboot.", "counter"),
    "ai_host_boot_time_seconds": ("Unix timestamp of the current host boot.", "gauge"),
    "ai_host_electricity_cost_jpy_total": ("Estimated host electricity cost from configured tariff scenarios.", "counter"),
    "ai_host_electricity_cost_jpy_today": ("Host electricity cost accumulated since the current calendar day in the configured timezone; resets at each period boundary.", "gauge"),
    "ai_host_electricity_cost_jpy_week_to_date": ("Host electricity cost accumulated since Monday in the configured timezone.", "gauge"),
    "ai_host_electricity_cost_jpy_month_to_date": ("Host electricity cost accumulated since the first day of the month in the configured timezone.", "gauge"),
    "ai_model_active_gpu_electricity_cost_jpy_total": ("Active direct-GPU electricity cost from configured tariff scenarios.", "counter"),
    "ai_model_api_workload_cost_usd_total": ("Equivalent remote API workload cost from observed input, cached-input, and output token deltas.", "counter"),
    "ai_model_api_output_only_cost_usd_total": ("Equivalent remote API output-only cost from observed output token deltas.", "counter"),
    "ai_model_api_workload_cost_jpy_total": ("Equivalent remote API workload cost converted by configured FX scenarios.", "counter"),
    "ai_model_api_output_only_cost_jpy_total": ("Equivalent remote API output-only cost converted by configured FX scenarios.", "counter"),
    "ai_host_power_accounting_configured": ("Whether host baseline accounting is configured.", "gauge"),
    "ai_host_baseline_power_watts": ("Configured host baseline power used by the selected accounting convention.", "gauge"),
    "ai_electricity_tariff_jpy_per_kwh": ("Configured electricity tariff rate.", "gauge"),
    "ai_api_price_usd_per_million_tokens": ("Configured remote API token price.", "gauge"),
    "ai_fx_jpy_per_usd": ("Configured USD to JPY exchange rate.", "gauge"),
}
EXPORTER_METRIC_DEFINITIONS = {
    "ai_serving_profile_info": ("Configured active serving profile.", "gauge"),
    "ai_backend_up": ("Whether the configured backend metrics endpoint is reachable.", "gauge"),
    "ai_replica_up": ("Whether the configured backend replica metrics endpoint is reachable.", "gauge"),
    "ai_backend_cache_observation_info": (
        "Cached-input accounting observation status for a backend.",
        "gauge",
    ),
    "ai_gateway_metrics_up": (
        "Whether the optional local gateway metrics input was reachable.",
        "gauge",
    ),
    "ai_gateway_metrics_observation_info": (
        "Observation status for the optional local gateway metrics input.",
        "gauge",
    ),
    "ai_gateway_pool_route_decisions_total": (
        "Pool route decisions observed from the local gateway.",
        "counter",
    ),
    "ai_gateway_pool_affinity_hits_total": (
        "Pool session-affinity hits observed from the local gateway.",
        "counter",
    ),
    "ai_gateway_pool_affinity_misses_total": (
        "Pool session-affinity misses observed from the local gateway.",
        "counter",
    ),
    "ai_gateway_pool_saturation_rejections_total": (
        "Pool saturation rejections observed from the local gateway.",
        "counter",
    ),
    "ai_gateway_pool_cold_failover_total": (
        "Cold health-failover events observed from the local gateway.",
        "counter",
    ),
}
GATEWAY_METRIC_ALIASES = {
    "ai_gateway_pool_route_decisions_total": "route",
    "ai_gateway_pool_route_total": "route",
    "ai_gateway_pool_routes_total": "route",
    "ai_gateway_route_decisions_total": "route",
    "gateway_pool_route_decisions_total": "route",
    "lan_inference_gateway_pool_route_decisions_total": "route",
    "ai_gateway_pool_affinity_hits_total": "affinity_hits",
    "ai_gateway_affinity_hits_total": "affinity_hits",
    "gateway_pool_affinity_hits_total": "affinity_hits",
    "lan_inference_gateway_pool_affinity_hits_total": "affinity_hits",
    "ai_gateway_pool_affinity_misses_total": "affinity_misses",
    "ai_gateway_affinity_misses_total": "affinity_misses",
    "gateway_pool_affinity_misses_total": "affinity_misses",
    "lan_inference_gateway_pool_affinity_misses_total": "affinity_misses",
    "ai_gateway_pool_saturation_rejections_total": "saturation",
    "ai_gateway_saturation_rejections_total": "saturation",
    "gateway_pool_saturation_rejections_total": "saturation",
    "lan_inference_gateway_pool_saturation_rejections_total": "saturation",
    "ai_gateway_pool_cold_failover_total": "cold_failover",
    "ai_gateway_cold_failover_total": "cold_failover",
    "gateway_pool_cold_failover_total": "cold_failover",
    "lan_inference_gateway_pool_cold_failover_total": "cold_failover",
}
GATEWAY_METRIC_NAMES = {
    "route": "ai_gateway_pool_route_decisions_total",
    "affinity_hits": "ai_gateway_pool_affinity_hits_total",
    "affinity_misses": "ai_gateway_pool_affinity_misses_total",
    "saturation": "ai_gateway_pool_saturation_rejections_total",
    "cold_failover": "ai_gateway_pool_cold_failover_total",
}
DEFAULT_GATEWAY_LABELS = {
    "pool": "unknown",
    "replica": "unknown",
    "reason": "unknown",
}
GATEWAY_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def load_backends() -> dict[str, dict[str, object]]:
    raw = os.getenv("AI_METRICS_BACKENDS")
    if not raw:
        validate_backend_gpu_ownership(DEFAULT_BACKENDS)
        return DEFAULT_BACKENDS
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("AI_METRICS_BACKENDS must be a JSON object")
    if not all(isinstance(name, str) and isinstance(backend, dict) for name, backend in value.items()):
        raise ValueError("AI_METRICS_BACKENDS values must be JSON objects")
    validate_backend_gpu_ownership(value)
    return value


def backend_gpus(backend: Mapping[str, object]) -> tuple[str, ...]:
    """Return physical GPU indices assigned to a backend, in primary-first order."""
    configured = backend.get("gpus")
    if configured is None:
        configured = [backend.get("gpu")]
    if not isinstance(configured, list) or not configured or not all(
        isinstance(gpu, str) and gpu for gpu in configured
    ):
        raise ValueError("each metrics backend requires a non-empty gpus list")
    gpus = tuple(str(gpu) for gpu in configured)
    if len(gpus) != len(set(gpus)):
        raise ValueError("each metrics backend must list each GPU only once")
    return gpus


def validate_backend_gpu_ownership(backends: Mapping[str, Mapping[str, object]]) -> None:
    """Reject ambiguous physical-GPU ownership across configured backends."""
    owners: dict[str, str] = {}
    for backend_name, backend in backends.items():
        for gpu in backend_gpus(backend):
            previous_owner = owners.get(gpu)
            if previous_owner is not None:
                raise ValueError(
                    f"GPU {gpu!r} is assigned to multiple metrics backends: "
                    f"{previous_owner!r} and {backend_name!r}"
                )
            owners[gpu] = backend_name


def cached_input_mode(backend: Mapping[str, object], backend_name: str = "") -> str:
    """Identify cache accounting without assuming an unobserved cache is empty."""
    configured = backend.get("cached_input_mode")
    if configured in {"observed", "unobserved", "unobserved_assumed_uncached"}:
        return str(configured)

    runtime = " ".join(
        str(backend.get(field, ""))
        for field in ("runtime", "engine", "format", "name", "model")
    ).lower()
    runtime = f"{runtime} {backend_name.lower()}"
    if "llamampere" in runtime or "llama_ampere" in runtime or "atx" in runtime:
        return "unobserved"
    if str(backend.get("format", "prometheus")).lower() == "freetoken":
        return "unobserved_assumed_uncached"
    return "observed"


def escape_label(value: object) -> str:
    return json.dumps(str(value), ensure_ascii=True)


def add_labels(existing: str | None, labels: Mapping[str, object]) -> str:
    values: list[str] = []
    if existing:
        values.append(existing)
    values.extend(f"{name}={escape_label(value)}" for name, value in labels.items())
    return "{" + ",".join(values) + "}"


def parse_prometheus_labels(encoded: str | None) -> dict[str, str]:
    if not encoded:
        return {}
    labels: dict[str, str] = {}
    for match in re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"])*)"', encoded):
        try:
            labels[match.group(1)] = str(json.loads(f'"{match.group(2)}"'))
        except ValueError:
            continue
    return labels


def _bounded_gateway_label(value: object) -> str:
    value = str(value)
    return value if GATEWAY_LABEL_PATTERN.fullmatch(value) else "unknown"


def parse_gateway_metrics(text: str) -> dict[str, dict[tuple[str, ...], float]]:
    """Keep only bounded gateway routing counters; never export request identity labels."""
    samples: dict[str, dict[tuple[str, ...], float]] = {
        kind: {} for kind in GATEWAY_METRIC_NAMES
    }
    for line in text.splitlines():
        match = METRIC_LINE.match(line)
        if not match:
            continue
        name, _, encoded_labels, value = match.groups()
        kind = GATEWAY_METRIC_ALIASES.get(name)
        if kind is None:
            continue
        source_labels = parse_prometheus_labels(encoded_labels)
        pool = _bounded_gateway_label(
            source_labels.get("pool", source_labels.get("pool_name", DEFAULT_GATEWAY_LABELS["pool"]))
        )
        replica = _bounded_gateway_label(
            source_labels.get(
                "replica",
                source_labels.get(
                    "selected_replica",
                    source_labels.get(
                        "backend", source_labels.get("backend_name", DEFAULT_GATEWAY_LABELS["replica"])
                    ),
                ),
            )
        )
        if kind == "route":
            reason = _bounded_gateway_label(
                source_labels.get("reason", source_labels.get("route_reason", DEFAULT_GATEWAY_LABELS["reason"]))
            )
        else:
            reason = "unknown"
        key = (pool, replica, reason)
        values = samples[kind]
        values[key] = values.get(key, 0.0) + _number(value)
    return samples


def gateway_metrics_lines(
    comments_seen: set[str],
    url: str | None,
    host_id: str | None = None,
) -> list[str]:
    output: list[str] = []
    add_metric_definitions(output, EXPORTER_METRIC_DEFINITIONS, comments_seen)
    gateway_up = 0.0
    observation_status = "unknown"
    samples: dict[str, dict[tuple[str, ...], float]] = {
        kind: {} for kind in GATEWAY_METRIC_NAMES
    }
    gateway_text = ""
    if url:
        try:
            gateway_text = fetch_metrics(url)
            samples = parse_gateway_metrics(gateway_text)
            gateway_up = 1.0
            observation_status = "observed"
        except (OSError, ValueError, UnicodeError):
            pass

    output.append(f"ai_gateway_metrics_up {gateway_up}")
    output.append(
        f"ai_gateway_metrics_observation_info{add_labels(None, {'status': observation_status})} 1"
    )
    for kind, metric_name in GATEWAY_METRIC_NAMES.items():
        values = samples[kind]
        if not values:
            values = {("unknown", "unknown", "unknown"): 0.0}
        for key, value in values.items():
            if kind == "route":
                labels = {
                    "pool": key[0],
                    "replica": key[1],
                    "reason": key[2],
                }
            else:
                labels = {"pool": key[0], "replica": key[1]}
            output.append(f"{metric_name}{add_labels(None, labels)} {value}")
    if host_id:
        for line in gateway_text.splitlines():
            match = METRIC_LINE.match(line)
            if not match:
                continue
            name, _, encoded_labels, value = match.groups()
            if name not in VLLM_COLD_ADMISSION_METRICS:
                continue
            source = parse_prometheus_labels(encoded_labels)
            labels = {"host_id": host_id}
            for key in ("backend", "model", "event", "outcome", "status", "request_class"):
                if key in source:
                    labels[key] = _bounded_gateway_label(source[key])
            output.append(f"{name}{add_labels(None, labels)} {value}")
    return output


def add_model_metric_definitions(
    output: list[str], names: list[str], comments_seen: set[str]
) -> None:
    for name in names:
        help_text, metric_type = MODEL_METRIC_DEFINITIONS[name]
        help_line = f"# HELP {name} {help_text}"
        type_line = f"# TYPE {name} {metric_type}"
        if help_line not in comments_seen:
            comments_seen.add(help_line)
            output.append(help_line)
        if type_line not in comments_seen:
            comments_seen.add(type_line)
            output.append(type_line)


def add_metric_definitions(
    output: list[str],
    definitions: Mapping[str, tuple[str, str]],
    comments_seen: set[str],
) -> None:
    for name, (help_text, metric_type) in definitions.items():
        help_line = f"# HELP {name} {help_text}"
        type_line = f"# TYPE {name} {metric_type}"
        if help_line not in comments_seen:
            comments_seen.add(help_line)
            output.append(help_line)
        if type_line not in comments_seen:
            comments_seen.add(type_line)
            output.append(type_line)


def fetch_metrics(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read().decode("utf-8")


def fetch_json(url: str) -> dict:
    value = json.loads(fetch_metrics(url))
    if not isinstance(value, dict):
        raise ValueError("model stats response must be a JSON object")
    return value


def parse_llama_metrics(
    text: str,
    labels: Mapping[str, str],
    comments_seen: set[str],
    rate_state: dict[str, float] | None = None,
    cached_input_mode: str = "observed",
) -> tuple[list[str], dict[str, float]]:
    output: list[str] = []
    rate_key = metric_key(labels.get("model", ""), labels.get("gpu_uuid", ""))
    observation = {
        "prompt_tokens": 0.0,
        "cached_prompt_tokens": 0.0,
        "completion_tokens": 0.0,
        "requests_active": 0.0,
    }
    for line in text.splitlines():
        if line.startswith("#"):
            if line not in comments_seen:
                comments_seen.add(line)
                output.append(line)
            continue
        match = METRIC_LINE.match(line)
        if not match:
            continue
        name, _, existing, value = match.groups()
        numeric_value = _number(value)
        generic_name = LLAMA_TO_MODEL_METRICS.get(name)
        trusted_cached_metric = not (
            name == "llamacpp:prompt_tokens_cached_total" and cached_input_mode == "unobserved"
        )
        if generic_name and trusted_cached_metric:
            add_model_metric_definitions(output, [generic_name], comments_seen)
        output.append(f"{name}{add_labels(existing, labels)} {value}")
        if generic_name and trusted_cached_metric:
            retained = retain_nonzero_rate(rate_state, rate_key, generic_name, numeric_value)
            output.append(f"{generic_name}{add_labels(None, labels)} {retained}")
        if name == "llamacpp:prompt_tokens_total":
            observation["prompt_tokens"] = numeric_value
        elif name == "llamacpp:prompt_tokens_cached_total" and cached_input_mode != "unobserved":
            observation["cached_prompt_tokens"] = numeric_value
        elif name == "llamacpp:tokens_predicted_total":
            observation["completion_tokens"] = numeric_value
        elif name == "llamacpp:requests_processing":
            observation["requests_active"] = numeric_value
    return output, observation


def _number(value: object) -> float:
    if not isinstance(value, (str, int, float)):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_vllm_metrics(
    text: str,
    labels: Mapping[str, str],
    comments_seen: set[str],
    rate_state: dict[str, float] | None = None,
    counter_state: dict[str, tuple[float, float]] | None = None,
    timestamp_seconds: float | None = None,
) -> tuple[list[str], dict[str, float]]:
    """Normalize vLLM counters without exporting its high-cardinality histograms."""
    observed: dict[str, float] = {}
    for line in text.splitlines():
        match = METRIC_LINE.match(line)
        if not match:
            continue
        name, _, encoded_labels, value = match.groups()
        if parse_prometheus_labels(encoded_labels).get("model_name") != labels["model"]:
            continue
        numeric = _number(value)
        if name in VLLM_TO_MODEL_COUNTERS:
            observed[VLLM_TO_MODEL_COUNTERS[name]] = numeric
        elif name == "vllm:num_requests_running":
            observed["ai_model_requests_active"] = numeric
        elif name == "vllm:request_success_total":
            observed["ai_model_requests_completed_total"] = observed.get("ai_model_requests_completed_total", 0.0) + numeric
        elif name == "vllm:time_to_first_token_seconds_count":
            observed["_ttft_count"] = numeric
        elif name == "vllm:time_to_first_token_seconds_sum":
            observed["_ttft_sum"] = numeric
    output: list[str] = []
    add_model_metric_definitions(output, list(MODEL_METRIC_DEFINITIONS), comments_seen)
    values = {name: observed.get(name, 0.0) for name in MODEL_METRIC_DEFINITIONS}
    count = observed.get("_ttft_count", 0.0)
    values["ai_model_ttft_seconds"] = observed.get("_ttft_sum", 0.0) / count if count else 0.0
    now = time.time() if timestamp_seconds is None else timestamp_seconds
    rate_key = metric_key(labels.get("model", ""), labels.get("gpu_uuid", ""))
    for counter_name, rate_name in (
        ("ai_model_prompt_tokens_total", "ai_model_prefill_tokens_per_second"),
        ("ai_model_completion_tokens_total", "ai_model_decode_tokens_per_second"),
    ):
        state_key = f"{rate_key}:{counter_name}"
        previous = counter_state.get(state_key) if counter_state is not None else None
        if counter_state is not None:
            counter_state[state_key] = (values[counter_name], now)
        rate = (
            0.0
            if previous is None or now <= previous[1]
            else max(values[counter_name] - previous[0], 0.0) / (now - previous[1])
        )
        values[rate_name] = retain_nonzero_rate(rate_state, rate_key, rate_name, rate)
    encoded_labels = add_labels(None, labels)
    output.extend(f"{name}{encoded_labels} {value}" for name, value in values.items())
    return output, {
        "prompt_tokens": values["ai_model_prompt_tokens_total"],
        "cached_prompt_tokens": values["ai_model_cached_prompt_tokens_total"],
        "completion_tokens": values["ai_model_completion_tokens_total"],
        "requests_active": values["ai_model_requests_active"],
    }


def parse_freetoken_stats(
    stats: Mapping[str, object],
    labels: Mapping[str, str],
    comments_seen: set[str],
    rate_state: dict[str, float] | None = None,
) -> tuple[list[str], dict[str, float]]:
    """Convert FreeToken's JSON stats endpoint into Prometheus samples."""
    output: list[str] = []
    add_model_metric_definitions(output, list(MODEL_METRIC_DEFINITIONS), comments_seen)

    throughput = stats.get("throughput")
    requests = stats.get("requests")
    card = stats.get("model")
    kv = stats.get("kv")
    throughput = throughput if isinstance(throughput, Mapping) else {}
    requests = requests if isinstance(requests, Mapping) else {}
    card = card if isinstance(card, Mapping) else {}
    kv = kv if isinstance(kv, Mapping) else {}
    context_tokens = _number(card.get("ctx"))
    total_pages = _number(kv.get("total_pages"))
    used_pages = _number(kv.get("used_pages"))
    # FreeToken exposes page_size in internal units; model.ctx is the token capacity.
    used_tokens = context_tokens * used_pages / total_pages if total_pages else 0.0
    values = {
        "ai_model_decode_tokens_per_second": _number(throughput.get("decode_tps")),
        "ai_model_prefill_tokens_per_second": _number(throughput.get("prefill_tps")),
        "ai_model_requests_active": _number(requests.get("active")),
        "ai_model_requests_completed_total": _number(requests.get("completed")),
        "ai_model_prompt_tokens_total": _number(requests.get("prompt_tokens_total")),
        "ai_model_cached_prompt_tokens_total": 0.0,
        "ai_model_completion_tokens_total": _number(requests.get("completion_tokens_total")),
        "ai_model_request_p95_seconds": _number(requests.get("p95_ms")) / 1000,
        "ai_model_ttft_seconds": _number(requests.get("ttft_mean_ms")) / 1000,
        "ai_model_vram_bytes": _number(stats.get("vram_bytes")),
        "ai_model_context_tokens": context_tokens,
        "ai_model_kv_capacity_tokens": context_tokens,
        "ai_model_kv_used_tokens": used_tokens,
    }
    rate_key = metric_key(labels.get("model", ""), labels.get("gpu_uuid", ""))
    for rate_name in RETAINED_RATE_METRICS:
        values[rate_name] = retain_nonzero_rate(rate_state, rate_key, rate_name, values[rate_name])
    encoded_labels = add_labels(None, labels)
    output.extend(f"{name}{encoded_labels} {value}" for name, value in values.items())
    return output, {
        "prompt_tokens": values["ai_model_prompt_tokens_total"],
        "cached_prompt_tokens": 0.0,
        "completion_tokens": values["ai_model_completion_tokens_total"],
        "requests_active": values["ai_model_requests_active"],
    }


def read_nvidia_gpus() -> list[dict[str, object]]:
    query = "index,uuid,name,power.limit,power.draw,utilization.gpu,temperature.gpu,memory.used,memory.total"
    command = ["/usr/bin/nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    rows = csv.reader(subprocess.check_output(command, text=True).splitlines())
    gpus: list[dict[str, object]] = []
    for row in rows:
        if len(row) != 9:
            continue
        gpu, uuid, name, limit, draw, utilization, temperature, memory_used, memory_total = (
            field.strip() for field in row
        )
        gpus.append(
            {
                "gpu": gpu,
                "gpu_uuid": uuid,
                "name": name,
                "power_limit_watts": float(limit),
                "power_watts": float(draw),
                "utilization_percent": float(utilization),
                "temperature_celsius": float(temperature),
                "memory_used_bytes": float(memory_used) * 1024 * 1024,
                "memory_total_bytes": float(memory_total) * 1024 * 1024,
            }
        )
    return gpus


def nvidia_metrics(
    comments_seen: set[str],
    gpus: list[Mapping[str, object]],
    gpu_models: Mapping[str, str],
    host_id: str,
) -> list[str]:
    output: list[str] = []
    definitions = {
        "ai_gpu_power_limit_watts": "Configured GPU power limit.",
        "ai_gpu_power_draw_watts": "Current GPU power draw.",
        "ai_gpu_utilization_percent": "Current GPU utilization.",
        "ai_gpu_temperature_celsius": "Current GPU temperature.",
        "ai_gpu_memory_used_bytes": "GPU memory used.",
        "ai_gpu_memory_total_bytes": "Total GPU memory.",
    }
    for name, help_text in definitions.items():
        help_line = f"# HELP {name} {help_text}"
        type_line = f"# TYPE {name} gauge"
        if help_line not in comments_seen:
            comments_seen.add(help_line)
            output.append(help_line)
        if type_line not in comments_seen:
            comments_seen.add(type_line)
            output.append(type_line)
    for sample in gpus:
        gpu = str(sample["gpu"])
        labels = {
            "host_id": host_id,
            "gpu": gpu,
            "gpu_uuid": str(sample["gpu_uuid"]),
            "name": str(sample["name"]),
        }
        if gpu in gpu_models:
            labels["model"] = gpu_models[gpu]
        values = {
            "ai_gpu_power_limit_watts": sample["power_limit_watts"],
            "ai_gpu_power_draw_watts": sample["power_watts"],
            "ai_gpu_utilization_percent": sample["utilization_percent"],
            "ai_gpu_temperature_celsius": sample["temperature_celsius"],
            "ai_gpu_memory_used_bytes": sample["memory_used_bytes"],
            "ai_gpu_memory_total_bytes": sample["memory_total_bytes"],
        }
        output.extend(
            f"{metric}{add_labels(None, labels)} {value}"
            for metric, value in values.items()
        )
    return output


def host_memory_metrics(comments_seen: set[str], host_id: str) -> list[str]:
    values: dict[str, int] = {}
    with open("/proc/meminfo", encoding="ascii") as meminfo:
        for line in meminfo:
            name, value, *_ = line.split()
            if name in {"MemTotal:", "MemAvailable:"}:
                values[name] = int(value) * 1024

    total = values["MemTotal:"]
    available = values.get("MemAvailable:", total)
    definitions = {
        "ai_host_memory_total_bytes": "Total host memory.",
        "ai_host_memory_available_bytes": "Available host memory.",
        "ai_host_memory_used_bytes": "Used host memory, calculated as total minus available.",
    }
    output: list[str] = []
    for name, help_text in definitions.items():
        help_line = f"# HELP {name} {help_text}"
        type_line = f"# TYPE {name} gauge"
        if help_line not in comments_seen:
            comments_seen.add(help_line)
            output.append(help_line)
        if type_line not in comments_seen:
            comments_seen.add(type_line)
            output.append(type_line)
    labels = add_labels(None, {"host_id": host_id})
    output.extend(
        [
            f"ai_host_memory_total_bytes{labels} {total}",
            f"ai_host_memory_available_bytes{labels} {available}",
            f"ai_host_memory_used_bytes{labels} {total - available}",
        ]
    )
    return output


def load_json_config(path: str | None) -> dict[str, object]:
    if not path:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


class MetricsState:
    def __init__(
        self,
        backends: dict[str, dict[str, object]],
        accounting: CostAccounting,
        gateway_metrics_url: str | None = None,
    ) -> None:
        validate_backend_gpu_ownership(backends)
        self.backends = backends
        self.accounting = accounting
        self.gateway_metrics_url = (
            gateway_metrics_url
            if gateway_metrics_url is not None
            else os.getenv("AI_GATEWAY_METRICS_URL")
        )
        self.lock = threading.Lock()
        self.rate_state: dict[str, float] = {}
        self.vllm_counter_state: dict[str, tuple[float, float]] = {}
        self.previous_proc: tuple[float, float] | None = None
        self.cpu_power: float | None = None
        self.cpu_utilization: float | None = None
        self.cpu_source = "unavailable"
        self.body = ""
        self.scrape()

    def resolve_cpu_power(self) -> None:
        """Select CPU package power: rapl -> bucket-curve interpolation -> linear -> none."""
        config = self.accounting.config.get("cpu_power")
        config = config if isinstance(config, Mapping) else {}
        url = str(config.get("profiler_url", DEFAULT_CPU_PROFILER_URL))
        profiler = read_cpu_profiler(url)
        utilization: float | None = profiler.utilization_percent if profiler is not None else None
        proc = read_proc_stat_cpu()
        if utilization is None and proc is not None:
            utilization = cpu_utilization_percent(self.previous_proc, proc)
        self.previous_proc = proc
        self.cpu_utilization = utilization

        power: float | None = None
        source = "unavailable"
        if profiler is not None and profiler.rapl_available > 0 and profiler.power_watts is not None:
            power, source = profiler.power_watts, "rapl"
        elif profiler is not None and profiler.buckets and utilization is not None:
            power = interpolate_power(bucket_points(profiler.buckets), utilization)
            if power is not None:
                source = "interpolated"
        if power is None:
            linear = config.get("linear")
            if isinstance(linear, Mapping) and utilization is not None:
                saturate = _number(linear.get("saturate_utilization_percent"))
                max_watts = _number(linear.get("max_watts"))
                if saturate > 0 and max_watts > 0:
                    power = min(max_watts, max_watts * utilization / saturate)
                    source = "linear"
        self.cpu_power = power
        self.cpu_source = source if power is not None else "unavailable"

    def scrape(self) -> None:
        timestamp_seconds = time.time()
        comments_seen: set[str] = set()
        lines: list[str] = [
            "# HELP ai_model_up Whether the configured model metrics endpoint is reachable.",
            "# TYPE ai_model_up gauge",
            "# HELP ai_model_scrape_error Whether scraping the configured model endpoint failed.",
            "# TYPE ai_model_scrape_error gauge",
            "# HELP ai_exporter_nvidia_smi_up Whether nvidia-smi scraping is working.",
            "# TYPE ai_exporter_nvidia_smi_up gauge",
            "# HELP ai_model_cached_input_accounting_info Cached-input accounting mode for the runtime.",
            "# TYPE ai_model_cached_input_accounting_info gauge",
        ]
        add_metric_definitions(lines, EXPORTER_METRIC_DEFINITIONS, comments_seen)
        profile = os.getenv("AI_SERVING_PROFILE", "unknown").strip() or "unknown"
        lines.append(f"ai_serving_profile_info{add_labels(None, {'profile': profile})} 1")
        gpu_models: dict[str, str] = {}
        configured_models: set[str] = set()
        for backend in self.backends.values():
            model = str(backend["model"])
            configured_models.add(model)
            for gpu in backend_gpus(backend):
                gpu_models[gpu] = model
        gpu_info: dict[str, Mapping[str, object]] = {}
        nvidia_ok = False
        try:
            gpu_info = {str(sample["gpu"]): sample for sample in read_nvidia_gpus()}
            nvidia_ok = True
        except (OSError, subprocess.CalledProcessError, ValueError):
            pass

        model_observations: dict[str, dict[str, object]] = {}
        for backend_name, backend in self.backends.items():
            model = str(backend["model"])
            gpus = backend_gpus(backend)
            primary_gpu = gpus[0]
            gpu_uuids = {
                gpu: str(gpu_info.get(gpu, {}).get("gpu_uuid", f"unresolved-gpu-{gpu}"))
                for gpu in gpus
            }
            scrape_error = 0
            cached_mode = cached_input_mode(backend, backend_name)
            backend_up = 0
            try:
                primary_labels = {
                    "host_id": self.accounting.host_id,
                    "model": model,
                    "gpu": primary_gpu,
                    "gpu_uuid": gpu_uuids[primary_gpu],
                }
                if backend.get("format", "prometheus") == "freetoken":
                    parsed, observation = parse_freetoken_stats(
                        fetch_json(str(backend["url"])), primary_labels, comments_seen, self.rate_state
                    )
                elif str(backend.get("runtime", "")).lower() == "vllm":
                    parsed, observation = parse_vllm_metrics(
                        fetch_metrics(str(backend["url"])),
                        primary_labels,
                        comments_seen,
                        self.rate_state,
                        self.vllm_counter_state,
                        timestamp_seconds,
                    )
                else:
                    parsed, observation = parse_llama_metrics(
                        fetch_metrics(str(backend["url"])),
                        primary_labels,
                        comments_seen,
                        self.rate_state,
                        cached_mode,
                    )
                lines.extend(parsed)
                backend_up = 1
                for gpu in gpus:
                    label_values = {
                        "host_id": self.accounting.host_id,
                        "model": model,
                        "gpu": gpu,
                        "gpu_uuid": gpu_uuids[gpu],
                    }
                    labels = add_labels(None, label_values)
                    lines.append(f"ai_model_up{labels} 1")
                    lines.append(
                        f"ai_model_cached_input_accounting_info{add_labels(None, label_values | {'mode': cached_mode})} 1"
                    )
                    model_observations[gpu_uuids[gpu]] = {
                        "model": model,
                        "prompt_tokens": observation["prompt_tokens"] if gpu == primary_gpu else 0.0,
                        "cached_prompt_tokens": observation["cached_prompt_tokens"] if gpu == primary_gpu else 0.0,
                        "completion_tokens": observation["completion_tokens"] if gpu == primary_gpu else 0.0,
                        "active": observation["requests_active"] > 0,
                        "cached_input_mode": cached_mode,
                        "account_tokens": gpu == primary_gpu and cached_mode != "unobserved",
                    }
            except (OSError, ValueError, KeyError):
                scrape_error = 1
            backend_labels = {
                "backend": backend_name,
                "model": model,
            }
            lines.append(f"ai_backend_up{add_labels(None, backend_labels)} {backend_up}")
            lines.append(
                "ai_backend_cache_observation_info"
                f"{add_labels(None, backend_labels | {'mode': cached_mode})} 1"
            )
            for gpu in gpus:
                label_values = {
                    "host_id": self.accounting.host_id,
                    "model": model,
                    "gpu": gpu,
                    "gpu_uuid": gpu_uuids[gpu],
                }
                labels = add_labels(None, label_values)
                if scrape_error:
                    lines.append(f"ai_model_up{labels} 0")
                lines.append(f"ai_model_scrape_error{labels} {scrape_error}")
                replica_labels = label_values | {
                    "backend": backend_name,
                    "replica": backend_name,
                }
                lines.append(f"ai_replica_up{add_labels(None, replica_labels)} {backend_up}")

        energy_gpus: dict[str, dict[str, object]] = {}
        if nvidia_ok:
            lines.extend(nvidia_metrics(comments_seen, list(gpu_info.values()), gpu_models, self.accounting.host_id))
            lines.append("ai_exporter_nvidia_smi_up 1")
            for gpu, sample in gpu_info.items():
                uuid = str(sample["gpu_uuid"])
                model = gpu_models.get(gpu)
                observation = model_observations.get(uuid, {})
                energy_gpus[uuid] = {
                    "power_watts": sample["power_watts"],
                    "model": model,
                    "active": bool(observation.get("active")),
                }
        else:
            lines.append("ai_exporter_nvidia_smi_up 0")

        lines.extend(gateway_metrics_lines(comments_seen, self.gateway_metrics_url, self.accounting.host_id))

        configured_model_gpus: dict[str, set[str]] = {}
        for uuid, sample in energy_gpus.items():
            model = sample.get("model")
            if isinstance(model, str):
                configured_model_gpus.setdefault(model, set()).add(uuid)

        self.resolve_cpu_power()
        add_metric_definitions(lines, CPU_METRIC_DEFINITIONS, comments_seen)
        host_labels = {"host_id": self.accounting.host_id}
        if self.cpu_utilization is not None:
            lines.append(
                f"ai_host_cpu_utilization_percent{add_labels(None, host_labels)} {self.cpu_utilization}"
            )
        lines.append(
            f"ai_host_cpu_power_source_info{add_labels(None, host_labels | {'source': self.cpu_source})} 1"
        )
        if self.cpu_power is not None:
            lines.append(
                f"ai_host_cpu_power_watts{add_labels(None, host_labels | {'source': self.cpu_source})} {self.cpu_power}"
            )
        if self.accounting.baseline.get("mode") == "wall_idle_plus_cpu_w":
            baseline = self.accounting.baseline
            gpu_excess = (
                gpu_excess_power_watts(
                    baseline,
                    {
                        uuid: float(power)
                        for uuid, sample in energy_gpus.items()
                        if isinstance((power := sample.get("power_watts")), (int, float))
                    },
                )
                if nvidia_ok
                else 0.0
            )
            for source_name, value in (
                ("base", float(baseline.get("base_idle_total_w", 0.0))),
                ("cpu", float(self.cpu_power) if self.cpu_power is not None else 0.0),
                ("gpu", gpu_excess),
            ):
                lines.append(
                    f"ai_host_power_attribution_watts{add_labels(None, host_labels | {'source': source_name})} {value}"
                )

        self.accounting.observe(timestamp_seconds, energy_gpus, model_observations, self.cpu_power)
        add_metric_definitions(lines, ACCOUNTING_METRIC_DEFINITIONS, comments_seen)
        for name, labels, value in self.accounting.configuration_metrics(timestamp_seconds):
            lines.append(f"{name}{add_labels(None, labels)} {value}")
        for name, labels, value in self.accounting.metrics(configured_models, configured_model_gpus):
            lines.append(f"{name}{add_labels(None, labels)} {value}")
        try:
            lines.extend(host_memory_metrics(comments_seen, self.accounting.host_id))
        except (OSError, KeyError, ValueError):
            pass
        body = "\n".join(lines) + "\n"
        with self.lock:
            self.body = body

    def get(self) -> str:
        with self.lock:
            return self.body


class Handler(BaseHTTPRequestHandler):
    state: MetricsState

    def do_GET(self) -> None:  # noqa: N802
        if self.path not in {"/metrics", "/healthz"}:
            self.send_error(404)
            return
        body = b"ok\n" if self.path == "/healthz" else self.state.get().encode("utf-8")
        content_type = "text/plain; version=0.0.4" if self.path == "/metrics" else "text/plain"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *_args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9108)
    parser.add_argument("--cost-config")
    parser.add_argument("--pricing-config")
    parser.add_argument("--state-path")
    parser.add_argument("--gateway-metrics-url")
    parser.add_argument("--host-id", default=os.getenv("AI_METRICS_HOST_ID", "unconfigured"))
    args = parser.parse_args()
    if bool(args.cost_config) != bool(args.pricing_config):
        parser.error("--cost-config and --pricing-config must be supplied together")
    accounting = (
        CostAccounting(
            load_json_config(args.cost_config),
            load_json_config(args.pricing_config),
            Path(args.state_path) if args.state_path else None,
        )
        if args.cost_config
        else CostAccounting.disabled()
    )
    accounting.host_id = args.host_id
    state = MetricsState(load_backends(), accounting, args.gateway_metrics_url)

    def refresh() -> None:
        while True:
            time.sleep(5)
            state.scrape()

    threading.Thread(target=refresh, daemon=True).start()
    Handler.state = state
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
