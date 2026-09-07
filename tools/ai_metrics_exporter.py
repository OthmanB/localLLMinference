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

from ai_cost_accounting import CostAccounting


DEFAULT_BACKENDS = {
    "qwen_27b_q4": {
        "url": "http://127.0.0.1:8080/metrics",
        "model": "qwen3.8-27b-q4-gpukv192",
        "gpu": "1",
    },
    "flash_next_262k": {
        "url": "http://127.0.0.1:1901/v1/stats",
        "model": "qwen3.8-flash-next-nvfp4-262k",
        "gpu": "0",
        "format": "freetoken",
    },
    "muse_glimmer_30b_131k": {
        "url": "http://127.0.0.1:8082/metrics",
        "model": "muse-glimmer-30b-kquant17",
        "gpu": "2",
    },
}
METRIC_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{([^}]*)\})?\s+(.+)$")
MODEL_METRIC_DEFINITIONS = {
    "ai_model_decode_tokens_per_second": ("Current decode throughput.", "gauge"),
    "ai_model_prefill_tokens_per_second": ("Current prefill throughput.", "gauge"),
    "ai_model_requests_active": ("Currently active model requests.", "gauge"),
    "ai_model_requests_completed_total": ("Completed model requests since process start.", "counter"),
    "ai_model_prompt_tokens_total": ("Prompt tokens reported by the runtime since process start.", "counter"),
    "ai_model_cached_prompt_tokens_total": ("Cached prompt tokens reported by the runtime since process start.", "counter"),
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
ACCOUNTING_METRIC_DEFINITIONS = {
    "ai_gpu_energy_joules_total": ("Measured GPU energy integrated from timestamped actual power samples.", "counter"),
    "ai_model_active_gpu_energy_joules_total": ("Measured GPU energy during sampled active model requests.", "counter"),
    "ai_model_energy_covered_completion_tokens_total": ("Completion tokens observed during sampled active-GPU energy intervals.", "counter"),
    "ai_host_estimated_energy_joules_total": ("Estimated whole-host energy using configured baseline semantics and actual GPU draw.", "counter"),
    "ai_energy_integration_gap_seconds_total": ("Time excluded from full host-energy integration because samples were missing or too far apart.", "counter"),
    "ai_host_reboot_downtime_seconds_total": ("Unintegrated interval following a confirmed host reboot.", "counter"),
    "ai_host_boot_time_seconds": ("Unix timestamp of the current host boot.", "gauge"),
    "ai_host_electricity_cost_jpy_total": ("Estimated host electricity cost from configured tariff scenarios.", "counter"),
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


def load_backends() -> dict[str, dict[str, str]]:
    raw = os.getenv("AI_METRICS_BACKENDS")
    if not raw:
        return DEFAULT_BACKENDS
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("AI_METRICS_BACKENDS must be a JSON object")
    return value


def escape_label(value: object) -> str:
    return json.dumps(str(value), ensure_ascii=True)


def add_labels(existing: str | None, labels: Mapping[str, object]) -> str:
    values: list[str] = []
    if existing:
        values.append(existing)
    values.extend(f"{name}={escape_label(value)}" for name, value in labels.items())
    return "{" + ",".join(values) + "}"


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
) -> tuple[list[str], dict[str, float]]:
    output: list[str] = []
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
        if generic_name:
            add_model_metric_definitions(output, [generic_name], comments_seen)
        output.append(f"{name}{add_labels(existing, labels)} {value}")
        if generic_name:
            output.append(f"{generic_name}{add_labels(None, labels)} {value}")
        if name == "llamacpp:prompt_tokens_total":
            observation["prompt_tokens"] = numeric_value
        elif name == "llamacpp:prompt_tokens_cached_total":
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


def parse_freetoken_stats(
    stats: Mapping[str, object],
    labels: Mapping[str, str],
    comments_seen: set[str],
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
    def __init__(self, backends: dict[str, dict[str, str]], accounting: CostAccounting) -> None:
        self.backends = backends
        self.accounting = accounting
        self.lock = threading.Lock()
        self.body = ""
        self.scrape()

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
        gpu_models = {str(backend["gpu"]): str(backend["model"]) for backend in self.backends.values()}
        gpu_info: dict[str, Mapping[str, object]] = {}
        nvidia_ok = False
        try:
            gpu_info = {str(sample["gpu"]): sample for sample in read_nvidia_gpus()}
            nvidia_ok = True
        except (OSError, subprocess.CalledProcessError, ValueError):
            pass

        model_observations: dict[str, dict[str, object]] = {}
        for backend in self.backends.values():
            model = str(backend["model"])
            gpu = str(backend["gpu"])
            gpu_uuid = str(gpu_info.get(gpu, {}).get("gpu_uuid", f"unresolved-gpu-{gpu}"))
            label_values = {
                "host_id": self.accounting.host_id,
                "model": model,
                "gpu": gpu,
                "gpu_uuid": gpu_uuid,
            }
            labels = add_labels(None, label_values)
            scrape_error = 0
            try:
                if backend.get("format", "prometheus") == "freetoken":
                    parsed, observation = parse_freetoken_stats(
                        fetch_json(str(backend["url"])), label_values, comments_seen
                    )
                    cached_mode = "unobserved_assumed_uncached"
                else:
                    parsed, observation = parse_llama_metrics(
                        fetch_metrics(str(backend["url"])), label_values, comments_seen
                    )
                    cached_mode = "observed"
                lines.extend(parsed)
                lines.append(f"ai_model_up{labels} 1")
                lines.append(
                    f"ai_model_cached_input_accounting_info{add_labels(None, label_values | {'mode': cached_mode})} 1"
                )
                model_observations[gpu_uuid] = {
                    "model": model,
                    "prompt_tokens": observation["prompt_tokens"],
                    "cached_prompt_tokens": observation["cached_prompt_tokens"],
                    "completion_tokens": observation["completion_tokens"],
                    "active": observation["requests_active"] > 0,
                    "cached_input_mode": cached_mode,
                }
            except (OSError, ValueError, KeyError):
                lines.append(f"ai_model_up{labels} 0")
                scrape_error = 1
            lines.append(f"ai_model_scrape_error{labels} {scrape_error}")

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

        self.accounting.observe(timestamp_seconds, energy_gpus, model_observations)
        add_metric_definitions(lines, ACCOUNTING_METRIC_DEFINITIONS, comments_seen)
        for name, labels, value in self.accounting.configuration_metrics(timestamp_seconds):
            lines.append(f"{name}{add_labels(None, labels)} {value}")
        for name, labels, value in self.accounting.metrics():
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
    state = MetricsState(load_backends(), accounting)

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
