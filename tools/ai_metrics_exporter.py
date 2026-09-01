#!/usr/bin/env python3
"""Expose llama.cpp and NVIDIA host metrics for the local AI server."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import re
import subprocess
import threading
import time
import urllib.request


DEFAULT_BACKENDS = {
    "q4": {
        "url": "http://127.0.0.1:8080/metrics",
        "model": "qwen3.8-27b-q4-gpukv192",
        "gpu": "0",
    }
}
METRIC_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{([^}]*)\})?\s+(.+)$")


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


def fetch_metrics(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read().decode("utf-8")


def parse_llama_metrics(
    text: str,
    model: str,
    gpu: str,
    comments_seen: set[str],
) -> list[str]:
    output: list[str] = []
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
        output.append(f"{name}{add_labels(existing, {'model': model, 'gpu': gpu})} {value}")
    return output


def nvidia_metrics(comments_seen: set[str], gpu_models: Mapping[str, str]) -> list[str]:
    query = "index,name,power.limit,power.draw,utilization.gpu,temperature.gpu,memory.used,memory.total"
    command = ["/usr/bin/nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    rows = csv.reader(subprocess.check_output(command, text=True).splitlines())
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
    for row in rows:
        if len(row) != 8:
            continue
        gpu, name, limit, draw, utilization, temperature, memory_used, memory_total = (
            field.strip() for field in row
        )
        labels = {"gpu": gpu, "name": name}
        if gpu in gpu_models:
            labels["model"] = gpu_models[gpu]
        values = {
            "ai_gpu_power_limit_watts": float(limit),
            "ai_gpu_power_draw_watts": float(draw),
            "ai_gpu_utilization_percent": float(utilization),
            "ai_gpu_temperature_celsius": float(temperature),
            "ai_gpu_memory_used_bytes": float(memory_used) * 1024 * 1024,
            "ai_gpu_memory_total_bytes": float(memory_total) * 1024 * 1024,
        }
        output.extend(
            f"{metric}{add_labels(None, labels)} {value}"
            for metric, value in values.items()
        )
    return output


def host_memory_metrics(comments_seen: set[str]) -> list[str]:
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
    output.extend(
        [
            f"ai_host_memory_total_bytes {total}",
            f"ai_host_memory_available_bytes {available}",
            f"ai_host_memory_used_bytes {total - available}",
        ]
    )
    return output


class MetricsState:
    def __init__(self, backends: dict[str, dict[str, str]]) -> None:
        self.backends = backends
        self.lock = threading.Lock()
        self.body = ""
        self.scrape()

    def scrape(self) -> None:
        comments_seen: set[str] = set()
        lines: list[str] = [
            "# HELP ai_model_up Whether the configured model metrics endpoint is reachable.",
            "# TYPE ai_model_up gauge",
            "# HELP ai_model_scrape_error Whether scraping the configured model endpoint failed.",
            "# TYPE ai_model_scrape_error gauge",
            "# HELP ai_exporter_nvidia_smi_up Whether nvidia-smi scraping is working.",
            "# TYPE ai_exporter_nvidia_smi_up gauge",
        ]
        gpu_models = {backend["gpu"]: backend["model"] for backend in self.backends.values()}
        for backend in self.backends.values():
            model = backend["model"]
            gpu = backend["gpu"]
            labels = add_labels(None, {"model": model, "gpu": gpu})
            scrape_error = 0
            try:
                lines.extend(parse_llama_metrics(fetch_metrics(backend["url"]), model, gpu, comments_seen))
                lines.append(f"ai_model_up{labels} 1")
            except (OSError, ValueError, KeyError):
                lines.append(f"ai_model_up{labels} 0")
                scrape_error = 1
            lines.append(f"ai_model_scrape_error{labels} {scrape_error}")
        try:
            lines.extend(nvidia_metrics(comments_seen, gpu_models))
            lines.append("ai_exporter_nvidia_smi_up 1")
        except (OSError, subprocess.CalledProcessError, ValueError):
            lines.append("ai_exporter_nvidia_smi_up 0")
        try:
            lines.extend(host_memory_metrics(comments_seen))
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
    args = parser.parse_args()
    state = MetricsState(load_backends())

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
