#!/usr/bin/env python3
"""Sample CPU package power (RAPL) and utilization and expose a learned power curve."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping


STATE_VERSION = 1
RAPL_GLOB = "/sys/class/powercap/intel-rapl*"
MICROJOULES_PER_JOULE = 1_000_000.0


def find_rapl_package(domain_name: str = "package-0") -> Path | None:
    for base in sorted(glob.glob(RAPL_GLOB)):
        path = Path(base)
        try:
            name = (path / "name").read_text(encoding="ascii").strip()
        except OSError:
            continue
        if name == domain_name and (path / "energy_uj").exists():
            return path
    return None


def rapl_power_watts(
    previous_uj: float | None,
    current_uj: float,
    max_range_uj: float,
    elapsed_seconds: float,
) -> float | None:
    """Convert two cumulative microjoule readings to watts, tolerating counter wraps."""
    if previous_uj is None or elapsed_seconds <= 0 or max_range_uj <= 0:
        return None
    delta_uj = (current_uj - previous_uj) % max_range_uj
    return delta_uj / MICROJOULES_PER_JOULE / elapsed_seconds


def read_proc_stat_cpu() -> tuple[float, float] | None:
    """Return (total_jiffies, busy_jiffies) from the aggregate cpu line."""
    try:
        with open("/proc/stat", encoding="ascii") as proc_stat:
            for line in proc_stat:
                if line.startswith("cpu "):
                    fields = [float(field) for field in line.split()[1:]]
                    idle = fields[3] + (fields[4] if len(fields) > 4 else 0.0)
                    total = sum(fields)
                    return total, total - idle
    except (OSError, ValueError):
        return None
    return None


def cpu_utilization_percent(
    previous: tuple[float, float] | None,
    current: tuple[float, float],
) -> float | None:
    if previous is None:
        return None
    total_delta = current[0] - previous[0]
    busy_delta = current[1] - previous[1]
    if total_delta <= 0:
        return None
    return min(max(100.0 * busy_delta / total_delta, 0.0), 100.0)


def bucket_label(index: int, width: float) -> str:
    lo = int(index * width)
    hi = int((index + 1) * width)
    return f"{lo:02d}-{hi:02d}"


def bucket_index(utilization: float, width: float) -> int:
    count = max(int(round(100.0 / width)), 1)
    return min(max(int(utilization // width), 0), count - 1)


def bucket_stats(samples: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(samples),
        "stddev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "max": max(samples),
        "count": float(len(samples)),
    }


def interpolate_power(points: list[tuple[float, float]], utilization: float) -> float | None:
    """Piecewise-linear interpolation over bucket (center, median) pairs, clamped."""
    if not points or utilization is None:
        return None
    points = sorted(points)
    if utilization <= points[0][0]:
        return points[0][1]
    if utilization >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= utilization <= x1:
            if x1 == x0:
                return y1
            return y0 + (y1 - y0) * (utilization - x0) / (x1 - x0)
    return points[-1][1]


class CpuPowerProfiler:
    def __init__(self, config: Mapping[str, Any], state_path: Path | None) -> None:
        self.host_id = str(config.get("host_id", "unconfigured"))
        self.sampling_interval_seconds = float(config.get("sampling_interval_seconds", 10.0))
        self.bucket_width_percent = float(config.get("bucket_width_percent", 5.0))
        self.bucket_window_samples = int(config.get("bucket_window_samples", 20))
        self.rapl_domain = str(config.get("rapl_domain", "package-0"))
        self.state_path = state_path
        self.rapl_path = find_rapl_package(self.rapl_domain)
        self.lock = threading.Lock()
        self.previous_energy_uj: float | None = None
        self.previous_energy_mono: float | None = None
        self.max_range_uj = self._read_max_range_uj()
        self.rapl_ready = self._read_energy_uj() is not None and self.max_range_uj > 0
        self.previous_proc: tuple[float, float] | None = None
        self.state = self._load_state()
        self.body = self._render()

    def _read_energy_uj(self) -> float | None:
        if self.rapl_path is None:
            return None
        try:
            return float((self.rapl_path / "energy_uj").read_text(encoding="ascii"))
        except (OSError, ValueError):
            return None

    def _domain_enabled(self) -> bool:
        """Informational only: whether the powercap `enabled` flag reads nonzero.

        Some kernels report 0 while the energy counter still accumulates, so this
        must never gate counter reads; it is exported as ai_cpu_profiler_rapl_enabled.
        """
        if self.rapl_path is None:
            return False
        try:
            return (self.rapl_path / "enabled").read_text(encoding="ascii").strip() != "0"
        except OSError:
            # Kernels without the enabled attribute expose readable counters directly.
            return True

    def _read_max_range_uj(self) -> float:
        if self.rapl_path is None:
            return 0.0
        try:
            return float((self.rapl_path / "max_energy_range_uj").read_text(encoding="ascii"))
        except (OSError, ValueError):
            return 0.0

    def _empty_state(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "buckets": {},
            "samples_total": 0,
            "last_utilization_percent": None,
            "last_power_watts": None,
        }

    def _load_state(self) -> dict[str, Any]:
        if self.state_path is None or not self.state_path.exists():
            return self._empty_state()
        try:
            loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return self._empty_state()
        if not isinstance(loaded, dict) or loaded.get("version") != STATE_VERSION:
            return self._empty_state()
        state = self._empty_state()
        state.update(loaded)
        return state

    def _save_state(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.state_path.parent,
            prefix=f".{self.state_path.name}.",
            text=True,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(self.state, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.state_path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def update_bucket(self, utilization: float, watts: float) -> None:
        label = bucket_label(bucket_index(utilization, self.bucket_width_percent), self.bucket_width_percent)
        samples = self.state["buckets"].setdefault(label, [])
        samples.append(watts)
        if len(samples) > self.bucket_window_samples:
            del samples[0 : len(samples) - self.bucket_window_samples]

    def sample(self) -> None:
        utilization: float | None = None
        proc = read_proc_stat_cpu()
        if proc is not None:
            utilization = cpu_utilization_percent(self.previous_proc, proc)
            self.previous_proc = proc

        watts: float | None = None
        current_uj = self._read_energy_uj()
        now = time.monotonic()
        if current_uj is None:
            self.rapl_ready = False
            self.previous_energy_uj = None
            self.previous_energy_mono = None
        else:
            if not self.rapl_ready:
                self.previous_energy_uj = None
                self.previous_energy_mono = None
                if self.max_range_uj <= 0:
                    self.max_range_uj = self._read_max_range_uj()
                self.rapl_ready = self.max_range_uj > 0
            elapsed = (
                now - self.previous_energy_mono
                if self.previous_energy_mono is not None
                else 0.0
            )
            watts = rapl_power_watts(self.previous_energy_uj, current_uj, self.max_range_uj, elapsed)
            self.previous_energy_uj = current_uj
            self.previous_energy_mono = now

        with self.lock:
            if utilization is not None:
                self.state["last_utilization_percent"] = utilization
            if watts is not None and utilization is not None and watts >= 0.0:
                self.state["last_power_watts"] = watts
                self.update_bucket(utilization, watts)
                self.state["samples_total"] = int(self.state["samples_total"]) + 1
            self._save_state()
            self.body = self._render()

    def get(self) -> str:
        with self.lock:
            return self.body

    def _render(self) -> str:
        lines = [
            "# HELP ai_cpu_power_watts Latest CPU package power from RAPL energy deltas.",
            "# TYPE ai_cpu_power_watts gauge",
            "# HELP ai_cpu_utilization_percent Latest CPU utilization from /proc/stat deltas.",
            "# TYPE ai_cpu_utilization_percent gauge",
            "# HELP ai_cpu_profiler_rapl_available Whether the RAPL energy counter is readable and has a usable energy range.",
            "# TYPE ai_cpu_profiler_rapl_available gauge",
            "# HELP ai_cpu_profiler_rapl_enabled Whether the powercap enabled flag reads nonzero (informational only).",
            "# TYPE ai_cpu_profiler_rapl_enabled gauge",
            "# HELP ai_cpu_profiler_config_info Configured profiler sampling parameters.",
            "# TYPE ai_cpu_profiler_config_info gauge",
            "# HELP ai_cpu_profiler_samples_total Power/utilization pairs recorded into buckets.",
            "# TYPE ai_cpu_profiler_samples_total counter",
            "# HELP ai_cpu_power_bucket_median_watts Median CPU power for a utilization bucket.",
            "# TYPE ai_cpu_power_bucket_median_watts gauge",
            "# HELP ai_cpu_power_bucket_stddev_watts Standard deviation of CPU power within a bucket.",
            "# TYPE ai_cpu_power_bucket_stddev_watts gauge",
            "# HELP ai_cpu_power_bucket_max_watts Maximum CPU power observed in a bucket.",
            "# TYPE ai_cpu_power_bucket_max_watts gauge",
            "# HELP ai_cpu_power_bucket_sample_count Samples retained in a bucket.",
            "# TYPE ai_cpu_power_bucket_sample_count gauge",
        ]
        labels = f'{{host_id="{self.host_id}"}}'
        last_power = self.state["last_power_watts"]
        last_util = self.state["last_utilization_percent"]
        lines.append(f"ai_cpu_power_watts{labels} {last_power if last_power is not None else 0.0}")
        lines.append(f"ai_cpu_utilization_percent{labels} {last_util if last_util is not None else 0.0}")
        lines.append(f"ai_cpu_profiler_rapl_available{labels} {1.0 if self.rapl_ready else 0.0}")
        lines.append(f"ai_cpu_profiler_rapl_enabled{labels} {1.0 if self._domain_enabled() else 0.0}")
        config_labels = f'{{host_id="{self.host_id}",interval_seconds="{self.sampling_interval_seconds:g}",bucket_width_percent="{self.bucket_width_percent:g}",bucket_window_samples="{self.bucket_window_samples}"}}'
        lines.append(f"ai_cpu_profiler_config_info{config_labels} 1")
        lines.append(f"ai_cpu_profiler_samples_total{labels} {int(self.state['samples_total'])}")
        for label in sorted(self.state["buckets"], key=lambda name: bucket_sort_key(name, self.bucket_width_percent)):
            samples = self.state["buckets"].get(label)
            if not isinstance(samples, list) or not samples:
                continue
            stats = bucket_stats([float(sample) for sample in samples])
            bucket_labels = f'{{host_id="{self.host_id}",bucket="{label}"}}'
            lines.append(f"ai_cpu_power_bucket_median_watts{bucket_labels} {stats['median']}")
            lines.append(f"ai_cpu_power_bucket_stddev_watts{bucket_labels} {stats['stddev']}")
            lines.append(f"ai_cpu_power_bucket_max_watts{bucket_labels} {stats['max']}")
            lines.append(f"ai_cpu_power_bucket_sample_count{bucket_labels} {stats['count']}")
        return "\n".join(lines) + "\n"


def bucket_sort_key(label: str, width: float) -> float:
    try:
        return float(label.split("-")[0]) / width
    except (ValueError, IndexError):
        return math.inf


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


class Handler(BaseHTTPRequestHandler):
    profiler: CpuPowerProfiler

    def do_GET(self) -> None:  # noqa: N802
        if self.path not in {"/metrics", "/healthz"}:
            self.send_error(404)
            return
        body = b"ok\n" if self.path == "/healthz" else self.profiler.get().encode("utf-8")
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
    parser.add_argument("--config")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--state-path", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    listen = config.get("listen", {}) if isinstance(config.get("listen"), Mapping) else {}
    host = args.host or str(listen.get("host", "127.0.0.1"))
    port = args.port if args.port is not None else int(listen.get("port", 9109))
    state_path = args.state_path or config.get("state_path")

    profiler = CpuPowerProfiler(config, Path(state_path) if state_path else None)

    def refresh() -> None:
        while True:
            time.sleep(profiler.sampling_interval_seconds)
            profiler.sample()

    profiler.sample()
    threading.Thread(target=refresh, daemon=True).start()
    Handler.profiler = profiler
    server = ThreadingHTTPServer((host, port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
