#!/usr/bin/env python3
"""Exercise Bonsai reviewer/subagent server topologies on isolated resources.

The runner deliberately starts the official Prism server as child processes.
Physical GPU identity is preserved by setting CUDA_VISIBLE_DEVICES to one
physical GPU and addressing that process's device as CUDA0.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(os.environ['EXPERIMENT_ROOT'])
RESEARCH = ROOT / "research"
SERVER = RESEARCH / "prismml-llama.cpp-prism-b10687-5d80cff-build/build-sm86/bin/llama-server"
SOURCE = RESEARCH / "prismml-llama.cpp-prism-b10687-5d80cff/source"
MODEL = RESEARCH / "prismml-bonsai-2-27b-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf"
BIN_DIR = SERVER.parent
PORTS = {1: 18221, 2: 18222}
SHARED_PORT = 18223
CONTEXT_SIZE = 262144
DEPTHS = (65536, 196608)
SHARED_DEPTH = 8192
OUTPUT_TOKENS = 128
SEED = 271828
POLL_SECONDS = 0.5
REQUEST_TIMEOUT = 3600


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def command_output(args: list[str], env: dict[str, str] | None = None, timeout: int = 120) -> dict[str, Any]:
    try:
        completed = subprocess.run(args, env=env, capture_output=True, text=True, timeout=timeout, check=False)
        return {"args": args, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    except Exception as exc:
        return {"args": args, "returncode": None, "stdout": "", "stderr": repr(exc)}


def unmasked_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def parse_number(value: str) -> int | float | str | None:
    value = value.strip()
    if value in {"", "N/A", "[Not Supported]"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def gpu_snapshot() -> dict[str, Any]:
    query = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem,utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ]
    result = command_output(query, env=unmasked_env(), timeout=30)
    fields = (
        "index",
        "uuid",
        "name",
        "memory_total_mib",
        "memory_used_mib",
        "power_w",
        "temperature_c",
        "clock_sm_mhz",
        "clock_mem_mhz",
        "utilization_gpu_pct",
        "utilization_mem_pct",
    )
    gpus: list[dict[str, Any]] = []
    for row in result.get("stdout", "").splitlines():
        values = next(csv.reader([row]), [])
        if len(values) != len(fields):
            continue
        item: dict[str, Any] = {}
        for field, value in zip(fields, values):
            item[field] = value.strip() if field in {"uuid", "name"} else parse_number(value)
        gpus.append(item)
    apps_result = command_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,process_name,used_memory", "--format=csv,noheader,nounits"],
        env=unmasked_env(),
        timeout=30,
    )
    apps: list[dict[str, Any]] = []
    for row in apps_result.get("stdout", "").splitlines():
        values = next(csv.reader([row]), [])
        if len(values) == 4:
            apps.append(
                {
                    "pid": parse_number(values[0]),
                    "gpu_uuid": values[1].strip(),
                    "process_name": values[2].strip(),
                    "used_memory_mib": parse_number(values[3]),
                }
            )
    return {
        "gpus": gpus,
        "compute_apps": apps,
        "gpu_query": query,
        "gpu_query_returncode": result.get("returncode"),
        "gpu_query_stderr": result.get("stderr", ""),
    }


def host_snapshot() -> dict[str, Any]:
    stat = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
    cpu = [int(value) for value in stat[1:]]
    mem: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(":")
        parts = value.strip().split()
        if parts:
            mem[key] = int(parts[0])
    return {
        "cpu_jiffies": cpu,
        "meminfo_kib": mem,
        "memory_used_kib": mem.get("MemTotal", 0) - mem.get("MemAvailable", 0),
        "swap_used_kib": mem.get("SwapTotal", 0) - mem.get("SwapFree", 0),
        "loadavg": list(os.getloadavg()),
    }


def port_snapshot() -> str:
    return command_output(["ss", "-ltnp"], env=unmasked_env(), timeout=30).get("stdout", "")


def process_snapshot() -> str:
    return command_output(["ps", "-eo", "pid,ppid,user,stat,cmd"], env=unmasked_env(), timeout=30).get("stdout", "")


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


class Telemetry:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.label = "idle"
        self.samples: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="bonsai-topology-telemetry", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def mark(self, label: str) -> None:
        with self.lock:
            self.label = label

    def _run(self) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            while not self.stop_event.is_set():
                mono = time.monotonic()
                with self.lock:
                    label = self.label
                sample = {"monotonic": mono, "wall_time": time.time(), "label": label, "gpu": gpu_snapshot(), "host": host_snapshot()}
                self.samples.append(sample)
                handle.write(json.dumps(sample, sort_keys=True) + "\n")
                handle.flush()
                self.stop_event.wait(POLL_SECONDS)

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=30)


def request_json(port: int, path: str, payload: dict[str, Any] | None = None, timeout: int = 120) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method="POST" if data is not None else "GET")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            return {"status": response.status, "headers": dict(response.headers.items()), "body": body.decode("utf-8", errors="replace"), "elapsed_s": time.monotonic() - started}
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "headers": dict(exc.headers.items()), "body": exc.read().decode("utf-8", errors="replace"), "error": "http_error", "elapsed_s": time.monotonic() - started}
    except Exception as exc:
        return {"status": None, "headers": {}, "body": "", "error": repr(exc), "elapsed_s": time.monotonic() - started}


def parsed_body(result: dict[str, Any]) -> Any:
    try:
        return json.loads(result.get("body", ""))
    except (json.JSONDecodeError, TypeError):
        return None


def tokenize(port: int, content: str) -> dict[str, Any]:
    result = request_json(port, "/tokenize", {"content": content, "add_special": True}, timeout=600)
    body = parsed_body(result)
    tokens = body.get("tokens") if isinstance(body, dict) else None
    result["token_count"] = len(tokens) if isinstance(tokens, list) else None
    result["parsed"] = body
    return result


def input_tokens(port: int, payload: dict[str, Any]) -> dict[str, Any]:
    result = request_json(port, "/v1/chat/completions/input_tokens", payload, timeout=600)
    result["parsed"] = parsed_body(result)
    return result


def make_corpus() -> tuple[str, str]:
    header = (
        "You are reviewing a deterministic synthetic repository snapshot for a coding subagent benchmark. "
        "Use supplied facts as authoritative. Do not invent files, APIs, tests, or commands. "
        "Return a concise implementation-oriented answer after examining the complete snapshot.\n\n"
    )
    blocks: list[str] = []
    for index in range(11000):
        blocks.append(
            f"### repository record {index:05d}\n"
            f"path: services/module_{index % 53:02d}/component_{index % 197:03d}.py\n"
            f"symbol: Handler_{index % 311:03d}.execute\n"
            "def execute(request, state):\n"
            f"    revision = \"r{index:05d}-{(index * 7919) % 100000:05d}\"\n"
            "    if request.get('cancelled'):\n"
            "        return {'status': 'cancelled', 'revision': revision}\n"
            "    result = state.lookup(request.get('key', 'default'))\n"
            "    return {'status': 'ok', 'revision': revision, 'result': result}\n"
            f"tests: test_handler_{index % 149:03d} checks revision {index:05d} and result shape\n\n"
        )
    return header, "".join(blocks)


def build_prompt(port: int, target_depth: int, instance: int, header: str, corpus: str) -> dict[str, Any]:
    marker = f"Benchmark instance: physical GPU {instance}; target populated context: {target_depth} tokens.\n\n"
    desired_content_tokens = max(256, target_depth - 512)
    probe_text = header + marker + corpus[:65536]
    probe = tokenize(port, probe_text)
    probe_count = probe.get("token_count") or max(1, len(probe_text.split()))
    ratio = probe_count / max(1, len(probe_text))
    char_count = min(len(corpus), max(128, int(desired_content_tokens / ratio)))
    calibration: list[dict[str, Any]] = [probe]
    for _ in range(6):
        content = header + marker + corpus[:char_count]
        result = tokenize(port, content)
        calibration.append(result)
        count = result.get("token_count")
        if not count or abs(count - desired_content_tokens) <= 16:
            break
        char_count = max(128, min(len(corpus), int(char_count * desired_content_tokens / count)))
    content = header + marker + corpus[:char_count]
    final = calibration[-1]
    return {
        "target_context_tokens": target_depth,
        "desired_content_tokens": desired_content_tokens,
        "instance": instance,
        "content": content,
        "content_chars": len(content),
        "content_token_count": final.get("token_count"),
        "calibration": calibration,
    }


def parse_sse(response: Any, started: float) -> tuple[str, dict[str, Any]]:
    raw_parts: list[str] = []
    chunks: list[dict[str, Any]] = []
    output: list[str] = []
    reasoning: list[str] = []
    first_event: float | None = None
    first_token: float | None = None
    finish_reason: str | None = None
    while True:
        line = response.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace")
        raw_parts.append(text)
        if not text.startswith("data:"):
            continue
        now = time.monotonic()
        if first_event is None:
            first_event = now
        value = text[5:].strip()
        if value == "[DONE]":
            continue
        try:
            chunk = json.loads(value)
        except json.JSONDecodeError:
            continue
        if not isinstance(chunk, dict):
            continue
        chunks.append(chunk)
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        finish_reason = choice.get("finish_reason") or finish_reason
        delta = choice.get("delta") or {}
        content = delta.get("content")
        thought = delta.get("reasoning_content")
        if content:
            first_token = first_token or now
            output.append(content)
        if thought:
            first_token = first_token or now
            reasoning.append(thought)
    ended = time.monotonic()
    return "".join(raw_parts), {
        "chunks": chunks,
        "output": "".join(output),
        "reasoning": "".join(reasoning),
        "finish_reason": finish_reason,
        "first_event_s": None if first_event is None else first_event - started,
        "ttft_s": None if first_token is None else first_token - started,
        "wall_s": ended - started,
    }


def chat_request(port: int, payload: dict[str, Any], instance: int) -> dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    started = time.monotonic()
    result: dict[str, Any] = {"instance": instance, "port": port, "request_started_monotonic": started}
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            raw, stream = parse_sse(response, started)
            result.update({"status": response.status, "headers": dict(response.headers.items()), "raw": raw, **stream})
    except urllib.error.HTTPError as exc:
        result.update({"status": exc.code, "headers": dict(exc.headers.items()), "raw": exc.read().decode("utf-8", errors="replace"), "error": "http_error", "wall_s": time.monotonic() - started})
    except Exception as exc:
        result.update({"status": None, "headers": {}, "raw": "", "error": repr(exc), "wall_s": time.monotonic() - started})
    result["request_ended_monotonic"] = time.monotonic()
    return result


def timing_from_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    timing: dict[str, Any] = {}
    for chunk in chunks:
        if isinstance(chunk.get("timings"), dict):
            timing.update(chunk["timings"])
        if isinstance(chunk.get("usage"), dict):
            timing["usage"] = chunk["usage"]
    return timing


def numeric(values: list[Any]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float))]


def gpu_rows(samples: list[dict[str, Any]], start: float, end: float, index: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        if not start <= sample.get("monotonic", -1) <= end:
            continue
        rows.extend(gpu for gpu in sample.get("gpu", {}).get("gpus", []) if gpu.get("index") == index)
    return rows


def trapezoid_energy(samples: list[dict[str, Any]], start: float, end: float, index: int, baseline: float | None = None) -> tuple[float | None, float | None]:
    points: list[tuple[float, float]] = []
    for sample in samples:
        mono = sample.get("monotonic")
        if not isinstance(mono, (int, float)) or not start <= mono <= end:
            continue
        gpu = next((item for item in sample.get("gpu", {}).get("gpus", []) if item.get("index") == index), None)
        power = gpu.get("power_w") if gpu else None
        if isinstance(power, (int, float)):
            points.append((float(mono), float(power)))
    if len(points) < 2:
        return None, None
    energy = 0.0
    dynamic = 0.0
    for left, right in zip(points, points[1:]):
        seconds = max(0.0, right[0] - left[0])
        average = (left[1] + right[1]) / 2
        energy += average * seconds
        if baseline is not None:
            dynamic += max(0.0, average - baseline) * seconds
    return energy, dynamic if baseline is not None else None


def telemetry_summary(samples: list[dict[str, Any]], start: float, end: float, physical_gpus: tuple[int, ...], baselines: dict[int, float | None]) -> dict[str, Any]:
    selected = [sample for sample in samples if start <= sample.get("monotonic", -1) <= end]
    result: dict[str, Any] = {"sample_count": len(selected), "window_s": end - start, "gpus": {}, "gpu0_proof": {}}
    fields = ("memory_used_mib", "power_w", "temperature_c", "clock_sm_mhz", "clock_mem_mhz", "utilization_gpu_pct", "utilization_mem_pct")
    for index in physical_gpus:
        rows = gpu_rows(samples, start, end, index)
        summary: dict[str, Any] = {"sample_count": len(rows)}
        for field in fields:
            values = numeric([row.get(field) for row in rows])
            if values:
                summary[field + "_min"] = min(values)
                summary[field + "_max"] = max(values)
                summary[field + "_avg"] = sum(values) / len(values)
        summary["energy_j"] = trapezoid_energy(samples, start, end, index, baselines.get(index))[0]
        summary["dynamic_energy_j"] = trapezoid_energy(samples, start, end, index, baselines.get(index))[1]
        result["gpus"][str(index)] = summary
    gpu0_rows = gpu_rows(samples, start, end, 0)
    result["gpu0_proof"] = {
        "sample_count": len(gpu0_rows),
        "max_memory_used_mib": max(numeric([row.get("memory_used_mib") for row in gpu0_rows]), default=None),
        "max_power_w": max(numeric([row.get("power_w") for row in gpu0_rows]), default=None),
        "compute_apps_seen": [
            app
            for sample in selected
            for app in sample.get("gpu", {}).get("compute_apps", [])
            if app.get("gpu_uuid") in {row.get("uuid") for row in gpu0_rows}
        ],
    }
    hosts = [sample.get("host", {}) for sample in selected]
    memory = numeric([host.get("memory_used_kib") for host in hosts])
    loads = numeric([host.get("loadavg", [None])[0] for host in hosts if host.get("loadavg")])
    if memory:
        result["host_memory_used_kib_max"] = max(memory)
        result["host_memory_used_kib_avg"] = sum(memory) / len(memory)
    if loads:
        result["host_load1_max"] = max(loads)
    cpu_pct: list[float] = []
    for left, right in zip(hosts, hosts[1:]):
        first = left.get("cpu_jiffies", [])
        second = right.get("cpu_jiffies", [])
        if len(first) < 5 or len(second) != len(first):
            continue
        deltas = [b - a for a, b in zip(first, second)]
        total = sum(deltas)
        if total > 0:
            cpu_pct.append(100.0 * (1.0 - (deltas[3] + deltas[4]) / total))
    if cpu_pct:
        result["host_cpu_pct_avg"] = sum(cpu_pct) / len(cpu_pct)
        result["host_cpu_pct_max"] = max(cpu_pct)
    return result


def launch_environment(physical_gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(physical_gpu),
            "LD_LIBRARY_PATH": str(BIN_DIR) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""),
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def server_command(port: int, parallel: int) -> list[str]:
    return [
        str(SERVER),
        "--model",
        str(MODEL),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--alias",
        "bonsai-2-27b-pq2_0-topology",
        "--device",
        "CUDA0",
        "--n-gpu-layers",
        "99",
        "--ctx-size",
        str(CONTEXT_SIZE),
        "--fit",
        "off",
        "--parallel",
        str(parallel),
        "--flash-attn",
        "on",
        "--temp",
        "1.0",
        "--top-p",
        "0.95",
        "--top-k",
        "20",
        "--jinja",
        "--reasoning-budget",
        "0",
        "--reasoning-format",
        "none",
        "--chat-template-kwargs",
        '{"enable_thinking":false}',
        "--seed",
        str(SEED),
        "--metrics",
        "--no-ui",
        "--no-context-shift",
        "--log-colors",
        "off",
        "--log-timestamps",
        "--log-verbosity",
        "3",
    ]


def wait_health(port: int, process: subprocess.Popen[bytes], path: Path, timeout: int = 900) -> dict[str, Any]:
    started = time.monotonic()
    attempts: list[dict[str, Any]] = []
    with path.open("w", encoding="utf-8") as handle:
        while time.monotonic() - started < timeout:
            if process.poll() is not None:
                return {"ready": False, "elapsed_s": time.monotonic() - started, "process_returncode": process.returncode, "attempts": attempts}
            result = request_json(port, "/health", timeout=10)
            record = {"status": result.get("status"), "body": result.get("body", "")[:1000], "error": result.get("error"), "elapsed_s": result.get("elapsed_s")}
            attempts.append(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            if result.get("status") == 200:
                return {"ready": True, "elapsed_s": time.monotonic() - started, "attempts": attempts}
            time.sleep(1)
    return {"ready": False, "elapsed_s": time.monotonic() - started, "timeout": True, "attempts": attempts}


def stop_process(process: subprocess.Popen[bytes] | None) -> dict[str, Any]:
    if process is None:
        return {"returncode": None, "signal": None}
    if process.poll() is not None:
        return {"returncode": process.returncode, "signal": None}
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=60)
        return {"returncode": process.returncode, "signal": "SIGTERM"}
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)
        return {"returncode": process.returncode, "signal": "SIGKILL"}
    except ProcessLookupError:
        return {"returncode": process.poll(), "signal": "already-exited"}


def process_gpu_proof(snapshot: dict[str, Any], server_pids: list[int]) -> dict[str, Any]:
    uuid_by_index = {item.get("index"): item.get("uuid") for item in snapshot.get("gpus", [])}
    apps = snapshot.get("compute_apps", [])
    return {
        "server_pids": server_pids,
        "gpu0_uuid": uuid_by_index.get(0),
        "gpu0_compute_apps_for_server": [app for app in apps if app.get("pid") in server_pids and app.get("gpu_uuid") == uuid_by_index.get(0)],
        "all_server_gpu_apps": [app for app in apps if app.get("pid") in server_pids],
    }


def normalize_timing(result: dict[str, Any]) -> dict[str, Any]:
    timing = timing_from_chunks(result.get("chunks", []))
    prompt_n = timing.get("prompt_n")
    predicted_n = timing.get("predicted_n")
    prompt_ms = timing.get("prompt_ms")
    predicted_ms = timing.get("predicted_ms")
    if isinstance(prompt_n, (int, float)) and isinstance(prompt_ms, (int, float)) and prompt_ms > 0:
        timing["prefill_tok_s"] = prompt_n / (prompt_ms / 1000)
    if isinstance(predicted_n, (int, float)) and isinstance(predicted_ms, (int, float)) and predicted_ms > 0:
        timing["decode_tok_s"] = predicted_n / (predicted_ms / 1000)
    timing["wall_output_tok_s"] = (predicted_n or 0) / max(float(result.get("wall_s") or 0), 0.000001)
    result["timings"] = timing
    result["output_tokens_observed"] = predicted_n
    return result


def concurrent_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in results if isinstance(item.get("request_started_monotonic"), (int, float)) and isinstance(item.get("request_ended_monotonic"), (int, float))]
    if not valid:
        return {"request_count": 0, "status": "no_completed_requests"}
    started = min(float(item["request_started_monotonic"]) for item in valid)
    finished = max(float(item["request_ended_monotonic"]) for item in valid)
    walls = [float(item.get("wall_s", 0)) for item in valid]
    prompt_n = [float(item["timings"]["prompt_n"]) for item in valid if isinstance(item.get("timings", {}).get("prompt_n"), (int, float))]
    predicted_n = [float(item["timings"]["predicted_n"]) for item in valid if isinstance(item.get("timings", {}).get("predicted_n"), (int, float))]
    prompt_ms = [float(item["timings"]["prompt_ms"]) for item in valid if isinstance(item.get("timings", {}).get("prompt_ms"), (int, float))]
    generation_starts = [float(item["request_ended_monotonic"]) - float(item["timings"]["predicted_ms"]) / 1000 for item in valid if isinstance(item.get("timings", {}).get("predicted_ms"), (int, float))]
    result: dict[str, Any] = {
        "request_count": len(valid),
        "aggregate_wall_s": finished - started,
        "sum_instance_wall_s": sum(walls),
        "max_instance_wall_s": max(walls),
        "min_instance_wall_s": min(walls),
        "straggler_ratio": max(walls) / max(min(walls), 0.000001),
        "overlap_efficiency": sum(walls) / max(finished - started, 0.000001),
        "cross_gpu_interference_factor": (finished - started) / max(walls),
        "total_prompt_tokens": sum(prompt_n) if prompt_n else None,
        "total_output_tokens": sum(predicted_n) if predicted_n else None,
        "aggregate_prefill_tok_s": sum(prompt_n) / (max(prompt_ms) / 1000) if prompt_n and prompt_ms else None,
        "aggregate_decode_tok_s": sum(predicted_n) / (max(finished - min(generation_starts), 0.000001)) if predicted_n and generation_starts else None,
    }
    return result


def request_payload(prompt: str, model_alias: str = "bonsai-2-27b-pq2_0-topology") -> dict[str, Any]:
    return {
        "model": model_alias,
        "messages": [
            {"role": "system", "content": "You are a deterministic coding subagent. Thinking is disabled."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "seed": SEED,
        "max_tokens": OUTPUT_TOKENS,
        "ignore_eos": True,
        "stream": True,
    }


def run_concurrent_requests(items: list[tuple[int, int, dict[str, Any]]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    lock = threading.Lock()
    start_event = threading.Event()

    def worker(instance: int, port: int, payload: dict[str, Any]) -> None:
        start_event.wait()
        result = normalize_timing(chat_request(port, payload, instance))
        with lock:
            results.append(result)

    workers = [threading.Thread(target=worker, args=item, daemon=True) for item in items]
    for worker_thread in workers:
        worker_thread.start()
    start_event.set()
    for worker_thread in workers:
        worker_thread.join(timeout=REQUEST_TIMEOUT + 60)
    results.sort(key=lambda item: int(item.get("instance", 0)))
    return results


def write_request_artifacts(case_dir: Path, prompt_meta: dict[str, Any], payload: dict[str, Any], result: dict[str, Any]) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "prompt.txt").write_text(prompt_meta["content"], encoding="utf-8")
    prompt_copy = {key: value for key, value in prompt_meta.items() if key != "content"}
    json_dump(case_dir / "prompt-meta.json", prompt_copy)
    json_dump(case_dir / "request.json", payload)
    (case_dir / "response.sse").write_text(result.get("raw", ""), encoding="utf-8")
    json_dump(case_dir / "response-chunks.json", result.get("chunks", []))
    json_dump(case_dir / "result.json", {key: value for key, value in result.items() if key != "raw"})


def classify_failure(log_path: Path, readiness: dict[str, Any], process_returncode: int | None) -> str:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    lower = text.lower()
    if any(token in lower for token in ("out of memory", "outofmemory", "cuda error out of memory", "oom")):
        return "oom"
    if readiness.get("ready") is not True:
        return "failed_load_or_startup"
    if process_returncode not in (None, 0):
        return "server_exit"
    return "request_failure"


def server_metadata(physical_gpu: int, port: int, parallel: int) -> dict[str, Any]:
    env = launch_environment(physical_gpu)
    command = server_command(port, parallel)
    return {
        "physical_gpu": physical_gpu,
        "port": port,
        "parallel": parallel,
        "ctx_size": CONTEXT_SIZE,
        "packing": "PQ2_0",
        "model": {"path": str(MODEL), "size": MODEL.stat().st_size, "sha256": sha256_file(MODEL)},
        "server": {"path": str(SERVER), "size": SERVER.stat().st_size, "sha256": sha256_file(SERVER), "version": command_output([str(SERVER), "--version"], env=env, timeout=60)},
        "source": {"path": str(SOURCE), "commit": command_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], timeout=30), "status": command_output(["git", "-C", str(SOURCE), "status", "--short"], timeout=30)},
        "environment": {key: env[key] for key in ("CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "PYTHONUNBUFFERED")},
        "command": command,
        "systemd_used": False,
        "production_interaction": False,
        "visible_device_argument": "CUDA0",
    }


def prepare_prompts(servers: dict[int, dict[str, Any]], header: str, corpus: str, artifact: Path) -> dict[str, dict[str, dict[str, Any]]]:
    prompts: dict[str, dict[str, dict[str, Any]]] = {}
    for depth in DEPTHS:
        prompts[str(depth)] = {}
        for instance, server in servers.items():
            meta = build_prompt(server["port"], depth, instance, header, corpus)
            prompt_path = artifact / "prompts" / f"gpu{instance}-depth{depth}.txt"
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(meta["content"], encoding="utf-8")
            payload = request_payload(meta["content"])
            meta["chat_input_tokens"] = input_tokens(server["port"], payload)
            prompts[str(depth)][str(instance)] = meta
    return prompts


def run_independent_phase(root: Path, header: str, corpus: str, before: dict[str, Any]) -> dict[str, Any]:
    phase = root / "independent-dual-gpu"
    phase.mkdir(parents=True, exist_ok=True)
    servers: dict[int, dict[str, Any]] = {instance: {"port": port, "physical_gpu": instance} for instance, port in PORTS.items()}
    for instance, server in servers.items():
        server.update(server_metadata(instance, server["port"], 1))
        json_dump(phase / f"gpu{instance}" / "server-metadata.json", server)
        json_dump(phase / f"gpu{instance}" / "device-list.json", command_output([str(SERVER), "--list-devices"], env=launch_environment(instance), timeout=120))
    telemetry = Telemetry(phase / "telemetry.jsonl")
    processes: dict[int, subprocess.Popen[bytes]] = {}
    logs: dict[int, Any] = {}
    readiness: dict[int, dict[str, Any]] = {}
    status: dict[str, Any] = {"phase": "independent_dual_gpu", "ports": PORTS, "physical_gpus": [1, 2], "cases": []}
    try:
        telemetry.start()
        telemetry.mark("server_startup")
        for instance, server in servers.items():
            log_path = phase / f"gpu{instance}" / "server.log"
            logs[instance] = log_path.open("wb")
            command = [str(value) for value in server["command"]]
            processes[instance] = subprocess.Popen(command, env=launch_environment(instance), stdout=logs[instance], stderr=subprocess.STDOUT, start_new_session=True)
            server["pid"] = processes[instance].pid
        health_threads: list[threading.Thread] = []

        def wait_one(instance: int) -> None:
            readiness[instance] = wait_health(PORTS[instance], processes[instance], phase / f"gpu{instance}" / "health-poll.jsonl")

        for instance in servers:
            thread = threading.Thread(target=wait_one, args=(instance,), daemon=True)
            health_threads.append(thread)
            thread.start()
        for thread in health_threads:
            thread.join()
        for instance, ready in readiness.items():
            json_dump(phase / f"gpu{instance}" / "readiness.json", ready)
        status["readiness"] = readiness
        if not all(item.get("ready") for item in readiness.values()):
            status["gate"] = "startup_failure"
            return status

        json_dump(phase / "props-gpu1.json", request_json(PORTS[1], "/props"))
        json_dump(phase / "props-gpu2.json", request_json(PORTS[2], "/props"))
        json_dump(phase / "slots-gpu1-initial.json", request_json(PORTS[1], "/slots"))
        json_dump(phase / "slots-gpu2-initial.json", request_json(PORTS[2], "/slots"))
        prompts = prepare_prompts(servers, header, corpus, phase)
        status["prompt_metadata"] = {depth: {instance: {key: value for key, value in meta.items() if key != "content"} for instance, meta in values.items()} for depth, values in prompts.items()}
        for depth in DEPTHS:
            for cache_state in ("cold", "warm"):
                if cache_state == "cold":
                    clears: dict[str, Any] = {}
                    for instance in PORTS:
                        clears[str(instance)] = request_json(PORTS[instance], "/slots/0?action=erase", {}, timeout=60)
                    json_dump(phase / "cache-clears" / f"depth-{depth}.json", clears)
                telemetry.mark(f"independent-{cache_state}-{depth}")
                baseline_snapshot = gpu_snapshot()
                baselines = {index: next((float(gpu["power_w"]) for gpu in baseline_snapshot.get("gpus", []) if gpu.get("index") == index and isinstance(gpu.get("power_w"), (int, float))), None) for index in PORTS}
                work: list[tuple[int, int, dict[str, Any]]] = []
                for instance, port in PORTS.items():
                    work.append((instance, port, request_payload(prompts[str(depth)][str(instance)]["content"])))
                results = run_concurrent_requests(work)
                valid_starts = [float(item["request_started_monotonic"]) for item in results if isinstance(item.get("request_started_monotonic"), (int, float))]
                valid_ends = [float(item["request_ended_monotonic"]) for item in results if isinstance(item.get("request_ended_monotonic"), (int, float))]
                if not valid_starts or not valid_ends:
                    case_start = time.monotonic()
                    case_end = case_start
                else:
                    case_start = min(valid_starts)
                    case_end = max(valid_ends)
                metrics = telemetry_summary(telemetry.samples, case_start, case_end, (1, 2), baselines)
                aggregate = concurrent_metrics(results)
                case_dir = phase / "cases" / f"depth-{depth}-{cache_state}"
                for result in results:
                    instance = int(result.get("instance", 0))
                    write_request_artifacts(case_dir / f"gpu{instance}", prompts[str(depth)][str(instance)], request_payload(prompts[str(depth)][str(instance)]["content"]), result)
                summary = {"depth": depth, "cache_state": cache_state, "results": [{key: value for key, value in item.items() if key != "raw"} for item in results], "aggregate": aggregate, "telemetry": metrics}
                json_dump(case_dir / "case-summary.json", summary)
                status["cases"].append(summary)
                telemetry.mark("idle")
                if any(process.poll() is not None for process in processes.values()):
                    status["gate"] = "server_failed_during_cases"
                    return status
        status["gate"] = "validated_independent_dual_gpu"
        return status
    finally:
        telemetry.mark("cleanup")
        for instance, process in processes.items():
            stop = stop_process(process)
            json_dump(phase / f"gpu{instance}" / "stop-record.json", {"pid": process.pid, **stop})
        telemetry.stop()
        for handle in logs.values():
            handle.close()
        after = gpu_snapshot()
        json_dump(phase / "gpu-after.json", after)
        (phase / "ports-after.txt").write_text(port_snapshot(), encoding="utf-8")
        (phase / "processes-after.txt").write_text(process_snapshot(), encoding="utf-8")
        status["cleanup"] = {
            "gpu_after": after,
            "server_pids": [server.get("pid") for server in servers.values()],
            "process_gpu_proof_after": process_gpu_proof(after, [int(server["pid"]) for server in servers.values() if server.get("pid")]),
            "ports_free": {str(port): port_is_free(port) for port in PORTS.values()},
            "llama_server_lines_after": [line for line in process_snapshot().splitlines() if "llama-server" in line],
            "systemd_used": False,
            "production_interaction": False,
        }
        json_dump(phase / "phase-status.json", status)


def run_shared_phase(root: Path, header: str, corpus: str, before: dict[str, Any]) -> dict[str, Any]:
    phase = root / "shared-weight-packing"
    phase.mkdir(parents=True, exist_ok=True)
    metadata = server_metadata(1, SHARED_PORT, 2)
    json_dump(phase / "server-metadata.json", metadata)
    json_dump(phase / "device-list.json", command_output([str(SERVER), "--list-devices"], env=launch_environment(1), timeout=120))
    telemetry = Telemetry(phase / "telemetry.jsonl")
    process: subprocess.Popen[bytes] | None = None
    log_path = phase / "server.log"
    log_handle = log_path.open("wb")
    status: dict[str, Any] = {"phase": "shared_weight_packing", "port": SHARED_PORT, "physical_gpu": 1, "parallel": 2, "cases": []}
    readiness: dict[str, Any] = {}
    try:
        telemetry.start()
        telemetry.mark("server_startup")
        process = subprocess.Popen(metadata["command"], env=launch_environment(1), stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True)
        status["server_pid"] = process.pid
        readiness = wait_health(SHARED_PORT, process, phase / "health-poll.jsonl")
        status["readiness"] = readiness
        json_dump(phase / "readiness.json", readiness)
        if not readiness.get("ready"):
            status["gate"] = classify_failure(log_path, readiness, process.poll())
            return status
        json_dump(phase / "props.json", request_json(SHARED_PORT, "/props"))
        json_dump(phase / "slots-initial.json", request_json(SHARED_PORT, "/slots"))
        prompts: dict[str, dict[str, Any]] = {}
        for instance in (1, 2):
            meta = build_prompt(SHARED_PORT, SHARED_DEPTH, instance, header, corpus)
            prompts[str(instance)] = meta
            (phase / f"prompt-gpu{instance}.txt").write_text(meta["content"], encoding="utf-8")
            payload = request_payload(meta["content"])
            meta["chat_input_tokens"] = input_tokens(SHARED_PORT, payload)
        json_dump(phase / "prompt-metadata.json", {instance: {key: value for key, value in meta.items() if key != "content"} for instance, meta in prompts.items()})
        clears = {str(slot): request_json(SHARED_PORT, f"/slots/{slot}?action=erase", {}, timeout=60) for slot in (0, 1)}
        json_dump(phase / "cache-clears.json", clears)
        telemetry.mark("shared-parallel2-8192")
        baseline_snapshot = gpu_snapshot()
        baseline = next((float(gpu["power_w"]) for gpu in baseline_snapshot.get("gpus", []) if gpu.get("index") == 1 and isinstance(gpu.get("power_w"), (int, float))), None)
        results = run_concurrent_requests([(instance, SHARED_PORT, request_payload(prompts[str(instance)]["content"])) for instance in (1, 2)])
        starts = [float(item["request_started_monotonic"]) for item in results if isinstance(item.get("request_started_monotonic"), (int, float))]
        ends = [float(item["request_ended_monotonic"]) for item in results if isinstance(item.get("request_ended_monotonic"), (int, float))]
        start = min(starts) if starts else time.monotonic()
        end = max(ends) if ends else start
        metrics = telemetry_summary(telemetry.samples, start, end, (1,), {1: baseline})
        aggregate = concurrent_metrics(results)
        case_dir = phase / "case-8192-parallel2"
        for result in results:
            instance = int(result.get("instance", 0))
            write_request_artifacts(case_dir / f"slot-{instance - 1}", prompts[str(instance)], request_payload(prompts[str(instance)]["content"]), result)
        case = {"depth": SHARED_DEPTH, "parallel": 2, "results": [{key: value for key, value in item.items() if key != "raw"} for item in results], "aggregate": aggregate, "telemetry": metrics}
        json_dump(case_dir / "case-summary.json", case)
        status["cases"].append(case)
        status["gate"] = "validated_shared_weight_parallel2" if all(item.get("status") == 200 for item in results) else classify_failure(log_path, readiness, process.poll())
        return status
    finally:
        telemetry.mark("cleanup")
        stop = stop_process(process)
        telemetry.stop()
        log_handle.close()
        after = gpu_snapshot()
        json_dump(phase / "stop-record.json", {"pid": process.pid if process else None, **stop})
        json_dump(phase / "gpu-after.json", after)
        (phase / "ports-after.txt").write_text(port_snapshot(), encoding="utf-8")
        (phase / "processes-after.txt").write_text(process_snapshot(), encoding="utf-8")
        status["cleanup"] = {
            "gpu_after": after,
            "process_gpu_proof_after": process_gpu_proof(after, [process.pid] if process else []),
            "port_free": port_is_free(SHARED_PORT),
            "llama_server_lines_after": [line for line in process_snapshot().splitlines() if "llama-server" in line],
            "systemd_used": False,
            "production_interaction": False,
        }
        json_dump(phase / "phase-status.json", status)


def write_summary(root: Path, independent: dict[str, Any], shared: dict[str, Any], before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    independent_ok = independent.get("gate") == "validated_independent_dual_gpu"
    shared_ok = shared.get("gate") == "validated_shared_weight_parallel2"
    shared_oom = shared.get("gate") == "oom"
    if independent_ok and shared_ok:
        conclusion = "yes: Bonsai provides both independent dual-GPU and shared-weight multi-context concurrency; the T8 contract regression remains a quality limitation."
    elif independent_ok and shared_oom:
        conclusion = "yes for independent reviewer/subagent concurrency across GPUs 1 and 2; shared-weight parallel-2 packing OOMed, and the T8 contract regression remains a quality limitation."
    elif independent_ok:
        conclusion = "yes for independent dual-GPU reviewer/subagent concurrency, but shared-weight packing was not validated; the T8 contract regression remains a quality limitation."
    else:
        conclusion = "no useful topology result was validated because the independent dual-GPU phase did not complete."
    summary = {
        "artifact": str(root.relative_to(ROOT)),
        "created_at_utc": utc_stamp(),
        "official_prism": {"binary": str(SERVER), "model": str(MODEL), "packing": "PQ2_0", "source_commit": "5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6"},
        "gpu_policy": {"allowed_physical_gpus": [1, 2], "forbidden_physical_gpus": [0], "cuda_visible_devices": {"gpu1_server": "1", "gpu2_server": "2", "shared_server": "1"}, "server_device_argument": "CUDA0", "systemd_used": False, "production_ports_or_services": False},
        "ports": {"gpu1": PORTS[1], "gpu2": PORTS[2], "shared": SHARED_PORT},
        "independent_depths": list(DEPTHS),
        "fixed_output_tokens": OUTPUT_TOKENS,
        "independent_dual_gpu": independent,
        "shared_weight_packing": shared,
        "t8_output_contract_reference": "research/qwen3.8-27b-serving-alternatives-phase2-quality-followup-20260918T173610Z/followup-summary.json",
        "t8_interpretation": "The prior PQ2_0 T8 run ended with finish_reason=length and malformed/truncated JSON at max_tokens=2048; topology success does not remove that reviewer quality caveat.",
        "gpu0_proof_before": process_gpu_proof(before["gpu"], []),
        "gpu0_proof_after": process_gpu_proof(after["gpu"], []),
        "cleanup": {"ports_free": {str(port): port_is_free(port) for port in (*PORTS.values(), SHARED_PORT)}, "llama_server_lines_after": [line for line in after.get("processes", "").splitlines() if "llama-server" in line]},
        "conclusion": conclusion,
    }
    json_dump(root / "topology-summary.json", summary)
    lines = [
        "# Bonsai Phase 2 Topology",
        "",
        f"Conclusion: **{conclusion}**",
        "",
        "Official Prism PQ2_0 servers used only physical GPUs 1 and 2. GPU0 was observed for proof only; no systemd or production service was used.",
        "",
        "| Phase | Result |",
        "|---|---|",
        f"| Independent GPU1/GPU2, 64k and 196k, cold/warm | {independent.get('gate')} |",
        f"| Shared GPU1, parallel=2, two 8k requests | {shared.get('gate')} |",
        "",
        "Interference is reported per case as `cross_gpu_interference_factor = aggregate_wall / max(instance_wall)` and `overlap_efficiency = sum(instance_wall) / aggregate_wall`; values near 1 and 2 respectively indicate successful overlap rather than serialization.",
        "",
        "The prior T8 PQ2_0 result remains a contract regression: truncated output and malformed JSON at the 2048-token cap. See `topology-summary.json` and the referenced follow-up artifact for complete evidence.",
    ]
    (root / "topology-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    required = [SERVER, MODEL]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print("required official Prism artifact missing: " + ", ".join(missing), file=sys.stderr)
        return 2
    occupied = [port for port in (*PORTS.values(), SHARED_PORT) if not port_is_free(port)]
    if occupied:
        print("isolated topology port(s) already in use: " + ", ".join(str(port) for port in occupied), file=sys.stderr)
        return 2
    root = RESEARCH / f"qwen3.8-27b-serving-alternatives-phase2-topology-{utc_stamp()}"
    root.mkdir(parents=True, exist_ok=False)
    before = {"gpu": gpu_snapshot(), "host": host_snapshot(), "ports": port_snapshot(), "processes": process_snapshot()}
    json_dump(root / "host-and-gpu-before.json", before)
    json_dump(root / "preflight.json", {"ports_free": {str(port): port_is_free(port) for port in (*PORTS.values(), SHARED_PORT)}, "gpu_policy": {"allowed": [1, 2], "forbidden": [0]}, "systemd_used": False, "production_interaction": False, **before})
    json_dump(root / "hashes.json", {"server": {"path": str(SERVER), "sha256": sha256_file(SERVER), "size": SERVER.stat().st_size}, "model": {"path": str(MODEL), "sha256": sha256_file(MODEL), "size": MODEL.stat().st_size}, "source_commit": command_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], timeout=30)})
    (root / "invocation.txt").write_text(f"cwd={ROOT}\ncommand=python3 {Path(__file__).relative_to(ROOT)}\nstarted_at_utc={utc_stamp()}\n", encoding="utf-8")
    header, corpus = make_corpus()
    json_dump(root / "matrix-config.json", {"packing": "PQ2_0", "model": str(MODEL), "server": str(SERVER), "independent": {"physical_gpus": [1, 2], "ports": PORTS, "ctx_size": CONTEXT_SIZE, "parallel": 1, "depths": DEPTHS, "output_tokens": OUTPUT_TOKENS}, "shared": {"physical_gpu": 1, "port": SHARED_PORT, "ctx_size": CONTEXT_SIZE, "parallel": 2, "depth": SHARED_DEPTH, "output_tokens": OUTPUT_TOKENS}, "prompt_generation": "deterministic populated synthetic repository records", "seed": SEED})
    independent = run_independent_phase(root, header, corpus, before)
    shared = run_shared_phase(root, header, corpus, before)
    after = {"gpu": gpu_snapshot(), "host": host_snapshot(), "ports": port_snapshot(), "processes": process_snapshot()}
    json_dump(root / "host-and-gpu-after.json", after)
    summary = write_summary(root, independent, shared, before, after)
    print(json.dumps({"artifact": str(root), "conclusion": summary["conclusion"], "independent_gate": independent.get("gate"), "shared_gate": shared.get("gate")}, indent=2, sort_keys=True))
    return 0 if independent.get("gate") == "validated_independent_dual_gpu" else 1


if __name__ == "__main__":
    raise SystemExit(main())
