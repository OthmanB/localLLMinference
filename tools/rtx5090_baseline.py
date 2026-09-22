#!/usr/bin/env python3
"""Run a guarded, one-shot RTX 5090 llama.cpp baseline measurement."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request


GPU_FIELDS = (
    "index",
    "uuid",
    "name",
    "pci.bus_id",
    "power.limit",
    "power.draw",
    "fan.speed",
    "temperature.gpu",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "clocks.sm",
)
MANAGED_MEMORY_VARIABLE = "GGML_CUDA_ENABLE_UNIFIED_MEMORY"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def http_json(url: str, payload: dict | None = None, timeout: float = 30) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError(f"unexpected JSON object from {url}")
    return value


def gpu_snapshot(gpu: int) -> dict[str, int | float | str | None]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            f"--query-gpu={','.join(GPU_FIELDS)}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    rows = list(csv.reader(io.StringIO(output)))
    if len(rows) != 1 or len(rows[0]) != len(GPU_FIELDS):
        raise RuntimeError(f"unexpected nvidia-smi output for GPU {gpu}")
    row = rows[0]
    result: dict[str, int | float | str | None] = {}
    for index, field in enumerate(GPU_FIELDS):
        value = row[index].strip()
        if field in {"name", "uuid", "pci.bus_id"}:
            result[field.replace(".", "_")] = value
        elif value in {"", "N/A", "Not Supported"}:
            result[field.replace(".", "_")] = None
        else:
            number = float(value)
            result[field.replace(".", "_")] = int(number) if number.is_integer() else number
    return result


def host_snapshot() -> dict[str, object]:
    load = os.getloadavg()
    values: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                key, separator, value = line.partition(":")
                fields = value.split()
                if separator and fields and fields[0].isdigit():
                    values[key] = int(fields[0])
    except OSError:
        pass
    return {
        "load1": load[0],
        "load5": load[1],
        "load15": load[2],
        "mem_total_kib": values.get("MemTotal"),
        "mem_available_kib": values.get("MemAvailable"),
        "swap_total_kib": values.get("SwapTotal"),
        "swap_free_kib": values.get("SwapFree"),
    }


def wait_ready(port: int, alias: str, process: subprocess.Popen[str], timeout: int) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup with code {process.returncode}")
        try:
            models = http_json(f"http://127.0.0.1:{port}/v1/models", timeout=5)
            data = models.get("data")
            if isinstance(data, list) and any(item.get("id") == alias for item in data if isinstance(item, dict)):
                return models
        except (OSError, RuntimeError, ValueError, urllib.error.HTTPError, urllib.error.URLError):
            pass
        time.sleep(1)
    raise TimeoutError(f"server on port {port} did not become ready as {alias}")


def tokenize(port: int, content: str) -> int:
    response = http_json(f"http://127.0.0.1:{port}/tokenize", {"content": content}, timeout=180)
    tokens = response.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError(f"unexpected /tokenize response: {response.keys()}")
    return len(tokens)


def build_prompt(port: int, target_tokens: int) -> tuple[str, int]:
    prefix = f"RTX 5090 native-context baseline with unique marker rtx5090-{target_tokens}.\n"
    pattern = "Neutral repeated context text for a deterministic inference speed measurement. "
    pattern_tokens = max(1, tokenize(port, pattern))
    repeats = max(1, math.ceil((target_tokens - tokenize(port, prefix)) / pattern_tokens))
    content = prefix + pattern * repeats
    for _ in range(10):
        count = tokenize(port, content)
        difference = target_tokens - count
        if abs(difference) <= 16:
            return content, count
        repeats = max(1, repeats + round(difference / pattern_tokens))
        content = prefix + pattern * repeats
    return content, tokenize(port, content)


def response_text(response: dict) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("completion response has no choices")
    message = choices[0].get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    if isinstance(choices[0].get("text"), str):
        return choices[0]["text"]
    raise RuntimeError("completion response has no text content")


def terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def process_environment(pid: int) -> dict[str, object]:
    try:
        with open(f"/proc/{pid}/environ", "rb") as stream:
            entries = stream.read().split(b"\0")
    except OSError as error:
        return {"read_error": str(error), "managed_memory_variable_present": None}
    environment = [entry.decode("utf-8", "replace") for entry in entries if entry]
    return {
        "managed_memory_variable_present": any(
            entry.startswith(f"{MANAGED_MEMORY_VARIABLE}=") for entry in environment
        ),
        "cuda_visible_devices": next(
            (entry.split("=", 1)[1] for entry in environment if entry.startswith("CUDA_VISIBLE_DEVICES=")),
            None,
        ),
        "cuda_device_order": next(
            (entry.split("=", 1)[1] for entry in environment if entry.startswith("CUDA_DEVICE_ORDER=")),
            None,
        ),
    }


def assert_gpu_idle(gpu: int) -> None:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
        timeout=10,
    )
    if output.strip():
        raise RuntimeError(f"GPU {gpu} has active compute processes: {output.strip()}")


def assert_port_free(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(1)
        if connection.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"TCP port {port} is already listening on loopback")


def build_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.llama),
        "--model",
        str(args.model),
        "--alias",
        args.alias,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--ctx-size",
        str(args.ctx_size),
        "--parallel",
        "1",
        "--n-gpu-layers",
        "all",
        "--kv-offload",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--flash-attn",
        "on",
        "--threads",
        "32",
        "--threads-batch",
        "32",
        "--batch-size",
        "2048",
        "--ubatch-size",
        "512",
        "--fit",
        "off",
        "--metrics",
        "--perf",
        "--no-ui",
        "--reasoning",
        "auto",
        "--reasoning-effort",
        "medium",
    ]


def run(args: argparse.Namespace) -> dict[str, object]:
    assert_gpu_idle(args.gpu)
    assert_port_free(args.port)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "baseline.server.log"
    result: dict[str, object] = {
        "status": "error",
        "started_at": now(),
        "finished_at": None,
        "gpu": args.gpu,
        "port": args.port,
        "alias": args.alias,
        "context_tokens": args.ctx_size,
        "prompt_target_tokens": args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "thermal_stop_c": args.thermal_stop_c,
        "thermal_stop_triggered": False,
        "llama": str(args.llama),
        "model": str(args.model),
        "llama_sha256": hashlib.sha256(args.llama.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "command": build_command(args),
        "server_log": str(log_path),
        "host_before": host_snapshot(),
        "gpu_before": gpu_snapshot(args.gpu),
        "samples": [],
    }
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    environment.pop(MANAGED_MEMORY_VARIABLE, None)
    process: subprocess.Popen[str] | None = None
    log = log_path.open("w", encoding="utf-8")
    samples: list[dict[str, object]] = []
    stop_sampling = threading.Event()

    def sample() -> None:
        while not stop_sampling.wait(1):
            try:
                gpu = gpu_snapshot(args.gpu)
                samples.append(
                    {
                        "timestamp": now(),
                        "gpu": gpu,
                        "host": host_snapshot(),
                    }
                )
                temperature = gpu.get("temperature_gpu")
                if isinstance(temperature, (int, float)) and temperature >= args.thermal_stop_c:
                    result["thermal_stop_triggered"] = True
                    terminate(process)
                    return
            except (OSError, RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
                pass

    sampler = threading.Thread(target=sample, daemon=True)
    try:
        process = subprocess.Popen(
            build_command(args),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        result["pid"] = process.pid
        result["models_response"] = wait_ready(args.port, args.alias, process, args.startup_timeout)
        result["process_environment"] = process_environment(process.pid)
        prompt, prompt_tokens = build_prompt(args.port, args.prompt_tokens)
        result["prompt_content_tokens"] = prompt_tokens
        result["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        sampler.start()
        started = time.monotonic()
        response = http_json(
            f"http://127.0.0.1:{args.port}/v1/chat/completions",
            {
                "model": args.alias,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "presence_penalty": 1.5,
                "ignore_eos": True,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "reasoning_format": "none",
            },
            timeout=args.request_timeout,
        )
        timings = response.get("timings", {})
        predicted_n = float(timings.get("predicted_n", 0) or 0)
        predicted_ms = float(timings.get("predicted_ms", 0) or 0)
        prompt_n = float(timings.get("prompt_n", 0) or 0)
        prompt_ms = float(timings.get("prompt_ms", 0) or 0)
        result.update(
            {
                "request_seconds": time.monotonic() - started,
                "usage": response.get("usage", {}),
                "timings": timings,
                "decode_tok_s": predicted_n / (predicted_ms / 1000) if predicted_ms else None,
                "prefill_tok_s": prompt_n / (prompt_ms / 1000) if prompt_ms else None,
                "completion_sha256": hashlib.sha256(response_text(response).encode()).hexdigest(),
                "status": "passed",
            }
        )
    except (OSError, RuntimeError, TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        stop_sampling.set()
        sampler.join(timeout=3)
        if process is not None:
            terminate(process)
            result["exit_code"] = process.returncode
        log.close()
        result["samples"] = samples
        result["sample_count"] = len(samples)
        result["gpu_after"] = gpu_snapshot(args.gpu)
        result["host_after"] = host_snapshot()
        result["finished_at"] = now()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--alias", default="qwen3.8-27b-q4-gpukv-native-rtx5090-baseline")
    parser.add_argument("--ctx-size", type=int, default=262144)
    parser.add_argument("--prompt-tokens", type=int, default=261765)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--thermal-stop-c", type=float, default=85.0)
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--request-timeout", type=int, default=1800)
    args = parser.parse_args()

    if args.gpu < 0 or args.port < 1 or args.port > 65535 or args.thermal_stop_c <= 0:
        parser.error("--gpu must be non-negative, --port must be 1..65535, and thermal stop must be positive")
    if args.prompt_tokens + args.max_tokens > args.ctx_size:
        parser.error("prompt plus output exceeds context size")
    if not args.llama.is_file() or not os.access(args.llama, os.X_OK):
        parser.error(f"llama-server is not executable: {args.llama}")
    if not args.model.is_file() or not os.access(args.model, os.R_OK):
        parser.error(f"model is not readable: {args.model}")

    result = run(args)
    output = args.output_dir / "results.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
