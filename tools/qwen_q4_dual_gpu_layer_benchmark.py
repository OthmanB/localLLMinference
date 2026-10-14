#!/usr/bin/env python3
"""Benchmark Qwen3.8-27B Q4 across supported one- and two-GPU split modes."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request


ROOT = Path((os.environ['EXPERIMENT_ROOT'] + ''))
LLAMA = Path((os.environ['RUNTIME_ROOT'] + '/llama.cpp-recent-20260905/build/bin/llama-server'))
MODEL = ROOT / "models/qwen3.8-27b-q4-k-m-gguf/Qwen3.8-27B-UD-Q4_K_M.gguf"
CPU_POWER = re.compile(r"^ai_cpu_power_watts\{[^}]*\}\s+([0-9.eE+-]+)$")
GPU_QUERY = "index,power.limit,power.draw,utilization.gpu,temperature.gpu,memory.used,memory.total,clocks.current.graphics"
MODES = {
    "baseline": {
        "visible_devices": "1",
        "devices": "CUDA0",
        "physical_gpus": ("1",),
        "split_mode": "none",
    },
    "layer": {
        "visible_devices": "1,2",
        "devices": "CUDA0,CUDA1",
        "physical_gpus": ("1", "2"),
        "split_mode": "layer",
        "tensor_split": "1,1",
    },
    "tensor": {
        "visible_devices": "1,2",
        "devices": "CUDA0,CUDA1",
        "physical_gpus": ("1", "2"),
        "split_mode": "tensor",
        "tensor_split": "1,1",
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "flash_attn": "on",
    },
}


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


def wait_ready(port: int, process: subprocess.Popen[str], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup with code {process.returncode}")
        try:
            models = http_json(f"http://127.0.0.1:{port}/v1/models", timeout=5)
            if models.get("data"):
                return
        except (OSError, RuntimeError, ValueError):
            pass
        time.sleep(1)
    raise TimeoutError(f"server on port {port} did not become ready")


def tokenize(port: int, content: str) -> int:
    response = http_json(f"http://127.0.0.1:{port}/tokenize", {"content": content}, timeout=180)
    tokens = response.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError(f"unexpected /tokenize response: {response.keys()}")
    return len(tokens)


def build_prompt(port: int, target_tokens: int) -> tuple[str, int]:
    prefix = f"Long context throughput probe with unique marker q4-layer-{target_tokens}.\n"
    pattern = "Neutral repeated context text for a deterministic inference speed measurement. "
    pattern_tokens = tokenize(port, pattern)
    repeats = max(1, math.ceil((target_tokens - tokenize(port, prefix)) / pattern_tokens))
    content = prefix + pattern * repeats
    for _ in range(8):
        count = tokenize(port, content)
        difference = target_tokens - count
        if abs(difference) <= 16:
            return content, count
        repeats = max(1, repeats + round(difference / pattern_tokens))
        content = prefix + pattern * repeats
    return content, tokenize(port, content)


def request_payload(prompt: str, max_tokens: int) -> dict:
    return {
        "model": "qwen3.8-27b-q4-dual-gpu-benchmark",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 1.5,
        "ignore_eos": True,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_format": "none",
    }


def timing_rates(timings: object) -> tuple[float | None, float | None]:
    if not isinstance(timings, dict):
        return None, None
    predicted_n = float(timings.get("predicted_n", 0) or 0)
    predicted_ms = float(timings.get("predicted_ms", 0) or 0)
    prompt_n = float(timings.get("prompt_n", 0) or 0)
    prompt_ms = float(timings.get("prompt_ms", 0) or 0)
    return (
        predicted_n / (predicted_ms / 1000) if predicted_ms else None,
        prompt_n / (prompt_ms / 1000) if prompt_ms else None,
    )


def request_measure(port: int, prompt: str, max_tokens: int, request_timeout: int) -> dict[str, object]:
    started = time.monotonic()
    response = http_json(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        request_payload(prompt, max_tokens),
        timeout=request_timeout,
    )
    finished = time.monotonic()
    timings = response.get("timings", {})
    usage = response.get("usage", {})
    cache_details = usage.get("prompt_tokens_details", {}) if isinstance(usage, dict) else {}
    result: dict[str, object] = {
        "request_started_monotonic": started,
        "request_finished_monotonic": finished,
        "request_seconds": finished - started,
        "usage": usage,
        "timings": timings,
        "decode_tok_s": timing_rates(timings)[0],
        "prefill_tok_s": timing_rates(timings)[1],
        "cached_tokens": cache_details.get("cached_tokens") if isinstance(cache_details, dict) else None,
    }
    return result


def required_number(values: dict[str, object], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)):
        raise RuntimeError(f"missing numeric {key}")
    return float(value)


def required_mapping(values: dict[str, object], key: str) -> dict[str, object]:
    value = values.get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"missing mapping {key}")
    return value


def server_command(
    mode: str,
    ctx_size: int,
    port: int,
    parallel: int,
    llama: Path,
    model: Path,
    batch_size: int,
    ubatch_size: int,
    cache_type_k: str | None,
    cache_type_v: str | None,
) -> tuple[list[str], dict[str, str]]:
    spec = MODES[mode]
    effective_cache_type_k = cache_type_k or str(spec.get("cache_type_k", "q8_0"))
    effective_cache_type_v = cache_type_v or str(spec.get("cache_type_v", "q8_0"))
    command = [
        str(llama),
        "--model",
        str(model),
        "--alias",
        "qwen3.8-27b-q4-dual-gpu-benchmark",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--device",
        str(spec["devices"]),
        "--split-mode",
        str(spec["split_mode"]),
        "--n-gpu-layers",
        "all",
        "--ctx-size",
        str(ctx_size),
        "--parallel",
        str(parallel),
        "--kv-offload",
        "--cache-type-k",
        effective_cache_type_k,
        "--cache-type-v",
        effective_cache_type_v,
        "--flash-attn",
        str(spec.get("flash_attn", "on")),
        "--threads",
        "32",
        "--threads-batch",
        "32",
        "--batch-size",
        str(batch_size),
        "--ubatch-size",
        str(ubatch_size),
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
    if "tensor_split" in spec:
        command.extend(["--tensor-split", str(spec["tensor_split"])])
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(spec["visible_devices"]),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    }
    # llama.cpp enables UVM when this variable exists, even when set to "0".
    environment.pop("GGML_CUDA_ENABLE_UNIFIED_MEMORY", None)
    return command, environment


def gpu_snapshot(selected_gpus: tuple[str, ...]) -> dict[str, dict[str, float]]:
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
        text=True,
    )
    snapshots: dict[str, dict[str, float]] = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 8 or fields[0] not in selected_gpus:
            continue
        snapshots[fields[0]] = {
            "power_limit_w": float(fields[1]),
            "power_draw_w": float(fields[2]),
            "utilization_gpu_percent": float(fields[3]),
            "temperature_c": float(fields[4]),
            "memory_used_mib": float(fields[5]),
            "memory_total_mib": float(fields[6]),
            "graphics_clock_mhz": float(fields[7]),
        }
    if set(snapshots) != set(selected_gpus):
        raise RuntimeError(f"could not sample all selected GPUs: expected {selected_gpus}, got {snapshots}")
    return snapshots


def cpu_power_watts() -> float | None:
    try:
        with urllib.request.urlopen("http://127.0.0.1:9109/metrics", timeout=3) as response:
            for line in response.read().decode("utf-8").splitlines():
                match = CPU_POWER.match(line)
                if match:
                    return float(match.group(1))
    except (OSError, ValueError):
        return None
    return None


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


def energy_joules(samples: list[dict], path: tuple[str, ...], started: float, finished: float) -> float | None:
    points: list[tuple[float, float]] = []
    for sample in samples:
        timestamp = float(sample["monotonic_seconds"])
        if timestamp < started or timestamp > finished:
            continue
        value: object = sample
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if isinstance(value, (int, float)):
            points.append((timestamp, float(value)))
    if len(points) < 2:
        return None
    return sum((before[1] + after[1]) * 0.5 * (after[0] - before[0]) for before, after in zip(points, points[1:]))


def sample_summary(samples: list[dict], gpu: str, started: float, finished: float) -> dict[str, float | None]:
    in_request = [sample for sample in samples if started <= sample["monotonic_seconds"] <= finished]
    gpu_rows = [sample["gpus"].get(gpu) for sample in in_request if gpu in sample["gpus"]]
    gpu_rows = [row for row in gpu_rows if isinstance(row, dict)]
    energy = energy_joules(samples, ("gpus", gpu, "power_draw_w"), started, finished)
    duration = finished - started
    return {
        "request_energy_joules": energy,
        "request_average_power_w": energy / duration if energy is not None and duration > 0 else None,
        "max_power_draw_w": max((float(row["power_draw_w"]) for row in gpu_rows), default=None),
        "max_temperature_c": max((float(row["temperature_c"]) for row in gpu_rows), default=None),
        "max_memory_used_mib": max((float(row["memory_used_mib"]) for row in gpu_rows), default=None),
        "min_graphics_clock_mhz": min((float(row["graphics_clock_mhz"]) for row in gpu_rows), default=None),
        "max_graphics_clock_mhz": max((float(row["graphics_clock_mhz"]) for row in gpu_rows), default=None),
    }


def measure(
    mode: str,
    target_tokens: int,
    ctx_size: int,
    port: int,
    output_dir: Path,
    request_timeout: int,
    llama: Path,
    model: Path,
    max_tokens: int,
    batch_size: int,
    ubatch_size: int,
    cache_type_k: str | None,
    cache_type_v: str | None,
    parallel: int = 1,
) -> dict:
    spec = MODES[mode]
    log_path = output_dir / f"q4-{mode}-ctx{ctx_size}-prompt{target_tokens}.server.log"
    samples_path = output_dir / f"q4-{mode}-ctx{ctx_size}-prompt{target_tokens}.samples.json"
    result: dict[str, object] = {
        "mode": mode,
        "context_tokens": ctx_size,
        "parallel_slots": parallel,
        "max_tokens": max_tokens,
        "batch_size": batch_size,
        "ubatch_size": ubatch_size,
        "cache_type_k": cache_type_k or str(spec.get("cache_type_k", "q8_0")),
        "cache_type_v": cache_type_v or str(spec.get("cache_type_v", "q8_0")),
        "model": str(model),
        "prompt_target_tokens": target_tokens,
        "physical_gpus": spec["physical_gpus"],
        "started_at": now(),
        "status": "error",
        "server_log": str(log_path),
        "samples": str(samples_path),
    }
    command, environment = server_command(
        mode, ctx_size, port, parallel, llama, model, batch_size, ubatch_size, cache_type_k, cache_type_v
    )
    result["command"] = command
    result["cuda_visible_devices"] = environment["CUDA_VISIBLE_DEVICES"]
    samples: list[dict] = []
    stop_sampling = threading.Event()
    process: subprocess.Popen[str] | None = None
    log = log_path.open("w", encoding="utf-8")

    def sample() -> None:
        while not stop_sampling.wait(1):
            try:
                samples.append(
                    {
                        "monotonic_seconds": time.monotonic(),
                        "gpus": gpu_snapshot(spec["physical_gpus"]),
                        "cpu_power_watts": cpu_power_watts(),
                    }
                )
            except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError):
                pass

    sampler = threading.Thread(target=sample, daemon=True)
    try:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        sampler.start()
        wait_ready(port, process, timeout=300)
        prompt, prompt_tokens = build_prompt(port, target_tokens)
        result["prompt_content_tokens"] = prompt_tokens
        result["startup_gpus"] = gpu_snapshot(spec["physical_gpus"])
        request_started = time.monotonic()
        response = http_json(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            request_payload(prompt, max_tokens),
            timeout=request_timeout,
        )
        request_finished = time.monotonic()
        result["request_seconds"] = request_finished - request_started
        result["usage"] = response.get("usage", {})
        result["timings"] = response.get("timings", {})
        result["decode_tok_s"], result["prefill_tok_s"] = timing_rates(result["timings"])
        result["gpu_request_summary"] = {
            gpu: sample_summary(samples, gpu, request_started, request_finished)
            for gpu in spec["physical_gpus"]
        }
        cpu_energy = energy_joules(samples, ("cpu_power_watts",), request_started, request_finished)
        result["cpu_request_energy_joules"] = cpu_energy
        result["total_request_energy_joules"] = (
            (cpu_energy or 0.0)
            + sum(
                summary["request_energy_joules"] or 0.0
                for summary in result["gpu_request_summary"].values()
            )
        )
        result["status"] = "passed"
    except (OSError, RuntimeError, TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        stop_sampling.set()
        sampler.join(timeout=3)
        if process is not None:
            terminate(process)
            result["exit_code"] = process.returncode
        samples_path.write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")
        log.close()
        result["sample_count"] = len(samples)
        result["finished_at"] = now()
    return result


def measure_concurrent(
    mode: str,
    target_tokens: tuple[int, int],
    ctx_size: int,
    port: int,
    output_dir: Path,
    request_timeout: int,
    llama: Path,
    model: Path,
    max_tokens: int,
    stagger_seconds: float,
    batch_size: int,
    ubatch_size: int,
    cache_type_k: str | None,
    cache_type_v: str | None,
) -> dict:
    spec = MODES[mode]
    targets = "-".join(str(tokens) for tokens in target_tokens)
    log_path = output_dir / f"q4-{mode}-ctx{ctx_size}-parallel2-prompts{targets}.server.log"
    samples_path = output_dir / f"q4-{mode}-ctx{ctx_size}-parallel2-prompts{targets}.samples.json"
    result: dict[str, object] = {
        "mode": mode,
        "context_tokens": ctx_size,
        "parallel_slots": 2,
        "max_tokens": max_tokens,
        "batch_size": batch_size,
        "ubatch_size": ubatch_size,
        "cache_type_k": cache_type_k or str(spec.get("cache_type_k", "q8_0")),
        "cache_type_v": cache_type_v or str(spec.get("cache_type_v", "q8_0")),
        "model": str(model),
        "prompt_target_tokens": list(target_tokens),
        "physical_gpus": spec["physical_gpus"],
        "started_at": now(),
        "status": "error",
        "server_log": str(log_path),
        "samples": str(samples_path),
        "stagger_seconds": stagger_seconds,
    }
    command, environment = server_command(
        mode, ctx_size, port, 2, llama, model, batch_size, ubatch_size, cache_type_k, cache_type_v
    )
    result["command"] = command
    result["cuda_visible_devices"] = environment["CUDA_VISIBLE_DEVICES"]
    samples: list[dict] = []
    stop_sampling = threading.Event()
    process: subprocess.Popen[str] | None = None
    log = log_path.open("w", encoding="utf-8")

    def sample() -> None:
        while not stop_sampling.wait(1):
            try:
                samples.append(
                    {
                        "monotonic_seconds": time.monotonic(),
                        "gpus": gpu_snapshot(spec["physical_gpus"]),
                        "cpu_power_watts": cpu_power_watts(),
                    }
                )
            except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError):
                pass

    sampler = threading.Thread(target=sample, daemon=True)
    requests: list[dict[str, object]] = []
    request_lock = threading.Lock()
    start_requests = threading.Event()

    def run_request(index: int, prompt: str, prompt_tokens: int) -> None:
        start_requests.wait()
        if index and stagger_seconds:
            time.sleep(stagger_seconds)
        started = time.monotonic()
        item: dict[str, object] = {
            "slot": index,
            "prompt_target_tokens": target_tokens[index],
            "prompt_content_tokens": prompt_tokens,
            "request_started_monotonic": started,
        }
        try:
            response = http_json(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                request_payload(prompt, max_tokens),
                timeout=request_timeout,
            )
            finished = time.monotonic()
            item["request_finished_monotonic"] = finished
            item["request_seconds"] = finished - started
            item["usage"] = response.get("usage", {})
            item["timings"] = response.get("timings", {})
            item["decode_tok_s"], item["prefill_tok_s"] = timing_rates(item["timings"])
            item["status"] = "passed"
        except (OSError, RuntimeError, TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
            item["status"] = "error"
        with request_lock:
            requests.append(item)

    try:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        sampler.start()
        wait_ready(port, process, timeout=300)
        prompts = [build_prompt(port, tokens) for tokens in target_tokens]
        result["startup_gpus"] = gpu_snapshot(spec["physical_gpus"])
        workers = [
            threading.Thread(target=run_request, args=(index, prompt, prompt_tokens))
            for index, (prompt, prompt_tokens) in enumerate(prompts)
        ]
        for worker in workers:
            worker.start()
        start_requests.set()
        for worker in workers:
            worker.join()
        requests.sort(key=lambda item: int(required_number(item, "slot")))
        result["requests"] = requests
        if any(item["status"] != "passed" for item in requests):
            raise RuntimeError("one or more concurrent requests failed")
        started = min(required_number(item, "request_started_monotonic") for item in requests)
        finished = max(required_number(item, "request_finished_monotonic") for item in requests)
        result["request_seconds"] = finished - started
        result["gpu_request_summary"] = {
            gpu: sample_summary(samples, gpu, started, finished)
            for gpu in spec["physical_gpus"]
        }
        cpu_energy = energy_joules(samples, ("cpu_power_watts",), started, finished)
        result["cpu_request_energy_joules"] = cpu_energy
        result["total_request_energy_joules"] = (
            (cpu_energy or 0.0)
            + sum(
                summary["request_energy_joules"] or 0.0
                for summary in result["gpu_request_summary"].values()
            )
        )
        generation_started = math.inf
        generation_finished = 0.0
        completion_tokens = 0.0
        for item in requests:
            request_finished = required_number(item, "request_finished_monotonic")
            timings = required_mapping(item, "timings")
            generation_started = min(
                generation_started,
                request_finished - required_number(timings, "predicted_ms") / 1000,
            )
            generation_finished = max(generation_finished, request_finished)
            completion_tokens += required_number(timings, "predicted_n")
        result["aggregate_decode_tok_s"] = completion_tokens / (generation_finished - generation_started)
        result["status"] = "passed"
    except (OSError, RuntimeError, TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        stop_sampling.set()
        sampler.join(timeout=3)
        if process is not None:
            terminate(process)
            result["exit_code"] = process.returncode
        samples_path.write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")
        log.close()
        result["sample_count"] = len(samples)
        result["finished_at"] = now()
    return result


def measure_warm_cache(
    mode: str,
    retained_prefix_tokens: int,
    tool_result_tokens: int,
    ctx_size: int,
    port: int,
    output_dir: Path,
    request_timeout: int,
    llama: Path,
    model: Path,
    max_tokens: int,
    batch_size: int,
    ubatch_size: int,
    cache_type_k: str | None,
    cache_type_v: str | None,
) -> dict:
    spec = MODES[mode]
    log_path = output_dir / f"q4-{mode}-ctx{ctx_size}-warm-prefix{retained_prefix_tokens}.server.log"
    samples_path = output_dir / f"q4-{mode}-ctx{ctx_size}-warm-prefix{retained_prefix_tokens}.samples.json"
    result: dict[str, object] = {
        "mode": mode,
        "context_tokens": ctx_size,
        "parallel_slots": 1,
        "max_tokens": max_tokens,
        "batch_size": batch_size,
        "ubatch_size": ubatch_size,
        "cache_type_k": cache_type_k or str(spec.get("cache_type_k", "q8_0")),
        "cache_type_v": cache_type_v or str(spec.get("cache_type_v", "q8_0")),
        "model": str(model),
        "retained_prefix_target_tokens": retained_prefix_tokens,
        "tool_result_target_tokens": tool_result_tokens,
        "physical_gpus": spec["physical_gpus"],
        "started_at": now(),
        "status": "error",
        "server_log": str(log_path),
        "samples": str(samples_path),
    }
    command, environment = server_command(
        mode, ctx_size, port, 1, llama, model, batch_size, ubatch_size, cache_type_k, cache_type_v
    )
    result["command"] = command
    result["cuda_visible_devices"] = environment["CUDA_VISIBLE_DEVICES"]
    samples: list[dict] = []
    stop_sampling = threading.Event()
    process: subprocess.Popen[str] | None = None
    log = log_path.open("w", encoding="utf-8")

    def sample() -> None:
        while not stop_sampling.wait(1):
            try:
                samples.append(
                    {
                        "monotonic_seconds": time.monotonic(),
                        "gpus": gpu_snapshot(spec["physical_gpus"]),
                        "cpu_power_watts": cpu_power_watts(),
                    }
                )
            except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError):
                pass

    sampler = threading.Thread(target=sample, daemon=True)
    try:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        sampler.start()
        wait_ready(port, process, timeout=300)
        retained_prefix, retained_prefix_count = build_prompt(port, retained_prefix_tokens)
        first_tool_result, first_tool_result_count = build_prompt(port, tool_result_tokens)
        second_tool_result, second_tool_result_count = build_prompt(port, tool_result_tokens)
        cold_prompt = retained_prefix + "\nTool result 1:\n" + first_tool_result
        warm_prompt = cold_prompt + "\nTool result 2:\n" + second_tool_result
        result["retained_prefix_content_tokens"] = retained_prefix_count
        result["cold_prompt_content_tokens"] = tokenize(port, cold_prompt)
        result["warm_prompt_content_tokens"] = tokenize(port, warm_prompt)
        result["startup_gpus"] = gpu_snapshot(spec["physical_gpus"])
        cold_request = request_measure(port, cold_prompt, max_tokens, request_timeout)
        warm_request = request_measure(port, warm_prompt, max_tokens, request_timeout)
        for name, request in (("cold_request", cold_request), ("warm_request", warm_request)):
            started = required_number(request, "request_started_monotonic")
            finished = required_number(request, "request_finished_monotonic")
            request["gpu_request_summary"] = {
                gpu: sample_summary(samples, gpu, started, finished)
                for gpu in spec["physical_gpus"]
            }
            cpu_energy = energy_joules(samples, ("cpu_power_watts",), started, finished)
            request["cpu_request_energy_joules"] = cpu_energy
            request["total_request_energy_joules"] = (
                (cpu_energy or 0.0)
                + sum(
                    summary["request_energy_joules"] or 0.0
                    for summary in request["gpu_request_summary"].values()
                )
            )
            result[name] = request
        result["status"] = "passed"
    except (OSError, RuntimeError, TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        stop_sampling.set()
        sampler.join(timeout=3)
        if process is not None:
            terminate(process)
            result["exit_code"] = process.returncode
        samples_path.write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")
        log.close()
        result["sample_count"] = len(samples)
        result["finished_at"] = now()
    return result


def save_results(path: Path, results: list[dict]) -> None:
    path.write_text(json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llama", type=Path, default=LLAMA)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--ctx-size", type=int, default=262144)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--concurrent-prompt-tokens", nargs=2, type=int)
    parser.add_argument("--concurrent-stagger-seconds", type=float, default=0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--cache-type-k")
    parser.add_argument("--cache-type-v")
    parser.add_argument("--warm-prefix-tokens", type=int)
    parser.add_argument("--tool-result-tokens", type=int, default=512)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--modes", nargs="+", choices=tuple(MODES), default=("baseline", "layer"))
    parser.add_argument("--prompt-tokens", nargs="+", type=int, default=(4096, 196608, 261760))
    args = parser.parse_args()

    if not args.llama.exists() or not args.model.exists():
        parser.error("llama-server or the Q4 GGUF is missing")
    if args.parallel < 1:
        parser.error("--parallel must be positive")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.batch_size < 1 or args.ubatch_size < 1:
        parser.error("--batch-size and --ubatch-size must be positive")
    if args.ubatch_size > args.batch_size:
        parser.error("--ubatch-size cannot exceed --batch-size")
    if (args.cache_type_k is None) != (args.cache_type_v is None):
        parser.error("--cache-type-k and --cache-type-v must be provided together")
    slot_context = args.ctx_size // args.parallel
    context_margin = args.max_tokens + 256
    if args.warm_prefix_tokens is not None:
        if args.parallel != 1:
            parser.error("warm-prefix measurements require --parallel 1")
        if args.warm_prefix_tokens < 1 or args.tool_result_tokens < 1:
            parser.error("warm-prefix and tool-result tokens must be positive")
        if args.warm_prefix_tokens + 2 * args.tool_result_tokens + context_margin > args.ctx_size:
            parser.error("warm-prefix request needs more context than --ctx-size")
    if args.warm_prefix_tokens is None:
        if args.concurrent_prompt_tokens is None:
            if any(tokens + context_margin > slot_context for tokens in args.prompt_tokens):
                parser.error(f"prompt target needs at least a {context_margin}-token context margin")
        else:
            if args.parallel != 2:
                parser.error("concurrent prompts require --parallel 2")
            if args.concurrent_stagger_seconds < 0:
                parser.error("concurrent stagger must be non-negative")
            if any(tokens + context_margin > slot_context for tokens in args.concurrent_prompt_tokens):
                parser.error(f"each concurrent prompt needs at least a {context_margin}-token slot margin")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    save_results(args.output, results)
    for mode in args.modes:
        if args.warm_prefix_tokens is not None:
            result = measure_warm_cache(
                mode,
                args.warm_prefix_tokens,
                args.tool_result_tokens,
                args.ctx_size,
                args.port,
                args.output.parent,
                args.request_timeout,
                args.llama,
                args.model,
                args.max_tokens,
                args.batch_size,
                args.ubatch_size,
                args.cache_type_k,
                args.cache_type_v,
            )
            results.append(result)
            save_results(args.output, results)
            print(json.dumps(result, sort_keys=True), flush=True)
            continue
        if args.concurrent_prompt_tokens is not None:
            result = measure_concurrent(
                mode,
                tuple(args.concurrent_prompt_tokens),
                args.ctx_size,
                args.port,
                args.output.parent,
                args.request_timeout,
                args.llama,
                args.model,
                args.max_tokens,
                args.concurrent_stagger_seconds,
                args.batch_size,
                args.ubatch_size,
                args.cache_type_k,
                args.cache_type_v,
            )
            results.append(result)
            save_results(args.output, results)
            print(json.dumps(result, sort_keys=True), flush=True)
            continue
        for target_tokens in args.prompt_tokens:
            result = measure(
                mode,
                target_tokens,
                args.ctx_size,
                args.port,
                args.output.parent,
                args.request_timeout,
                args.llama,
                args.model,
                args.max_tokens,
                args.batch_size,
                args.ubatch_size,
                args.cache_type_k,
                args.cache_type_v,
                args.parallel,
            )
            results.append(result)
            save_results(args.output, results)
            print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
