#!/usr/bin/env python3
"""Measure long-context llama.cpp throughput with a fresh server per point."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request


ROOT = Path(os.environ.get("LOCAL_LLM_ROOT", Path(__file__).resolve().parents[1]))
LLAMA = Path(os.environ.get("LLAMA_SERVER_BIN", "llama-server"))
MODELS = {
    "q4": {
        "gpu": "0",
        "port": 8080,
        "alias": "qwen3.8-27b-q4-context-sweep",
        "path": ROOT / "models/qwen3.8-27b-q4-k-m-gguf/Qwen3.8-27B-UD-Q4_K_M.gguf",
    },
    "q5": {
        "gpu": "1",
        "port": 8081,
        "alias": "qwen3.8-27b-q5-context-sweep",
        "path": ROOT / "models/qwen3.8-27b-q5-k-m-gguf/Qwen3.8-27B-UD-Q5_K_M.gguf",
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
        return json.load(response)


def wait_ready(port: int, process: subprocess.Popen, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup with code {process.returncode}")
        try:
            models = http_json(f"http://127.0.0.1:{port}/v1/models", timeout=5)
            if models.get("data"):
                return
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise TimeoutError(f"server on port {port} did not become ready")


def tokenize(port: int, content: str) -> int:
    response = http_json(f"http://127.0.0.1:{port}/tokenize", {"content": content}, timeout=180)
    tokens = response.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError(f"unexpected /tokenize response: {response.keys()}")
    return len(tokens)


def build_prompt(port: int, model: str, context: int) -> tuple[str, int]:
    target = context - 384
    prefix = f"Long context throughput probe for {model} at {context} tokens. Unique marker {model}-{context}.\n"
    pattern = "Neutral repeated context text for a deterministic inference speed measurement. "
    pattern_tokens = tokenize(port, pattern)
    repeats = max(1, math.ceil(target / pattern_tokens))
    content = prefix + pattern * repeats
    for _ in range(8):
        count = tokenize(port, content)
        difference = target - count
        if abs(difference) <= 16:
            return content, count
        repeats = max(1, repeats + round(difference / pattern_tokens))
        content = prefix + pattern * repeats
    return content, tokenize(port, content)


def gpu_snapshot(gpu: str) -> dict:
    query = "index,power.limit,power.draw,utilization.gpu,temperature.gpu,memory.used,memory.total"
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        text=True,
    )
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields and fields[0] == gpu:
            return {
                "index": int(fields[0]),
                "power_limit_w": float(fields[1]),
                "power_draw_w": float(fields[2]),
                "utilization_gpu_percent": float(fields[3]),
                "temperature_c": float(fields[4]),
                "memory_used_mib": float(fields[5]),
                "memory_total_mib": float(fields[6]),
            }
    raise RuntimeError(f"GPU {gpu} not found in nvidia-smi output")


def terminate(process: subprocess.Popen) -> None:
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


def measure(model: str, context: int, output_dir: Path, request_timeout: int) -> dict:
    spec = MODELS[model]
    port = spec["port"]
    log_path = output_dir / f"{model}-{context}.server.log"
    result = {
        "model": model,
        "context_tokens": context,
        "gpu": spec["gpu"],
        "port": port,
        "started_at": now(),
        "status": "error",
    }
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            str(LLAMA),
            "--model",
            str(spec["path"]),
            "--alias",
            spec["alias"],
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--ctx-size",
            str(context),
            "--parallel",
            "1",
            "--n-gpu-layers",
            "99",
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
        ],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": spec["gpu"],
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        },
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    samples: list[dict] = []
    stop_sampling = threading.Event()

    def sample() -> None:
        while not stop_sampling.wait(1):
            try:
                samples.append(gpu_snapshot(spec["gpu"]))
            except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError):
                pass

    sampler = threading.Thread(target=sample, daemon=True)
    try:
        wait_ready(port, process, 240)
        content, content_tokens = build_prompt(port, model, context)
        result["prompt_target_tokens"] = context - 384
        result["prompt_content_tokens"] = content_tokens
        result["startup_snapshot"] = gpu_snapshot(spec["gpu"])
        payload = {
            "model": spec["alias"],
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "presence_penalty": 1.5,
            "ignore_eos": True,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_format": "none",
        }
        sampler.start()
        request_started = time.monotonic()
        response = http_json(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            payload,
            timeout=request_timeout,
        )
        request_seconds = time.monotonic() - request_started
        result["request_seconds"] = request_seconds
        result["response_id"] = response.get("id")
        result["usage"] = response.get("usage", {})
        result["timings"] = response.get("timings", {})
        timings = result["timings"]
        predicted_n = float(timings.get("predicted_n", 0) or 0)
        predicted_ms = float(timings.get("predicted_ms", 0) or 0)
        prompt_n = float(timings.get("prompt_n", 0) or 0)
        prompt_ms = float(timings.get("prompt_ms", 0) or 0)
        result["decode_tok_s"] = predicted_n / (predicted_ms / 1000) if predicted_ms else None
        result["prefill_tok_s"] = prompt_n / (prompt_ms / 1000) if prompt_ms else None
        result["status"] = "passed"
    except (OSError, RuntimeError, TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        stop_sampling.set()
        sampler.join(timeout=3)
        if samples:
            result["sample_count"] = len(samples)
            result["max_power_draw_w"] = max(sample["power_draw_w"] for sample in samples)
            result["max_temperature_c"] = max(sample["temperature_c"] for sample in samples)
            result["max_memory_used_mib"] = max(sample["memory_used_mib"] for sample in samples)
        terminate(process)
        log.close()
        result["exit_code"] = process.returncode
        result["finished_at"] = now()
        result["server_log"] = str(log_path)
    return result


def save_results(path: Path, results: list[dict]) -> None:
    path.write_text(json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-timeout", type=int, default=1800)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    save_results(args.output, results)
    for model in ("q4", "q5"):
        for context in (196608, 262144):
            result = measure(model, context, args.output.parent, args.request_timeout)
            results.append(result)
            save_results(args.output, results)
            print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
