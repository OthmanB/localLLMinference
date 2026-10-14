#!/usr/bin/env python3
"""Run one guarded, uncached long-context probe against a FreeToken server."""

from __future__ import annotations

import os

import argparse
import json
from pathlib import Path
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid

from transformers import AutoTokenizer  # type: ignore[import-not-found]


DEFAULT_MODEL_PATH = Path(
    (os.environ['EXPERIMENT_ROOT'] + '/models/qwen3.8-flash-next-nvfp4')
)
DEFAULT_CGROUP = Path(
    "/sys/fs/cgroup/system.slice/freetoken-flash-next-candidate.service"
)


def http_json(url: str, payload: dict | None = None, timeout: float = 30) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {body}") from error


def read_int(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="ascii").strip()
        return None if value == "max" else int(value)
    except (OSError, ValueError):
        return None


def gpu_snapshot(gpu: int) -> dict:
    query = "memory.used,power.draw,temperature.gpu,utilization.gpu"
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
            "-i",
            str(gpu),
        ],
        text=True,
    )
    memory, power, temperature, utilization = (
        field.strip() for field in output.strip().split(",")
    )
    return {
        "memory_used_mib": float(memory),
        "power_draw_w": float(power),
        "temperature_c": float(temperature),
        "utilization_percent": float(utilization),
    }


def chat_token_count(tokenizer, content: str) -> int:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    return len(encoded)


def build_content(tokenizer, target_tokens: int) -> tuple[str, int, str]:
    marker = f"FT-{uuid.uuid4().hex.upper()}"
    prefix = f"Long-context runtime probe. Start marker: {marker}.\n"
    pattern = "Neutral deterministic context for sparse-attention throughput measurement.\n"
    suffix = (
        f"\nEnd marker: {marker}. Return the marker first, then repeat the word probe "
        "until the output limit."
    )
    fixed = chat_token_count(tokenizer, prefix + suffix)
    pattern_tokens = len(tokenizer.encode(pattern, add_special_tokens=False))
    repeats = max(1, (target_tokens - fixed) // pattern_tokens)
    content = prefix + pattern * repeats + suffix
    count = 0

    for _ in range(8):
        content = prefix + pattern * repeats + suffix
        count = chat_token_count(tokenizer, content)
        difference = target_tokens - count
        if abs(difference) <= pattern_tokens:
            return content, count, marker
        repeats = max(1, repeats + difference // pattern_tokens)
    return content, count, marker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--port", type=int, default=1901)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=14400)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--cgroup", type=Path, default=DEFAULT_CGROUP)
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    cache = http_json(f"{base_url}/v1/cache/status")
    stats = http_json(f"{base_url}/v1/stats")
    geometry = cache.get("geometry", {})
    kv_tokens = int(geometry.get("num_pages", 0)) * int(geometry.get("page_size", 0))
    server_context = int(stats.get("model", {}).get("ctx", 0))
    if cache.get("state") != "serving":
        raise RuntimeError(f"server is not serving: {cache.get('state')}")
    if kv_tokens < args.context or server_context < args.context:
        raise RuntimeError(
            f"server capacity is too small: KV={kv_tokens}, frontend={server_context}, "
            f"requested={args.context}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    target_prompt_tokens = args.context - args.output_tokens - 64
    content, local_prompt_tokens, marker = build_content(tokenizer, target_prompt_tokens)
    payload = {
        "model": stats["model"]["id"],
        "messages": [{"role": "user", "content": content}],
        "max_tokens": args.output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_format": "none",
    }

    stop = threading.Event()
    telemetry = {
        "max_gpu_memory_mib": 0.0,
        "max_gpu_power_w": 0.0,
        "max_gpu_temperature_c": 0.0,
        "max_gpu_utilization_percent": 0.0,
        "max_cgroup_memory_bytes": 0,
        "max_cgroup_swap_bytes": 0,
        "max_decode_tps": 0.0,
    }

    def sample() -> None:
        next_gpu = 0.0
        # Short completions can finish between two-second samples.
        while not stop.wait(0.5):
            current = read_int(args.cgroup / "memory.current") or 0
            swap = read_int(args.cgroup / "memory.swap.current") or 0
            telemetry["max_cgroup_memory_bytes"] = max(
                telemetry["max_cgroup_memory_bytes"], current
            )
            telemetry["max_cgroup_swap_bytes"] = max(
                telemetry["max_cgroup_swap_bytes"], swap
            )
            try:
                live = http_json(f"{base_url}/v1/stats", timeout=5)
                throughput = live.get("throughput", {})
                telemetry["max_decode_tps"] = max(
                    telemetry["max_decode_tps"], float(throughput.get("decode_tps", 0))
                )
            except (OSError, RuntimeError, ValueError):
                pass
            if time.monotonic() >= next_gpu:
                try:
                    gpu = gpu_snapshot(args.gpu)
                    telemetry["max_gpu_memory_mib"] = max(
                        telemetry["max_gpu_memory_mib"], gpu["memory_used_mib"]
                    )
                    telemetry["max_gpu_power_w"] = max(
                        telemetry["max_gpu_power_w"], gpu["power_draw_w"]
                    )
                    telemetry["max_gpu_temperature_c"] = max(
                        telemetry["max_gpu_temperature_c"], gpu["temperature_c"]
                    )
                    telemetry["max_gpu_utilization_percent"] = max(
                        telemetry["max_gpu_utilization_percent"], gpu["utilization_percent"]
                    )
                except (OSError, subprocess.CalledProcessError, ValueError):
                    pass
                next_gpu = time.monotonic() + 5

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    started = time.monotonic()
    try:
        response = http_json(
            f"{base_url}/v1/chat/completions", payload, timeout=args.timeout
        )
    finally:
        elapsed = time.monotonic() - started
        stop.set()
        sampler.join(timeout=10)

    text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    completion_tokens = int(response.get("usage", {}).get("completion_tokens", 0))
    decode_tps = telemetry["max_decode_tps"]
    estimated_prompt_tps = None
    if decode_tps > 0:
        prompt_seconds = elapsed - completion_tokens / decode_tps
        if prompt_seconds > 0:
            estimated_prompt_tps = local_prompt_tokens / prompt_seconds
    result = {
        "context_target": args.context,
        "local_prompt_tokens": local_prompt_tokens,
        "usage": response.get("usage", {}),
        "elapsed_seconds": elapsed,
        "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
        "marker": marker,
        "marker_found": marker in text,
        "marker_first": text.startswith(marker),
        "estimated_prompt_tps": estimated_prompt_tps,
        "output": text,
        "cache_geometry": geometry,
        "telemetry": telemetry,
        "cgroup_memory_peak_bytes": read_int(args.cgroup / "memory.peak"),
        "cgroup_swap_peak_bytes": read_int(args.cgroup / "memory.swap.peak"),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
