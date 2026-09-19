#!/usr/bin/env python3
"""Benchmark Qwen3.8-27B Q4 tensor inference with a DFlash2 drafter."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from typing import TextIO
import urllib.error
import urllib.request


ROOT = Path("/home/obenomar/localLLMinference")
LLAMA = Path("/home/obenomar/.local/share/llama.cpp-recent-20260905/build/bin/llama-server")
MODEL = ROOT / "models/qwen3.8-27b-q4-k-m-gguf/Qwen3.8-27B-UD-Q4_K_M.gguf"
DRAFT = ROOT / "models/qwen3.8-27b-dflash2/Qwen3.8-27B-DFlash2-Q8_0.gguf"
GPU_QUERY = "index,power.limit,power.draw,utilization.gpu,temperature.gpu,memory.used,memory.total,clocks.current.graphics"
PHYSICAL_GPUS = ("1", "2")

CODE_PATTERN = """\
# repository module: deterministic code context
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

@dataclass(frozen=True)
class Record:
    key: str
    value: int

def normalize_records(records: Iterable[Record]) -> list[Record]:
    result = []
    for record in records:
        if record.key and record.value >= 0:
            result.append(record)
    return sorted(result, key=lambda item: (item.key, item.value))

def group_records(records: Iterable[Record]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for record in normalize_records(records):
        groups.setdefault(record.key, []).append(record.value)
    return groups

"""


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


def build_prompt(port: int, target_tokens: int, style: str) -> tuple[str, int]:
    if style == "code":
        prefix = (
            "You are reviewing a Python repository. The following is source context from the repository.\n"
            "Use it to solve the implementation task at the end. Preserve APIs and explain no extra text.\n\n"
        )
        suffix = (
            "\n\nImplementation task:\n"
            "Implement `merge_record_groups(records)` so it returns a stable mapping from each non-empty key "
            "to sorted unique non-negative values. Handle any iterable, do not mutate input, and return a plain "
            "dictionary. Reply with only the complete Python function.\n"
        )
        pattern = CODE_PATTERN
    else:
        prefix = f"Long context throughput probe with unique marker q4-dflash-{target_tokens}.\n"
        suffix = ""
        pattern = "Neutral repeated context text for a deterministic inference speed measurement. "

    fixed_tokens = tokenize(port, prefix + suffix)
    pattern_tokens = max(1, tokenize(port, pattern))
    repeats = max(1, math.ceil((target_tokens - fixed_tokens) / pattern_tokens))
    content = prefix + pattern * repeats + suffix
    for _ in range(10):
        count = tokenize(port, content)
        difference = target_tokens - count
        if abs(difference) <= 16:
            return content, count
        repeats = max(1, repeats + round(difference / pattern_tokens))
        content = prefix + pattern * repeats + suffix
    return content, tokenize(port, content)


def request_payload(prompt: str, max_tokens: int, greedy: bool = False) -> dict:
    payload = {
        "model": "qwen3.8-27b-q4-dflash-benchmark",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0 if greedy else 0.7,
        "top_p": 1.0 if greedy else 0.8,
        "top_k": 0 if greedy else 20,
        "presence_penalty": 0.0 if greedy else 1.5,
        "ignore_eos": False if greedy else True,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_format": "none",
    }
    if greedy:
        payload["seed"] = 12345
    return payload


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


def speculative_metrics(timings: object, n_max: int | None) -> dict[str, float | int | None]:
    if not isinstance(timings, dict):
        return {"draft_n": None, "draft_n_accepted": None, "draft_accept_rate": None}
    draft_n = int(timings.get("draft_n", 0) or 0)
    draft_n_accepted = int(timings.get("draft_n_accepted", 0) or 0)
    result: dict[str, float | int | None] = {
        "draft_n": draft_n or None,
        "draft_n_accepted": draft_n_accepted or None,
        "draft_accept_rate": draft_n_accepted / draft_n if draft_n else None,
    }
    if draft_n and n_max:
        verification_steps = math.ceil(draft_n / n_max)
        predicted_n = float(timings.get("predicted_n", 0) or 0)
        result["estimated_verification_steps"] = verification_steps
        result["estimated_mean_tokens_per_verification"] = predicted_n / verification_steps
        result["estimated_accepted_draft_tokens_per_verification"] = draft_n_accepted / verification_steps
    else:
        result["estimated_verification_steps"] = None
        result["estimated_mean_tokens_per_verification"] = None
        result["estimated_accepted_draft_tokens_per_verification"] = None
    return result


def gpu_snapshot() -> dict[str, dict[str, float]]:
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
        text=True,
    )
    snapshots: dict[str, dict[str, float]] = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 8 or fields[0] not in PHYSICAL_GPUS:
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
    if set(snapshots) != set(PHYSICAL_GPUS):
        raise RuntimeError(f"could not sample all selected GPUs: {snapshots}")
    return snapshots


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


def server_command(
    llama: Path,
    context_tokens: int,
    port: int,
    draft_model: Path | None,
    n_max: int | None,
    draft_device: str,
    target_split_mode: str,
    batch_size: int,
    ubatch_size: int,
) -> tuple[list[str], dict[str, str]]:
    command = [
        str(llama),
        "--model",
        str(MODEL),
        "--alias",
        "qwen3.8-27b-q4-dflash-benchmark",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--device",
        "CUDA0,CUDA1",
        "--split-mode",
        target_split_mode,
        "--n-gpu-layers",
        "all",
        "--ctx-size",
        str(context_tokens),
        "--parallel",
        "1",
        "--kv-offload",
        "--cache-type-k",
        "f16",
        "--cache-type-v",
        "f16",
        "--flash-attn",
        "on",
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
    if target_split_mode == "tensor":
        command.extend(["--tensor-split", "1,1"])
    if draft_model is not None:
        command.extend(
            [
                "--model-draft",
                str(draft_model),
                "--n-gpu-layers-draft",
                "0" if draft_device == "none" else "all",
                "--spec-draft-device",
                draft_device,
                "--spec-type",
                "draft-dflash",
                "--spec-draft-n-max",
                str(n_max),
            ]
        )
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "1,2",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    }
    environment.pop("GGML_CUDA_ENABLE_UNIFIED_MEMORY", None)
    return command, environment


def start_server(
    llama: Path,
    context_tokens: int,
    port: int,
    draft_model: Path | None,
    n_max: int | None,
    draft_device: str,
    target_split_mode: str,
    batch_size: int,
    ubatch_size: int,
    log_path: Path,
    startup_timeout: int,
) -> tuple[subprocess.Popen[str], TextIO]:
    command, environment = server_command(
        llama,
        context_tokens,
        port,
        draft_model,
        n_max,
        draft_device,
        target_split_mode,
        batch_size,
        ubatch_size,
    )
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    try:
        wait_ready(port, process, startup_timeout)
    except Exception:
        terminate(process)
        log.close()
        raise
    return process, log


def measure_gpu_samples(
    stop_sampling: threading.Event,
    samples: list[dict],
) -> None:
    while not stop_sampling.wait(1):
        try:
            samples.append({"monotonic_seconds": time.monotonic(), "gpus": gpu_snapshot()})
        except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError):
            pass


def run_measurement(args: argparse.Namespace, style: str, prompt_tokens: int, n_max: int | None) -> dict:
    draft_model = Path(args.draft_model) if n_max is not None else None
    label = f"{'dflash' if draft_model else 'baseline'}-ctx{args.ctx_size}-prompt{prompt_tokens}-style{style}"
    if n_max is not None:
        label += f"-nmax{n_max}"
    log_path = args.output_dir / f"{label}.server.log"
    result: dict[str, object] = {
        "run": label,
        "status": "error",
        "style": style,
        "context_tokens": args.ctx_size,
        "prompt_target_tokens": prompt_tokens,
        "max_tokens": args.max_tokens,
        "batch_size": args.batch_size,
        "ubatch_size": args.ubatch_size,
        "draft_model": str(draft_model) if draft_model else None,
        "spec_draft_n_max": n_max,
        "physical_gpus": list(PHYSICAL_GPUS),
        "server_log": str(log_path),
        "started_at": now(),
    }
    process: subprocess.Popen[str] | None = None
    log = None
    samples: list[dict] = []
    stop_sampling = threading.Event()
    sampler = threading.Thread(target=measure_gpu_samples, args=(stop_sampling, samples), daemon=True)
    request_started = time.monotonic()
    try:
        process, log = start_server(
            args.llama,
            args.ctx_size,
            args.port,
            draft_model,
            n_max,
            args.draft_device,
            args.target_split_mode,
            args.batch_size,
            args.ubatch_size,
            log_path,
            args.startup_timeout,
        )
        sampler.start()
        prompt, prompt_content_tokens = build_prompt(args.port, prompt_tokens, style)
        request_started = time.monotonic()
        response = http_json(
            f"http://127.0.0.1:{args.port}/v1/chat/completions",
            request_payload(prompt, args.max_tokens),
            timeout=args.request_timeout,
        )
        request_finished = time.monotonic()
        timings = response.get("timings", {})
        decode_rate, prefill_rate = timing_rates(timings)
        result.update(
            {
                "prompt_content_tokens": prompt_content_tokens,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "request_seconds": request_finished - request_started,
                "usage": response.get("usage", {}),
                "timings": timings,
                "decode_tok_s": decode_rate,
                "prefill_tok_s": prefill_rate,
                **speculative_metrics(timings, n_max),
                "completion_sha256": hashlib.sha256(response_text(response).encode()).hexdigest(),
                "completion_chars": len(response_text(response)),
                "status": "passed",
            }
        )
    except (OSError, RuntimeError, TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        stop_sampling.set()
        sampler.join(timeout=3)
        if process is not None:
            terminate(process)
            result["exit_code"] = process.returncode
        if log is not None:
            log.close()
        result["samples"] = samples
        result["sample_count"] = len(samples)
        result["peak_gpu_memory_mib"] = {
            gpu: max(
                (float(sample["gpus"][gpu]["memory_used_mib"]) for sample in samples if gpu in sample["gpus"]),
                default=None,
            )
            for gpu in PHYSICAL_GPUS
        }
        result["finished_at"] = now()
    return result


def run_lossless_pair(args: argparse.Namespace, prompt_tokens: int) -> dict:
    style = args.lossless_style
    result: dict[str, object] = {
        "context_tokens": args.ctx_size,
        "prompt_target_tokens": prompt_tokens,
        "style": style,
        "status": "error",
        "started_at": now(),
    }
    baseline_log = args.output_dir / f"lossless-baseline-prompt{prompt_tokens}.server.log"
    draft_log = args.output_dir / f"lossless-dflash-prompt{prompt_tokens}.server.log"
    prompt: str | None = None
    baseline_text: str | None = None
    baseline_process: subprocess.Popen[str] | None = None
    draft_process: subprocess.Popen[str] | None = None
    baseline_log_handle = None
    draft_log_handle = None
    try:
        baseline_process, baseline_log_handle = start_server(
            args.llama,
            args.ctx_size,
            args.port,
            None,
            None,
            args.draft_device,
            args.target_split_mode,
            args.batch_size,
            args.ubatch_size,
            baseline_log,
            args.startup_timeout,
        )
        prompt, prompt_content_tokens = build_prompt(args.port, prompt_tokens, style)
        baseline_response = http_json(
            f"http://127.0.0.1:{args.port}/v1/chat/completions",
            request_payload(prompt, args.lossless_max_tokens, greedy=True),
            timeout=args.request_timeout,
        )
        baseline_text = response_text(baseline_response)
        terminate(baseline_process)
        baseline_process = None
        baseline_log_handle.close()
        baseline_log_handle = None

        draft_process, draft_log_handle = start_server(
            args.llama,
            args.ctx_size,
            args.port,
            Path(args.draft_model),
            args.lossless_n_max,
            args.draft_device,
            args.target_split_mode,
            args.batch_size,
            args.ubatch_size,
            draft_log,
            args.startup_timeout,
        )
        draft_response = http_json(
            f"http://127.0.0.1:{args.port}/v1/chat/completions",
            request_payload(prompt, args.lossless_max_tokens, greedy=True),
            timeout=args.request_timeout,
        )
        draft_text = response_text(draft_response)
        result.update(
            {
                "prompt_tokens": prompt_content_tokens,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "baseline_completion_sha256": hashlib.sha256(baseline_text.encode()).hexdigest(),
                "draft_completion_sha256": hashlib.sha256(draft_text.encode()).hexdigest(),
                "token_identical_text": baseline_text == draft_text,
                "baseline_chars": len(baseline_text),
                "draft_chars": len(draft_text),
                "draft_timings": draft_response.get("timings", {}),
                "status": "passed" if baseline_text == draft_text else "failed_losslessness",
            }
        )
    except (OSError, RuntimeError, TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if baseline_process is not None:
            terminate(baseline_process)
        if draft_process is not None:
            terminate(draft_process)
        if baseline_log_handle is not None:
            baseline_log_handle.close()
        if draft_log_handle is not None:
            draft_log_handle.close()
        result["finished_at"] = now()
    return result


def save_results(path: Path, results: list[dict]) -> None:
    path.write_text(json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llama", type=Path, default=LLAMA)
    parser.add_argument("--draft-model", type=Path, default=DRAFT)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--ctx-size", type=int, default=262144)
    parser.add_argument("--contexts", nargs="+", type=int, default=(65536, 196608, 257000))
    parser.add_argument("--n-max", nargs="+", type=int, default=(2, 4, 7))
    parser.add_argument("--draft-device", default="none")
    parser.add_argument("--target-split-mode", choices=("tensor", "layer"), default="tensor")
    parser.add_argument("--prompt-styles", nargs="+", choices=("neutral", "code"), default=("neutral", "code"))
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=1024)
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--lossless-check", action="store_true")
    parser.add_argument("--lossless-contexts", nargs="+", type=int, default=(8192, 65536))
    parser.add_argument("--lossless-style", choices=("neutral", "code"), default="code")
    parser.add_argument("--lossless-max-tokens", type=int, default=256)
    parser.add_argument("--lossless-n-max", type=int, default=7)
    args = parser.parse_args()

    if not args.llama.exists() or not MODEL.exists():
        parser.error("llama-server or target Q4 GGUF is missing")
    if args.lossless_check and not args.draft_model.exists():
        parser.error("DFlash2 draft GGUF is missing")
    if not args.skip_baseline and not args.draft_model.exists():
        parser.error("DFlash2 draft GGUF is missing")
    if args.ubatch_size > args.batch_size:
        parser.error("--ubatch-size cannot exceed --batch-size")
    for prompt_tokens in args.contexts:
        if prompt_tokens + args.max_tokens + 256 > args.ctx_size:
            parser.error(f"prompt plus output exceeds context {args.ctx_size}: {prompt_tokens}")
    for prompt_tokens in args.lossless_contexts:
        if prompt_tokens + args.lossless_max_tokens + 256 > args.ctx_size:
            parser.error(f"lossless prompt plus output exceeds context {args.ctx_size}: {prompt_tokens}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.json"
    results: list[dict] = []
    save_results(results_path, results)

    if args.lossless_check:
        for prompt_tokens in args.lossless_contexts:
            result = run_lossless_pair(args, prompt_tokens)
            results.append(result)
            save_results(results_path, results)
            print(json.dumps(result, sort_keys=True), flush=True)
            if result.get("status") == "failed_losslessness":
                raise SystemExit("DFlash2 failed the losslessness gate")
        return

    for prompt_tokens in args.contexts:
        if not args.skip_baseline:
            for style in args.prompt_styles:
                result = run_measurement(args, style, prompt_tokens, None)
                results.append(result)
                save_results(results_path, results)
                print(json.dumps(result, sort_keys=True), flush=True)
        for n_max in args.n_max:
            for style in args.prompt_styles:
                result = run_measurement(args, style, prompt_tokens, n_max)
                results.append(result)
                save_results(results_path, results)
                print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
