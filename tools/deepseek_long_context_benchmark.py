#!/usr/bin/env python3
"""Populate DeepSeek-V4 context and measure long-context prefill/decode."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path((os.environ['EXPERIMENT_ROOT'] + ''))
LLAMA = Path((os.environ['RUNTIME_ROOT'] + '/llama.cpp/build/bin/llama-server'))
MODEL_STORE = Path(os.environ.get("MODEL_STORE", (os.environ['MODEL_STORE'] + '/models')))
MODEL = MODEL_STORE / "deepseek-v4-flash-0731-iq3-xxs-gguf/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf"
ALIAS = "deepseek-v4-flash-0731-iq3-xxs"
PORT = 8090


def http_json(path: str, payload: dict | None = None, timeout: int = 60) -> dict:
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        body,
        headers={"Content-Type": "application/json"} if body else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_ready(process: subprocess.Popen, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with {process.returncode}")
        try:
            if http_json("/v1/models", timeout=5).get("data"):
                return
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise TimeoutError("llama-server did not become ready")


def vmstat() -> dict[str, int]:
    values = {"pswpin": 0, "pswpout": 0}
    for line in Path("/proc/vmstat").read_text().splitlines():
        key, value = line.split()
        if key in values:
            values[key] = int(value)
    return values


def gpu_snapshot(gpu_ids: list[int]) -> list[dict]:
    query = "index,utilization.gpu,memory.used,power.draw,temperature.gpu"
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        text=True,
    )
    cards = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5 or int(fields[0]) not in gpu_ids:
            continue
        cards.append({
            "index": int(fields[0]),
            "gpu_util_percent": float(fields[1]),
            "memory_used_mib": float(fields[2]),
            "power_w": float(fields[3]),
            "temperature_c": float(fields[4]),
        })
    return cards


def tokenize(content: str) -> int:
    return len(http_json("/tokenize", {"content": content}, timeout=1800)["tokens"])


def make_prompt(target_tokens: int) -> tuple[str, int]:
    prefix = "DeepSeek long-context prefill benchmark. Preserve this neutral context exactly.\n"
    unit = "This is deterministic neutral context used only to populate the model sequence before a short decode. "
    unit_tokens = max(1, tokenize(unit))
    count = max(1, target_tokens // unit_tokens)
    content = prefix + unit * count
    for _ in range(8):
        actual = tokenize(content)
        if abs(actual - target_tokens) <= 64:
            return content, actual
        count = max(1, count + round((target_tokens - actual) / unit_tokens))
        content = prefix + unit * count
    return content, tokenize(content)


def terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def parse_case(value: str) -> tuple[int, int]:
    context, prompt = value.split(":", 1)
    return int(context), int(prompt)


def run_case(args: argparse.Namespace, context_tokens: int, prompt_target: int) -> dict:
    name = f"ctx{context_tokens}-prompt{prompt_target}-{args.cache_k}-{args.cache_v}"
    log_path = args.outdir / f"{name}.server.log"
    command = [
        str(LLAMA), "--model", str(MODEL), "--alias", ALIAS,
        "--host", "127.0.0.1", "--port", str(PORT), "--ctx-size", str(context_tokens),
        "--parallel", "1", "--no-cache-prompt", "--n-gpu-layers", "99", "--fit", "off",
        "--split-mode", "layer", "--tensor-split", args.tensor_split,
        "--n-cpu-moe", str(args.n_cpu_moe), "--cache-type-k", args.cache_k,
        "--cache-type-v", args.cache_v, "--flash-attn", "on",
        "--threads", str(args.threads), "--threads-batch", str(args.threads),
        "--batch-size", str(args.batch_size), "--ubatch-size", str(args.ubatch_size),
        "--metrics", "--perf", "--reasoning", "auto", "--log-colors", "off", "-lv", "3",
    ]
    if args.no_kv_offload:
        command.append("--no-kv-offload")
    gpu_ids = [int(value) for value in args.gpus.split(",")]
    result: dict = {
        "name": name,
        "model": str(MODEL),
        "context_tokens": context_tokens,
        "prompt_target_tokens": prompt_target,
        "tensor_split": args.tensor_split,
        "n_cpu_moe": args.n_cpu_moe,
        "cache_k": args.cache_k,
        "cache_v": args.cache_v,
        "batch_size": args.batch_size,
        "ubatch_size": args.ubatch_size,
        "no_kv_offload": args.no_kv_offload,
        "command": command,
        "status": "error",
        "server_log": str(log_path),
    }
    samples: list[list[dict]] = []
    stop_sampling = threading.Event()
    swap_abort = threading.Event()
    process: subprocess.Popen | None = None
    swap_before: dict[str, int] | None = None

    def sample() -> None:
        while not stop_sampling.wait(2):
            try:
                samples.append(gpu_snapshot(gpu_ids))
            except (OSError, ValueError, subprocess.CalledProcessError):
                pass

    def abort_on_swap(before: dict) -> None:
        while not stop_sampling.wait(2):
            after = vmstat()
            if after["pswpin"] > before["pswpin"] or after["pswpout"] > before["pswpout"]:
                swap_abort.set()
                if process is not None and process.poll() is None:
                    # Interrupt the blocking HTTP request as well as inference.
                    process.kill()
                return

    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": args.gpus,
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "0",
            },
            start_new_session=True,
        )
        try:
            wait_ready(process, args.startup_timeout)
            result["startup_seconds"] = round(time.monotonic() - started, 2)
            prompt, prompt_tokens = make_prompt(prompt_target)
            result["prompt_tokens"] = prompt_tokens
            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            swap_before = vmstat()
            if args.abort_on_swap:
                swap_monitor = threading.Thread(target=abort_on_swap, args=(swap_before,), daemon=True)
                swap_monitor.start()
            response = http_json(
                "/v1/chat/completions",
                {
                    "model": ALIAS,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": args.decode_tokens,
                    "ignore_eos": True,
                    "temperature": 0,
                    "seed": 1234,
                    "stream": False,
                },
                timeout=args.request_timeout,
            )
            swap_after = vmstat()
            result.update({
                "status": "passed" if not swap_abort.is_set() else "active_swap_abort",
                "timings": response.get("timings", {}),
                "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
                "response": response.get("choices", [{}])[0].get("message", {}),
                "swap_pages_in": swap_after["pswpin"] - swap_before["pswpin"],
                "swap_pages_out": swap_after["pswpout"] - swap_before["pswpout"],
            })
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            if swap_abort.is_set():
                result["status"] = "active_swap_abort"
            if swap_before is not None:
                swap_after = vmstat()
                result["swap_pages_in"] = swap_after["pswpin"] - swap_before["pswpin"]
                result["swap_pages_out"] = swap_after["pswpout"] - swap_before["pswpout"]
        finally:
            stop_sampling.set()
            terminate(process)
    result["samples"] = samples
    if samples:
        flattened = [card for sample in samples for card in sample]
        result["gpu_peak_util_percent"] = {
            str(index): max(card["gpu_util_percent"] for card in flattened if card["index"] == index)
            for index in gpu_ids
        }
        result["gpu_peak_memory_mib"] = {
            str(index): max(card["memory_used_mib"] for card in flattened if card["index"] == index)
            for index in gpu_ids
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", type=parse_case, required=True,
                        metavar="CTX:PROMPT", help="for example 131072:120000")
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--tensor-split", default="1.2,1,1")
    parser.add_argument("--n-cpu-moe", type=int, default=20)
    parser.add_argument("--cache-k", default="q8_0")
    parser.add_argument("--cache-v", default="q8_0")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--startup-timeout", type=int, default=3600)
    parser.add_argument("--request-timeout", type=int, default=21600)
    parser.add_argument("--no-kv-offload", action="store_true")
    parser.add_argument("--no-abort-on-swap", dest="abort_on_swap", action="store_false")
    parser.set_defaults(abort_on_swap=True)
    args = parser.parse_args()
    if not MODEL.is_file():
        raise SystemExit(f"missing model: {MODEL}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    results = []
    for context_tokens, prompt_target in args.cases:
        result = run_case(args, context_tokens, prompt_target)
        results.append({key: value for key, value in result.items() if key != "samples"})
        (args.outdir / "results.json").write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
        print(json.dumps(results[-1], sort_keys=True), flush=True)
        if result["status"] != "passed":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
