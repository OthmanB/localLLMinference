#!/usr/bin/env python3
"""Initial llama.cpp probes for large GGUF MoE models on all three GPUs.

This is a capacity/throughput smoke harness, not a quality benchmark. It keeps
the existing GPU-0 Qwen service out of the way by requiring an explicit stop
before launch, uses --fit off, and makes CPU expert placement explicit.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path((os.environ['EXPERIMENT_ROOT'] + ''))
LLAMA = Path((os.environ['RUNTIME_ROOT'] + '/llama.cpp/build/bin/llama-server'))
MODEL_STORE = Path(os.environ.get("MODEL_STORE", ROOT / "models"))
MODELS = {
    "deepseek-v4-flash-iq3-xxs": {
        "path": MODEL_STORE / "deepseek-v4-flash-0731-iq3-xxs-gguf/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf",
        "alias": "deepseek-v4-flash-0731-iq3-xxs",
    },
    "minimax-m2.5-iq3-s": {
        "path": MODEL_STORE / "minimax-m2.5-iq3-s-gguf/MiniMax-M2.5-IQ3_S.gguf",
        "alias": "minimax-m2.5-iq3-s",
    },
}
PORT = 8090
GPUS = "0,1,2"


def http_json(url: str, payload: dict | None = None, timeout: float = 30) -> dict:
    import urllib.request

    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def gpu_snapshot() -> list[dict]:
    query = "index,memory.used,memory.total,power.draw,power.limit,temperature.gpu,pcie.link.gen.current,pcie.link.width.current"
    output = subprocess.check_output(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], text=True)
    cards = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 8:
            continue
        cards.append({
            "index": int(fields[0]),
            "memory_used_mib": float(fields[1]),
            "memory_total_mib": float(fields[2]),
            "power_draw_w": float(fields[3]),
            "power_limit_w": float(fields[4]),
            "temperature_c": float(fields[5]),
            "pcie_gen": int(fields[6]),
            "pcie_width": int(fields[7]),
        })
    return cards


def wait_ready(port: int, process: subprocess.Popen, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with {process.returncode}")
        try:
            if http_json(f"http://127.0.0.1:{port}/v1/models", timeout=5).get("data"):
                return
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise TimeoutError(f"server on port {port} did not become ready")


def startup_summary(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    offload = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", text)
    model_buffers = re.findall(r"CUDA\d+ model buffer size = ([0-9.]+) MiB", text)
    cpu_buffers = re.findall(r"CPU model buffer size = ([0-9.]+) MiB", text)
    return {
        "offloaded_layers": [list(map(int, item)) for item in offload[-1:]],
        "cuda_model_buffers_mib": [float(value) for value in model_buffers],
        "cpu_model_buffers_mib": [float(value) for value in cpu_buffers],
    }


def run(args: argparse.Namespace) -> dict:
    spec = MODELS[args.model]
    if not spec["path"].is_file():
        raise FileNotFoundError(spec["path"])
    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}-ctx{args.ctx}-ncmoe{args.n_cpu_moe}"
    log_path = args.outdir / f"{stem}.server.log"
    command = [
        str(LLAMA), "--model", str(spec["path"]), "--alias", spec["alias"],
        "--host", "127.0.0.1", "--port", str(PORT), "--ctx-size", str(args.ctx),
        "--parallel", "1", "--n-gpu-layers", "99", "--fit", "off",
        "--split-mode", args.split_mode, "--tensor-split", args.tensor_split,
        "--n-cpu-moe", str(args.n_cpu_moe),
        "--cache-type-k", args.cache_k, "--cache-type-v", args.cache_v,
        "--flash-attn", "on", "--threads", "32", "--threads-batch", "32",
        "--batch-size", "2048", "--ubatch-size", "512", "--metrics", "--perf",
        "--reasoning", "auto", "--log-colors", "off", "-lv", "3",
    ]
    if args.main_gpu is not None:
        command.extend(["--main-gpu", str(args.main_gpu)])
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": GPUS, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "0"},
            start_new_session=True,
        )
        try:
            wait_ready(PORT, process, args.startup_timeout)
            ready_seconds = round(time.monotonic() - started, 2)
            payload = {
                "model": spec["alias"],
                "messages": [{"role": "user", "content": "Reply with one short sentence confirming that the local model is ready."}],
                "max_tokens": 128,
                "temperature": 0.7,
                "stream": False,
            }
            response = http_json(f"http://127.0.0.1:{PORT}/v1/chat/completions", payload, timeout=args.request_timeout)
            return {
                "model": args.model,
                "alias": spec["alias"],
                "n_cpu_moe": args.n_cpu_moe,
                "ctx": args.ctx,
                "tensor_split": args.tensor_split,
                "split_mode": args.split_mode,
                "main_gpu": args.main_gpu,
                "cache_k": args.cache_k,
                "cache_v": args.cache_v,
                "startup_seconds": ready_seconds,
                "timings": response.get("timings", {}),
                "response": response.get("choices", [{}])[0].get("message", {}),
                "gpu_after_request": gpu_snapshot(),
                "startup": startup_summary(log_path),
                "log": str(log_path),
                "status": "ready_and_generated",
            }
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--n-cpu-moe", type=int, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--split-mode", choices=("layer", "row", "tensor"), default="layer")
    parser.add_argument("--tensor-split", default="1,1,1")
    parser.add_argument("--main-gpu", type=int, choices=(0, 1, 2))
    parser.add_argument("--cache-k", default="q8_0")
    parser.add_argument("--cache-v", default="q8_0")
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--request-timeout", type=int, default=1200)
    args = parser.parse_args()
    result = run(args)
    stem = f"{args.model}-ctx{args.ctx}-ncmoe{args.n_cpu_moe}"
    (args.outdir / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
