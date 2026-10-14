#!/usr/bin/env python3
"""Long-context llama.cpp throughput sweep for Mistral-Small-4-119B (IQ3) on GPU 1+2.

Fork of tools/context_speed_sweep.py (Qwen single-GPU) with:
  * two-GPU targets (GPU 1 + GPU 2), even tensor split by default
  * NO --kv-offload (KV stays on GPU) so a point that cannot stay fully resident
    is a hard failure, not a silent hybrid
  * per-point full-residency assertion parsed from the startup log
    ("offloaded N/M layers to GPU", CUDA/CPU model buffer sizes); a point that is
    not fully resident is recorded and NOT scored
  * a short decode-warmup before the timed request
  * Mistral chat template (reasoning_effort=none for the speed pass) instead of the
    Qwen enable_thinking / reasoning_format knobs
  * per-card nvidia-smi sampling (VRAM sum, power/temp peaks, PCIe gen/width under
    load) plus host RSS / swap / MemAvailable
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import socket
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request


ROOT = Path((os.environ['EXPERIMENT_ROOT'] + ''))
LLAMA = Path((os.environ['RUNTIME_ROOT'] + '/llama.cpp/build/bin/llama-server'))

MODELS = {
    "mistral-xxs": {
        "gpu": "1,2",
        "port": 8082,
        "alias": "mistral-small-4-119b-2603-iq3-xxs",
        "path": ROOT / "models/mistral-small-4-119b-2603-iq3-xxs-gguf/Mistral-Small-4-119B-2603-UD-IQ3_XXS.gguf",
    },
    "mistral-s": {
        "gpu": "1,2",
        "port": 8083,
        "alias": "mistral-small-4-119b-2603-iq3-s",
        "path": ROOT / "models/mistral-small-4-119b-2603-iq3-s-gguf/Mistral-Small-4-119B-2603-UD-IQ3_S.gguf",
    },
}

# Execution order from the plan: XXS establishes max fit/speed across all tiers;
# S is the production candidate, swept from 128k up while it stays fully resident.
DEFAULT_CONTEXTS = {
    "mistral-xxs": [8192, 32768, 131072, 196608, 262144],
    "mistral-s": [131072, 196608, 262144],
}

N_GPU_LAYERS = "99"          # > 36 blocks -> all layers to GPU
THREADS = "32"
BATCH_SIZE = "2048"
UBATCH_SIZE = "512"
OUTPUT_TOKENS = 128
PROMPT_MARGIN = 128          # slack beyond (output + measured template overhead)


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


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


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
    response = http_json(f"http://127.0.0.1:{port}/tokenize", {"content": content}, timeout=300)
    tokens = response.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError(f"unexpected /tokenize response: {response.keys()}")
    return len(tokens)


def measure_template_overhead(port: int, alias: str) -> int:
    """Fixed tokens the chat template adds around a single user message.

    The Mistral template wraps the content (system slot, MODEL_SETTINGS with
    reasoning_effort, [INST]...[/INST]) which is a non-trivial, constant number
    of tokens. We must account for it or the timed prompt overruns --ctx-size.
    Measured by sending a known 1-message probe and reading timings.prompt_n.
    """
    probe = "Context length calibration probe."
    raw = tokenize(port, probe)
    payload = {
        "model": alias,
        "messages": [{"role": "user", "content": probe}],
        "max_tokens": 1,
        "ignore_eos": True,
        "stream": False,
        "chat_template_kwargs": {"reasoning_effort": "none"},
    }
    response = http_json(f"http://127.0.0.1:{port}/v1/chat/completions", payload, timeout=180)
    prompt_n = int(response.get("timings", {}).get("prompt_n", 0) or 0)
    return prompt_n - raw


def build_prompt(port: int, model: str, context: int, target_content: int) -> tuple[str, int]:
    target = max(1, int(target_content))
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


def gpu_snapshot(gpu_ids: list[int]) -> dict:
    query = (
        "index,power.limit,power.draw,utilization.gpu,temperature.gpu,"
        "memory.used,memory.total,pcie.link.gen.current,pcie.link.width.current"
    )
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        text=True,
    )
    cards: list[dict] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if not fields:
            continue
        try:
            idx = int(fields[0])
        except ValueError:
            continue
        if idx not in gpu_ids:
            continue
        cards.append({
            "index": idx,
            "power_limit_w": float(fields[1]),
            "power_draw_w": float(fields[2]),
            "utilization_gpu_percent": float(fields[3]),
            "temperature_c": float(fields[4]),
            "memory_used_mib": float(fields[5]),
            "memory_total_mib": float(fields[6]),
            "pcie_gen_current": int(fields[7]),
            "pcie_width_current": int(fields[8]),
        })
    if not cards:
        raise RuntimeError(f"GPUs {gpu_ids} not found in nvidia-smi output")
    cards.sort(key=lambda c: c["index"])
    return {
        "cards": cards,
        "memory_used_mib_total": sum(c["memory_used_mib"] for c in cards),
        "max_power_draw_w": max(c["power_draw_w"] for c in cards),
        "max_temperature_c": max(c["temperature_c"] for c in cards),
    }


def host_snapshot() -> dict:
    info: dict = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                info["mem_available_kb"] = int(line.split()[1])
            elif line.startswith("SwapFree:"):
                info["swap_free_kb"] = int(line.split()[1])
    except (OSError, ValueError):
        pass
    return info


def proc_rss_mib(pid: int) -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        pass
    return None


def check_residency(log_path: Path) -> dict:
    """Assert full GPU residency from the llama-server startup log.

    Fully resident means every layer was offloaded to GPU (offloaded == total)
    and there is no meaningful CPU *compute* model buffer. The `CPU_Mapped`
    buffer is the mmap file mapping and is expected, so it is recorded but does
    not by itself mean non-residency.
    """
    out = {
        "fully_resident": False,
        "offloaded_layers": None,
        "total_layers": None,
        "cuda_model_mib": None,
        "cpu_model_mib": None,
        "cpu_mapped_mib": None,
        "evidence": [],
    }
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    m = re.search(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU", text)
    if m:
        out["offloaded_layers"] = int(m.group(1))
        out["total_layers"] = int(m.group(2))
        out["evidence"].append(m.group(0))
    cuda = 0.0
    seen_cuda = False
    for mm in re.finditer(r"CUDA\d+\s+model buffer size =\s+([\d.]+)\s+MiB", text):
        cuda += float(mm.group(1))
        seen_cuda = True
        out["evidence"].append(mm.group(0))
    if seen_cuda:
        out["cuda_model_mib"] = cuda
    mc = re.search(r"CPU model buffer size =\s+([\d.]+)\s+MiB", text)
    if mc:
        out["cpu_model_mib"] = float(mc.group(1))
        out["evidence"].append(mc.group(0))
    mp = re.search(r"CPU_Mapped model buffer size =\s+([\d.]+)\s+MiB", text)
    if mp:
        out["cpu_mapped_mib"] = float(mp.group(1))
    off, tot = out["offloaded_layers"], out["total_layers"]
    cpu_model = out["cpu_model_mib"] or 0.0
    out["fully_resident"] = (
        off is not None and tot is not None and off == tot and cpu_model < 128.0
    )
    return out


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


def warmup(port: int, alias: str) -> str | None:
    """Short decode to warm CUDA graphs / the decode path before the timed run."""
    payload = {
        "model": alias,
        "messages": [{"role": "user", "content": "Warm up the pipeline. Reply with one short sentence."}],
        "max_tokens": 16,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": False,
        "chat_template_kwargs": {"reasoning_effort": "none"},
    }
    try:
        http_json(f"http://127.0.0.1:{port}/v1/chat/completions", payload, timeout=180)
        return None
    except Exception as exc:  # noqa: BLE001 - best-effort warmup
        return f"{type(exc).__name__}: {exc}"


def _classify_startup_failure(log_path: Path, exc: Exception) -> str:
    """Map a startup failure to not_fully_resident (OOM / partial offload) or error."""
    resid = check_residency(log_path)
    if resid["offloaded_layers"] is not None and resid["offloaded_layers"] < resid["total_layers"]:
        return "not_fully_resident"
    try:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
    except OSError:
        tail = ""
    needle = str(exc).lower()
    if any(k in needle for k in ("memory", "oom")) or any(
        k in tail.lower() for k in ("out of memory", "cannot allocate", "cuda error", "no space")
    ):
        return "not_fully_resident"
    return "error"


def measure(
    model: str,
    context: int,
    output_dir: Path,
    request_timeout: int,
    startup_timeout: int,
    tensor_split: str,
    verbosity: int,
) -> dict:
    spec = MODELS[model]
    port = spec["port"]
    gpu_ids = [int(x) for x in spec["gpu"].split(",") if x.strip()]
    log_path = output_dir / f"{model}-{context}.server.log"
    result = {
        "model": model,
        "context_tokens": context,
        "gpu": spec["gpu"],
        "port": port,
        "tensor_split": tensor_split,
        "started_at": now(),
        "status": "error",
        "fully_resident": None,
    }
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            str(LLAMA),
            "--model", str(spec["path"]),
            "--alias", spec["alias"],
            "--host", "127.0.0.1",
            "--port", str(port),
            "--ctx-size", str(context),
            "--parallel", "1",
            "--no-cache-prompt",
            "--n-gpu-layers", N_GPU_LAYERS,
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
            "--flash-attn", "on",
            "--threads", THREADS,
            "--threads-batch", THREADS,
            "--batch-size", BATCH_SIZE,
            "--ubatch-size", UBATCH_SIZE,
            "--fit", "off",
            "--tensor-split", tensor_split,
            "--metrics",
            "--perf",
            "-lv", str(verbosity),
        ],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": spec["gpu"],
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "0",
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
                snap = gpu_snapshot(gpu_ids)
                snap.update(host_snapshot())
                snap["process_rss_mib"] = proc_rss_mib(process.pid)
                samples.append(snap)
            except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError):
                pass

    sampler = threading.Thread(target=sample, daemon=True)
    try:
        wait_ready(port, process, startup_timeout)
        resid = check_residency(log_path)
        result["residency"] = resid
        result["fully_resident"] = resid["fully_resident"]
        if not resid["fully_resident"]:
            result["status"] = "not_fully_resident"
        else:
            result["startup_snapshot"] = gpu_snapshot(gpu_ids)
            result["startup_process_rss_mib"] = proc_rss_mib(process.pid)
            warmup_error = warmup(port, spec["alias"])
            if warmup_error:
                result["warmup_error"] = warmup_error
            overhead = measure_template_overhead(port, spec["alias"])
            target_content = context - OUTPUT_TOKENS - overhead - PROMPT_MARGIN
            content, content_tokens = build_prompt(port, model, context, target_content)
            result["template_overhead_tokens"] = overhead
            result["prompt_target_tokens"] = target_content
            result["prompt_content_tokens"] = content_tokens
            payload = {
                "model": spec["alias"],
                "messages": [{"role": "user", "content": content}],
                "max_tokens": OUTPUT_TOKENS,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "presence_penalty": 1.5,
                "ignore_eos": True,
                "stream": False,
                "chat_template_kwargs": {"reasoning_effort": "none"},
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
        if result["fully_resident"] is None:
            result["status"] = _classify_startup_failure(log_path, exc)
        else:
            result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        stop_sampling.set()
        sampler.join(timeout=3)
        if samples:
            result["sample_count"] = len(samples)
            result["max_power_draw_w"] = max(s["max_power_draw_w"] for s in samples)
            result["max_temperature_c"] = max(s["max_temperature_c"] for s in samples)
            result["max_memory_used_mib_total"] = max(s["memory_used_mib_total"] for s in samples)
            per_card_peak = {}
            for idx in gpu_ids:
                vals = [c["memory_used_mib"] for s in samples for c in s["cards"] if c["index"] == idx]
                per_card_peak[str(idx)] = max(vals) if vals else None
            result["peak_memory_used_mib_per_card"] = per_card_peak
            gens = [c["pcie_gen_current"] for s in samples for c in s["cards"]]
            widths = [c["pcie_width_current"] for s in samples for c in s["cards"]]
            result["pcie_gen_under_load"] = {"max": max(gens), "min": min(gens)} if gens else None
            result["pcie_width_under_load"] = {"max": max(widths), "min": min(widths)} if widths else None
            rsss = [s["process_rss_mib"] for s in samples if s.get("process_rss_mib") is not None]
            result["max_process_rss_mib"] = max(rsss) if rsss else None
            swapfrees = [s["swap_free_kb"] for s in samples if "swap_free_kb" in s]
            result["min_swap_free_kb"] = min(swapfrees) if swapfrees else None
            memavs = [s["mem_available_kb"] for s in samples if "mem_available_kb" in s]
            result["min_mem_available_kb"] = min(memavs) if memavs else None
        terminate(process)
        log.close()
        result["exit_code"] = process.returncode
        result["finished_at"] = now()
        result["server_log"] = str(log_path)
    return result


def save_results(path: Path, results: list[dict]) -> None:
    path.write_text(json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8")


def preflight(models: list[str]) -> None:
    for m in models:
        spec = MODELS[m]
        if not Path(spec["path"]).exists():
            raise SystemExit(f"preflight: model file missing for {m}: {spec['path']}")
        if port_in_use(spec["port"]):
            raise SystemExit(f"preflight: port {spec['port']} already in use")
    try:
        snap = gpu_snapshot([1, 2])
        for c in snap["cards"]:
            free = c["memory_total_mib"] - c["memory_used_mib"]
            print(
                f"preflight: GPU {c['index']} free {free:.0f} MiB "
                f"(pcie gen{c['pcie_gen_current']} x{c['pcie_width_current']})",
                flush=True,
            )
            if free < 20000:
                print(f"preflight WARNING: GPU {c['index']} has <20000 MiB free", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"preflight WARNING: GPU check failed: {exc}", flush=True)
    print("preflight OK", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--startup-timeout", type=int, default=360)
    parser.add_argument("--tensor-split", type=str, default="1,1")
    parser.add_argument("--verbosity", type=int, default=4)
    parser.add_argument("--models", type=str, nargs="*", default=None)
    parser.add_argument("--contexts", type=str, default=None,
                        help="comma-separated context sizes to override the per-model defaults")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model_order = args.models or ["mistral-xxs", "mistral-s"]
    for m in model_order:
        if m not in MODELS:
            raise SystemExit(f"unknown model: {m} (choose from {list(MODELS)})")
    ctx_override = (
        [int(x) for x in args.contexts.split(",") if x.strip()] if args.contexts else None
    )

    preflight(model_order)

    results: list[dict] = []
    save_results(args.output, results)
    for model in model_order:
        contexts = sorted(ctx_override if ctx_override else DEFAULT_CONTEXTS[model])
        for context in contexts:
            result = measure(
                model, context, args.output.parent,
                args.request_timeout, args.startup_timeout,
                args.tensor_split, args.verbosity,
            )
            results.append(result)
            save_results(args.output, results)
            print(json.dumps(result, sort_keys=True), flush=True)
            if result.get("status") == "not_fully_resident":
                print(f"[{model}] stopping sweep at {context}: not fully resident", flush=True)
                break


if __name__ == "__main__":
    main()
