#!/usr/bin/env python3
"""Fixed-workload placement sweep for DeepSeek-V4-Flash on three RTX 3090s."""
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
MODEL_STORE = Path(os.environ.get("MODEL_STORE", ROOT / "models"))
MODEL = MODEL_STORE / "deepseek-v4-flash-0731-iq3-xxs-gguf/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf"
ALIAS = "deepseek-v4-flash-0731-iq3-xxs"
PORT = 8090
DEFAULT_GPUS = "0,1,2"

CASES = (
    ("layer-3gpu-050-24", DEFAULT_GPUS, "layer", "0.5,1,1", 24, None),
    ("layer-3gpu-111-24", DEFAULT_GPUS, "layer", "1,1,1", 24, None),
    ("layer-3gpu-111-20", DEFAULT_GPUS, "layer", "1,1,1", 20, None),
    ("layer-3gpu-111-16", DEFAULT_GPUS, "layer", "1,1,1", 16, None),
    ("layer-3gpu-111-12", DEFAULT_GPUS, "layer", "1,1,1", 12, None),
    ("row-3gpu-111-24-main1", DEFAULT_GPUS, "row", "1,1,1", 24, 1),
    ("row-3gpu-111-20-main1", DEFAULT_GPUS, "row", "1,1,1", 20, 1),
    ("layer-gpu12-11-24", "1,2", "layer", "1,1", 24, None),
    ("layer-gpu12-11-20", "1,2", "layer", "1,1", 20, None),
    ("layer-gpu02-11-24", "0,2", "layer", "1,1", 24, None),
    ("row-gpu12-11-24-main1", "1,2", "row", "1,1", 24, 1),
    ("layer-3gpu-090-20", DEFAULT_GPUS, "layer", "0.9,1,1", 20, None),
    ("layer-3gpu-110-20", DEFAULT_GPUS, "layer", "1.1,1,1", 20, None),
    ("layer-3gpu-120-20", DEFAULT_GPUS, "layer", "1.2,1,1", 20, None),
    ("layer-3gpu-111-18", DEFAULT_GPUS, "layer", "1,1,1", 18, None),
    ("layer-3gpu-111-19", DEFAULT_GPUS, "layer", "1,1,1", 19, None),
    ("layer-3gpu-111-21", DEFAULT_GPUS, "layer", "1,1,1", 21, None),
    ("layer-3gpu-111-22", DEFAULT_GPUS, "layer", "1,1,1", 22, None),
    ("layer-3gpu-110-24", DEFAULT_GPUS, "layer", "1.1,1,1", 24, None),
    ("layer-3gpu-120-24", DEFAULT_GPUS, "layer", "1.2,1,1", 24, None),
)


def http_json(path: str, payload: dict | None = None, timeout: int = 60) -> dict:
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}", body,
        headers={"Content-Type": "application/json"} if body else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def server_metrics() -> dict[str, float]:
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=60) as response:
        text = response.read().decode()
    metrics = {}
    for line in text.splitlines():
        if not line.startswith("llamacpp:") or "{" in line:
            continue
        name, value = line.split()
        metrics[name.removeprefix("llamacpp:")] = float(value)
    return metrics


def speculative_metrics(before: dict[str, float], after: dict[str, float]) -> dict:
    drafted = after.get("spec_decode_num_draft_tokens_total", 0) - before.get("spec_decode_num_draft_tokens_total", 0)
    accepted = after.get("spec_decode_num_accepted_tokens_total", 0) - before.get("spec_decode_num_accepted_tokens_total", 0)
    verification_steps = after.get("spec_decode_num_drafts_total", 0) - before.get("spec_decode_num_drafts_total", 0)
    return {
        "drafted_tokens": int(drafted),
        "accepted_tokens": int(accepted),
        "verification_steps": int(verification_steps),
        "acceptance_fraction": accepted / drafted if drafted else None,
        "accepted_per_verification": accepted / verification_steps if verification_steps else None,
    }


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


def gpu_snapshot(gpu_ids: list[int]) -> list[dict]:
    query = "index,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.current.sm,pcie.link.gen.current,pcie.link.width.current"
    output = subprocess.check_output(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], text=True)
    result = []
    for line in output.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 9:
            continue
        if int(fields[0]) not in gpu_ids:
            continue
        result.append({
            "index": int(fields[0]), "gpu_util_percent": float(fields[1]),
            "memory_util_percent": float(fields[2]), "memory_used_mib": float(fields[3]),
            "power_w": float(fields[4]), "temperature_c": float(fields[5]),
            "sm_clock_mhz": float(fields[6]), "pcie_gen": int(fields[7]),
            "pcie_width": int(fields[8]),
        })
    return result


def vmstat() -> dict[str, int]:
    wanted = {"pswpin": 0, "pswpout": 0}
    for line in Path("/proc/vmstat").read_text().splitlines():
        key, value = line.split()
        if key in wanted:
            wanted[key] = int(value)
    return wanted


def host_memory_snapshot() -> dict[str, float]:
    wanted = {"MemTotal": 0.0, "MemAvailable": 0.0, "SwapTotal": 0.0, "SwapFree": 0.0}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value, *_ = line.split()
        key = key.rstrip(":")
        if key in wanted:
            wanted[key] = round(int(value) / 1024, 2)
    return wanted


def process_rss_mib(pid: int) -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 2)
    except (FileNotFoundError, OSError, ValueError):
        return None
    return None


def tokenize(content: str) -> int:
    response = http_json("/tokenize", {"content": content}, timeout=300)
    return len(response["tokens"])


def make_prompt(target_tokens: int) -> tuple[str, int]:
    prefix = "DeepSeek fixed prefill benchmark. Preserve these instructions exactly.\n"
    unit = "Neutral benchmark context used solely to measure deterministic prefill throughput. "
    count = max(1, target_tokens // max(1, tokenize(unit)))
    content = prefix + unit * count
    for _ in range(6):
        actual = tokenize(content)
        if abs(actual - target_tokens) <= 16:
            return content, actual
        count = max(1, count + round((target_tokens - actual) / max(1, tokenize(unit))))
        content = prefix + unit * count
    return content, tokenize(content)


def sanity_checks(alias: str, count: int = 3) -> list[dict]:
    checks = (
        ("arithmetic", "Answer with only the decimal result: 17 + 25 =", "42"),
        ("capital", "Answer with only the city: What is the capital of France?", "paris"),
        ("echo", "Reply only with this exact token: placement-sanity-9", "placement-sanity-9"),
    )
    results = []
    for index in range(count):
        name, prompt, expected = checks[index % len(checks)]
        response = http_json("/v1/chat/completions", {
            "model": alias, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64, "temperature": 0, "seed": 1234, "stream": False,
        }, timeout=300)
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        results.append({
            "name": name, "expected_substring": expected, "content": content,
            "passed": expected in content.lower(),
        })
    return results


def terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_case(
    case: tuple[str, str, str, str, int, int | None],
    outdir: Path,
    prompt_target_tokens: int,
    threads: int,
    model_path: Path,
    alias: str,
    llama_path: Path,
    override_tensor: str | None,
    omit_n_cpu_moe: bool,
    cache_type_k: str,
    cache_type_v: str,
    ctx_size: int,
    server_options: list[str],
    server_environment: dict[str, str],
    decode_tokens: int,
    warmup_decode_requests: int,
    stability_requests: int,
    request_timeout: int,
) -> dict:
    name, gpus, split_mode, tensor_split, n_cpu_moe, main_gpu = case
    gpu_ids = [int(value) for value in gpus.split(",")]
    result_name = name if threads == 32 else f"{name}-t{threads}"
    server_log = outdir / f"{result_name}.server.log"
    command = [
        str(llama_path), "--model", str(model_path), "--alias", alias,
        "--host", "127.0.0.1", "--port", str(PORT), "--ctx-size", str(ctx_size),
        "--parallel", "1", "--no-cache-prompt", "--n-gpu-layers", "99",
        "--fit", "off", "--split-mode", split_mode, "--tensor-split", tensor_split,
        "--cache-type-k", cache_type_k, "--cache-type-v", cache_type_v,
        "--flash-attn", "on", "--threads", str(threads), "--threads-batch", str(threads),
        "--batch-size", "2048", "--ubatch-size", "512", "--metrics", "--perf",
        "--reasoning", "auto", "--log-colors", "off", "-lv", "3",
    ]
    command.extend(server_options)
    if not omit_n_cpu_moe:
        command.extend(["--n-cpu-moe", str(n_cpu_moe)])
    if override_tensor:
        command.extend(["--override-tensor", override_tensor])
    if main_gpu is not None:
        command.extend(["--main-gpu", str(main_gpu)])
    result: dict = {
        "name": result_name, "model": alias, "model_path": str(model_path),
        "llama_path": str(llama_path), "override_tensor": override_tensor,
        "gpus": gpus, "split_mode": split_mode, "tensor_split": tensor_split,
        "n_cpu_moe": None if omit_n_cpu_moe else n_cpu_moe, "main_gpu": main_gpu, "threads": threads,
        "cache_type_k": cache_type_k, "cache_type_v": cache_type_v,
        "server_options": server_options, "server_environment": server_environment,
        "ctx_size": ctx_size,
        "decode_tokens": decode_tokens, "warmup_decode_requests": warmup_decode_requests,
        "stability_requests": stability_requests,
        "request_timeout": request_timeout,
        "prefill_target_content_tokens": prompt_target_tokens,
        "status": "error", "command": command,
    }
    samples: list[list[dict]] = []
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(1):
            try:
                samples.append(gpu_snapshot(gpu_ids))
            except (OSError, ValueError, subprocess.CalledProcessError):
                pass

    started = time.monotonic()
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    environment.pop("GGML_CUDA_ENABLE_UNIFIED_MEMORY", None)
    environment.update(server_environment)
    with server_log.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        try:
            wait_ready(process, 1800)
            result["startup_seconds"] = round(time.monotonic() - started, 2)
            result["sanity_checks"] = sanity_checks(alias)
            if not all(check["passed"] for check in result["sanity_checks"]):
                raise RuntimeError("output sanity check failed")
            if stability_requests:
                stability_rss_before = process_rss_mib(process.pid)
                stability_gpus_before = gpu_snapshot(gpu_ids)
                result["stability_checks"] = sanity_checks(alias, stability_requests)
                if not all(check["passed"] for check in result["stability_checks"]):
                    raise RuntimeError("stability output sanity check failed")
                result["stability_rss_before_mib"] = stability_rss_before
                result["stability_rss_after_mib"] = process_rss_mib(process.pid)
                result["stability_gpus_before"] = stability_gpus_before
                result["stability_gpus_after"] = gpu_snapshot(gpu_ids)
            prompt, prompt_tokens = make_prompt(prompt_target_tokens)
            result["prefill_content_tokens"] = prompt_tokens
            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            swap_before = vmstat()
            host_before = host_memory_snapshot()
            rss_before = process_rss_mib(process.pid)
            workload_started = time.monotonic()
            prefill = http_json("/v1/chat/completions", {
                "model": alias, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1, "ignore_eos": True, "temperature": 0, "seed": 1234, "stream": False,
            }, timeout=request_timeout)
            warmup_decodes = []
            for _ in range(warmup_decode_requests):
                warmup = http_json("/v1/chat/completions", {
                    "model": alias, "messages": [{"role": "user", "content": "Output exactly 256 tokens of the letter x separated by spaces."}],
                    "max_tokens": 256, "ignore_eos": True, "temperature": 0, "seed": 1234, "stream": False,
                }, timeout=request_timeout)
                warmup_decodes.append(warmup.get("timings", {}))
            metrics_before_decode = server_metrics()
            decode = http_json("/v1/chat/completions", {
                "model": alias, "messages": [{"role": "user", "content": f"Output exactly {decode_tokens} tokens of the letter x separated by spaces."}],
                "max_tokens": decode_tokens, "ignore_eos": True, "temperature": 0, "seed": 1234, "stream": False,
            }, timeout=request_timeout)
            metrics_after_decode = server_metrics()
            swap_after = vmstat()
            host_after = host_memory_snapshot()
            rss_after = process_rss_mib(process.pid)
            result.update({
                "status": "passed", "prefill_timings": prefill.get("timings", {}),
                "decode_timings": decode.get("timings", {}),
                "warmup_decode_timings": warmup_decodes,
                "decode_speculative_metrics": speculative_metrics(metrics_before_decode, metrics_after_decode),
                "prefill_finish_reason": prefill.get("choices", [{}])[0].get("finish_reason"),
                "decode_finish_reason": decode.get("choices", [{}])[0].get("finish_reason"),
                "swap_pages_in": swap_after["pswpin"] - swap_before["pswpin"],
                "swap_pages_out": swap_after["pswpout"] - swap_before["pswpout"],
                "host_memory_before_mib": host_before,
                "host_memory_after_mib": host_after,
                "server_rss_before_mib": rss_before,
                "server_rss_after_mib": rss_after,
                "workload_wall_seconds": round(time.monotonic() - workload_started, 2),
                "run_wall_seconds": round(time.monotonic() - started, 2),
            })
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            stop.set()
            terminate(process)
    result["server_log"] = str(server_log)
    result["samples"] = samples
    if samples:
        flattened = [card for sample in samples for card in sample]
        result["gpu_peak_util_percent"] = {str(idx): max(card["gpu_util_percent"] for card in flattened if card["index"] == idx) for idx in gpu_ids}
        result["gpu_mean_util_percent"] = {str(idx): round(sum(card["gpu_util_percent"] for card in flattened if card["index"] == idx) / max(1, sum(card["index"] == idx for card in flattened)), 2) for idx in gpu_ids}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--two-gpu-only", action="store_true")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--model-path", type=Path, default=MODEL)
    parser.add_argument("--alias", default=ALIAS)
    parser.add_argument("--llama-path", type=Path, default=LLAMA)
    parser.add_argument("--override-tensor", default=None)
    parser.add_argument("--omit-n-cpu-moe", action="store_true")
    parser.add_argument("--case-split-mode", choices=("layer", "row", "tensor"))
    parser.add_argument("--case-tensor-split")
    parser.add_argument("--cache-type-k", default="q8_0")
    parser.add_argument("--cache-type-v", default="q8_0")
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--warmup-decode-requests", type=int, default=0)
    parser.add_argument("--stability-requests", type=int, default=0)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--server-options", nargs=argparse.REMAINDER, default=[])
    parser.add_argument("--server-env", action="append", default=[])
    args = parser.parse_args()
    args.server_environment = {}
    for item in args.server_env:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise SystemExit(f"invalid --server-env value: {item}")
        args.server_environment[key] = value
    if args.decode_tokens < 1 or args.warmup_decode_requests < 0 or args.stability_requests < 0 or args.request_timeout < 1:
        raise SystemExit("decode token counts must be non-negative")
    if not args.model_path.is_file():
        raise SystemExit(f"missing model: {args.model_path}")
    if not args.llama_path.is_file():
        raise SystemExit(f"missing llama-server: {args.llama_path}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []
    cases = [case for case in CASES if len(case[1].split(",")) == 2] if args.two_gpu_only else list(CASES)
    if args.cases:
        known = {case[0] for case in CASES}
        missing = [name for name in args.cases if name not in known]
        if missing:
            raise SystemExit(f"unknown case IDs: {missing}")
        cases = [case for case in cases if case[0] in args.cases]
    if not cases:
        raise SystemExit("no performance cases selected")
    for case in cases:
        name, gpus, split_mode, tensor_split, n_cpu_moe, main_gpu = case
        if args.case_split_mode:
            split_mode = args.case_split_mode
        if args.case_tensor_split:
            tensor_split = args.case_tensor_split
        case = (name, gpus, split_mode, tensor_split, n_cpu_moe, main_gpu)
        result = run_case(case, args.outdir, args.prompt_tokens, args.threads, args.model_path, args.alias, args.llama_path, args.override_tensor, args.omit_n_cpu_moe, args.cache_type_k, args.cache_type_v, args.ctx_size, args.server_options, args.server_environment, args.decode_tokens, args.warmup_decode_requests, args.stability_requests, args.request_timeout)
        all_results.append(result)
        (args.outdir / "results.json").write_text(json.dumps({"results": all_results}, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in result.items() if key != "samples"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
