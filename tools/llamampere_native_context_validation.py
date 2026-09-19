#!/usr/bin/env python3
"""Validate the native llamAmpere context without touching production services."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


TARGET_TOKENS = 261_760
CONTEXT_TOKENS = 262_144
DEFAULT_BINARY_SHA256 = "bcee5b8939763da8f87e87d77325142d9b2d4b686036bb4450d3c4754d77ec46"
DEFAULT_MODEL_SHA256 = "5cf05ad901dcaa76f41db13a5629146ed882219339377a80e37b12a8528d963b"
DEFAULT_VOCAB_SHA256 = "8405ff0f8970da24b72d68c9e61fa4309fdcf3719ecc1d7626f0ae8bd2a8e326"
SENTINEL_BEGIN = "BEGIN_NATIVE_262K_SENTINEL"
SENTINEL_END = "END_NATIVE_262K_SENTINEL"


@dataclass(frozen=True)
class RunResult:
    gpu: int
    port: int
    run: int
    response: dict[str, object] | None
    error: str | None
    metrics: list[dict[str, object]]
    host_metrics: list[dict[str, object]]
    log_path: str
    cleanup_ok: bool


def request_json(base_url: str, path: str, payload: dict[str, object] | None = None, timeout: float = 30.0) -> dict[str, object]:
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} did not return a JSON object")
    return value


def wait_ready(base_url: str, process: subprocess.Popen[bytes], timeout: float = 900.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with status {process.returncode}")
        try:
            request_json(base_url, "/health", timeout=5.0)
            return
        except (OSError, URLError, ValueError, RuntimeError):
            time.sleep(2)
    raise TimeoutError(f"server did not become ready at {base_url}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} hash mismatch: expected {expected}, got {actual}")


def token_count(base_url: str, text: str) -> int:
    value = request_json(base_url, "/tokenize", {"content": text}, timeout=120.0)
    tokens = value.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError("/tokenize response did not contain a token list")
    return len(tokens)


def build_prompt(base_url: str, output_dir: Path) -> tuple[str, int]:
    prefix = (
        f"{SENTINEL_BEGIN}\n"
        "This is a deterministic native-context capacity test. Read the complete "
        "prompt before answering. Preserve the sentinel values exactly.\n"
    )
    unit = (
        "Record code: def normalize(value, offset): return value + offset. "
        "Record neutral: blue cedar seven is a fixed observation. "
        "Record rule: preserve ordering and do not invent records.\n"
    )
    suffix = (
        "\nAfter reading the complete context, reply with exactly two lines. "
        f"The first line must be {SENTINEL_BEGIN}. "
        f"The second line must be {SENTINEL_END}.\n{SENTINEL_END}\n"
    )

    sample_count = max(1, token_count(base_url, prefix + unit * 64 + suffix) - token_count(base_url, prefix + suffix))
    unit_tokens = max(1, sample_count // 64)
    cycles = max(1, (TARGET_TOKENS - token_count(base_url, prefix + suffix)) // unit_tokens)
    prompt = prefix + unit * cycles + suffix
    count = token_count(base_url, prompt)
    for _ in range(12):
        if count >= TARGET_TOKENS and count <= CONTEXT_TOKENS - 128:
            break
        delta = TARGET_TOKENS - count
        cycles = max(1, cycles + max(1, abs(delta) // unit_tokens) * (1 if delta > 0 else -1))
        prompt = prefix + unit * cycles + suffix
        count = token_count(base_url, prompt)
    if count < TARGET_TOKENS or count > CONTEXT_TOKENS - 128:
        raise RuntimeError(f"could not construct a safe prompt: {count} tokens")
    (output_dir / "native-context-prompt.txt").write_text(prompt, encoding="utf-8")
    (output_dir / "native-context-prompt.json").write_text(
        json.dumps(
            {
                "target_tokens": TARGET_TOKENS,
                "actual_tokens": count,
                "context_tokens": CONTEXT_TOKENS,
                "sentinel_begin": SENTINEL_BEGIN,
                "sentinel_end": SENTINEL_END,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return prompt, count


def sample_system(gpu: int, pid: int) -> tuple[dict[str, object], dict[str, object]]:
    timestamp = time.time()
    metrics: dict[str, object] = {"timestamp": timestamp, "gpu": gpu}
    try:
        command = [
            "/usr/bin/nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=index,uuid,memory.used,power.draw,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        row = next(csv.reader(subprocess.check_output(command, text=True).splitlines()))
        for name, value in zip(("index", "uuid", "memory_used_mib", "power_w", "temperature_c", "utilization_pct"), row):
            metrics[name] = value.strip()
    except (OSError, subprocess.CalledProcessError, StopIteration):
        metrics["nvidia_smi_error"] = True

    host: dict[str, object] = {"timestamp": timestamp}
    try:
        memory = {}
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            name, value, *_ = line.split()
            if name in {"MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:"}:
                memory[name[:-1]] = int(value) * 1024
        host.update(memory)
    except (OSError, ValueError):
        host["meminfo_error"] = True
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="ascii")
        for line in status.splitlines():
            if line.startswith(("VmRSS:", "VmSize:")):
                name, value, *_ = line.split()
                host[name[:-1]] = int(value) * 1024
    except (OSError, ValueError):
        host["process_status_error"] = True
    return metrics, host


def sample_loop(gpu: int, pid: int, stop: threading.Event, metrics: list[dict[str, object]], host_metrics: list[dict[str, object]]) -> None:
    while not stop.is_set():
        gpu_sample, host_sample = sample_system(gpu, pid)
        metrics.append(gpu_sample)
        host_metrics.append(host_sample)
        stop.wait(2.0)


def stop_process(process: subprocess.Popen[bytes]) -> bool:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)
    return process.poll() is not None


def port_is_free(port: int) -> bool:
    try:
        result = subprocess.run(
            ["ss", "-H", "-ltn", f"sport = :{port}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return not result.stdout.strip()


def gpu_is_free(gpu: int) -> bool:
    try:
        result = subprocess.run(
            [
                "/usr/bin/nvidia-smi",
                f"--id={gpu}",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return not result.stdout.strip()


def wait_resources_free(gpu: int, port: int, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port_is_free(port) and gpu_is_free(gpu):
            return True
        time.sleep(1.0)
    return port_is_free(port) and gpu_is_free(gpu)


def cleanup_server(process: subprocess.Popen[bytes], gpu: int, port: int) -> bool:
    return stop_process(process) and wait_resources_free(gpu, port)


def run_once(args: argparse.Namespace, prompt: str, prompt_tokens: int, gpu: int, port: int, run: int, output_dir: Path) -> RunResult:
    base_url = f"http://127.0.0.1:{port}"
    log_path = output_dir / f"gpu{gpu}-run{run}.server.log"
    command = [
        str(args.binary),
        "--model", str(args.model),
        "--ctx-size", str(CONTEXT_TOKENS),
        "--batch-size", "4096",
        "--ubatch-size", "1024",
        "--threads", "8",
        "--threads-batch", "8",
        "--n-gpu-layers", "99",
        "--flash-attn", "on",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "turbo3",
        "--parallel", "1",
        "--jinja",
        "--fit", "off",
        "--cache-prompt",
        "--cache-ram", "8192",
        "--ctx-checkpoints", "24",
        "--checkpoint-min-step", "10240",
        "--spec-type", "draft-mtp",
        "--spec-draft-n-max", "3",
        "--spec-draft-p-min", "0",
        "--spec-draft-type-k", "q8_0",
        "--spec-draft-type-v", "q8_0",
        "--spec-draft-vocab-map", str(args.vocab_map),
        "--device", "CUDA0",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--metrics",
        "--perf",
        "--no-ui",
        "--log-file", str(log_path),
    ]
    env = os.environ.copy()
    env.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(gpu), "GGML_Q8_TURBO3_MMA_FUSED": "1"})
    process = subprocess.Popen(command, cwd=args.binary.parent.parent.parent, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    metrics: list[dict[str, object]] = []
    host_metrics: list[dict[str, object]] = []
    stop = threading.Event()
    sampler = threading.Thread(target=sample_loop, args=(gpu, process.pid, stop, metrics, host_metrics), daemon=True)
    try:
        sampler.start()
        wait_ready(base_url, process)
        props = request_json(base_url, "/props", timeout=30.0)
        generation_settings = props.get("default_generation_settings")
        context = generation_settings.get("n_ctx") if isinstance(generation_settings, dict) else None
        if context != CONTEXT_TOKENS:
            raise RuntimeError(f"server reported context {context!r}, expected {CONTEXT_TOKENS}")
        response = request_json(
            base_url,
            "/completion",
            {
                "prompt": prompt,
                "n_predict": 128,
                "temperature": 0.0,
                "top_k": 0,
                "top_p": 1.0,
                "min_p": 0.0,
                "seed": 3817,
                "ignore_eos": False,
                "stream": False,
                "cache_prompt": False,
            },
            timeout=3600.0,
        )
        response["_prompt_tokens_expected"] = prompt_tokens
        return RunResult(
            gpu,
            port,
            run,
            response,
            None,
            metrics,
            host_metrics,
            str(log_path),
            cleanup_server(process, gpu, port),
        )
    except (OSError, URLError, RuntimeError, TimeoutError, ValueError) as error:
        return RunResult(
            gpu,
            port,
            run,
            None,
            repr(error),
            metrics,
            host_metrics,
            str(log_path),
            cleanup_server(process, gpu, port),
        )
    finally:
        stop.set()
        sampler.join(timeout=5)


def run_concurrent(
    args: argparse.Namespace,
    prompt: str,
    prompt_tokens: int,
    output_dir: Path,
) -> list[RunResult]:
    results: list[RunResult] = []
    result_lock = threading.Lock()

    def worker(index: int, gpu: int) -> None:
        result = run_once(args, prompt, prompt_tokens, gpu, args.port_base + 10 + index, 3, output_dir)
        with result_lock:
            results.append(result)

    threads = [
        threading.Thread(target=worker, args=(index, gpu), daemon=True)
        for index, gpu in enumerate(args.gpus)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return sorted(results, key=lambda result: result.gpu)


def validate_result(result: RunResult) -> list[str]:
    failures: list[str] = []
    if result.error:
        failures.append(result.error)
    response = result.response or {}
    if response.get("truncated") is not False:
        failures.append(f"truncated={response.get('truncated')!r}")
    evaluated = response.get("tokens_evaluated")
    evaluated_count = evaluated if isinstance(evaluated, int) else 0
    if evaluated_count < TARGET_TOKENS:
        failures.append(f"tokens_evaluated={evaluated!r}")
    expected = response.get("_prompt_tokens_expected")
    if not isinstance(expected, int):
        failures.append(f"_prompt_tokens_expected={expected!r}")
    elif evaluated_count != expected:
        failures.append(f"tokens_evaluated={evaluated!r}, expected={expected}")
    content = str(response.get("content", ""))
    if SENTINEL_BEGIN not in content or SENTINEL_END not in content:
        failures.append("completion did not contain both sentinels")
    if not result.cleanup_ok:
        failures.append("server cleanup failed")
    if not result.metrics:
        failures.append("no GPU samples recorded")
    if not result.host_metrics:
        failures.append("no host samples recorded")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vocab-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs-per-gpu", type=int, default=2)
    parser.add_argument("--gpu", type=int, action="append", dest="gpus", default=[])
    parser.add_argument("--port-base", type=int, default=18121)
    parser.add_argument("--expected-binary-sha256", default=DEFAULT_BINARY_SHA256)
    parser.add_argument("--expected-model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--expected-vocab-sha256", default=DEFAULT_VOCAB_SHA256)
    args = parser.parse_args()
    args.gpus = args.gpus or [1, 2]
    if args.runs_per_gpu < 1:
        parser.error("--runs-per-gpu must be positive")
    args.binary = args.binary.resolve()
    args.model = args.model.resolve()
    args.vocab_map = args.vocab_map.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    verify_hash(args.binary, args.expected_binary_sha256, "runtime")
    verify_hash(args.model, args.expected_model_sha256, "model")
    verify_hash(args.vocab_map, args.expected_vocab_sha256, "vocabulary map")

    probe_binary = [str(args.binary), "--help"]
    subprocess.run(probe_binary, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results: list[RunResult] = []
    failures: dict[str, list[str]] = {}

    # Build the prompt once from a fresh server on the first GPU, then reuse it.
    prompt_server_port = args.port_base + 90
    env = os.environ.copy()
    env.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(args.gpus[0]), "GGML_Q8_TURBO3_MMA_FUSED": "1"})
    prompt_log = args.output_dir / "prompt-builder.server.log"
    prompt_process = subprocess.Popen(
        [
            str(args.binary),
            "--model", str(args.model),
            "--ctx-size", str(CONTEXT_TOKENS),
            "--batch-size", "4096",
            "--ubatch-size", "1024",
            "--threads", "8",
            "--threads-batch", "8",
            "--n-gpu-layers", "99",
            "--flash-attn", "on",
            "--cache-type-k", "q8_0",
            "--cache-type-v", "turbo3",
            "--parallel", "1",
            "--jinja",
            "--fit", "off",
            "--cache-prompt",
            "--cache-ram", "8192",
            "--ctx-checkpoints", "24",
            "--checkpoint-min-step", "10240",
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", "3",
            "--spec-draft-p-min", "0",
            "--spec-draft-type-k", "q8_0",
            "--spec-draft-type-v", "q8_0",
            "--spec-draft-vocab-map", str(args.vocab_map),
            "--device", "CUDA0",
            "--host", "127.0.0.1",
            "--port", str(prompt_server_port),
            "--no-ui",
            "--log-file", str(prompt_log),
        ],
        cwd=args.binary.parent.parent.parent,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    prompt_cleanup_ok = False
    try:
        wait_ready(f"http://127.0.0.1:{prompt_server_port}", prompt_process)
        prompt, prompt_tokens = build_prompt(f"http://127.0.0.1:{prompt_server_port}", args.output_dir)
    finally:
        prompt_cleanup_ok = cleanup_server(prompt_process, args.gpus[0], prompt_server_port)
    if not prompt_cleanup_ok:
        raise RuntimeError("prompt-builder server cleanup failed")
    for gpu_index, gpu in enumerate(args.gpus):
        for run in range(1, args.runs_per_gpu + 1):
            result = run_once(args, prompt, prompt_tokens, gpu, args.port_base + gpu_index, run, args.output_dir)
            results.append(result)
            errors = validate_result(result)
            if errors:
                failures[f"gpu{gpu}-run{run}"] = errors
    if len(args.gpus) == 2:
        for result in run_concurrent(args, prompt, prompt_tokens, args.output_dir):
            results.append(result)
            errors = validate_result(result)
            if errors:
                failures[f"gpu{result.gpu}-concurrent"] = errors

    output = {
        "context_tokens": CONTEXT_TOKENS,
        "target_tokens": TARGET_TOKENS,
        "prompt_tokens": prompt_tokens,
        "prompt_builder_cleanup_ok": prompt_cleanup_ok,
        "runs": [result.__dict__ for result in results],
        "failures": failures,
    }
    (args.output_dir / "native-context-results.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for result in results:
        (args.output_dir / f"gpu{result.gpu}-run{result.run}.gpu.json").write_text(json.dumps(result.metrics, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / f"gpu{result.gpu}-run{result.run}.host.json").write_text(json.dumps(result.host_metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
