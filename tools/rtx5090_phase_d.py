#!/usr/bin/env python3
"""Run guarded RTX 5090 Phase D temporary replica or TP2 measurements."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from transformers import AutoTokenizer


GPU_FIELDS = (
    "index",
    "uuid",
    "name",
    "pci.bus_id",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "fan.speed",
    "utilization.gpu",
    "clocks.sm",
)
PRODUCTION_SERVICE = "llama-qwen3.8-q4-native.service"
RECOVERY_GSM8K_DATASET = Path("/home/michel/.cache/sgl_eval/gsm8k/test.jsonl")
VLLM_COUNTER_NAMES = frozenset(
    {
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
        "vllm:external_prefix_cache_queries_total",
        "vllm:external_prefix_cache_hits_total",
        "vllm:prompt_tokens_total",
        "vllm:prompt_tokens_cached_total",
        "vllm:prompt_tokens_by_source_total",
        "vllm:generation_tokens_total",
        "vllm:request_success_total",
        "vllm:spec_decode_num_drafts_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_per_pos_total",
    }
)
VLLM_SCHEDULER_METRIC_NAMES = frozenset(
    {
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
        "vllm:num_preemptions_total",
        "vllm:request_queue_time_seconds_count",
        "vllm:request_queue_time_seconds_sum",
    }
)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def interval_summary(values: list[float]) -> dict[str, object]:
    return {
        "sample_count": len(values),
        "p50_seconds": percentile(values, 0.50),
        "p95_seconds": percentile(values, 0.95),
        "p99_seconds": percentile(values, 0.99),
    }


def run_text(command: list[str], timeout: int = 30) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=timeout)


def gpu_snapshots(gpus: tuple[int, ...]) -> dict[str, dict[str, int | float | str | None]]:
    output = run_text(
        ["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits"], timeout=10
    )
    snapshots: dict[str, dict[str, int | float | str | None]] = {}
    for row in csv.reader(io.StringIO(output)):
        if len(row) != len(GPU_FIELDS):
            continue
        index = int(row[0].strip())
        if index not in gpus:
            continue
        snapshot: dict[str, int | float | str | None] = {}
        for field, raw in zip(GPU_FIELDS, row):
            value = raw.strip()
            key = field.replace(".", "_")
            if field in {"uuid", "name", "pci.bus_id"}:
                snapshot[key] = value
            elif value in {"", "N/A", "Not Supported"}:
                snapshot[key] = None
            else:
                number = float(value)
                snapshot[key] = int(number) if number.is_integer() else number
        snapshots[str(index)] = snapshot
    expected = {str(gpu) for gpu in gpus}
    if set(snapshots) != expected:
        raise RuntimeError(f"could not sample expected GPUs {gpus}: {snapshots}")
    return snapshots


def gpu_processes(gpus: tuple[int, ...]) -> dict[str, list[dict[str, int | str]]]:
    snapshots = gpu_snapshots(gpus)
    by_uuid = {str(snapshot["uuid"]): index for index, snapshot in snapshots.items()}
    result: dict[str, list[dict[str, int | str]]] = {str(gpu): [] for gpu in gpus}
    output = run_text(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout=10,
    )
    for row in csv.reader(io.StringIO(output)):
        if len(row) != 4 or row[0].strip() not in by_uuid:
            continue
        result[by_uuid[row[0].strip()]].append(
            {"pid": int(row[1].strip()), "process_name": row[2].strip(), "used_memory_mib": row[3].strip()}
        )
    return result


def host_snapshot() -> dict[str, int | float | None]:
    values: dict[str, int] = {}
    with open("/proc/meminfo", encoding="ascii") as stream:
        for line in stream:
            key, separator, value = line.partition(":")
            fields = value.split()
            if separator and fields and fields[0].isdigit():
                values[key] = int(fields[0])
    load = os.getloadavg()
    return {
        "load1": load[0],
        "load5": load[1],
        "load15": load[2],
        "mem_available_kib": values.get("MemAvailable"),
        "swap_free_kib": values.get("SwapFree"),
    }


def http_json(url: str, payload: dict | None = None, timeout: int = 30) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError(f"unexpected JSON object from {url}")
    return value


def http_text(url: str, timeout: int = 30) -> str:
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def assert_port_free(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(1)
        if connection.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"loopback port {port} is already in use")


def terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    # Give vLLM's API process a chance to coordinate worker shutdown first.
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def wait_ready(port: int, alias: str, processes: list[subprocess.Popen[str]], timeout: int) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        exited = [process.returncode for process in processes if process.poll() is not None]
        if exited:
            raise RuntimeError(f"server exited during startup: {exited}")
        try:
            models = http_json(f"http://127.0.0.1:{port}/v1/models", timeout=5)
            data = models.get("data")
            if isinstance(data, list) and any(item.get("id") == alias for item in data if isinstance(item, dict)):
                return models
        except (OSError, ValueError, urllib.error.HTTPError, urllib.error.URLError):
            pass
        time.sleep(1)
    raise TimeoutError(f"server on port {port} did not become ready as {alias}")


def response_summary(response: dict) -> dict[str, object]:
    choice = response.get("choices", [{}])[0]
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    content = message.get("content") if isinstance(message, dict) else None
    return {
        "usage": response.get("usage"),
        "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
        "content_sha256": hashlib.sha256(content.encode()).hexdigest() if isinstance(content, str) else None,
        "content_preview": content[:160] if isinstance(content, str) else None,
        "tool_calls": message.get("tool_calls") if isinstance(message, dict) else None,
    }


def vllm_counter_samples(payload: str) -> dict[str, float]:
    samples: dict[str, float] = {}
    for line in payload.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            sample, raw_value = line.rsplit(maxsplit=1)
            metric_name = sample.partition("{")[0]
            if metric_name in VLLM_COUNTER_NAMES:
                samples[sample] = float(raw_value)
        except ValueError:
            continue
    return samples


def vllm_scheduler_samples(payload: str) -> dict[str, float]:
    samples: dict[str, float] = {}
    for line in payload.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            sample, raw_value = line.rsplit(maxsplit=1)
            if sample.partition("{")[0] in VLLM_SCHEDULER_METRIC_NAMES:
                samples[sample] = float(raw_value)
        except ValueError:
            continue
    return samples


def counter_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, dict[str, float | None]]:
    return {
        sample: {
            "before": before.get(sample),
            "after": after.get(sample),
            "delta": after[sample] - before[sample] if sample in before and sample in after else None,
        }
        for sample in sorted(set(before) | set(after))
    }


def counter_delta_total(
    deltas: dict[str, dict[str, float | None]], metric_name: str, required_label: str | None = None
) -> float | None:
    values = [
        value["delta"]
        for sample, value in deltas.items()
        if sample.partition("{")[0] == metric_name
        and (required_label is None or required_label in sample)
        and value["delta"] is not None
    ]
    return sum(values) if values else None


def metric_total(samples: dict[str, float], metric_name: str) -> float | None:
    values = [value for sample, value in samples.items() if sample.partition("{")[0] == metric_name]
    return sum(values) if values else None


def vllm_runtime_info(vllm: Path) -> dict[str, object]:
    python = vllm.with_name("python")
    if not python.is_file():
        return {"status": "error", "error": f"companion interpreter is unavailable: {python}"}
    command = [
        str(python),
        "-c",
        (
            "import json, torch, vllm; "
            "import flashinfer; "
            "print(json.dumps({'vllm': vllm.__version__, 'torch': torch.__version__, "
            "'cuda': torch.version.cuda, 'flashinfer': flashinfer.__version__}))"
        ),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, check=False, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "error", "error": f"{type(error).__name__}: {error}", "command": command}
    if completed.returncode != 0:
        return {"status": "error", "command": command, "output": completed.stdout + completed.stderr}
    try:
        return {"status": "captured", "command": command, **json.loads(completed.stdout)}
    except json.JSONDecodeError as error:
        return {"status": "error", "command": command, "error": f"JSONDecodeError: {error}"}


class PhaseDRun:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.gpus = tuple(args.gpus)
        self.ports = (args.port, args.port + 1) if args.engine == "llama-replicas" else (args.port,)
        self.request_ports = (
            (args.request_port, args.request_port + 1)
            if args.engine == "llama-replicas" and args.request_port is not None
            else (args.request_port,)
            if args.request_port is not None
            else self.ports
        )
        self.metrics_port = args.metrics_port or self.ports[0]
        if args.engine == "llama-replicas":
            self.aliases = (f"qwen3.8-27b-q4-phase-d-gpu{self.gpus[0]}", f"qwen3.8-27b-q4-phase-d-gpu{self.gpus[1]}")
        elif args.engine == "sglang":
            # SGLang exposes the checkpoint path as the OpenAI model ID.
            self.aliases = (str(args.model),)
        else:
            self.aliases = (f"qwen3.8-27b-nvfp4-{args.engine}-tp2-phase-d",)
        self.processes: list[subprocess.Popen[str]] = []
        self.logs: list[object] = []
        self.samples: list[dict[str, object]] = []
        self.stop_sampling = threading.Event()
        self.thermal_stop = threading.Event()
        self.first_decode_token = threading.Event()
        self.server_ready_monotonic: float | None = None
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)

    def endpoints(self) -> list[tuple[int, str]]:
        return list(zip(self.request_ports, self.aliases))

    def server_endpoints(self) -> list[tuple[int, str]]:
        return list(zip(self.ports, self.aliases))

    def command(self, index: int = 0) -> tuple[list[str], dict[str, str]]:
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, self.gpus)),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            # FlashInfer's sampler JIT does not recognize Blackwell SM 12 yet.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
        env.pop("GGML_CUDA_ENABLE_UNIFIED_MEMORY", None)
        if self.args.engine == "vllm":
            if self.args.vllm_breakable_cudagraph is not None:
                env["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1" if self.args.vllm_breakable_cudagraph else "0"
            if self.args.vllm_logging_level is not None:
                env["VLLM_LOGGING_LEVEL"] = self.args.vllm_logging_level
            prefix_cache_arguments = []
            if self.args.vllm_enable_prefix_caching is not None:
                prefix_cache_arguments = [
                    "--enable-prefix-caching" if self.args.vllm_enable_prefix_caching else "--no-enable-prefix-caching"
                ]
            graph_arguments = []
            if self.args.vllm_enable_cuda_graph:
                if self.args.vllm_cudagraph_metrics:
                    graph_arguments.append("--cudagraph-metrics")
                if self.args.vllm_cudagraph_capture_sizes:
                    graph_arguments.extend(
                        [
                        "--cudagraph-capture-sizes",
                        *map(str, self.args.vllm_cudagraph_capture_sizes),
                        ]
                    )
            else:
                graph_arguments = ["--enforce-eager"]
            compilation_arguments = []
            if self.args.vllm_compilation_config is not None:
                compilation_arguments = ["--compilation-config", self.args.vllm_compilation_config]
            gdn_prefill_arguments = []
            if self.args.vllm_gdn_prefill_backend is not None:
                gdn_prefill_arguments = ["--gdn-prefill-backend", self.args.vllm_gdn_prefill_backend]
            scheduler_arguments = []
            if self.args.vllm_max_num_batched_tokens is not None:
                scheduler_arguments.extend(["--max-num-batched-tokens", str(self.args.vllm_max_num_batched_tokens)])
            if self.args.vllm_long_prefill_token_threshold is not None:
                scheduler_arguments.extend(
                    ["--long-prefill-token-threshold", str(self.args.vllm_long_prefill_token_threshold)]
                )
            if self.args.vllm_enable_chunked_prefill is not None:
                scheduler_arguments.append(
                    "--enable-chunked-prefill" if self.args.vllm_enable_chunked_prefill else "--no-enable-chunked-prefill"
                )
            speculative_arguments = []
            if self.args.vllm_spec_tokens is not None:
                speculative_arguments = ["--spec-method", "mtp", "--spec-tokens", str(self.args.vllm_spec_tokens)]
            return (
                [
                    str(self.args.vllm),
                    "serve",
                    str(self.args.model),
                    "--trust-remote-code",
                    "--tensor-parallel-size",
                    "2",
                    "--max-model-len",
                    str(self.args.context_tokens),
                    "--gpu-memory-utilization",
                    str(self.args.gpu_memory_utilization),
                    "--max-num-seqs",
                    str(self.args.max_num_seqs),
                    "--kv-cache-dtype",
                    "fp8_e4m3",
                    "--attention-config",
                    self.args.vllm_attention_config,
                    *prefix_cache_arguments,
                    *graph_arguments,
                    *compilation_arguments,
                    *gdn_prefill_arguments,
                    *scheduler_arguments,
                    *speculative_arguments,
                    "--reasoning-parser",
                    "qwen3",
                    "--tool-call-parser",
                    "qwen3_coder",
                    "--enable-auto-tool-choice",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(self.ports[0]),
                    "--served-model-name",
                    self.aliases[0],
                ],
                env,
            )
        if self.args.engine == "sglang":
            if self.args.mamba_ssm_dtype:
                mamba_arguments = ["--mamba-ssm-dtype", self.args.mamba_ssm_dtype]
            else:
                mamba_arguments = []
            return (
                [
                    str(self.args.sglang),
                    "serve",
                    "--model-path",
                    str(self.args.model),
                    "--trust-remote-code",
                    "--tp-size",
                    "2",
                    "--context-length",
                    str(self.args.context_tokens),
                    "--quantization",
                    "modelopt",
                    "--kv-cache-dtype",
                    "fp8_e4m3",
                    "--mem-fraction-static",
                    str(self.args.gpu_memory_utilization),
                    "--chunked-prefill-size",
                    str(self.args.chunked_prefill_size),
                    "--disable-cuda-graph",
                    "--reasoning-parser",
                    "qwen3",
                    "--tool-call-parser",
                    "qwen3_coder",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(self.ports[0]),
                    *mamba_arguments,
                ],
                env,
            )
        gpu = self.gpus[index]
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        return (
            [
                str(self.args.llama),
                "--model",
                str(self.args.gguf_model),
                "--alias",
                self.aliases[index],
                "--host",
                "127.0.0.1",
                "--port",
                str(self.ports[index]),
                "--ctx-size",
                str(self.args.context_tokens),
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
            ],
            env,
        )

    def start(self) -> dict[str, object]:
        for port in self.ports:
            assert_port_free(port)
        count = 2 if self.args.engine == "llama-replicas" else 1
        commands: list[list[str]] = []
        for index in range(count):
            command, environment = self.command(index)
            commands.append(command)
            log = (self.args.output_dir / f"{self.args.engine}-{index}.server.log").open("w", encoding="utf-8")
            self.logs.append(log)
            self.processes.append(
                subprocess.Popen(
                    command,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    text=True,
                )
            )
        models = [wait_ready(port, alias, self.processes, self.args.startup_timeout) for port, alias in self.server_endpoints()]
        return {"commands": commands, "models": models, "pids": [process.pid for process in self.processes]}

    def stop(self) -> list[int | None]:
        for process in self.processes:
            terminate(process)
        for log in self.logs:
            log.close()
        return [process.returncode for process in self.processes]

    def restart(self) -> dict[str, object]:
        exit_codes = self.stop()
        self.processes = []
        self.logs = []
        server = self.start()
        return {"status": "passed", "prior_exit_codes": exit_codes, "server": server}

    def metrics_snapshot(self, label: str) -> dict[str, object]:
        port = self.metrics_port
        path = self.args.output_dir / f"metrics-{label}.prom"
        try:
            payload = http_text(f"http://127.0.0.1:{port}/metrics", timeout=30)
        except (OSError, RuntimeError, TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as error:
            return {"status": "error", "error": f"{type(error).__name__}: {error}"}
        path.write_text(payload, encoding="utf-8")
        interesting = [
            line
            for line in payload.splitlines()
            if line
            and not line.startswith("#")
            and any(
                term in line.lower()
                for term in ("spec", "draft", "accept", "prefix_cache", "prompt_tokens", "time_to_first", "inter_token")
            )
        ]
        return {
            "status": "captured",
            "path": str(path),
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "bytes": len(payload.encode()),
            "speculative_and_latency_samples": interesting,
            "vllm_counter_samples": vllm_counter_samples(payload),
            "vllm_scheduler_samples": vllm_scheduler_samples(payload),
            "cache_config_samples": [line for line in payload.splitlines() if line.startswith("vllm:cache_config_info{")],
        }

    def scheduler_snapshot(self, label: str) -> dict[str, object]:
        snapshot = self.metrics_snapshot(label)
        samples = snapshot.get("vllm_scheduler_samples", {})
        if not isinstance(samples, dict):
            samples = {}
        return {
            "monotonic": time.monotonic(),
            "metrics": snapshot,
            "running_requests": metric_total(samples, "vllm:num_requests_running"),
            "waiting_requests": metric_total(samples, "vllm:num_requests_waiting"),
            "kv_cache_usage_perc": metric_total(samples, "vllm:kv_cache_usage_perc"),
            "preemptions_total": metric_total(samples, "vllm:num_preemptions_total"),
            "request_queue_time_seconds_count": metric_total(samples, "vllm:request_queue_time_seconds_count"),
            "request_queue_time_seconds_sum": metric_total(samples, "vllm:request_queue_time_seconds_sum"),
        }

    def sampler(self) -> None:
        while not self.stop_sampling.wait(1):
            try:
                gpus = gpu_snapshots(self.gpus)
                self.samples.append({"timestamp": now(), "gpus": gpus, "host": host_snapshot()})
                temperatures = [snapshot.get("temperature_gpu") for snapshot in gpus.values()]
                if any(isinstance(value, (float, int)) and value >= self.args.thermal_stop_c for value in temperatures):
                    self.thermal_stop.set()
                    for process in self.processes:
                        terminate(process)
                    return
            except (OSError, RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
                pass

    def prompt(self, target_tokens: int, marker: str, directive: str) -> tuple[str, int]:
        prefix = f"Phase D unique context marker {marker}.\n"
        suffix = f"\n\n{directive}\n"
        pattern = "Neutral long-context throughput content for isolated RTX 5090 serving evaluation. "
        fixed = len(self.tokenizer.encode(prefix + suffix, add_special_tokens=False))
        pattern_tokens = max(1, len(self.tokenizer.encode(pattern, add_special_tokens=False)))
        repeats = max(1, math.ceil((target_tokens - fixed) / pattern_tokens))
        for _ in range(12):
            content = prefix + pattern * repeats + suffix
            count = len(self.tokenizer.encode(content, add_special_tokens=False))
            difference = target_tokens - count
            if abs(difference) <= 16:
                return content, count
            effective_pattern_tokens = max(1.0, (count - fixed) / repeats)
            repeats = max(1, repeats + round(difference / effective_pattern_tokens))
        content = prefix + pattern * repeats + suffix
        return content, len(self.tokenizer.encode(content, add_special_tokens=False))

    def chat(
        self,
        port: int,
        alias: str,
        prompt: str,
        max_tokens: int,
        timeout: int,
        *,
        ignore_eos: bool = False,
        messages: list[dict[str, str]] | None = None,
    ) -> tuple[dict, float]:
        payload = {
            "model": alias,
            "messages": messages if messages is not None else [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if ignore_eos:
            payload["ignore_eos"] = True
        started = time.monotonic()
        response = http_json(f"http://127.0.0.1:{port}/v1/chat/completions", payload, timeout=timeout)
        return response, time.monotonic() - started

    def normal_eos(self) -> dict[str, object]:
        port, alias = self.endpoints()[0]
        try:
            response, seconds = self.chat(port, alias, "Reply with exactly: phase-d-ok", 32, 180)
            summary = response_summary(response)
            content = summary.get("content_preview")
            passed = summary.get("finish_reason") == "stop" and isinstance(content, str) and "phase-d-ok" in content.lower()
            return {"status": "passed" if passed else "failed_contract", "request_seconds": seconds, **summary}
        except (OSError, RuntimeError, TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as error:
            return {"status": "error", "error": f"{type(error).__name__}: {error}"}

    def forced_tool(self) -> dict[str, object]:
        port, alias = self.endpoints()[0]
        payload = {
            "model": alias,
            "messages": [{"role": "user", "content": "Call lookup_symbol for the symbol Record and no other action."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_symbol",
                        "description": "Look up a source symbol.",
                        "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]},
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "lookup_symbol"}},
            "temperature": 0.0,
            "max_tokens": 64,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            started = time.monotonic()
            response = http_json(f"http://127.0.0.1:{port}/v1/chat/completions", payload, timeout=180)
            summary = response_summary(response)
            calls = summary.get("tool_calls")
            valid = False
            if isinstance(calls, list) and len(calls) == 1:
                function = calls[0].get("function") if isinstance(calls[0], dict) else None
                if isinstance(function, dict) and function.get("name") == "lookup_symbol":
                    try:
                        valid = json.loads(function.get("arguments", "{}")) == {"symbol": "Record"}
                    except (TypeError, ValueError):
                        pass
            return {"status": "passed" if valid else "failed_contract", "request_seconds": time.monotonic() - started, **summary}
        except (OSError, RuntimeError, TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as error:
            return {"status": "error", "error": f"{type(error).__name__}: {error}"}

    def cache_probe(self, label: str) -> dict[str, object]:
        expected = "phase-e-cache-ok"
        prompt, prompt_tokens = self.prompt(
            self.args.cache_prompt_tokens,
            label,
            f"Reply with exactly: {expected}",
        )
        port, alias = self.endpoints()[0]
        result: dict[str, object] = {
            "expected": expected,
            "prompt_content_tokens": prompt_tokens,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        }
        metrics_before_first = self.metrics_snapshot(f"{label}-before-first")
        result["metrics_before_first"] = metrics_before_first
        try:
            first, first_seconds = self.chat(port, alias, prompt, 32, self.args.request_timeout)
            metrics_after_first = self.metrics_snapshot(f"{label}-after-first")
            second, second_seconds = self.chat(port, alias, prompt, 32, self.args.request_timeout)
            metrics_after_second = self.metrics_snapshot(f"{label}-after-second")
            first_summary = response_summary(first)
            second_summary = response_summary(second)
            first_content = first_summary.get("content_preview")
            second_content = second_summary.get("content_preview")
            passed = (
                first_summary.get("finish_reason") == "stop"
                and second_summary.get("finish_reason") == "stop"
                and isinstance(first_content, str)
                and isinstance(second_content, str)
                and first_content.strip() == expected
                and second_content.strip() == expected
            )
            result.update(
                {
                    "status": "passed" if passed else "failed_contract",
                    "first_request_seconds": first_seconds,
                    "second_request_seconds": second_seconds,
                    "first": first_summary,
                    "second": second_summary,
                    "responses_identical": first_content == second_content,
                    "metrics_after_first": metrics_after_first,
                    "metrics_after_second": metrics_after_second,
                    "counter_deltas": {
                        "first_request": counter_delta(
                            metrics_before_first.get("vllm_counter_samples", {}), metrics_after_first.get("vllm_counter_samples", {})
                        ),
                        "second_request": counter_delta(
                            metrics_after_first.get("vllm_counter_samples", {}), metrics_after_second.get("vllm_counter_samples", {})
                        ),
                    },
                }
            )
        except (OSError, RuntimeError, TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as error:
            result.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
        return result

    def cache_reproducer(self) -> dict[str, object]:
        probe = self.cache_probe("cache-repro")
        deltas = probe.get("counter_deltas", {})
        second_request = deltas.get("second_request", {}) if isinstance(deltas, dict) else {}
        prefix_hits = counter_delta_total(second_request, "vllm:prefix_cache_hits_total")
        local_cache_hits = counter_delta_total(
            second_request, "vllm:prompt_tokens_by_source_total", 'source="local_cache_hit"'
        )
        return {
            "status": probe.get("status"),
            "second_request_prefix_cache_hits": prefix_hits,
            "second_request_local_cache_hits": local_cache_hits,
            "cache_reuse_observed": prefix_hits is not None and prefix_hits > 0 and local_cache_hits is not None and local_cache_hits > 0,
            "probe": probe,
        }

    def cache_correctness(self) -> dict[str, object]:
        before_restart = self.cache_probe("cache-before-restart")
        try:
            restart = self.restart()
            after_restart = self.cache_probe("cache-after-restart")
        except (OSError, RuntimeError, TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as error:
            return {
                "status": "error",
                "before_restart": before_restart,
                "error": f"{type(error).__name__}: {error}",
            }
        passed = before_restart.get("status") == "passed" and after_restart.get("status") == "passed"
        return {
            "status": "passed" if passed else "failed_contract",
            "before_restart": before_restart,
            "restart": restart,
            "after_restart": after_restart,
        }

    def cuda_graph_evidence(self) -> dict[str, object]:
        log_text = ""
        for log in self.logs:
            log.flush()
            log_text += Path(log.name).read_text(encoding="utf-8", errors="replace")
        capture_lines = [
            line
            for line in log_text.splitlines()
            if "Capturing a cudagraph on (FULL," in line
            or "CG Capture: mode=FULL" in line
            or "Capturing CUDA graphs (FULL)" in line
        ]
        statistic_lines = [
            line for line in log_text.splitlines() if "CUDAGraph" in line or "| FULL" in line
        ]
        fallback_lines = [
            line
            for line in log_text.splitlines()
            if "Cudagraph is disabled under eager mode" in line
            or "Overriding cudagraph_mode" in line
            or "cudagraph_mode=none" in line
        ]
        pre_shutdown_log = log_text.split("[shutdown] API server: shutdown triggered", 1)[0]
        workload_error_lines = [
            line
            for line in pre_shutdown_log.splitlines()
            if re.search(r"\)\s+ERROR\s", line) or "Traceback (most recent call last)" in line
        ]
        full_capture_token_counts = sorted(
            {
                int(match.group(1))
                for line in capture_lines
                if (match := re.search(r"num_tokens=(\d+), num_reqs=(\d+)", line))
            }
        )
        full_capture_request_counts = sorted(
            {
                int(match.group(2))
                for line in capture_lines
                if (match := re.search(r"num_tokens=(\d+), num_reqs=(\d+)", line))
            }
        )
        full_capture_batch_sizes = sorted(
            {
                int(match.group(1))
                for line in capture_lines
                if (match := re.search(r"num_tokens=(\d+), num_reqs=(\d+)", line))
                and match.group(1) == match.group(2)
            }
        )
        full_runtime_token_counts = sorted(
            {
                int(match.group(1))
                for line in statistic_lines
                if (match := re.search(r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*\d+\s*\|\s*FULL\s*\|\s*\d+\s*\|", line))
            }
        )
        full_runtime_request_counts = sorted(
            {
                int(match.group(2))
                for line in statistic_lines
                if (match := re.search(r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*\d+\s*\|\s*FULL\s*\|\s*\d+\s*\|", line))
            }
        )
        full_runtime_batch_sizes = sorted(
            {
                int(match.group(1))
                for line in statistic_lines
                if (match := re.search(r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*\d+\s*\|\s*FULL\s*\|\s*\d+\s*\|", line))
                and match.group(1) == match.group(2)
            }
        )
        full_runtime_stats = "**CUDAGraph Stats:**" in log_text and "| FULL" in log_text
        full_capture_observed = bool(capture_lines)
        return {
            "status": "passed" if full_capture_observed and not fallback_lines and not workload_error_lines else "failed_graph_evidence",
            "full_capture_observed": full_capture_observed,
            "full_runtime_stats_observed": full_runtime_stats,
            "full_capture_token_counts": full_capture_token_counts,
            "full_capture_request_counts": full_capture_request_counts,
            "full_capture_batch_sizes": full_capture_batch_sizes,
            "full_runtime_token_counts": full_runtime_token_counts,
            "full_runtime_request_counts": full_runtime_request_counts,
            "full_runtime_batch_sizes": full_runtime_batch_sizes,
            "full_capture_events": capture_lines,
            "cudagraph_stat_lines": statistic_lines,
            "fallback_lines": fallback_lines,
            "workload_error_lines": workload_error_lines,
        }

    def vllm_throughput_intervals(self, reference: dt.datetime) -> list[dict[str, object]]:
        intervals: list[dict[str, object]] = []
        pattern = re.compile(
            r"\)\s+\w+\s+(\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}).*?"
            r"Avg prompt throughput:\s+([0-9.]+)\s+tokens/s,\s+"
            r"Avg generation throughput:\s+([0-9.]+)\s+tokens/s,\s+"
            r"Running:\s+(\d+)\s+reqs,\s+Waiting:\s+(\d+)\s+reqs"
        )
        for log in self.logs:
            log.flush()
            for line in Path(log.name).read_text(encoding="utf-8", errors="replace").splitlines():
                match = pattern.search(line)
                if match is None:
                    continue
                timestamp = dt.datetime.strptime(f"{reference.year}-{match.group(1)}", "%Y-%m-%d %H:%M:%S")
                if timestamp < reference - dt.timedelta(hours=12):
                    timestamp += dt.timedelta(days=1)
                intervals.append(
                    {
                        "timestamp": timestamp.isoformat(timespec="seconds"),
                        "offset_seconds": (timestamp - reference).total_seconds(),
                        "prompt_tokens_per_second": float(match.group(2)),
                        "generation_tokens_per_second": float(match.group(3)),
                        "running_requests": int(match.group(4)),
                        "waiting_requests": int(match.group(5)),
                    }
                )
        return intervals

    def smoke(self) -> dict[str, object]:
        normal_eos = self.normal_eos()
        normal_eos_repeat = self.normal_eos()
        forced_tool = self.forced_tool()
        graph_evidence = None
        if self.args.vllm_enable_cuda_graph:
            # CUDAGraph statistics are emitted by vLLM's periodic logger.
            time.sleep(12)
            graph_evidence = self.cuda_graph_evidence()
        deterministic_output = (
            normal_eos.get("status") == "passed"
            and normal_eos_repeat.get("status") == "passed"
            and normal_eos.get("content_sha256") == normal_eos_repeat.get("content_sha256")
        )
        passed = deterministic_output and forced_tool.get("status") == "passed"
        if graph_evidence is not None:
            passed = passed and graph_evidence.get("status") == "passed"
        return {
            "status": "passed" if passed else "failed_smoke",
            "normal_eos": normal_eos,
            "normal_eos_repeat": normal_eos_repeat,
            "deterministic_output": deterministic_output,
            "forced_tool": forced_tool,
            "cuda_graph_evidence": graph_evidence,
        }

    def gsm8k(self) -> dict[str, object]:
        if self.args.gsm8k_executable is None:
            raise RuntimeError("--gsm8k-executable is required for the GSM8K gate")
        port, alias = self.endpoints()[0]
        output_dir = self.args.output_dir / "gsm8k"
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.args.gsm8k_executable),
            "run",
            "gsm8k",
            "--base-url",
            f"http://127.0.0.1:{port}/v1",
            "--model",
            alias,
            "--api-key",
            "EMPTY",
            "--num-examples",
            str(self.args.gsm8k_examples),
            "--num-threads",
            str(self.args.max_num_seqs),
            "--max-tokens",
            str(self.args.gsm8k_max_tokens),
            "--temperature",
            "1.0",
            "--top-p",
            "0.95",
            "--seed",
            "0",
            "--thinking",
            "--chat-template-kwarg",
            "enable_thinking=true",
            "--out-dir",
            str(output_dir),
        ]
        log_path = output_dir / "sgl-eval.log"
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.args.gsm8k_timeout,
                check=False,
            )
        return {
            "status": "completed" if completed.returncode == 0 else "error",
            "command": command,
            "returncode": completed.returncode,
            "request_seconds": time.monotonic() - started,
            "examples": self.args.gsm8k_examples,
            "log": str(log_path),
            "output_dir": str(output_dir),
        }

    def recovery_gsm8k(self) -> dict[str, object]:
        port, alias = self.endpoints()[0]
        output_dir = self.args.output_dir / "gsm8k-recovery"
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.args.recovery_gsm8k_python or sys.executable),
            str(self.args.recovery_gsm8k_executable),
            "--base-url",
            f"http://127.0.0.1:{port}/v1",
            "--model",
            alias,
            "--dataset",
            str(self.args.recovery_gsm8k_dataset),
            "--count",
            str(self.args.recovery_gsm8k_examples),
            "--output-dir",
            str(output_dir),
            "--max-tokens",
            str(self.args.recovery_gsm8k_max_tokens),
            "--continuation-max-tokens",
            str(self.args.recovery_gsm8k_continuation_max_tokens),
            "--temperature",
            str(self.args.recovery_gsm8k_temperature),
            "--top-p",
            str(self.args.recovery_gsm8k_top_p),
            "--seed",
            str(self.args.recovery_gsm8k_seed),
            "--concurrency",
            str(self.args.recovery_gsm8k_concurrency),
            "--timeout",
            str(self.args.recovery_gsm8k_request_timeout),
            "--api-key",
            "EMPTY",
        ]
        log_path = output_dir / "evaluator.log"
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.args.recovery_gsm8k_run_timeout,
                check=False,
            )
        summary_path = output_dir / "summary.json"
        summary: dict[str, object] | None = None
        if summary_path.is_file():
            try:
                loaded = json.loads(summary_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    summary = loaded
            except json.JSONDecodeError:
                summary = None
        return {
            "status": "passed" if completed.returncode == 0 and summary and summary.get("complete") else "failed",
            "returncode": completed.returncode,
            "request_seconds": time.monotonic() - started,
            "command": command,
            "log": str(log_path),
            "output_dir": str(output_dir),
            "summary": summary,
        }

    def long_request(self, port: int, alias: str, target_tokens: int, label: str) -> dict[str, object]:
        prompt, content_tokens = self.prompt(target_tokens, label, "Reply with the exact phrase phase-d-long-ok.")
        result: dict[str, object] = {
            "label": label,
            "target_content_tokens": target_tokens,
            "prompt_content_tokens": content_tokens,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        }
        try:
            response, seconds = self.chat(
                port, alias, prompt, self.args.long_max_tokens, self.args.request_timeout, ignore_eos=True
            )
            result.update({"status": "passed", "request_seconds": seconds, **response_summary(response)})
        except (OSError, RuntimeError, TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as error:
            result.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
        return result

    def long_stream_request(self, port: int, alias: str, target_tokens: int, label: str) -> dict[str, object]:
        prompt, content_tokens = self.prompt(target_tokens, label, "Reply with the exact phrase phase-d-long-ok.")
        result: dict[str, object] = {
            "label": label,
            "target_content_tokens": target_tokens,
            "prompt_content_tokens": content_tokens,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        }
        result.update(self.stream(port, alias, prompt, self.args.long_max_tokens))
        return result

    def long_context_retrieval(self) -> dict[str, object]:
        expected = "phase-d-retrieval-raven-73"
        before, before_tokens = self.prompt(120000, "retrieval-before", "Continue reading the source context.")
        after, after_tokens = self.prompt(129000, "retrieval-after", "Continue reading the source context.")
        prompt = (
            before
            + f"\n\nRetrieval fact: the exact verification code is {expected}.\n\n"
            + after
            + "\n\nWhat is the exact verification code? Reply with only that code."
        )
        prompt_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        port, alias = self.endpoints()[0]
        result: dict[str, object] = {
            "expected": expected,
            "prompt_content_tokens": prompt_tokens,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "marker_after_content_tokens": before_tokens,
            "trailing_content_tokens": after_tokens,
        }
        try:
            response, seconds = self.chat(port, alias, prompt, 32, self.args.request_timeout)
            summary = response_summary(response)
            content = summary.get("content_preview")
            passed = summary.get("finish_reason") == "stop" and isinstance(content, str) and content.strip() == expected
            result.update({"status": "passed" if passed else "failed_retrieval", "request_seconds": seconds, **summary})
        except (OSError, RuntimeError, TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as error:
            result.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
        return result

    def capacity(self) -> dict[str, object]:
        endpoints = self.endpoints()
        jobs = []
        for index in range(self.args.capacity_concurrency):
            port, alias = endpoints[index % len(endpoints)]
            jobs.append((port, alias, self.args.capacity_prompt_tokens, f"capacity-{index}"))
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            request_method = self.long_stream_request if self.args.stream_capacity else self.long_request
            futures = [executor.submit(request_method, *job) for job in jobs]
            requests = [future.result() for future in as_completed(futures)]
        return {
            "requested_concurrency": self.args.capacity_concurrency,
            "prompt_target_tokens": self.args.capacity_prompt_tokens,
            "wall_seconds": time.monotonic() - started,
            "requests": requests,
            "passed_requests": sum(request["status"] == "passed" for request in requests),
            "status": "passed" if all(request["status"] == "passed" for request in requests) else "failed_capacity",
        }

    def decode_only(self) -> dict[str, object]:
        port, alias = self.endpoints()[0]
        contexts: list[tuple[str, str, int]] = []
        warm_requests: list[dict[str, object]] = []
        for index in range(self.args.decode_concurrency):
            context, context_tokens = self.prompt(
                self.args.decode_prompt_tokens,
                f"decode-context-{index}",
                "Read and retain the supplied context.",
            )
            contexts.append((f"decode-{index}", context, context_tokens))
        if self.args.decode_warm_prefixes:
            for label, context, context_tokens in contexts:
                try:
                    response, seconds = self.chat(
                        port,
                        alias,
                        context + "\n\nReply with exactly: phase-d-warm-ok.",
                        16,
                        self.args.request_timeout,
                    )
                    summary = response_summary(response)
                    warm_requests.append(
                        {
                            "label": label,
                            "context_content_tokens": context_tokens,
                            "status": "passed" if summary.get("finish_reason") == "stop" else "failed_warm",
                            "request_seconds": seconds,
                            **summary,
                        }
                    )
                except (OSError, RuntimeError, TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as error:
                    warm_requests.append({"label": label, "status": "error", "error": f"{type(error).__name__}: {error}"})
            if not all(request["status"] == "passed" for request in warm_requests):
                return {"status": "failed_warm", "prefix_warmup_enabled": True, "warm_requests": warm_requests}

        metrics_before_decode = self.metrics_snapshot("after-context-warm" if self.args.decode_warm_prefixes else "before-raw-decode")
        started = time.monotonic()
        timeline_started = dt.datetime.now()
        start_gate = threading.Barrier(len(contexts))
        with ThreadPoolExecutor(max_workers=self.args.decode_concurrency) as executor:
            futures = {
                executor.submit(
                    self.stream,
                    port,
                    alias,
                    context + "\n\nReply with the exact phrase phase-d-long-ok.",
                    self.args.decode_output_tokens,
                    timeline_origin=started,
                    start_gate=start_gate,
                ): (label, context_tokens)
                for label, context, context_tokens in contexts
            }
            responses = []
            for future in as_completed(futures):
                response = future.result()
                label, context_tokens = futures[future]
                response.update({"label": label, "context_content_tokens": context_tokens})
                responses.append(response)
        metrics_after_decode = self.metrics_snapshot("after-decode")
        counter_deltas = counter_delta(
            metrics_before_decode.get("vllm_counter_samples", {}), metrics_after_decode.get("vllm_counter_samples", {})
        )
        cache_neutral_required = not self.args.decode_warm_prefixes and self.args.vllm_enable_prefix_caching is False
        cache_neutral = {
            "required": cache_neutral_required,
            "prefix_cache_queries": counter_delta_total(counter_deltas, "vllm:prefix_cache_queries_total"),
            "prefix_cache_hits": counter_delta_total(counter_deltas, "vllm:prefix_cache_hits_total"),
            "local_cache_hit_tokens": counter_delta_total(
                counter_deltas, "vllm:prompt_tokens_by_source_total", 'source="local_cache_hit"'
            ),
        }
        cache_neutral["status"] = (
            "passed"
            if not cache_neutral_required
            or all(cache_neutral[key] == 0.0 for key in ("prefix_cache_queries", "prefix_cache_hits", "local_cache_hit_tokens"))
            else "failed_cache_neutrality"
        )
        speculation = None
        if self.args.vllm_spec_tokens is not None:
            drafted_tokens = counter_delta_total(counter_deltas, "vllm:spec_decode_num_draft_tokens_total")
            accepted_tokens = counter_delta_total(counter_deltas, "vllm:spec_decode_num_accepted_tokens_total")
            draft_steps = counter_delta_total(counter_deltas, "vllm:spec_decode_num_drafts_total")
            speculation = {
                "configured_speculative_tokens": self.args.vllm_spec_tokens,
                "draft_steps": draft_steps,
                "drafted_tokens": drafted_tokens,
                "accepted_tokens": accepted_tokens,
                "accepted_per_drafted_token": accepted_tokens / drafted_tokens if drafted_tokens else None,
            }
            speculation["status"] = (
                "passed"
                if draft_steps is not None and drafted_tokens is not None and drafted_tokens > 0 and accepted_tokens is not None
                else "missing_speculation_metrics"
            )
        completed_responses = [
            response
            for response in responses
            if response.get("status") == "passed" and response.get("finish_reason") == "length"
        ]
        decode_rates = [response["decode_tokens_per_second"] for response in completed_responses]
        ttfts = [response["ttft_seconds"] for response in completed_responses]
        p95_itls = [response["itl_p95_seconds"] for response in completed_responses if response["itl_p95_seconds"] is not None]
        per_user = {
            str(response["label"]): {
                "decode_tokens_per_second": response["decode_tokens_per_second"],
                "ttft_seconds": response["ttft_seconds"],
                "itl_p95_seconds": response["itl_p95_seconds"],
            }
            for response in sorted(completed_responses, key=lambda response: str(response["label"]))
        }
        first_token_offsets = [response.get("first_token_offset_seconds") for response in completed_responses]
        all_decode_offset = (
            max(first_token_offsets)
            if first_token_offsets and all(isinstance(value, float) for value in first_token_offsets)
            else None
        )
        prefill_interference_itl: dict[str, float | None] = {}
        if all_decode_offset is not None:
            for response in completed_responses:
                token_offsets = response.get("token_offsets_seconds")
                if not isinstance(token_offsets, list):
                    prefill_interference_itl[str(response["label"])] = None
                    continue
                intervals = [
                    later - earlier
                    for earlier, later in zip(token_offsets, token_offsets[1:])
                    if isinstance(earlier, float) and isinstance(later, float) and later <= all_decode_offset
                ]
                prefill_interference_itl[str(response["label"])] = percentile(intervals, 0.95)
        throughput_intervals = self.vllm_throughput_intervals(timeline_started) if self.args.engine == "vllm" else []
        stable_three_user_intervals = (
            [
                interval
                for interval in throughput_intervals
                if all_decode_offset is not None
                and interval["offset_seconds"] >= all_decode_offset
                and interval["running_requests"] == self.args.decode_concurrency
                and interval["waiting_requests"] == 0
                and interval["prompt_tokens_per_second"] == 0.0
            ]
            if self.args.decode_concurrency == 3
            else []
        )
        stable_three_user_rates = [interval["generation_tokens_per_second"] for interval in stable_three_user_intervals]
        decode_summary = {
            "per_user": per_user,
            "median_decode_tokens_per_second": statistics.median(decode_rates) if decode_rates else None,
            "aggregate_decode_tokens_per_second": sum(decode_rates) if decode_rates else None,
            "median_ttft_seconds": statistics.median(ttfts) if ttfts else None,
            "max_ttft_seconds": max(ttfts) if ttfts else None,
            "median_p95_itl_seconds": statistics.median(p95_itls) if p95_itls else None,
            "max_p95_itl_seconds": max(p95_itls) if p95_itls else None,
            "time_until_all_users_enter_decode_seconds": all_decode_offset,
            "p95_itl_while_another_user_prefills_seconds": prefill_interference_itl,
            "stable_three_user_decode": {
                "server_intervals": stable_three_user_intervals,
                "median_aggregate_tokens_per_second": statistics.median(stable_three_user_rates) if stable_three_user_rates else None,
                "min_aggregate_tokens_per_second": min(stable_three_user_rates) if stable_three_user_rates else None,
                "max_aggregate_tokens_per_second": max(stable_three_user_rates) if stable_three_user_rates else None,
            },
            "interference": {
                "decode_rate_min_tokens_per_second": min(decode_rates) if decode_rates else None,
                "decode_rate_max_tokens_per_second": max(decode_rates) if decode_rates else None,
                "decode_rate_max_to_min_ratio": max(decode_rates) / min(decode_rates) if decode_rates and min(decode_rates) else None,
            },
        }
        passed = (
            len(completed_responses) == len(responses)
            and cache_neutral["status"] == "passed"
            and (speculation is None or speculation["status"] == "passed")
        )
        finished = time.monotonic()
        result = {
            "status": "passed" if passed else "failed_decode",
            "prefix_warmup_enabled": self.args.decode_warm_prefixes,
            "requested_concurrency": self.args.decode_concurrency,
            "prompt_target_tokens": self.args.decode_prompt_tokens,
            "output_tokens": self.args.decode_output_tokens,
            "warm_requests": warm_requests,
            "metrics_before_decode": metrics_before_decode,
            "metrics_after_decode": metrics_after_decode,
            "counter_delta": counter_deltas,
            "cache_neutral": cache_neutral,
            "speculation": speculation,
            "decode_summary": decode_summary,
            "wall_seconds": finished - started,
            "total_completion_wall_seconds": finished - started,
            "requests": responses,
        }
        if self.args.decode_warm_prefixes:
            result["metrics_after_context_warm"] = metrics_before_decode
        return result

    def stream(
        self,
        port: int,
        alias: str,
        prompt: str,
        max_tokens: int,
        ready: threading.Event | None = None,
        *,
        timeline_origin: float | None = None,
        start_gate: threading.Barrier | None = None,
        ignore_eos: bool = True,
        messages: list[dict[str, str]] | None = None,
        timeout: int | None = None,
    ) -> dict[str, object]:
        payload = {
            "model": alias,
            "messages": messages if messages is not None else [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "ignore_eos": ignore_eos,
        }
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        if start_gate is not None:
            try:
                start_gate.wait(timeout=30)
            except threading.BrokenBarrierError as error:
                return {"status": "error", "error": f"BrokenBarrierError: {error}"}
        started = time.monotonic()
        first: float | None = None
        token_times: list[float] = []
        pieces: list[str] = []
        finish_reason: str | None = None
        try:
            with urllib.request.urlopen(request, timeout=timeout if timeout is not None else self.args.request_timeout) as response:
                admission_wait_header = response.headers.get("X-Inference-Gateway-Admission-Wait-Ms")
                for raw_line in response:
                    line = raw_line.decode("utf-8", "replace").strip()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    data = json.loads(line[6:])
                    choices = data.get("choices", [])
                    choice = choices[0] if choices else {}
                    if isinstance(choice, dict) and isinstance(choice.get("finish_reason"), str):
                        finish_reason = choice["finish_reason"]
                    delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
                    content = delta.get("content") if isinstance(delta, dict) else None
                    if not isinstance(content, str) or not content:
                        continue
                    observed = time.monotonic()
                    if first is None:
                        first = observed
                        if ready is not None:
                            ready.set()
                    token_times.append(observed)
                    pieces.append(content)
            if first is None:
                raise RuntimeError("stream completed without a content token")
            completed = time.monotonic()
            content = "".join(pieces)
            intervals = [later - earlier for earlier, later in zip(token_times, token_times[1:])]
            completion_token_estimate = len(self.tokenizer.encode(content, add_special_tokens=False))
            completion_tokens = max_tokens if finish_reason == "length" else completion_token_estimate
            decode_seconds = completed - first
            result: dict[str, object] = {
                "status": "passed",
                "ttft_seconds": first - started,
                "request_seconds": completed - started,
                "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "content_preview": content[:160],
                "content_chars": len(content),
                "stream_chunks": len(token_times),
                "itl_p50_seconds": percentile(intervals, 0.5),
                "itl_p95_seconds": percentile(intervals, 0.95),
                "itl_p99_seconds": percentile(intervals, 0.99),
                "finish_reason": finish_reason,
                "completion_token_estimate": completion_token_estimate,
                "completion_tokens": completion_tokens,
                "decode_tokens_per_second": completion_tokens / decode_seconds if decode_seconds else None,
                "gateway_admission_wait_ms": (
                    float(admission_wait_header) if admission_wait_header is not None else None
                ),
            }
            if timeline_origin is not None:
                result.update(
                    {
                        "request_start_offset_seconds": started - timeline_origin,
                        "first_token_offset_seconds": first - timeline_origin,
                        "completion_offset_seconds": completed - timeline_origin,
                        "request_start_monotonic": started,
                        "first_token_monotonic": first,
                        "completion_monotonic": completed,
                        "token_offsets_seconds": [observed - timeline_origin for observed in token_times],
                    }
                )
            return result
        except (OSError, RuntimeError, TimeoutError, ValueError, urllib.error.HTTPError, urllib.error.URLError) as error:
            if ready is not None:
                ready.set()
            return {"status": "error", "error": f"{type(error).__name__}: {error}"}

    def interference(self) -> dict[str, object]:
        port, alias = self.endpoints()[0]
        decode_prompt, decode_tokens = self.prompt(
            self.args.interference_decode_prompt_tokens, "decode", "Explain the marker in a short sentence."
        )
        prefill_prompt, prefill_tokens = self.prompt(
            self.args.interference_prefill_prompt_tokens, "interference-prefill", "Reply with phase-d-prefill-ok."
        )
        decode_result: dict[str, object] = {}
        self.first_decode_token.clear()
        timeline_started = time.monotonic()
        metrics_before = self.metrics_snapshot("before-prefill-decode-interference")

        def decode() -> None:
            decode_result.update(
                self.stream(
                    port,
                    alias,
                    decode_prompt,
                    self.args.interference_decode_tokens,
                    self.first_decode_token,
                    timeline_origin=timeline_started,
                )
            )

        thread = threading.Thread(target=decode, daemon=True)
        thread.start()
        if not self.first_decode_token.wait(timeout=300):
            thread.join(timeout=self.args.request_timeout + 30)
            return {
                "status": "failed_primary_decode_start",
                "decode_prompt_content_tokens": decode_tokens,
                "prefill_prompt_content_tokens": prefill_tokens,
                "decode": decode_result,
            }
        time.sleep(self.args.interference_established_decode_seconds)
        injected_at = time.monotonic() - timeline_started
        prefill_result = self.stream(
            port,
            alias,
            prefill_prompt,
            self.args.interference_prefill_tokens,
            timeline_origin=timeline_started,
        )
        thread.join(timeout=self.args.request_timeout + 30)
        metrics_after = self.metrics_snapshot("after-prefill-decode-interference")
        primary_offsets = decode_result.get("token_offsets_seconds")
        injected_first_token = prefill_result.get("first_token_offset_seconds")
        before_intervals: list[float] = []
        during_intervals: list[float] = []
        if isinstance(primary_offsets, list) and isinstance(injected_first_token, float):
            for earlier, later in zip(primary_offsets, primary_offsets[1:]):
                if not isinstance(earlier, float) or not isinstance(later, float):
                    continue
                if later <= injected_at:
                    before_intervals.append(later - earlier)
                elif earlier >= injected_at and later <= injected_first_token:
                    during_intervals.append(later - earlier)
        counter_deltas = counter_delta(
            metrics_before.get("vllm_counter_samples", {}), metrics_after.get("vllm_counter_samples", {})
        )
        cache_neutral = {
            "prefix_cache_queries": counter_delta_total(counter_deltas, "vllm:prefix_cache_queries_total"),
            "prefix_cache_hits": counter_delta_total(counter_deltas, "vllm:prefix_cache_hits_total"),
            "local_cache_hit_tokens": counter_delta_total(
                counter_deltas, "vllm:prompt_tokens_by_source_total", 'source="local_cache_hit"'
            ),
        }
        cache_neutral["status"] = (
            "passed"
            if self.args.vllm_enable_prefix_caching is not False
            or all(cache_neutral[key] == 0.0 for key in ("prefix_cache_queries", "prefix_cache_hits", "local_cache_hit_tokens"))
            else "failed_cache_neutrality"
        )
        passed = (
            decode_result.get("status") == "passed"
            and prefill_result.get("status") == "passed"
            and cache_neutral["status"] == "passed"
        )
        return {
            "status": "passed" if passed else "failed_interference",
            "decode_prompt_content_tokens": decode_tokens,
            "prefill_prompt_content_tokens": prefill_tokens,
            "established_decode_seconds": self.args.interference_established_decode_seconds,
            "prefill_injected_at_seconds": injected_at,
            "decode": decode_result,
            "prefill": prefill_result,
            "established_decode_p95_itl_seconds": percentile(before_intervals, 0.95),
            "decode_p95_itl_during_new_prefill_seconds": percentile(during_intervals, 0.95),
            "decode_itl_samples_before_new_prefill": len(before_intervals),
            "decode_itl_samples_during_new_prefill": len(during_intervals),
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "counter_delta": counter_deltas,
            "cache_neutral": cache_neutral,
        }

    def cached_continuation_interference(self) -> dict[str, object]:
        port, alias = self.endpoints()[0]
        warm_expected = "phase-d-cache-warm-ok"
        continuation_expected = "phase-d-cache-continuation-ok"
        active_prompt, active_tokens = self.prompt(
            self.args.cached_active_prompt_tokens,
            "cached-active-cold",
            "Reply with the exact phrase phase-d-cache-active-ok.",
        )
        base_prompt, base_tokens = self.prompt(
            self.args.cached_base_prompt_tokens,
            "cached-base",
            f"Reply with exactly: {warm_expected}",
        )
        append_prompt, append_tokens = self.prompt(
            self.args.cached_continuation_append_tokens,
            "cached-continuation",
            f"Reply with exactly: {continuation_expected}",
        )
        base_messages = [{"role": "user", "content": base_prompt}]
        result: dict[str, object] = {
            "active_prompt_content_tokens": active_tokens,
            "active_prompt_sha256": hashlib.sha256(active_prompt.encode()).hexdigest(),
            "cached_base_content_tokens": base_tokens,
            "cached_base_sha256": hashlib.sha256(base_prompt.encode()).hexdigest(),
            "continuation_append_content_tokens": append_tokens,
            "continuation_append_sha256": hashlib.sha256(append_prompt.encode()).hexdigest(),
        }
        metrics_before_warm = self.metrics_snapshot("before-cached-continuation-warm")
        cache_enabled = any(
            'enable_prefix_caching="True"' in sample
            for sample in metrics_before_warm.get("cache_config_samples", [])
            if isinstance(sample, str)
        )
        try:
            warm_response, warm_seconds = self.chat(
                port,
                alias,
                base_prompt,
                32,
                self.args.request_timeout,
                messages=base_messages,
            )
            warm_summary = response_summary(warm_response)
            warm_content = warm_summary.get("content_preview")
            warm_passed = (
                warm_summary.get("finish_reason") == "stop"
                and isinstance(warm_content, str)
                and warm_content.strip() == warm_expected
            )
            metrics_after_warm = self.metrics_snapshot("after-cached-continuation-warm")
        except (OSError, RuntimeError, TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as error:
            result.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
            return result

        warm_delta = counter_delta(
            metrics_before_warm.get("vllm_counter_samples", {}), metrics_after_warm.get("vllm_counter_samples", {})
        )
        warm_local_compute = counter_delta_total(
            warm_delta, "vllm:prompt_tokens_by_source_total", 'source="local_compute"'
        )
        warm_local_cache = counter_delta_total(
            warm_delta, "vllm:prompt_tokens_by_source_total", 'source="local_cache_hit"'
        )
        warm_prefix_hits = counter_delta_total(warm_delta, "vllm:prefix_cache_hits_total")

        self.first_decode_token.clear()
        timeline_started = time.monotonic()
        active_result: dict[str, object] = {}

        def decode_active() -> None:
            active_result.update(
                self.stream(
                    port,
                    alias,
                    active_prompt,
                    self.args.cached_active_output_tokens,
                    self.first_decode_token,
                    timeline_origin=timeline_started,
                )
            )

        active_thread = threading.Thread(target=decode_active, daemon=True)
        active_thread.start()
        if not self.first_decode_token.wait(timeout=300):
            active_thread.join(timeout=self.args.request_timeout + 30)
            result.update(
                {
                    "status": "failed_active_decode_start",
                    "warm": {"status": "passed" if warm_passed else "failed_contract", "request_seconds": warm_seconds, **warm_summary},
                    "warm_counter_delta": warm_delta,
                    "active": active_result,
                }
            )
            return result

        metrics_after_active_first = self.metrics_snapshot("after-cached-continuation-active-first-token")
        active_cold_delta = counter_delta(
            metrics_after_warm.get("vllm_counter_samples", {}), metrics_after_active_first.get("vllm_counter_samples", {})
        )
        time.sleep(self.args.interference_established_decode_seconds)
        continuation_injected_at = time.monotonic() - timeline_started
        metrics_before_continuation = self.metrics_snapshot("before-cached-continuation")
        continuation_messages = [
            *base_messages,
            {"role": "assistant", "content": warm_expected},
            {"role": "user", "content": append_prompt},
        ]
        continuation_result = self.stream(
            port,
            alias,
            "",
            32,
            timeline_origin=timeline_started,
            ignore_eos=False,
            messages=continuation_messages,
        )
        metrics_after_continuation = self.metrics_snapshot("after-cached-continuation")
        active_thread.join(timeout=self.args.request_timeout + 30)

        continuation_delta = counter_delta(
            metrics_before_continuation.get("vllm_counter_samples", {}),
            metrics_after_continuation.get("vllm_counter_samples", {}),
        )
        active_prefix_hits = counter_delta_total(active_cold_delta, "vllm:prefix_cache_hits_total")
        active_local_cache = counter_delta_total(
            active_cold_delta, "vllm:prompt_tokens_by_source_total", 'source="local_cache_hit"'
        )
        active_local_compute = counter_delta_total(
            active_cold_delta, "vllm:prompt_tokens_by_source_total", 'source="local_compute"'
        )
        continuation_prefix_queries = counter_delta_total(continuation_delta, "vllm:prefix_cache_queries_total")
        continuation_prefix_hits = counter_delta_total(continuation_delta, "vllm:prefix_cache_hits_total")
        continuation_local_compute = counter_delta_total(
            continuation_delta, "vllm:prompt_tokens_by_source_total", 'source="local_compute"'
        )
        continuation_local_cache = counter_delta_total(
            continuation_delta, "vllm:prompt_tokens_by_source_total", 'source="local_cache_hit"'
        )
        continuation_cached_tokens = counter_delta_total(continuation_delta, "vllm:prompt_tokens_cached_total")
        continuation_external_hits = counter_delta_total(continuation_delta, "vllm:external_prefix_cache_hits_total")
        active_offsets = active_result.get("token_offsets_seconds")
        continuation_first_token = continuation_result.get("first_token_offset_seconds")
        before_intervals: list[float] = []
        during_intervals: list[float] = []
        if isinstance(active_offsets, list) and isinstance(continuation_first_token, float):
            for earlier, later in zip(active_offsets, active_offsets[1:]):
                if not isinstance(earlier, float) or not isinstance(later, float):
                    continue
                if later <= continuation_injected_at:
                    before_intervals.append(later - earlier)
                elif earlier >= continuation_injected_at and later <= continuation_first_token:
                    during_intervals.append(later - earlier)

        continuation_content = continuation_result.get("content_preview")
        warm_contract = (
            warm_passed
            and warm_local_compute is not None
            and warm_local_compute >= base_tokens * 0.9
            and warm_local_cache == 0.0
            and warm_prefix_hits == 0.0
        )
        active_cold_contract = (
            active_prefix_hits == 0.0
            and active_local_cache == 0.0
            and active_local_compute is not None
            and active_local_compute >= active_tokens * 0.9
        )
        continuation_cache_contract = (
            continuation_result.get("status") == "passed"
            and continuation_result.get("finish_reason") == "stop"
            and isinstance(continuation_content, str)
            and continuation_content.strip() == continuation_expected
            and continuation_prefix_queries is not None
            and continuation_prefix_hits is not None
            and continuation_prefix_hits >= base_tokens * 0.9
            and continuation_local_cache is not None
            and continuation_local_cache >= base_tokens * 0.9
            and continuation_cached_tokens is not None
            and continuation_cached_tokens >= base_tokens * 0.9
            and continuation_local_compute is not None
            and continuation_local_compute <= append_tokens + 8192
            and continuation_external_hits == 0.0
        )
        active_completion_contract = (
            active_result.get("status") == "passed" and active_result.get("finish_reason") == "length"
        )
        cache_contract = {
            "runtime_prefix_caching_enabled": cache_enabled,
            "warm_created_local_prefix": warm_contract,
            "active_request_was_cold": active_cold_contract,
            "continuation_reused_local_prefix": continuation_cache_contract,
            "active_request_completed": active_completion_contract,
            "status": "passed"
            if all((cache_enabled, warm_contract, active_cold_contract, continuation_cache_contract, active_completion_contract))
            else "failed_cache_contract",
        }
        result.update(
            {
                "status": "passed" if cache_contract["status"] == "passed" else "failed_cached_continuation",
                "warm": {"status": "passed" if warm_passed else "failed_contract", "request_seconds": warm_seconds, **warm_summary},
                "active": active_result,
                "continuation": continuation_result,
                "established_decode_seconds": self.args.interference_established_decode_seconds,
                "continuation_injected_at_seconds": continuation_injected_at,
                "active_p95_itl_before_continuation_seconds": percentile(before_intervals, 0.95),
                "active_p95_itl_during_continuation_seconds": percentile(during_intervals, 0.95),
                "active_itl_samples_before_continuation": len(before_intervals),
                "active_itl_samples_during_continuation": len(during_intervals),
                "cache_contract": cache_contract,
                "warm_counter_delta": warm_delta,
                "active_cold_counter_delta": active_cold_delta,
                "continuation_counter_delta": continuation_delta,
                "metrics_before_warm": metrics_before_warm,
                "metrics_after_warm": metrics_after_warm,
                "metrics_after_active_first": metrics_after_active_first,
                "metrics_before_continuation": metrics_before_continuation,
                "metrics_after_continuation": metrics_after_continuation,
            }
        )
        return result

    def cached_c3_queued_cold(self, admission_policy: str = "c0") -> dict[str, object]:
        """Compare native, C<=2, and C=0 cold admission on the fixed Phase J workload."""
        port, alias = self.endpoints()[0]
        phase_started = self.server_ready_monotonic if self.server_ready_monotonic is not None else time.monotonic()
        deadline = phase_started + self.args.cached_c3_max_workload_seconds
        if admission_policy not in {"c0", "native_vllm", "c2"}:
            raise ValueError(f"unsupported cached C=3 admission policy: {admission_policy}")
        is_native = admission_policy == "native_vllm"
        is_c2 = admission_policy == "c2"
        result: dict[str, object] = {
            "status": "error",
            "policy": {
                "name": {
                    "c0": "client_side_cold_admission_after_c0",
                    "native_vllm": "native_vllm_waiting_admission",
                    "c2": "explicit_c2_cold_admission",
                }[admission_policy],
                "admission_policy": admission_policy,
                "max_num_seqs": self.args.max_num_seqs,
                "cold_request_posted_before_c0": is_native or is_c2,
                "required_consecutive_idle_samples": 2 if not is_native and not is_c2 else 0,
            },
            "workload_max_seconds_after_ready": self.args.cached_c3_max_workload_seconds,
            "phase_started_monotonic": phase_started,
        }

        def remaining_timeout() -> int:
            remaining = math.floor(deadline - time.monotonic())
            if remaining < 1:
                raise TimeoutError("Phase J exceeded its workload time limit")
            return min(self.args.request_timeout, remaining)

        def cache_values(delta: dict[str, dict[str, float | None]]) -> dict[str, float | None]:
            return {
                "prefix_queries": counter_delta_total(delta, "vllm:prefix_cache_queries_total"),
                "prefix_hits": counter_delta_total(delta, "vllm:prefix_cache_hits_total"),
                "local_compute_tokens": counter_delta_total(
                    delta, "vllm:prompt_tokens_by_source_total", 'source="local_compute"'
                ),
                "local_cache_hit_tokens": counter_delta_total(
                    delta, "vllm:prompt_tokens_by_source_total", 'source="local_cache_hit"'
                ),
                "prompt_tokens_cached": counter_delta_total(delta, "vllm:prompt_tokens_cached_total"),
                "external_prefix_cache_hits": counter_delta_total(delta, "vllm:external_prefix_cache_hits_total"),
            }

        def normal_probe(label: str, messages: list[dict[str, str]], expected: str) -> dict[str, object]:
            metrics_before = self.metrics_snapshot(f"phase-j-{label}-before")
            try:
                response, request_seconds = self.chat(
                    port,
                    alias,
                    "",
                    32,
                    remaining_timeout(),
                    messages=messages,
                )
                summary = response_summary(response)
                content = summary.get("content_preview")
                normal_eos = (
                    summary.get("finish_reason") == "stop"
                    and isinstance(content, str)
                    and content.strip() == expected
                )
                status = "passed" if normal_eos else "failed_contract"
            except (OSError, RuntimeError, TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as error:
                request_seconds = None
                summary = {"error": f"{type(error).__name__}: {error}"}
                normal_eos = False
                status = "error"
            metrics_after = self.metrics_snapshot(f"phase-j-{label}-after")
            delta = counter_delta(
                metrics_before.get("vllm_counter_samples", {}), metrics_after.get("vllm_counter_samples", {})
            )
            return {
                "status": status,
                "normal_eos": normal_eos,
                "expected": expected,
                "request_seconds": request_seconds,
                "response": summary,
                "metrics_before": metrics_before,
                "metrics_after": metrics_after,
                "counter_delta": delta,
                "cache": cache_values(delta),
            }

        def warm_contract(probe: dict[str, object], base_tokens: int) -> bool:
            cache = probe.get("cache", {})
            if not isinstance(cache, dict):
                return False
            return bool(
                probe.get("normal_eos")
                and cache.get("local_compute_tokens") is not None
                and cache["local_compute_tokens"] >= base_tokens * 0.9
                and cache.get("prefix_hits") == 0.0
                and cache.get("local_cache_hit_tokens") == 0.0
                and cache.get("external_prefix_cache_hits") == 0.0
            )

        def cached_probe_contract(probe: dict[str, object], base_tokens: int, append_tokens: int) -> bool:
            cache = probe.get("cache", {})
            if not isinstance(cache, dict):
                return False
            return bool(
                probe.get("normal_eos")
                and cache.get("prefix_hits") is not None
                and cache["prefix_hits"] >= base_tokens * 0.9
                and cache.get("local_cache_hit_tokens") is not None
                and cache["local_cache_hit_tokens"] >= base_tokens * 0.9
                and cache.get("prompt_tokens_cached") is not None
                and cache["prompt_tokens_cached"] >= base_tokens * 0.9
                and cache.get("local_compute_tokens") is not None
                and cache["local_compute_tokens"] <= append_tokens + 8192
                and cache.get("external_prefix_cache_hits") == 0.0
            )

        conversations: list[dict[str, object]] = []
        for index, append_target in enumerate(self.args.cached_c3_agent_append_tokens):
            marker = f"phase-j-agent-{index}-base"
            warm_expected = f"phase-j-agent-{index}-warm-ok"
            preflight_expected = f"phase-j-agent-{index}-preflight-ok"
            post_cold_expected = f"phase-j-agent-{index}-post-cold-ok"
            base_prompt, base_tokens = self.prompt(
                self.args.cached_c3_base_prompt_tokens,
                marker,
                f"Reply with exactly: {warm_expected}",
            )
            preflight_prompt, preflight_tokens = self.prompt(
                self.args.cached_c3_probe_tokens,
                f"phase-j-agent-{index}-preflight",
                f"Reply with exactly: {preflight_expected}",
            )
            agent_prompt, agent_tokens = self.prompt(
                append_target,
                f"phase-j-agent-{index}-activity",
                "Continue this agent session with a detailed working answer.",
            )
            post_cold_prompt, post_cold_tokens = self.prompt(
                self.args.cached_c3_probe_tokens,
                f"phase-j-agent-{index}-post-cold",
                f"Reply with exactly: {post_cold_expected}",
            )
            conversations.append(
                {
                    "label": f"agent-{index}",
                    "base_prompt": base_prompt,
                    "base_tokens": base_tokens,
                    "base_prompt_sha256": hashlib.sha256(base_prompt.encode()).hexdigest(),
                    "warm_expected": warm_expected,
                    "preflight_prompt": preflight_prompt,
                    "preflight_tokens": preflight_tokens,
                    "preflight_prompt_sha256": hashlib.sha256(preflight_prompt.encode()).hexdigest(),
                    "preflight_expected": preflight_expected,
                    "agent_prompt": agent_prompt,
                    "agent_tokens": agent_tokens,
                    "agent_prompt_sha256": hashlib.sha256(agent_prompt.encode()).hexdigest(),
                    "post_cold_prompt": post_cold_prompt,
                    "post_cold_tokens": post_cold_tokens,
                    "post_cold_prompt_sha256": hashlib.sha256(post_cold_prompt.encode()).hexdigest(),
                    "post_cold_expected": post_cold_expected,
                }
            )

        def conversation_artifacts() -> list[dict[str, object]]:
            return [
                {
                    key: conversation[key]
                    for key in (
                        "label",
                        "base_tokens",
                        "base_prompt_sha256",
                        "preflight_tokens",
                        "preflight_prompt_sha256",
                        "agent_tokens",
                        "agent_prompt_sha256",
                        "post_cold_tokens",
                        "post_cold_prompt_sha256",
                        "warm",
                        "warm_contract",
                        "preflight",
                        "preflight_contract",
                        "post_cold",
                        "post_cold_contract",
                    )
                    if key in conversation
                }
                for conversation in conversations
            ]

        cold_prompt, cold_tokens = self.prompt(
            self.args.cached_c3_cold_prompt_tokens,
            "phase-j-independent-cold",
            "Produce a detailed isolated answer for this independent new request.",
        )
        result["cold_request"] = {
            "prompt_content_tokens": cold_tokens,
            "prompt_sha256": hashlib.sha256(cold_prompt.encode()).hexdigest(),
            "output_tokens": self.args.cached_c3_cold_output_tokens,
        }

        cache_enabled_snapshot = self.metrics_snapshot("phase-j-before-cache-creation")
        cache_enabled = any(
            'enable_prefix_caching="True"' in sample
            for sample in cache_enabled_snapshot.get("cache_config_samples", [])
            if isinstance(sample, str)
        )
        result["runtime_prefix_caching_enabled"] = cache_enabled

        for conversation in conversations:
            base_messages = [{"role": "user", "content": str(conversation["base_prompt"])}]
            warm = normal_probe(f"{conversation['label']}-warm", base_messages, str(conversation["warm_expected"]))
            conversation["warm"] = warm
            conversation["warm_contract"] = warm_contract(warm, int(conversation["base_tokens"]))
            conversation["base_messages"] = base_messages
        result["conversations"] = conversation_artifacts()
        if not cache_enabled or not all(bool(conversation["warm_contract"]) for conversation in conversations):
            result["status"] = "failed_cache_creation"
            return result

        for conversation in conversations:
            preflight_messages = [
                *conversation["base_messages"],
                {"role": "assistant", "content": str(conversation["warm_expected"])},
                {"role": "user", "content": str(conversation["preflight_prompt"])},
            ]
            preflight = normal_probe(
                f"{conversation['label']}-preflight", preflight_messages, str(conversation["preflight_expected"])
            )
            conversation["preflight"] = preflight
            conversation["preflight_contract"] = cached_probe_contract(
                preflight, int(conversation["base_tokens"]), int(conversation["preflight_tokens"])
            )
            conversation["preflight_messages"] = preflight_messages
        if not all(bool(conversation["preflight_contract"]) for conversation in conversations):
            result["conversations"] = conversation_artifacts()
            result["status"] = "failed_preflight_cache_contract"
            return result

        timeline_started_wallclock = dt.datetime.now()
        timeline_started = time.monotonic()
        metrics_before_c3 = self.metrics_snapshot("phase-j-before-c3-agent-activity")
        ready_events = [threading.Event() for _ in conversations]
        start_gate = threading.Barrier(len(conversations))
        scheduler_samples: list[dict[str, object]] = []
        responses: dict[str, dict[str, object]] = {}
        cold_response: dict[str, object] | None = None
        cold_future = None
        cold_enqueued_monotonic: float | None = None
        metrics_before_cold: dict[str, object] | None = None
        metrics_at_active_completion: dict[str, object] | None = None
        first_active_completion_monotonic: float | None = None
        active_completed_before_cold = 0

        def cold_request() -> dict[str, object]:
            return self.stream(
                port,
                alias,
                cold_prompt,
                self.args.cached_c3_cold_output_tokens,
                timeline_origin=timeline_started,
                ignore_eos=True,
                timeout=remaining_timeout(),
            )

        with ThreadPoolExecutor(max_workers=len(conversations) + 1) as executor:
            futures = {
                executor.submit(
                    self.stream,
                    port,
                    alias,
                    "",
                    self.args.cached_c3_agent_output_tokens,
                    ready_events[index],
                    timeline_origin=timeline_started,
                    start_gate=start_gate,
                    ignore_eos=True,
                    messages=[
                        *conversation["preflight_messages"],
                        {"role": "assistant", "content": str(conversation["preflight_expected"])},
                        {"role": "user", "content": str(conversation["agent_prompt"])},
                    ],
                    timeout=remaining_timeout(),
                ): str(conversation["label"])
                for index, conversation in enumerate(conversations)
            }
            if not all(event.wait(timeout=min(300, remaining_timeout())) for event in ready_events):
                for future, label in futures.items():
                    if future.done():
                        responses[label] = future.result()
                result.update(
                    {
                        "status": "failed_c3_first_token",
                        "conversations": conversation_artifacts(),
                        "c3_agent_responses": responses,
                        "metrics_before_c3": metrics_before_c3,
                    }
                )
                return result

            if is_native:
                scheduler_samples.append(self.scheduler_snapshot("phase-k-native-before-cold-submit"))
                metrics_before_cold = self.metrics_snapshot("phase-k-native-before-cold-submit")
                cold_enqueued_monotonic = time.monotonic()
                cold_future = executor.submit(cold_request)
            elif is_c2:
                # Poll briefly so the cold request is posted after the first
                # completion, rather than waiting for the one-second sampler.
                next_scheduler_sample = time.monotonic()
                while not any(future.done() for future in futures):
                    if time.monotonic() >= next_scheduler_sample:
                        scheduler_samples.append(
                            self.scheduler_snapshot(f"phase-k-c2-before-cold-{len(scheduler_samples):03d}")
                        )
                        next_scheduler_sample += 1
                    if self.thermal_stop.is_set():
                        raise RuntimeError("thermal stop interrupted Phase K C<=2 activity")
                    remaining_timeout()
                    time.sleep(0.01)
                completed = [future for future in futures if future.done()]
                completed.sort(
                    key=lambda future: futures[future]
                )
                first_completed_future = completed[0]
                first_completed_response = first_completed_future.result()
                first_active_completion_monotonic = first_completed_response.get("completion_monotonic")
                active_completed_before_cold = len(completed)
                responses[futures[first_completed_future]] = first_completed_response
                metrics_before_cold = self.metrics_snapshot("phase-k-c2-before-cold-submit")
                cold_enqueued_monotonic = time.monotonic()
                cold_future = executor.submit(cold_request)
            else:
                cold_enqueued_monotonic = time.monotonic()

            while True:
                scheduler_samples.append(self.scheduler_snapshot(f"phase-j-admission-{len(scheduler_samples):03d}"))
                if self.thermal_stop.is_set():
                    raise RuntimeError("thermal stop interrupted cached C=3 activity")
                active_done = all(future.done() for future in futures)
                if active_done:
                    for future, label in futures.items():
                        if label not in responses:
                            responses[label] = future.result()
                    if metrics_at_active_completion is None:
                        metrics_at_active_completion = self.metrics_snapshot("phase-k-after-active-c3-completion")
                    if metrics_before_cold is None:
                        # The C=0 reference starts measuring the cold request
                        # only after the two idle samples below.
                        metrics_before_cold = self.metrics_snapshot("phase-j-after-c3-agent-activity")
                    if not is_native and not is_c2:
                        break
                    if cold_future is not None and cold_future.done():
                        break
                remaining_timeout()
                time.sleep(1)
            metrics_after_c3 = self.metrics_snapshot("phase-j-after-c3-agent-activity")
            if cold_future is not None:
                cold_response = cold_future.result()

        if cold_enqueued_monotonic is None:
            raise RuntimeError("cached C=3 admission did not establish a cold request enqueue time")
        c3_delta = counter_delta(
            metrics_before_c3.get("vllm_counter_samples", {}), metrics_after_c3.get("vllm_counter_samples", {})
        )
        c3_cache = cache_values(c3_delta)
        total_base_tokens = sum(int(conversation["base_tokens"]) for conversation in conversations)
        c3_completed = [response for response in responses.values() if response.get("status") == "passed" and response.get("finish_reason") == "length"]
        first_tokens = [response.get("first_token_monotonic") for response in c3_completed]
        completions = [response.get("completion_monotonic") for response in c3_completed]
        all_active_start = max(first_tokens) if len(first_tokens) == len(conversations) and all(isinstance(value, float) for value in first_tokens) else None
        all_active_end = min(completions) if len(completions) == len(conversations) and all(isinstance(value, float) for value in completions) else None
        all_active_scheduler_samples = [
            sample
            for sample in scheduler_samples
            if isinstance(all_active_start, float)
            and isinstance(all_active_end, float)
            and all_active_start <= sample["monotonic"] <= all_active_end
            and sample.get("running_requests") == 3.0
            and sample.get("waiting_requests") == 0.0
        ]
        c3_cache_contract = (
            len(c3_completed) == len(conversations)
            and c3_cache.get("prefix_hits") is not None
            and c3_cache["prefix_hits"] >= total_base_tokens * 0.9
            and c3_cache.get("local_cache_hit_tokens") is not None
            and c3_cache["local_cache_hit_tokens"] >= total_base_tokens * 0.9
            and c3_cache.get("prompt_tokens_cached") is not None
            and c3_cache["prompt_tokens_cached"] >= total_base_tokens * 0.9
            and c3_cache.get("external_prefix_cache_hits") == 0.0
        )
        c3_graph_evidence = self.cuda_graph_evidence()
        c3_graph_contract = (
            c3_graph_evidence.get("status") == "passed"
            and 3 in c3_graph_evidence.get("full_runtime_request_counts", [])
            and 3 in c3_graph_evidence.get("full_runtime_batch_sizes", [])
        )
        server_intervals = self.vllm_throughput_intervals(timeline_started_wallclock)
        c3_decode_intervals = [
            interval
            for interval in server_intervals
            if interval["running_requests"] == 3
            and interval["waiting_requests"] == 0
            and interval["prompt_tokens_per_second"] == 0.0
            and interval["offset_seconds"] <= cold_enqueued_monotonic - timeline_started
        ]
        c3_capacity_contract = (
            bool(scheduler_samples)
            and all(sample.get("running_requests") is not None and sample["running_requests"] <= 3.0 for sample in scheduler_samples)
            and len(all_active_scheduler_samples) >= 2
            and (len(c3_decode_intervals) >= 2 or is_native or is_c2)
        )
        c3_streams = {
            label: {
                "ttft_seconds": response.get("ttft_seconds"),
                "itl_p50_seconds": response.get("itl_p50_seconds"),
                "itl_p95_seconds": response.get("itl_p95_seconds"),
                "itl_p99_seconds": response.get("itl_p99_seconds"),
                "decode_tokens_per_second": response.get("decode_tokens_per_second"),
                "request_start_monotonic": response.get("request_start_monotonic"),
                "first_token_monotonic": response.get("first_token_monotonic"),
                "completion_monotonic": response.get("completion_monotonic"),
                "token_offsets_seconds": response.get("token_offsets_seconds"),
                "status": response.get("status"),
                "finish_reason": response.get("finish_reason"),
            }
            for label, response in sorted(responses.items())
        }
        result.update(
            {
                "c3_agent_activity": {
                    "output_tokens_per_agent": self.args.cached_c3_agent_output_tokens,
                    "total_cached_base_tokens_required": total_base_tokens,
                    "cold_enqueued_monotonic": cold_enqueued_monotonic,
                    "streams": c3_streams,
                    "aggregate_decode_tokens_per_second": sum(
                        response["decode_tokens_per_second"]
                        for response in c3_completed
                        if isinstance(response.get("decode_tokens_per_second"), float)
                    ),
                    "metrics_before": metrics_before_c3,
                    "metrics_after": metrics_after_c3,
                    "counter_delta": c3_delta,
                    "cache": c3_cache,
                    "cache_contract": c3_cache_contract,
                    "scheduler_samples": scheduler_samples,
                    "all_three_active_scheduler_samples": all_active_scheduler_samples,
                    "capacity_contract": c3_capacity_contract,
                    "server_decode_intervals": c3_decode_intervals,
                    "server_intervals": server_intervals,
                    "graph_evidence_before_cold_admission": c3_graph_evidence,
                    "graph_contract_before_cold_admission": c3_graph_contract,
                }
            }
        )
        if not c3_cache_contract or not c3_capacity_contract or not c3_graph_contract:
            result["conversations"] = conversation_artifacts()
            result["status"] = "failed_c3_cache_capacity_or_graph_contract"
            return result

        idle_samples: list[dict[str, object]] = []
        if not is_native and not is_c2:
            consecutive_idle = 0
            while consecutive_idle < 2:
                idle_sample = self.scheduler_snapshot(f"phase-j-c0-{len(idle_samples):03d}")
                idle_samples.append(idle_sample)
                if idle_sample.get("running_requests") == 0.0 and idle_sample.get("waiting_requests") == 0.0:
                    consecutive_idle += 1
                else:
                    consecutive_idle = 0
                if consecutive_idle < 2:
                    remaining_timeout()
                    time.sleep(1)
            cold_dispatched_monotonic = time.monotonic()
            metrics_before_cold = idle_samples[-1]["metrics"]
            metrics_at_active_completion = metrics_before_cold
            cold_response = self.stream(
                port,
                alias,
                cold_prompt,
                self.args.cached_c3_cold_output_tokens,
                timeline_origin=timeline_started,
                ignore_eos=True,
                timeout=remaining_timeout(),
            )
        else:
            cold_dispatched_monotonic = cold_enqueued_monotonic
            if cold_response is None or metrics_before_cold is None:
                raise RuntimeError("native/C<=2 cold admission did not complete its dispatch setup")
        metrics_after_cold = self.metrics_snapshot("phase-j-after-cold-admission")
        cold_delta = counter_delta(
            metrics_before_cold.get("vllm_counter_samples", {}), metrics_after_cold.get("vllm_counter_samples", {})
        )
        queue_metrics_before = metrics_at_active_completion or metrics_before_cold
        cold_scheduler_delta = counter_delta(
            queue_metrics_before.get("vllm_scheduler_samples", {}), metrics_after_cold.get("vllm_scheduler_samples", {})
        )
        cold_cache = cache_values(cold_delta)
        cold_contract = (
            cold_response.get("status") == "passed"
            and cold_response.get("finish_reason") == "length"
            and cold_cache.get("prefix_hits") == 0.0
            and cold_cache.get("local_cache_hit_tokens") == 0.0
            and cold_cache.get("prompt_tokens_cached") == 0.0
            and cold_cache.get("external_prefix_cache_hits") == 0.0
            and cold_cache.get("local_compute_tokens") is not None
            and cold_cache["local_compute_tokens"] >= cold_tokens * 0.9
        )
        cold_first = cold_response.get("first_token_monotonic")
        cold_completed = cold_response.get("completion_monotonic")
        gateway_wait_ms = cold_response.get("gateway_admission_wait_ms")
        if isinstance(gateway_wait_ms, float) and gateway_wait_ms >= 0:
            cold_dispatched_monotonic = cold_enqueued_monotonic + gateway_wait_ms / 1000.0
        backend_queue_count = counter_delta_total(cold_scheduler_delta, "vllm:request_queue_time_seconds_count")
        backend_queue_sum = counter_delta_total(cold_scheduler_delta, "vllm:request_queue_time_seconds_sum")
        admission_samples = [
            sample for sample in scheduler_samples if sample["monotonic"] >= cold_enqueued_monotonic
        ]
        waiting_seen = next(
            (
                sample
                for sample in admission_samples
                if sample.get("waiting_requests") is not None and sample["waiting_requests"] >= 1.0
            ),
            None,
        )
        running_after_wait = next(
            (
                sample
                for sample in admission_samples
                if sample.get("waiting_requests") == 0.0
                and (
                    (is_native and waiting_seen is not None and sample["monotonic"] >= waiting_seen["monotonic"])
                    or (not is_native and sample.get("running_requests") == 3.0)
                )
            ),
            None,
        )
        histogram_queue_wait = backend_queue_sum / backend_queue_count if backend_queue_count else None
        cold_prefill_start = (
            cold_dispatched_monotonic
            if not is_native and not is_c2
            else running_after_wait["monotonic"]
            if running_after_wait is not None
            else cold_dispatched_monotonic + histogram_queue_wait
            if isinstance(histogram_queue_wait, float)
            else cold_dispatched_monotonic
        )
        if not isinstance(cold_prefill_start, float):
            cold_prefill_start = cold_enqueued_monotonic

        active_prefill_intervals: dict[str, list[float]] = {}
        aggregate_prefill_intervals: list[float] = []
        if isinstance(cold_first, float):
            for label, response in responses.items():
                token_offsets = response.get("token_offsets_seconds")
                intervals = []
                if isinstance(token_offsets, list):
                    absolute_offsets = [timeline_started + value for value in token_offsets if isinstance(value, float)]
                    intervals = [
                        later - earlier
                        for earlier, later in zip(absolute_offsets, absolute_offsets[1:])
                        if earlier >= cold_prefill_start and later <= cold_first
                    ]
                active_prefill_intervals[label] = intervals
                aggregate_prefill_intervals.extend(intervals)
        active_prefill_itl = {
            "per_user": {label: interval_summary(values) for label, values in sorted(active_prefill_intervals.items())},
            "aggregate": interval_summary(aggregate_prefill_intervals),
        }
        stable_intervals = self.vllm_throughput_intervals(timeline_started_wallclock)
        post_cold_stable = [
            interval
            for interval in stable_intervals
            if isinstance(cold_completed, float)
            and interval["offset_seconds"] >= cold_completed - timeline_started
            and interval["running_requests"] == 3
            and interval["waiting_requests"] == 0
            and interval["prompt_tokens_per_second"] == 0.0
        ]
        queue_contract = (
            (
                len(idle_samples) >= 2
                and all(
                    sample.get("running_requests") == 0.0 and sample.get("waiting_requests") == 0.0
                    for sample in idle_samples[-2:]
                )
                if not is_native and not is_c2
                else isinstance(cold_enqueued_monotonic, float)
                and (first_active_completion_monotonic is None or cold_enqueued_monotonic >= first_active_completion_monotonic)
            )
            and cold_dispatched_monotonic >= cold_enqueued_monotonic
            and isinstance(cold_first, float)
            and cold_first >= cold_dispatched_monotonic
            and backend_queue_count is not None
            and backend_queue_sum is not None
        )
        result["cold_admission"] = {
            "admission_policy": admission_policy,
            "idle_scheduler_samples": idle_samples,
            "admission_scheduler_samples": admission_samples,
            "waiting_observed_at_monotonic": waiting_seen["monotonic"] if waiting_seen else None,
            "running_after_wait_observed_at_monotonic": running_after_wait["monotonic"] if running_after_wait else None,
            "running_at_admission_observed": running_after_wait.get("running_requests") if running_after_wait else None,
            "cold_prefill_start_monotonic": cold_prefill_start,
            "cold_prefill_start_source": (
                "scheduler_transition"
                if running_after_wait is not None
                else "gateway_dispatch_plus_vllm_queue_histogram"
                if isinstance(gateway_wait_ms, float) and isinstance(histogram_queue_wait, float)
                else "vllm_queue_histogram"
                if isinstance(histogram_queue_wait, float)
                else "request_start"
            ),
            "cold_enqueued_monotonic": cold_enqueued_monotonic,
            "cold_dispatched_monotonic": cold_dispatched_monotonic,
            "gateway_admission_wait_ms": gateway_wait_ms,
            "cold_first_token_monotonic": cold_first,
            "cold_completed_monotonic": cold_completed,
            "gateway_queue_delay_seconds": cold_dispatched_monotonic - cold_enqueued_monotonic,
            "vllm_wait_observed_seconds": (
                cold_prefill_start - cold_dispatched_monotonic
                if isinstance(cold_prefill_start, float)
                else None
            ),
            "post_admission_ttft_seconds": cold_first - cold_dispatched_monotonic if isinstance(cold_first, float) else None,
            "cold_prefill_ttft_seconds": cold_first - cold_prefill_start if isinstance(cold_first, float) else None,
            "user_visible_ttft_seconds": cold_first - cold_enqueued_monotonic if isinstance(cold_first, float) else None,
            "cold_total_wall_seconds": cold_completed - cold_enqueued_monotonic if isinstance(cold_completed, float) else None,
            "backend_queue_delay_seconds": backend_queue_sum / backend_queue_count if backend_queue_count else None,
            "backend_queue_time_seconds_count_delta": backend_queue_count,
            "backend_queue_time_seconds_sum_delta": backend_queue_sum,
            "response": cold_response,
            "metrics_before": metrics_before_cold,
            "metrics_before_queue_delta": queue_metrics_before,
            "metrics_after": metrics_after_cold,
            "counter_delta": cold_delta,
            "scheduler_counter_delta": cold_scheduler_delta,
            "cache": cold_cache,
            "active_users_during_cold_prefill": active_prefill_itl,
            "active_user_decode_rates": {
                label: response.get("decode_tokens_per_second") for label, response in sorted(responses.items())
            },
            "stable_three_user_decode": {
                "pre_admission_server_intervals": c3_decode_intervals,
                "all_server_intervals": server_intervals,
                "post_cold_server_intervals": post_cold_stable,
                "time_until_stable_three_user_decode_resumes_seconds": (
                    post_cold_stable[0]["offset_seconds"] - (cold_enqueued_monotonic - timeline_started)
                    if post_cold_stable
                    else None
                ),
            },
            "first_active_completion_monotonic": first_active_completion_monotonic,
            "active_completed_before_cold": active_completed_before_cold,
            "cold_contract": cold_contract,
            "queue_contract": queue_contract,
        }
        if not cold_contract or not queue_contract:
            result["conversations"] = conversation_artifacts()
            result["status"] = "failed_cold_admission_contract"
            return result

        for conversation in conversations:
            post_cold_messages = [
                *conversation["base_messages"],
                {"role": "assistant", "content": str(conversation["warm_expected"])},
                {"role": "user", "content": str(conversation["post_cold_prompt"])},
            ]
            post_cold = normal_probe(
                f"{conversation['label']}-post-cold", post_cold_messages, str(conversation["post_cold_expected"])
            )
            conversation["post_cold"] = post_cold
            conversation["post_cold_contract"] = cached_probe_contract(
                post_cold, int(conversation["base_tokens"]), int(conversation["post_cold_tokens"])
            )
        metrics_after_retention = self.metrics_snapshot("phase-j-after-retention-probes")
        preemption_delta = counter_delta_total(
            counter_delta(
                cache_enabled_snapshot.get("vllm_scheduler_samples", {}),
                metrics_after_retention.get("vllm_scheduler_samples", {}),
            ),
            "vllm:num_preemptions_total",
        )
        log_text = ""
        for log in self.logs:
            log.flush()
            log_text += Path(log.name).read_text(encoding="utf-8", errors="replace")
        retention_contract = all(bool(conversation.get("post_cold_contract")) for conversation in conversations)
        result.update(
            {
                "conversations": conversation_artifacts(),
                "cache_retention": {
                    "post_cold_contract": retention_contract,
                    "preemptions_delta": preemption_delta,
                    "supporting_log_lines": [
                        line for line in log_text.splitlines() if "evict" in line.lower() or "preempt" in line.lower()
                    ],
                    "metrics_after_retention_probes": metrics_after_retention,
                },
                "server_c3_decode_intervals": c3_decode_intervals,
            }
        )
        result["status"] = (
            "passed"
            if retention_contract and preemption_delta == 0.0 and not self.thermal_stop.is_set()
            else "failed_retention_or_preemption_contract"
        )
        return result

    def sustained(self) -> dict[str, object]:
        deadline = time.monotonic() + self.args.sustained_seconds
        rounds: list[dict[str, object]] = []
        round_number = 0
        while time.monotonic() < deadline and not self.thermal_stop.is_set():
            jobs = []
            for index in range(self.args.sustained_concurrency):
                port, alias = self.endpoints()[index % len(self.endpoints())]
                jobs.append((port, alias, 8192, f"sustained-{round_number}-{index}"))
            with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
                requests = [future.result() for future in as_completed([executor.submit(self.long_request, *job) for job in jobs])]
            rounds.append({"round": round_number, "requests": requests})
            round_number += 1
        requests = [request for round_result in rounds for request in round_result["requests"]]
        total_tokens = sum(
            int((request.get("usage") or {}).get("completion_tokens", 0) or 0) for request in requests if isinstance(request, dict)
        )
        return {
            "duration_target_seconds": self.args.sustained_seconds,
            "rounds": rounds,
            "completed_requests": len(requests),
            "passed_requests": sum(request.get("status") == "passed" for request in requests),
            "completion_tokens": total_tokens,
            "status": "passed" if requests and all(request.get("status") == "passed" for request in requests) else "failed_sustained",
        }

    def run(self) -> dict[str, object]:
        if subprocess.run(["systemctl", "is-active", "--quiet", PRODUCTION_SERVICE], check=False).returncode == 0:
            raise RuntimeError(f"{PRODUCTION_SERVICE} is active; use the guarded maintenance wrapper")
        before_processes = gpu_processes(self.gpus)
        for gpu, processes in before_processes.items():
            unexpected = [process for process in processes if process["pid"] not in self.args.allow_preexisting_compute_pid]
            if unexpected:
                raise RuntimeError(f"GPU {gpu} has unapproved pre-existing compute processes: {unexpected}")
        result: dict[str, object] = {
            "status": "error",
            "started_at": now(),
            "engine": self.args.engine,
            "gpus": list(self.gpus),
            "ports": list(self.ports),
            "request_ports": list(self.request_ports),
            "metrics_port": self.metrics_port,
            "aliases": list(self.aliases),
            "context_tokens": self.args.context_tokens,
            "thermal_stop_c": self.args.thermal_stop_c,
            "preexisting_compute": before_processes,
            "gpu_before": gpu_snapshots(self.gpus),
            "host_before": host_snapshot(),
            "model": str(self.args.model),
            "model_config_sha256": hashlib.sha256((self.args.model / "config.json").read_bytes()).hexdigest(),
            "vllm_configuration": {
                "attention_config": self.args.vllm_attention_config,
                "speculative_tokens": self.args.vllm_spec_tokens,
                "cuda_graph": self.args.vllm_enable_cuda_graph,
                "cudagraph_capture_sizes": self.args.vllm_cudagraph_capture_sizes,
                "cudagraph_metrics": self.args.vllm_cudagraph_metrics,
                "compilation_config": self.args.vllm_compilation_config,
                "gdn_prefill_backend": self.args.vllm_gdn_prefill_backend,
                "breakable_cudagraph": self.args.vllm_breakable_cudagraph,
                "prefix_caching": self.args.vllm_enable_prefix_caching,
                "max_num_batched_tokens": self.args.vllm_max_num_batched_tokens,
                "long_prefill_token_threshold": self.args.vllm_long_prefill_token_threshold,
                "enable_chunked_prefill": self.args.vllm_enable_chunked_prefill,
                "decode_warm_prefixes": self.args.decode_warm_prefixes,
                "cached_active_prompt_tokens": self.args.cached_active_prompt_tokens,
                "cached_base_prompt_tokens": self.args.cached_base_prompt_tokens,
                "cached_continuation_append_tokens": self.args.cached_continuation_append_tokens,
                "cached_c3_queued_cold": self.args.cached_c3_queued_cold_only,
                "cached_c3_native_admission": self.args.cached_c3_native_admission_only,
                "cached_c3_c2_admission": self.args.cached_c3_c2_admission_only,
                "cached_c3_agent_append_tokens": self.args.cached_c3_agent_append_tokens,
                "stream_capacity": self.args.stream_capacity,
            },
        }
        if self.args.engine == "vllm":
            result["vllm_runtime"] = vllm_runtime_info(self.args.vllm)
            environment = self.command()[1]
            result["vllm_environment"] = {
                key: environment.get(key)
                for key in ("VLLM_USE_FLASHINFER_SAMPLER", "VLLM_USE_BREAKABLE_CUDAGRAPH", "VLLM_LOGGING_LEVEL")
                if key in environment
            }
        sampler = threading.Thread(target=self.sampler, daemon=True)
        sampler_started = False
        try:
            result["server"] = self.start()
            self.server_ready_monotonic = time.monotonic()
            result["metrics_after_start"] = self.metrics_snapshot("after-start")
            if self.args.vllm_enable_cuda_graph:
                result["cuda_graph_capture"] = self.cuda_graph_evidence()
                if result["cuda_graph_capture"].get("status") != "passed":
                    raise RuntimeError("CUDA-graph capture evidence failed")
            sampler.start()
            sampler_started = True
            if self.args.capacity_only:
                result["capacity"] = self.capacity()
            elif self.args.decode_only:
                result["decode_only"] = self.decode_only()
            elif self.args.interference_only:
                result["prefill_decode_interference"] = self.interference()
            elif self.args.cached_c3_queued_cold_only:
                result["cached_c3_queued_cold"] = self.cached_c3_queued_cold()
                if result["cached_c3_queued_cold"].get("status") != "passed":
                    raise RuntimeError("Phase J client-side cold-admission/cache-retention contract failed")
            elif self.args.cached_c3_native_admission_only:
                result["cached_c3_native_admission"] = self.cached_c3_queued_cold("native_vllm")
                if result["cached_c3_native_admission"].get("status") != "passed":
                    raise RuntimeError("Phase K native vLLM admission/cache-retention contract failed")
            elif self.args.cached_c3_c2_admission_only:
                result["cached_c3_c2_admission"] = self.cached_c3_queued_cold("c2")
                if result["cached_c3_c2_admission"].get("status") != "passed":
                    raise RuntimeError("Phase K explicit C<=2 admission/cache-retention contract failed")
            elif self.args.cached_continuation_interference_only:
                result["cached_continuation_interference"] = self.cached_continuation_interference()
                if result["cached_continuation_interference"].get("status") != "passed":
                    raise RuntimeError("cached continuation cache/QoS contract failed")
            elif self.args.retrieval_only:
                result["long_context_retrieval"] = self.long_context_retrieval()
            elif self.args.cache_only:
                result["cache_correctness"] = self.cache_correctness()
            elif self.args.cache_repro_only:
                result["cache_reproducer"] = self.cache_reproducer()
            elif self.args.smoke_only:
                result["smoke"] = self.smoke()
                if result["smoke"].get("status") != "passed":
                    raise RuntimeError("startup/correctness or CUDA-graph evidence failed")
            elif self.args.gsm8k_executable is not None:
                result["gsm8k"] = self.gsm8k()
            elif self.args.recovery_gsm8k_executable is not None:
                result["recovery_gsm8k"] = self.recovery_gsm8k()
                if result["recovery_gsm8k"].get("status") != "passed":
                    raise RuntimeError("recovery-aware GSM8K evaluation failed or was incomplete")
            else:
                result["normal_eos"] = self.normal_eos()
                result["forced_tool"] = self.forced_tool()
                port, alias = self.endpoints()[0]
                result["single_native_context"] = self.long_request(port, alias, self.args.native_prompt_tokens, "single-native")
                result["capacity"] = self.capacity()
                result["prefill_decode_interference"] = self.interference()
                result["sustained"] = self.sustained()
            if self.args.vllm_enable_cuda_graph:
                # Startup capture alone is insufficient; require observed FULL graph replay after work.
                # vLLM emits CUDAGraph runtime statistics on its periodic ten-second logger.
                time.sleep(12)
                result["cuda_graph_runtime"] = self.cuda_graph_evidence()
                if (
                    result["cuda_graph_runtime"].get("status") != "passed"
                    or not result["cuda_graph_runtime"].get("full_runtime_stats_observed")
                ):
                    raise RuntimeError("CUDA-graph runtime evidence failed")
                if (
                    self.args.decode_only
                    or self.args.interference_only
                    or self.args.cached_c3_queued_cold_only
                    or self.args.cached_c3_native_admission_only
                    or self.args.cached_c3_c2_admission_only
                    or self.args.vllm_spec_tokens is not None
                ):
                    required_request_count = (
                        3
                        if (
                            self.args.cached_c3_queued_cold_only
                            or self.args.cached_c3_native_admission_only
                            or self.args.cached_c3_c2_admission_only
                        )
                        else self.args.decode_concurrency
                        if self.args.decode_only
                        else 1
                    )
                    required_token_count = required_request_count * (self.args.vllm_spec_tokens + 1 if self.args.vllm_spec_tokens else 1)
                    result["cuda_graph_decode_shape"] = {
                        "required_request_count": required_request_count,
                        "required_token_count": required_token_count,
                        "full_capture_request_counts": result["cuda_graph_runtime"].get("full_capture_request_counts"),
                        "full_capture_token_counts": result["cuda_graph_runtime"].get("full_capture_token_counts"),
                        "full_runtime_token_counts": result["cuda_graph_runtime"].get("full_runtime_token_counts"),
                        "full_runtime_request_counts": result["cuda_graph_runtime"].get("full_runtime_request_counts"),
                        "full_capture_batch_sizes": result["cuda_graph_runtime"].get("full_capture_batch_sizes"),
                        "full_runtime_batch_sizes": result["cuda_graph_runtime"].get("full_runtime_batch_sizes"),
                    }
                    if (
                        required_request_count not in result["cuda_graph_decode_shape"]["full_capture_request_counts"]
                        or required_token_count not in result["cuda_graph_decode_shape"]["full_capture_token_counts"]
                        or required_token_count not in result["cuda_graph_decode_shape"]["full_runtime_token_counts"]
                        or (
                            (
                                self.args.cached_c3_queued_cold_only
                                or self.args.cached_c3_native_admission_only
                                or self.args.cached_c3_c2_admission_only
                            )
                            and (
                                required_request_count not in result["cuda_graph_decode_shape"]["full_runtime_request_counts"]
                                or required_request_count not in result["cuda_graph_decode_shape"]["full_runtime_batch_sizes"]
                            )
                        )
                    ):
                        raise RuntimeError(
                            "CUDA-graph decode-shape evidence failed for "
                            f"requests={required_request_count}, tokens={required_token_count}"
                        )
            result["status"] = "completed"
        except KeyboardInterrupt:
            result["error"] = "KeyboardInterrupt: candidate interrupted"
        except (OSError, RuntimeError, TimeoutError, ValueError, urllib.error.HTTPError, urllib.error.URLError) as error:
            result["error"] = f"{type(error).__name__}: {error}"
        except Exception as error:
            # Preserve candidate artifacts if an unexpected harness defect occurs.
            result["error"] = f"{type(error).__name__}: {error}"
        finally:
            self.stop_sampling.set()
            if sampler_started:
                sampler.join(timeout=3)
            if self.processes and all(process.poll() is None for process in self.processes):
                result["metrics_before_stop"] = self.metrics_snapshot("before-stop")
            result["exit_codes"] = self.stop()
            result["thermal_stop_triggered"] = self.thermal_stop.is_set()
            result["samples"] = self.samples
            result["sample_count"] = len(self.samples)
            result["gpu_after"] = gpu_snapshots(self.gpus)
            result["host_after"] = host_snapshot()
            result["finished_at"] = now()
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("llama-replicas", "vllm", "sglang"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="ModelOpt checkpoint for vLLM/SGLang and tokenizer source")
    parser.add_argument("--gguf-model", type=Path, required=True)
    parser.add_argument("--llama", type=Path, required=True)
    parser.add_argument("--vllm", type=Path, required=True)
    parser.add_argument("--sglang", type=Path, required=True)
    parser.add_argument("--gpus", nargs=2, type=int, default=(0, 1))
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--request-port", type=int, help="Optional API port or loopback gateway port for requests.")
    parser.add_argument("--metrics-port", type=int, help="Optional direct vLLM metrics port.")
    parser.add_argument("--context-tokens", type=int, default=262144)
    parser.add_argument("--native-prompt-tokens", type=int, default=250000)
    parser.add_argument("--long-max-tokens", type=int, default=512)
    parser.add_argument("--capacity-prompt-tokens", type=int, default=196000)
    parser.add_argument("--capacity-concurrency", type=int, default=2)
    parser.add_argument("--capacity-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true", help="Run normal EOS and forced-tool contracts only.")
    parser.add_argument("--decode-only", action="store_true", help="Warm distinct long prefixes, then measure simultaneous decode streams.")
    parser.add_argument("--decode-prompt-tokens", type=int, default=196000)
    parser.add_argument("--decode-output-tokens", type=int, default=4096)
    parser.add_argument("--decode-concurrency", type=int, default=3)
    parser.add_argument("--decode-warm-prefixes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interference-only", action="store_true", help="Measure an established decode while a new prefill arrives.")
    parser.add_argument(
        "--cached-continuation-interference-only",
        action="store_true",
        help="Measure an established decode while a cached long conversation adds a new turn.",
    )
    parser.add_argument(
        "--cached-c3-queued-cold-only",
        action="store_true",
        help="Qualify C=3 cached agents while a fourth cold request remains client-side until C=0.",
    )
    parser.add_argument(
        "--cached-c3-native-admission-only",
        action="store_true",
        help="Submit the fourth cold request directly to vLLM while the three cached agents are active.",
    )
    parser.add_argument(
        "--cached-c3-c2-admission-only",
        action="store_true",
        help="Submit the fourth cold request after the first cached agent completes, at C<=2.",
    )
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--cache-repro-only", action="store_true", help="Send one identical prompt pair without a server restart.")
    parser.add_argument("--gsm8k-executable", type=Path)
    parser.add_argument("--gsm8k-examples", type=int, default=128)
    parser.add_argument("--gsm8k-max-tokens", type=int, default=4096)
    parser.add_argument("--gsm8k-timeout", type=int, default=21600)
    parser.add_argument("--recovery-gsm8k-executable", type=Path)
    parser.add_argument("--recovery-gsm8k-python", type=Path)
    parser.add_argument("--recovery-gsm8k-dataset", type=Path, default=RECOVERY_GSM8K_DATASET)
    parser.add_argument("--recovery-gsm8k-examples", type=int, default=256)
    parser.add_argument("--recovery-gsm8k-max-tokens", type=int, default=4096)
    parser.add_argument("--recovery-gsm8k-continuation-max-tokens", type=int, default=4096)
    parser.add_argument("--recovery-gsm8k-temperature", type=float, default=1.0)
    parser.add_argument("--recovery-gsm8k-top-p", type=float, default=0.95)
    parser.add_argument("--recovery-gsm8k-seed", type=int, default=0)
    parser.add_argument("--recovery-gsm8k-concurrency", type=int, default=1)
    parser.add_argument("--recovery-gsm8k-request-timeout", type=float, default=3600.0)
    parser.add_argument("--recovery-gsm8k-run-timeout", type=int, default=21600)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--stream-capacity", action="store_true", help="Use streaming requests for capacity measurements.")
    parser.add_argument("--cache-prompt-tokens", type=int, default=8192)
    parser.add_argument("--vllm-attention-config", default='{"use_trtllm_attention": false}')
    parser.add_argument("--vllm-spec-tokens", choices=(1, 2, 3), type=int)
    parser.add_argument("--vllm-enable-prefix-caching", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--vllm-enable-cuda-graph", action="store_true")
    parser.add_argument("--vllm-cudagraph-capture-sizes", nargs="+", type=int)
    parser.add_argument("--vllm-cudagraph-metrics", action="store_true")
    parser.add_argument("--vllm-compilation-config")
    parser.add_argument("--vllm-gdn-prefill-backend", choices=("flashinfer", "triton", "cutedsl"))
    parser.add_argument("--vllm-breakable-cudagraph", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--vllm-logging-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--vllm-max-num-batched-tokens", type=int)
    parser.add_argument("--vllm-long-prefill-token-threshold", type=int)
    parser.add_argument("--vllm-enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--chunked-prefill-size", type=int, default=4096)
    parser.add_argument("--mamba-ssm-dtype", choices=("bfloat16",))
    parser.add_argument("--interference-decode-prompt-tokens", type=int, default=8192)
    parser.add_argument("--interference-prefill-prompt-tokens", type=int, default=65536)
    parser.add_argument("--interference-decode-tokens", type=int, default=512)
    parser.add_argument("--interference-prefill-tokens", type=int, default=256)
    parser.add_argument("--interference-established-decode-seconds", type=int, default=0)
    parser.add_argument("--cached-active-prompt-tokens", type=int, default=196000)
    parser.add_argument("--cached-active-output-tokens", type=int, default=8192)
    parser.add_argument("--cached-base-prompt-tokens", type=int, default=196000)
    parser.add_argument("--cached-continuation-append-tokens", type=int, default=2048)
    parser.add_argument("--cached-c3-base-prompt-tokens", type=int, default=196000)
    parser.add_argument("--cached-c3-probe-tokens", type=int, default=256)
    parser.add_argument("--cached-c3-agent-append-tokens", nargs=3, type=int, default=(1024, 2048, 4096))
    parser.add_argument("--cached-c3-agent-output-tokens", type=int, default=2048)
    parser.add_argument("--cached-c3-cold-prompt-tokens", type=int, default=196000)
    parser.add_argument("--cached-c3-cold-output-tokens", type=int, default=256)
    parser.add_argument("--cached-c3-max-workload-seconds", type=int, default=900)
    parser.add_argument("--sustained-seconds", type=int, default=300)
    parser.add_argument("--sustained-concurrency", type=int, default=2)
    parser.add_argument("--thermal-stop-c", type=float, default=85.0)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=3600)
    parser.add_argument("--allow-preexisting-compute-pid", type=int, action="append", default=[])
    args = parser.parse_args()
    ports_to_validate = [args.port, args.request_port, args.metrics_port]
    if any(port is not None and (port < 1 or port > 65535) for port in ports_to_validate) or len(set(args.gpus)) != 2:
        parser.error("ports must be valid and --gpus must name two distinct physical GPUs")
    exclusive_modes = sum(
        (
            args.capacity_only,
            args.smoke_only,
            args.decode_only,
            args.interference_only,
            args.cached_continuation_interference_only,
            args.cached_c3_queued_cold_only,
            args.cached_c3_native_admission_only,
            args.cached_c3_c2_admission_only,
            args.retrieval_only,
            args.cache_only,
            args.cache_repro_only,
            args.gsm8k_executable is not None,
            args.recovery_gsm8k_executable is not None,
        )
    )
    if exclusive_modes > 1:
        parser.error("only one specialized benchmark mode may be selected")
    if args.vllm_cudagraph_capture_sizes and not args.vllm_enable_cuda_graph:
        parser.error("--vllm-cudagraph-capture-sizes requires --vllm-enable-cuda-graph")
    if args.vllm_cudagraph_metrics and not args.vllm_enable_cuda_graph:
        parser.error("--vllm-cudagraph-metrics requires --vllm-enable-cuda-graph")
    if args.vllm_compilation_config is not None:
        try:
            compilation_config = json.loads(args.vllm_compilation_config)
        except json.JSONDecodeError as error:
            parser.error(f"--vllm-compilation-config must be valid JSON: {error.msg}")
        if not isinstance(compilation_config, dict):
            parser.error("--vllm-compilation-config must be a JSON object")
        if not args.vllm_enable_cuda_graph:
            parser.error("--vllm-compilation-config requires --vllm-enable-cuda-graph")
    if args.engine != "vllm" and any(
        (
            args.vllm_enable_cuda_graph,
            args.vllm_cudagraph_capture_sizes,
            args.vllm_cudagraph_metrics,
            args.vllm_compilation_config is not None,
            args.vllm_gdn_prefill_backend is not None,
            args.vllm_breakable_cudagraph is not None,
            args.vllm_logging_level is not None,
            args.vllm_max_num_batched_tokens is not None,
            args.vllm_long_prefill_token_threshold is not None,
            args.vllm_enable_chunked_prefill is not None,
        )
    ):
        parser.error("vLLM CUDA-graph options require --engine vllm")
    if args.vllm_spec_tokens is not None and args.engine != "vllm":
        parser.error("--vllm-spec-tokens requires --engine vllm")
    if args.vllm_enable_prefix_caching is not None and args.engine != "vllm":
        parser.error("--vllm-enable-prefix-caching requires --engine vllm")
    if args.vllm_max_num_batched_tokens is not None and args.vllm_max_num_batched_tokens < 1:
        parser.error("--vllm-max-num-batched-tokens must be positive")
    if args.vllm_long_prefill_token_threshold is not None and args.vllm_long_prefill_token_threshold < 0:
        parser.error("--vllm-long-prefill-token-threshold must be non-negative")
    if args.recovery_gsm8k_concurrency < 1:
        parser.error("--recovery-gsm8k-concurrency must be positive")
    if args.interference_established_decode_seconds < 0:
        parser.error("--interference-established-decode-seconds must be non-negative")
    if args.cache_repro_only and args.engine != "vllm":
        parser.error("--cache-repro-only requires --engine vllm")
    if args.cache_repro_only and args.vllm_enable_prefix_caching is not True:
        parser.error("--cache-repro-only requires explicit --vllm-enable-prefix-caching")
    if args.cached_continuation_interference_only and args.engine != "vllm":
        parser.error("--cached-continuation-interference-only requires --engine vllm")
    if args.cached_continuation_interference_only and args.vllm_enable_prefix_caching is not True:
        parser.error("--cached-continuation-interference-only requires explicit --vllm-enable-prefix-caching")
    if args.cached_continuation_interference_only and args.max_num_seqs < 2:
        parser.error("--cached-continuation-interference-only requires --max-num-seqs of at least two")
    if args.cached_c3_queued_cold_only and args.engine != "vllm":
        parser.error("--cached-c3-queued-cold-only requires --engine vllm")
    if (args.cached_c3_native_admission_only or args.cached_c3_c2_admission_only) and args.engine != "vllm":
        parser.error("Phase K cached C=3 admission modes require --engine vllm")
    if args.cached_c3_queued_cold_only and args.vllm_enable_prefix_caching is not True:
        parser.error("--cached-c3-queued-cold-only requires explicit --vllm-enable-prefix-caching")
    if args.cached_c3_queued_cold_only and args.max_num_seqs != 3:
        parser.error("--cached-c3-queued-cold-only requires --max-num-seqs exactly three")
    if (args.cached_c3_native_admission_only or args.cached_c3_c2_admission_only) and args.max_num_seqs != 3:
        parser.error("Phase K cached C=3 admission modes require --max-num-seqs exactly three")
    if args.cached_c3_queued_cold_only and (
        not args.vllm_enable_cuda_graph
        or not args.vllm_cudagraph_metrics
        or 3 not in (args.vllm_cudagraph_capture_sizes or [])
        or args.vllm_max_num_batched_tokens != 4096
        or args.vllm_long_prefill_token_threshold != 256
        or args.vllm_enable_chunked_prefill is not True
    ):
        parser.error("--cached-c3-queued-cold-only requires the Phase I graph-B0 4096/256 chunked-prefill profile with B=3 metrics")
    if (args.cached_c3_native_admission_only or args.cached_c3_c2_admission_only) and (
        not args.vllm_enable_cuda_graph
        or not args.vllm_cudagraph_metrics
        or 3 not in (args.vllm_cudagraph_capture_sizes or [])
        or args.vllm_max_num_batched_tokens != 4096
        or args.vllm_long_prefill_token_threshold != 256
        or args.vllm_enable_chunked_prefill is not True
    ):
        parser.error("Phase K cached C=3 admission modes require the fixed graph-B0 4096/256 chunked-prefill profile with B=3 metrics")
    if args.cached_c3_queued_cold_only and (
        args.cached_c3_base_prompt_tokens != 196000
        or args.cached_c3_cold_prompt_tokens != 196000
        or list(args.cached_c3_agent_append_tokens) != [1024, 2048, 4096]
        or args.cached_c3_agent_output_tokens != 2048
        or args.cached_c3_cold_output_tokens != 256
        or args.cached_c3_probe_tokens < 1
        or args.cached_c3_max_workload_seconds != 900
        or args.request_timeout < 1
        or args.request_timeout > 600
    ):
        parser.error("--cached-c3-queued-cold-only requires the fixed Phase J workload and a request timeout from one to 600 seconds")
    if (args.cached_c3_native_admission_only or args.cached_c3_c2_admission_only) and (
        args.cached_c3_base_prompt_tokens != 196000
        or args.cached_c3_cold_prompt_tokens != 196000
        or list(args.cached_c3_agent_append_tokens) != [1024, 2048, 4096]
        or args.cached_c3_agent_output_tokens != 2048
        or args.cached_c3_cold_output_tokens != 256
        or args.cached_c3_probe_tokens < 1
        or args.cached_c3_max_workload_seconds != 900
        or args.request_timeout < 1
        or args.request_timeout > 600
    ):
        parser.error("Phase K cached C=3 admission modes require the fixed four-user workload and a request timeout from one to 600 seconds")
    context_workloads: list[tuple[int, int]] = []
    if exclusive_modes == 0:
        context_workloads.extend(
            (
                (args.native_prompt_tokens, args.long_max_tokens),
                (args.capacity_prompt_tokens, args.long_max_tokens),
                (args.interference_decode_prompt_tokens, args.interference_decode_tokens),
                (args.interference_prefill_prompt_tokens, args.interference_prefill_tokens),
            )
        )
    elif args.capacity_only:
        context_workloads.append((args.capacity_prompt_tokens, args.long_max_tokens))
    elif args.decode_only:
        context_workloads.append((args.decode_prompt_tokens, args.decode_output_tokens))
    elif args.interference_only:
        context_workloads.extend(
            (
                (args.interference_decode_prompt_tokens, args.interference_decode_tokens),
                (args.interference_prefill_prompt_tokens, args.interference_prefill_tokens),
            )
        )
    elif args.cached_continuation_interference_only:
        context_workloads.extend(
            (
                (args.cached_active_prompt_tokens, args.cached_active_output_tokens),
                (
                    args.cached_base_prompt_tokens + args.cached_continuation_append_tokens,
                    32,
                ),
            )
        )
    elif args.cached_c3_queued_cold_only or args.cached_c3_native_admission_only or args.cached_c3_c2_admission_only:
        context_workloads.extend(
            (
                (
                    args.cached_c3_base_prompt_tokens
                    + args.cached_c3_probe_tokens
                    + max(args.cached_c3_agent_append_tokens),
                    args.cached_c3_agent_output_tokens,
                ),
                (args.cached_c3_cold_prompt_tokens, args.cached_c3_cold_output_tokens),
            )
        )
    elif args.retrieval_only:
        context_workloads.append((249000, 32))
    elif args.cache_only or args.cache_repro_only:
        context_workloads.append((args.cache_prompt_tokens, 32))
    for prompt_tokens, max_tokens in context_workloads:
        if prompt_tokens + max_tokens + 2048 > args.context_tokens:
            parser.error("a selected benchmark prompt plus output and template reserve exceeds context")
    if not args.model.is_dir() or not (args.model / "config.json").is_file():
        parser.error(f"ModelOpt checkpoint is not readable: {args.model}")
    if not args.gguf_model.is_file() or not args.llama.is_file() or not os.access(args.llama, os.X_OK):
        parser.error("GGUF reference model or llama-server is not usable")
    if args.engine == "vllm" and not os.access(args.vllm, os.X_OK):
        parser.error(f"vLLM executable is not usable: {args.vllm}")
    if args.engine == "sglang" and not os.access(args.sglang, os.X_OK):
        parser.error(f"SGLang executable is not usable: {args.sglang}")
    if args.decode_only and (args.decode_concurrency < 1 or args.decode_concurrency > args.max_num_seqs):
        parser.error("--decode-concurrency must be between one and --max-num-seqs")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    def interrupt(_signum: int, _frame: object) -> None:
        # Let PhaseDRun.run() stop its separate server process group before exit.
        raise KeyboardInterrupt

    prior_handlers = {signum: signal.signal(signum, interrupt) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        result = PhaseDRun(args).run()
    finally:
        for signum, handler in prior_handlers.items():
            signal.signal(signum, handler)
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"engine": args.engine, "status": result["status"], "error": result.get("error")}, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
