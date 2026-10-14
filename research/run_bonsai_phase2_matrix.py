#!/usr/bin/env python3
"""Run the isolated Phase 2 Bonsai 2 capacity/speed matrix.

This runner deliberately starts the official server as a child process rather
than using a service manager.  CUDA device numbering is made explicit in the
recorded environment: physical GPU 1 is exposed to the process as CUDA0.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(os.environ['EXPERIMENT_ROOT'])
RESEARCH = ROOT / "research"
SERVER = RESEARCH / "prismml-llama.cpp-prism-b10687-5d80cff-build/build-sm86/bin/llama-server"
BIN_DIR = SERVER.parent
SOURCE = RESEARCH / "prismml-llama.cpp-prism-b10687-5d80cff/source"
MODEL_DIR = RESEARCH / "prismml-bonsai-2-27b-gguf"
MODELS = {
    "PQ2_0": MODEL_DIR / "Ternary-Bonsai-2-27B-PQ2_0.gguf",
    "PTQ1_0": MODEL_DIR / "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
}
PORTS = {"PQ2_0": 18201, "PTQ1_0": 18202}
DEPTHS = [8192, 65536, 131072, 196608, 262144]
OUTPUT_TOKENS = 128
SEED = 271828
POLL_SECONDS = 0.5


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(args: list[str], env: dict[str, str] | None = None, timeout: int = 120) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {"args": args, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    except Exception as exc:  # keep diagnostics in the artifact even if a tool is absent
        return {"args": args, "error": repr(exc)}


def parse_number(value: str) -> int | float | str | None:
    value = value.strip()
    if value in {"", "N/A", "[Not Supported]"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def gpu_snapshot() -> dict[str, Any]:
    query = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem,utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ]
    result = command_output(query, timeout=15)
    gpus: list[dict[str, Any]] = []
    for row in result.get("stdout", "").splitlines():
        values = next(csv.reader([row]), [])
        if len(values) != 11:
            continue
        fields = [
            "index",
            "uuid",
            "name",
            "memory_total_mib",
            "memory_used_mib",
            "power_w",
            "temperature_c",
            "clock_sm_mhz",
            "clock_mem_mhz",
            "utilization_gpu_pct",
            "utilization_mem_pct",
        ]
        item: dict[str, Any] = {}
        for field, value in zip(fields, values):
            item[field] = parse_number(value) if field not in {"uuid", "name"} else value.strip()
        gpus.append(item)
    apps = command_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,process_name,used_memory", "--format=csv,noheader,nounits"],
        timeout=15,
    )
    compute_apps: list[dict[str, Any]] = []
    for row in apps.get("stdout", "").splitlines():
        values = next(csv.reader([row]), [])
        if len(values) == 4:
            compute_apps.append(
                {
                    "pid": parse_number(values[0]),
                    "gpu_uuid": values[1].strip(),
                    "process_name": values[2].strip(),
                    "used_memory_mib": parse_number(values[3]),
                }
            )
    return {"gpus": gpus, "compute_apps": compute_apps, "query": query}


def host_snapshot() -> dict[str, Any]:
    stat = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
    cpu = [int(value) for value in stat[1:]]
    mem: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(":")
        if value.strip().split():
            mem[key] = int(value.strip().split()[0])
    load = os.getloadavg()
    return {
        "cpu_jiffies": cpu,
        "meminfo_kib": mem,
        "memory_used_kib": mem.get("MemTotal", 0) - mem.get("MemAvailable", 0),
        "swap_used_kib": mem.get("SwapTotal", 0) - mem.get("SwapFree", 0),
        "loadavg": list(load),
    }


class Telemetry:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.label = "idle"
        self.samples: list[dict[str, Any]] = []
        self.thread = threading.Thread(target=self._run, name="phase2-telemetry", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def mark(self, label: str) -> None:
        with self.lock:
            self.label = label

    def _run(self) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            while not self.stop_event.is_set():
                mono = time.monotonic()
                with self.lock:
                    label = self.label
                sample = {
                    "monotonic": mono,
                    "wall_time": time.time(),
                    "label": label,
                    "gpu": gpu_snapshot(),
                    "host": host_snapshot(),
                }
                self.samples.append(sample)
                handle.write(json.dumps(sample, sort_keys=True) + "\n")
                handle.flush()
                self.stop_event.wait(POLL_SECONDS)

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=20)


def request_json(port: int, path: str, payload: dict[str, Any] | None = None, timeout: int = 3600) -> dict[str, Any]:
    url = f"http://127.0.0.1:{port}{path}"
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            status = response.status
            headers = dict(response.headers.items())
        return {
            "url": url,
            "status": status,
            "headers": headers,
            "body": body.decode("utf-8", errors="replace"),
            "elapsed_s": time.monotonic() - started,
        }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"url": url, "status": exc.code, "headers": dict(exc.headers.items()), "body": body, "elapsed_s": time.monotonic() - started}
    except Exception as exc:
        return {"url": url, "status": None, "headers": {}, "body": "", "error": repr(exc), "elapsed_s": time.monotonic() - started}


def parsed_body(result: dict[str, Any]) -> Any:
    try:
        return json.loads(result.get("body", ""))
    except (json.JSONDecodeError, TypeError):
        return None


def tokenize(port: int, content: str) -> dict[str, Any]:
    result = request_json(port, "/tokenize", {"content": content, "add_special": True}, timeout=600)
    body = parsed_body(result)
    tokens = body.get("tokens") if isinstance(body, dict) else None
    result["token_count"] = len(tokens) if isinstance(tokens, list) else None
    result["parsed"] = body
    return result


def chat_input_tokens(port: int, payload: dict[str, Any]) -> dict[str, Any]:
    result = request_json(port, "/v1/chat/completions/input_tokens", payload, timeout=600)
    result["parsed"] = parsed_body(result)
    return result


def make_corpus() -> tuple[str, str]:
    header = (
        "You are processing a deterministic synthetic repository snapshot for a coding benchmark. "
        "Use the supplied facts as authoritative. Do not invent files, commands, test results, or APIs. "
        "At the end, return exactly a concise implementation-oriented answer with the requested code reasoning.\n\n"
    )
    blocks: list[str] = []
    for index in range(6000):
        blocks.append(
            f"### snapshot block {index:05d}\n"
            f"path: services/module_{index % 37:02d}/component_{index % 113:03d}.py\n"
            f"symbol: Handler_{index % 211:03d}.execute\n"
            "def execute(request, state):\n"
            f"    revision = \"r{index:05d}-{(index * 7919) % 100000:05d}\"\n"
            "    if request.get('cancelled'):\n"
            "        return {'status': 'cancelled', 'revision': revision}\n"
            "    result = state.lookup(request.get('key', 'default'))\n"
            "    return {'status': 'ok', 'revision': revision, 'result': result}\n"
            f"tests: test_handler_{index % 149:03d} checks revision {index:05d} and result shape\n\n"
        )
    return header, "".join(blocks)


def build_prompt(port: int, target_depth: int, header: str, corpus: str) -> dict[str, Any]:
    # Leave room for the chat template and the fixed completion so the 262k
    # request tests a populated near-limit context rather than overflows it.
    desired_content_tokens = max(256, target_depth - 256)
    probe = tokenize(port, header + corpus[:65536])
    probe_count = probe.get("token_count") or max(1, len((header + corpus[:65536]).split()))
    ratio = probe_count / max(1, len(header + corpus[:65536]))
    char_count = min(len(corpus), max(128, int((desired_content_tokens / ratio) - len(header))))
    calibration: list[dict[str, Any]] = [probe]
    for _ in range(5):
        content = header + corpus[:char_count]
        result = tokenize(port, content)
        calibration.append(result)
        count = result.get("token_count")
        if not count or abs(count - desired_content_tokens) <= 8:
            break
        char_count = max(128, min(len(corpus), int(char_count * desired_content_tokens / count)))
    content = header + corpus[:char_count]
    final = calibration[-1]
    return {
        "target_context_tokens": target_depth,
        "desired_content_tokens": desired_content_tokens,
        "content": content,
        "content_chars": len(content),
        "content_tokenize": final,
        "calibration": calibration,
    }


def parse_stream_response(response: Any, started: float) -> tuple[str, dict[str, Any]]:
    raw_parts: list[str] = []
    chunks: list[dict[str, Any]] = []
    first_event: float | None = None
    first_token: float | None = None
    output: list[str] = []
    reasoning: list[str] = []
    finish_reason: str | None = None
    while True:
        line = response.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace")
        raw_parts.append(text)
        if not text.startswith("data:"):
            continue
        now = time.monotonic()
        if first_event is None:
            first_event = now
        value = text[5:].strip()
        if value == "[DONE]":
            continue
        try:
            chunk = json.loads(value)
        except json.JSONDecodeError:
            continue
        if isinstance(chunk, dict):
            chunks.append(chunk)
            choices = chunk.get("choices") or []
            if choices:
                choice = choices[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                content = delta.get("content")
                thought = delta.get("reasoning_content")
                if content:
                    if first_token is None:
                        first_token = now
                    output.append(content)
                if thought:
                    if first_token is None:
                        first_token = now
                    reasoning.append(thought)
    ended = time.monotonic()
    return "".join(raw_parts), {
        "chunks": chunks,
        "output": "".join(output),
        "reasoning": "".join(reasoning),
        "finish_reason": finish_reason,
        "first_event_s": None if first_event is None else first_event - started,
        "ttft_s": None if first_token is None else first_token - started,
        "wall_s": ended - started,
    }


def chat_request(port: int, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    started = time.monotonic()
    try:
        response = urllib.request.urlopen(request, timeout=3600)
        raw, stream = parse_stream_response(response, started)
        result = {
            "url": url,
            "status": response.status,
            "headers": dict(response.headers.items()),
            "request_started_monotonic": started,
            "request_ended_monotonic": time.monotonic(),
            "raw": raw,
            **stream,
        }
        return result
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return {
            "url": url,
            "status": exc.code,
            "headers": dict(exc.headers.items()),
            "request_started_monotonic": started,
            "request_ended_monotonic": time.monotonic(),
            "raw": raw,
            "error": "http_error",
            "wall_s": time.monotonic() - started,
        }
    except Exception as exc:
        return {
            "url": url,
            "status": None,
            "headers": {},
            "request_started_monotonic": started,
            "request_ended_monotonic": time.monotonic(),
            "raw": "",
            "error": repr(exc),
            "wall_s": time.monotonic() - started,
        }


def timing_from_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    timing: dict[str, Any] = {}
    for chunk in chunks:
        value = chunk.get("timings")
        if isinstance(value, dict):
            timing.update(value)
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            timing["usage"] = usage
    return timing


def numeric(values: list[Any]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float))]


def request_telemetry(samples: list[dict[str, Any]], start: float, end: float, baseline_power: float | None) -> dict[str, Any]:
    selected = [sample for sample in samples if start <= sample["monotonic"] <= end]
    gpu1: list[dict[str, Any]] = []
    all_gpu_max_memory: dict[str, float] = {}
    for sample in selected:
        for gpu in sample.get("gpu", {}).get("gpus", []):
            index = str(gpu.get("index"))
            used = gpu.get("memory_used_mib")
            if isinstance(used, (int, float)):
                all_gpu_max_memory[index] = max(all_gpu_max_memory.get(index, 0), float(used))
            if gpu.get("index") == 1:
                gpu1.append(gpu)
    fields = ["memory_used_mib", "power_w", "temperature_c", "clock_sm_mhz", "clock_mem_mhz", "utilization_gpu_pct", "utilization_mem_pct"]
    summary: dict[str, Any] = {"sample_count": len(selected), "gpu1_max_memory_mib_all_gpus": all_gpu_max_memory}
    for field in fields:
        values = numeric([item.get(field) for item in gpu1])
        if values:
            summary[f"gpu1_{field}_min"] = min(values)
            summary[f"gpu1_{field}_max"] = max(values)
            summary[f"gpu1_{field}_avg"] = sum(values) / len(values)
    energy_j = 0.0
    dynamic_j = 0.0
    for left, right in zip(selected, selected[1:]):
        left_gpu = next((item for item in left.get("gpu", {}).get("gpus", []) if item.get("index") == 1), None)
        right_gpu = next((item for item in right.get("gpu", {}).get("gpus", []) if item.get("index") == 1), None)
        if not left_gpu or not right_gpu:
            continue
        p0 = left_gpu.get("power_w")
        p1 = right_gpu.get("power_w")
        if isinstance(p0, (int, float)) and isinstance(p1, (int, float)):
            seconds = max(0.0, right["monotonic"] - left["monotonic"])
            energy_j += ((float(p0) + float(p1)) / 2) * seconds
            if baseline_power is not None:
                dynamic_j += max(0.0, ((float(p0) + float(p1)) / 2) - baseline_power) * seconds
    summary["gpu1_energy_j"] = energy_j
    summary["gpu1_dynamic_energy_j"] = dynamic_j if baseline_power is not None else None
    hosts = [sample.get("host", {}) for sample in selected]
    memory = numeric([host.get("memory_used_kib") for host in hosts])
    swap = numeric([host.get("swap_used_kib") for host in hosts])
    loads = numeric([host.get("loadavg", [None])[0] for host in hosts if host.get("loadavg")])
    if memory:
        summary["host_memory_used_kib_max"] = max(memory)
        summary["host_memory_used_kib_avg"] = sum(memory) / len(memory)
    if swap:
        summary["host_swap_used_kib_max"] = max(swap)
    if loads:
        summary["host_load1_max"] = max(loads)
    cpu_pct: list[float] = []
    for left, right in zip(hosts, hosts[1:]):
        first = left.get("cpu_jiffies", [])
        second = right.get("cpu_jiffies", [])
        if len(first) < 5 or len(second) != len(first):
            continue
        deltas = [b - a for a, b in zip(first, second)]
        total = sum(deltas)
        idle = deltas[3] + deltas[4]
        if total > 0:
            cpu_pct.append(100.0 * (1.0 - idle / total))
    if cpu_pct:
        summary["host_cpu_pct_avg"] = sum(cpu_pct) / len(cpu_pct)
        summary["host_cpu_pct_max"] = max(cpu_pct)
    return summary


def filtered_environment() -> dict[str, str]:
    keys = {
        key
        for key in os.environ
        if key.startswith(("CUDA", "NVIDIA", "NCCL", "LLAMA", "GGML", "OMP", "MKL", "CUBLAS"))
    }
    env = {key: os.environ[key] for key in sorted(keys)}
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["LD_LIBRARY_PATH"] = f"{BIN_DIR}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    return env


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def server_command(model: Path, port: int, slot_save_path: Path | None = None) -> list[str]:
    alias = "bonsai-2-27b-pq2_0" if "PQ2_0" in model.name else "bonsai-2-27b-ptq1_0"
    command = [
        str(SERVER),
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--alias",
        alias,
        "--device",
        "CUDA0",
        "--n-gpu-layers",
        "99",
        "--ctx-size",
        "262144",
        "--fit",
        "off",
        "--parallel",
        "1",
        "--flash-attn",
        "on",
        "--temp",
        "1.0",
        "--top-p",
        "0.95",
        "--top-k",
        "20",
        "--jinja",
        "--reasoning-budget",
        "0",
        "--reasoning-format",
        "none",
        "--chat-template-kwargs",
        '{"enable_thinking":false}',
        "--seed",
        str(SEED),
        "--metrics",
        "--no-ui",
        "--no-context-shift",
        "--log-colors",
        "off",
        "--log-timestamps",
        "--log-verbosity",
        "3",
    ]
    if slot_save_path is not None:
        command[command.index("--metrics"):command.index("--metrics")] = ["--slot-save-path", str(slot_save_path)]
    return command


def wait_ready(port: int, process: subprocess.Popen[bytes], timeout: int = 900) -> dict[str, Any]:
    started = time.monotonic()
    attempts: list[dict[str, Any]] = []
    while time.monotonic() - started < timeout:
        if process.poll() is not None:
            return {"ready": False, "elapsed_s": time.monotonic() - started, "attempts": attempts, "process_returncode": process.returncode}
        result = request_json(port, "/health", timeout=10)
        attempts.append({"status": result.get("status"), "body": result.get("body", "")[:1000], "elapsed_s": result.get("elapsed_s")})
        if result.get("status") == 200:
            return {"ready": True, "elapsed_s": time.monotonic() - started, "attempts": attempts}
        time.sleep(1)
    return {"ready": False, "elapsed_s": time.monotonic() - started, "attempts": attempts, "timeout": True}


def stop_process(process: subprocess.Popen[bytes]) -> dict[str, Any]:
    if process.poll() is not None:
        return {"returncode": process.returncode, "signal": None}
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
        return {"returncode": process.returncode, "signal": "SIGTERM"}
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)
        return {"returncode": process.returncode, "signal": "SIGKILL"}
    except ProcessLookupError:
        return {"returncode": process.poll(), "signal": "already-exited"}


def collect_hashes() -> dict[str, Any]:
    runtime_files = []
    for path in sorted(BIN_DIR.iterdir()):
        if path.is_file():
            runtime_files.append({"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)})
    source_commit = command_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"])
    source_status = command_output(["git", "-C", str(SOURCE), "status", "--short"])
    return {
        "runtime_source": str(SOURCE),
        "runtime_commit": source_commit.get("stdout", "").strip(),
        "runtime_git_rev_parse": source_commit,
        "runtime_git_status": source_status,
        "server_binary": {"path": str(SERVER), "size": SERVER.stat().st_size, "sha256": sha256_file(SERVER)},
        "runtime_bin_files": runtime_files,
        "models": {
            name: {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in MODELS.items()
        },
    }


def write_request_artifacts(case_dir: Path, payload: dict[str, Any], result: dict[str, Any], timing: dict[str, Any], telemetry: dict[str, Any], prompt_meta: dict[str, Any]) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    json_dump(case_dir / "request.json", payload)
    (case_dir / "response.sse").write_text(result.get("raw", ""), encoding="utf-8")
    json_dump(case_dir / "response-chunks.json", result.get("chunks", []))
    json_dump(case_dir / "prompt-tokenization.json", prompt_meta)
    json_dump(case_dir / "slots-after.json", prompt_meta.get("slots_after", {}))
    status = {
        "status": result.get("status"),
        "headers": result.get("headers", {}),
        "error": result.get("error"),
        "finish_reason": result.get("finish_reason"),
        "output_tokens_observed": timing.get("predicted_n"),
        "reasoning_chars": len(result.get("reasoning", "")),
    }
    json_dump(case_dir / "http-status.json", status)
    summary = {"prompt": prompt_meta, "http": status, "timings": timing, "telemetry": telemetry, "stream": {key: value for key, value in result.items() if key not in {"raw", "chunks"}}}
    json_dump(case_dir / "result.json", summary)
    return summary


def run_packing(name: str, root: Path, hashes: dict[str, Any], header: str, corpus: str) -> dict[str, Any]:
    port = PORTS[name]
    model = MODELS[name]
    output = root / name
    output.mkdir(parents=True, exist_ok=True)
    env = filtered_environment()
    slot_save_path = output / "slot-cache"
    slot_save_path.mkdir(parents=True, exist_ok=True)
    command = server_command(model, port, slot_save_path)
    json_dump(output / "environment.json", env)
    json_dump(output / "command.json", {"argv": command, "shell": " ".join(shell_quote(value) for value in command)})
    json_dump(output / "hashes.json", {**hashes, "packing": name, "model": hashes["models"][name]})
    (output / "command-and-environment.txt").write_text(
        "Environment:\n" + "\n".join(f"{key}={value}" for key, value in env.items()) + "\n\nCommand:\n" + " ".join(shell_quote(value) for value in command) + "\n",
        encoding="utf-8",
    )
    json_dump(output / "gpu-before.json", gpu_snapshot())
    json_dump(output / "device-list.json", command_output([str(SERVER), "--list-devices"], env=env, timeout=120))
    json_dump(output / "cuda-visible-device-list.json", command_output([str(SERVER), "--list-devices"], env=env, timeout=120))
    server_log = (output / "server.log").open("wb")
    process: subprocess.Popen[bytes] | None = None
    telemetry = Telemetry(output / "telemetry.jsonl")
    results: list[dict[str, Any]] = []
    packing_status: dict[str, Any] = {"packing": name, "port": port, "model": str(model), "started": utc_stamp(), "cases": results}
    try:
        process = subprocess.Popen(command, env=env, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
        packing_status["server_pid"] = process.pid
        telemetry.start()
        telemetry.mark("server_startup")
        ready = wait_ready(port, process)
        packing_status["readiness"] = ready
        json_dump(output / "readiness.json", ready)
        if not ready.get("ready"):
            packing_status["gate"] = "failed_load_or_startup"
            return packing_status
        json_dump(output / "props.json", request_json(port, "/props"))
        json_dump(output / "slots-initial.json", request_json(port, "/slots"))
        baseline_candidates = [sample for sample in telemetry.samples if sample.get("label") == "server_startup"]
        baseline_power: float | None = None
        if baseline_candidates:
            values = [gpu.get("power_w") for gpu in baseline_candidates[-3].get("gpu", {}).get("gpus", []) if gpu.get("index") == 1]
            if values and isinstance(values[0], (int, float)):
                baseline_power = float(values[0])
        for depth in DEPTHS:
            prompt = build_prompt(port, depth, header, corpus)
            prompt_file = output / "prompts" / f"depth-{depth}.txt"
            prompt_file.parent.mkdir(parents=True, exist_ok=True)
            prompt_file.write_text(prompt["content"], encoding="utf-8")
            message_payload = {
                "messages": [
                    {"role": "system", "content": "You are a deterministic coding benchmark assistant. Thinking is disabled."},
                    {"role": "user", "content": prompt["content"]},
                ]
            }
            prompt["chat_input_tokens"] = chat_input_tokens(port, message_payload)
            for cache_state in ("cold", "warm"):
                if cache_state == "cold":
                    cache_clear = request_json(port, "/slots/0?action=erase", {}, timeout=60)
                    json_dump(output / "cache-clears" / f"depth-{depth}.json", cache_clear)
                label = f"{cache_state}-{depth}"
                telemetry.mark(label)
                payload = {
                    "model": f"bonsai-2-27b-{name.lower()}",
                    "messages": [
                        {"role": "system", "content": "You are a deterministic coding benchmark assistant. Thinking is disabled."},
                        {"role": "user", "content": prompt["content"]},
                    ],
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 20,
                    "seed": SEED,
                    "max_tokens": OUTPUT_TOKENS,
                    "ignore_eos": True,
                    "stream": True,
                }
                started = time.monotonic()
                result = chat_request(port, payload)
                ended = time.monotonic()
                timing = timing_from_chunks(result.get("chunks", []))
                prompt_meta = {
                    "target_context_tokens": depth,
                    "content_chars": prompt["content_chars"],
                    "desired_content_tokens": prompt["desired_content_tokens"],
                    "content_token_count": prompt.get("content_tokenize", {}).get("token_count"),
                    "server_prompt_tokens": timing.get("prompt_n"),
                    "server_predicted_tokens": timing.get("predicted_n"),
                    "cache_state": cache_state,
                    "prompt_file": str(prompt_file),
                }
                slots = request_json(port, "/slots")
                slots_parsed = parsed_body(slots)
                prompt_meta["slots_after"] = {"http": {key: value for key, value in slots.items() if key != "body"}, "parsed": slots_parsed}
                slot_total = None
                if isinstance(slots_parsed, list) and slots_parsed and isinstance(slots_parsed[0], dict):
                    slot_total = slots_parsed[0].get("n_prompt_tokens")
                metrics = request_telemetry(telemetry.samples, started, ended, baseline_power)
                timing["wall_s"] = result.get("wall_s", ended - started)
                if isinstance(timing.get("prompt_ms"), (int, float)) and timing.get("prompt_n"):
                    timing["prefill_tok_s"] = timing["prompt_n"] / (timing["prompt_ms"] / 1000)
                if isinstance(timing.get("predicted_ms"), (int, float)) and timing.get("predicted_n"):
                    timing["decode_tok_s"] = timing["predicted_n"] / (timing["predicted_ms"] / 1000)
                timing["wall_output_tok_s"] = (timing.get("predicted_n") or 0) / max(0.000001, result.get("wall_s", ended - started))
                expected = prompt_meta.get("content_token_count") or 0
                prompt_meta["chat_input_token_count"] = (prompt.get("chat_input_tokens", {}).get("parsed") or {}).get("input_tokens")
                prompt_meta["server_prompt_tokens_total"] = slot_total
                prompt_meta["server_prompt_tokens_uncached"] = timing.get("prompt_n")
                prompt_meta["server_prompt_tokens_cached"] = timing.get("cache_n")
                actual = slot_total or 0
                full_expected = prompt_meta["chat_input_token_count"] or expected
                prompt_meta["truncation_suspected"] = bool(actual and full_expected and actual < full_expected)
                case_summary = write_request_artifacts(output / "requests" / label, payload, result, timing, metrics, prompt_meta)
                results.append({"depth": depth, "cache": cache_state, **case_summary})
                telemetry.mark("idle")
                if process.poll() is not None:
                    packing_status["server_failed_after"] = label
                    break
            if process.poll() is not None:
                break
        packing_status["gate"] = "validated_matrix" if len(results) == len(DEPTHS) * 2 else "partial_or_failed_matrix"
        packing_status["props_after"] = request_json(port, "/props")
        packing_status["slots_after"] = request_json(port, "/slots")
        return packing_status
    except Exception as exc:
        packing_status["runner_error"] = repr(exc)
        packing_status["gate"] = "runner_error"
        return packing_status
    finally:
        if process is not None:
            packing_status["cleanup"] = stop_process(process)
        telemetry.mark("cleanup")
        telemetry.stop()
        server_log.close()
        json_dump(output / "gpu-after.json", gpu_snapshot())
        json_dump(output / "packing-status.json", packing_status)


def main() -> int:
    if not SERVER.is_file() or not all(path.is_file() for path in MODELS.values()):
        print("official server or model file missing", file=sys.stderr)
        return 2
    run_root = RESEARCH / f"qwen3.8-27b-serving-alternatives-phase2-{utc_stamp()}"
    run_root.mkdir(parents=True, exist_ok=False)
    hashes = collect_hashes()
    json_dump(run_root / "hashes.json", hashes)
    json_dump(run_root / "host-and-gpu-before.json", {"gpu": gpu_snapshot(), "host": host_snapshot()})
    json_dump(run_root / "physical-gpu-proof.json", {
        "required_mapping": {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1", "candidate_device": "CUDA0", "physical_gpu": 1},
        "before": gpu_snapshot(),
        "server_processes_are_children_only": True,
    })
    header, corpus = make_corpus()
    json_dump(run_root / "matrix-config.json", {
        "depths": DEPTHS,
        "output_tokens": OUTPUT_TOKENS,
        "seed": SEED,
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        "server_flags": server_command(MODELS["PQ2_0"], PORTS["PQ2_0"]),
        "thinking_disable_flags": ["--reasoning-budget 0", "--reasoning-format none", '--chat-template-kwargs {"enable_thinking":false}'],
        "ports": PORTS,
        "runtime_commit_required": "5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6",
    })
    statuses = []
    for name in ("PQ2_0", "PTQ1_0"):
        statuses.append(run_packing(name, run_root, hashes, header, corpus))
    json_dump(run_root / "run-status.json", {"run_root": str(run_root), "packings": statuses, "finished": utc_stamp()})
    json_dump(run_root / "host-and-gpu-after.json", {"gpu": gpu_snapshot(), "host": host_snapshot()})
    print(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
