#!/usr/bin/env python3
"""Bounded, isolated llamAmpere TP=2 experiment for the ATX Qwen GGUF."""

from __future__ import annotations

import csv
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
import urllib.error
import urllib.request


ROOT = Path((os.environ['EXPERIMENT_ROOT'] + ''))
ARTIFACT = ROOT / "research/qwen3.8-27b-llamampere-tp2-20260918T235826Z"
RUNTIME_ROOT = ROOT / "research/llamAmpere-v0.3-36a6bca81"
BINARY = RUNTIME_ROOT / "build-sm86/bin/llama-server"
MODEL = ROOT / "research/qwen3.8-27b-atx-iq4_xs-m-gguf/Qwen3.8-27B-ATX-4-XS.gguf"
VOCAB_MAP = RUNTIME_ROOT / "docs/mtp-vocab/atx_65536.txt"
PORT = 18121
CTX = 245760
MAX_TOKENS = 128
DEPTHS = (64000, 196000, 239960)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def run_capture(command: list[str], env: dict[str, str] | None = None, timeout: int = 120) -> dict:
    try:
        completed = subprocess.run(
            command,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except Exception as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": repr(exc)}


def clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith("BASH_FUNC_") or key.startswith("_BASH_FUNC_"):
            env.pop(key, None)
    return env


def no_cuda_env() -> dict[str, str]:
    env = clean_env()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_preflight() -> None:
    env = no_cuda_env()
    commands = {
        "gpu_inventory": [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,pci.bus_id,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
        ],
        "compute_apps": [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        "topology": ["nvidia-smi", "topo", "-m"],
        "ports": ["ss", "-ltn"],
        "candidate_processes": ["pgrep", "-af", "llama-server"],
        "qwen_service": ["systemctl", "is-active", "llama-qwen3.8-q4-tensor-262k.service"],
        "gateway_service": ["systemctl", "is-active", "lan-inference-gateway.service"],
    }
    write_json(
        ARTIFACT / "preflight.json",
        {name: run_capture(command, env=env) for name, command in commands.items()},
    )


def server_command(mtp: bool) -> list[str]:
    command = [
        str(BINARY),
        "-m",
        str(MODEL),
        "-c",
        str(CTX),
        "-b",
        "4096",
        "-ub",
        "1024",
        "-t",
        "8",
        "-tb",
        "8",
        "-ngl",
        "99",
        "-fa",
        "on",
        "-ctk",
        "q8_0",
        "-ctv",
        "turbo3",
        "--parallel",
        "1",
        "--jinja",
        "--fit",
        "off",
        "--cache-prompt",
        "--cache-ram",
        "8192",
        "--ctx-checkpoints",
        "24",
        "--checkpoint-min-step",
        "10240",
        "--split-mode",
        "tensor",
        "--tensor-split",
        "1,1",
        "--device",
        "CUDA0,CUDA1",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--metrics",
        "--no-ui",
        "--log-colors",
        "off",
        "--log-file",
        str(ARTIFACT / ("server-mtp3.log" if mtp else "server-non-mtp.log")),
    ]
    if mtp:
        command.extend(
            [
                "--spec-type",
                "draft-mtp",
                "--spec-draft-n-max",
                "3",
                "--spec-draft-p-min",
                "0",
                "--spec-draft-type-k",
                "q8_0",
                "--spec-draft-type-v",
                "q8_0",
                "--spec-draft-vocab-map",
                str(VOCAB_MAP),
            ]
        )
    return command


def server_environment() -> dict[str, str]:
    env = clean_env()
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "1,2",
            "GGML_Q8_TURBO3_MMA_FUSED": "1",
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,COLL",
        }
    )
    env.pop("GGML_CUDA_ENABLE_UNIFIED_MEMORY", None)
    return env


def gpu_snapshot() -> list[dict]:
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,uuid,pci.bus_id,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem",
        "--format=csv,noheader,nounits",
    ]
    result = run_capture(command, env=no_cuda_env())
    rows: list[dict] = []
    for line in result["stdout"].splitlines():
        fields = next(csv.reader([line], skipinitialspace=True), [])
        if len(fields) != 12:
            continue
        try:
            rows.append(
                {
                    "timestamp": fields[0].strip(),
                    "index": int(fields[1]),
                    "name": fields[2].strip(),
                    "uuid": fields[3].strip(),
                    "pci_bus_id": fields[4].strip(),
                    "memory_used_mib": float(fields[5]),
                    "memory_total_mib": float(fields[6]),
                    "utilization_gpu_percent": float(fields[7]),
                    "power_draw_w": float(fields[8]),
                    "temperature_c": float(fields[9]),
                    "clock_sm_mhz": float(fields[10]),
                    "clock_mem_mhz": float(fields[11]),
                }
            )
        except ValueError:
            continue
    return rows


def host_snapshot(pid: int) -> dict:
    memory = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            fields = value.strip().split()
            if fields and fields[0].isdigit():
                memory[key] = int(fields[0]) * (1024 if len(fields) > 1 and fields[1] == "kB" else 1)
    except (OSError, ValueError):
        pass
    process = {}
    result = run_capture(["ps", "-p", str(pid), "-o", "pcpu=,rss=,etime="], env=no_cuda_env())
    fields = result["stdout"].strip().split()
    if len(fields) >= 3:
        try:
            process = {"cpu_percent": float(fields[0]), "rss_kib": float(fields[1]), "elapsed": fields[2]}
        except ValueError:
            pass
    return {"memory": memory, "process": process}


def monitor(pid: int, path: Path, stop: threading.Event) -> None:
    with path.open("w", encoding="utf-8") as output:
        while not stop.is_set():
            sample = {
                "epoch_seconds": time.time(),
                "monotonic_seconds": time.monotonic(),
                "gpus": gpu_snapshot(),
                "host": host_snapshot(pid),
            }
            output.write(json.dumps(sample, sort_keys=True) + "\n")
            output.flush()
            stop.wait(1.0)


def get_json(path: str, timeout: int = 30) -> tuple[int, dict, dict]:
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            value = json.loads(body.decode("utf-8")) if body else {}
            return response.status, value, dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            value = json.loads(body.decode("utf-8")) if body else {}
        except json.JSONDecodeError:
            value = {"raw": body.decode("utf-8", errors="replace")}
        return exc.code, value, dict(exc.headers.items())


def wait_ready(process: subprocess.Popen[bytes], timeout: int = 300) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup with code {process.returncode}")
        try:
            status, value, _ = get_json("/v1/models", timeout=5)
        except (OSError, urllib.error.URLError):
            status, value = None, {}
        if status == 200 and isinstance(value, dict) and value.get("data"):
            return
        time.sleep(1)
    raise TimeoutError("candidate server did not become ready")


def post_json(path: str, payload: dict, timeout: int = 1800) -> tuple[int | None, dict, dict, str | None]:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            value = json.loads(body.decode("utf-8")) if body else {}
            return response.status, value, dict(response.headers.items()), None
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            value = json.loads(body.decode("utf-8")) if body else {}
        except json.JSONDecodeError:
            value = {"raw": body.decode("utf-8", errors="replace")}
        return exc.code, value, dict(exc.headers.items()), None
    except Exception as exc:
        return None, {}, {}, repr(exc)


def stream_completion(payload: dict, timeout: int = 1800) -> dict:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    first_event = None
    raw = bytearray()
    events: list[dict] = []
    status: int | None = None
    headers: dict = {}
    error: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            headers = dict(response.headers.items())
            while True:
                line = response.readline()
                if not line:
                    break
                raw.extend(line)
                if not line.startswith(b"data:"):
                    continue
                data_line = line[5:].strip()
                if data_line == b"[DONE]":
                    continue
                try:
                    event = json.loads(data_line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                if first_event is None:
                    first_event = time.time()
                events.append(event)
    except urllib.error.HTTPError as exc:
        status = exc.code
        headers = dict(exc.headers.items())
        raw.extend(exc.read())
        error = f"HTTPError: {exc}"
    except Exception as exc:
        error = repr(exc)
    finished = time.time()
    content = "".join(str(event.get("content", "")) for event in events)
    last_event = events[-1] if events else {}
    return {
        "status": status,
        "headers": headers,
        "error": error,
        "started_epoch": started,
        "finished_epoch": finished,
        "wall_seconds": finished - started,
        "ttft_ms": (first_event - started) * 1000 if first_event is not None else None,
        "events_count": len(events),
        "content": content,
        "last_event": last_event,
        "raw": bytes(raw),
    }


def tokenize(text: str, name: str) -> int:
    status, value, headers, error = post_json("/tokenize", {"content": text}, timeout=180)
    write_json(
        ARTIFACT / name,
        {"status": status, "headers": headers, "error": error, "content_sha256": hashlib.sha256(text.encode()).hexdigest(), "response": value},
    )
    if error or status != 200 or not isinstance(value.get("tokens"), list):
        raise RuntimeError(f"tokenize failed: status={status}, error={error}, response={value}")
    return len(value["tokens"])


def build_prompt(target: int, label: str) -> tuple[str, int]:
    prefix = (
        "Synthetic TP2 coding and neutral prefill workload. Read every record in order and retain the final instruction. "
        "The records are deliberately repetitive but include code-shaped and neutral text. Depth label: %s.\n\n" % label
    )
    units = [
        "Record class code: def normalize_value(value: int, offset: int) -> int: return value + offset.\n",
        "Record class neutral: The stable observation is blue cedar seven, with no external action.\n",
        "Record class code: if total < 0: total = 0; return min(total, 1000003).\n",
        "Record class neutral: Preserve ordering, preserve exact integers, and do not invent records.\n",
        'Record class code: result = {"name": "sample", "enabled": True, "count": 3}.\n',
        "Record class neutral: This sentence is a fixed synthetic datum for long-context measurement.\n",
        "Record class code: for item in values: checksum = (checksum * 31 + item) % 1000000007.\n",
        "Record class neutral: The expected final response is short and must not quote the records.\n",
    ]
    cycle = "".join(units)
    prefix_tokens = tokenize(prefix, f"{label}-prefix-tokenize.json")
    sample_tokens = tokenize(prefix + cycle * 64, f"{label}-sample-tokenize.json")
    cycle_tokens = max(1, (sample_tokens - prefix_tokens) // 64)
    cycles = max(1, (target - prefix_tokens) // cycle_tokens)
    suffix = (
        "\nFinal instruction: after reading the complete context, output exactly one line with the label %s and the integer "
        "checksum 3817. Do not repeat the records.\n" % label
    )
    text = prefix + cycle * cycles + suffix
    count = tokenize(text, f"{label}-tokenize-0.json")
    for attempt in range(8):
        if abs(count - target) <= max(64, cycle_tokens // 2):
            break
        delta = target - count
        step = max(1, abs(delta) // cycle_tokens)
        cycles = max(1, cycles + (step if delta > 0 else -step))
        text = prefix + cycle * cycles + suffix
        count = tokenize(text, f"{label}-tokenize-{attempt + 1}.json")
    prompt_path = ARTIFACT / f"{label}-prompt.txt"
    prompt_path.write_text(text, encoding="utf-8")
    write_json(
        ARTIFACT / f"{label}-prompt-metadata.json",
        {
            "label": label,
            "target_tokens": target,
            "actual_tokenize_count": count,
            "prefix_tokens": prefix_tokens,
            "cycle_tokens_estimate": cycle_tokens,
            "cycles": cycles,
            "prompt_path": str(prompt_path),
            "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
        },
    )
    return text, count


def request_payload(prompt: str, stream: bool, cache_prompt: bool, seed: int) -> dict:
    return {
        "prompt": prompt,
        "n_predict": MAX_TOKENS,
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "seed": seed,
        "ignore_eos": True,
        "stream": stream,
        "cache_prompt": cache_prompt,
    }


def request_cell(label: str, target: int, prompt: str, prompt_tokens: int, phase: str, smoke: bool) -> dict:
    settings = request_payload(prompt, stream=True, cache_prompt=not smoke, seed=6100)
    write_json(
        ARTIFACT / f"{label}-{phase}-request-settings.json",
        {key: value for key, value in settings.items() if key != "prompt"} | {"prompt_tokens": prompt_tokens},
    )
    started = time.time()
    response = stream_completion(settings)
    ended = time.time()
    raw_path = ARTIFACT / f"{label}-{phase}-response.sse"
    raw_path.write_bytes(response.pop("raw"))
    content = response.get("content", "")
    last = response.get("last_event", {})
    correctness = {
        "http_ok": response.get("status") == 200,
        "has_content": bool(content),
        "expected_marker": ("clamp_int" in content and "return max" in content) if smoke else (label in content and "3817" in content),
    }
    response["correctness"] = correctness
    response["prompt_tokens"] = prompt_tokens
    response["phase"] = phase
    response["label"] = label
    response["target_token_depth"] = target
    response["request_started_epoch"] = started
    response["request_finished_epoch"] = ended
    response["timings"] = last.get("timings", {}) if isinstance(last, dict) else {}
    response["tokens_evaluated"] = last.get("tokens_evaluated") if isinstance(last, dict) else None
    response["tokens_predicted"] = last.get("tokens_predicted") if isinstance(last, dict) else None
    response["stop_type"] = last.get("stop_type") if isinstance(last, dict) else None
    response["truncated"] = last.get("truncated") if isinstance(last, dict) else None
    write_json(ARTIFACT / f"{label}-{phase}-response.json", response)
    return response


def terminate(process: subprocess.Popen[bytes]) -> None:
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


def load_samples(path: Path) -> list[dict]:
    samples = []
    if not path.exists():
        return samples
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return samples


def energy(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    return sum((b[0] - a[0]) * (a[1] + b[1]) * 0.5 for a, b in zip(points, points[1:])) / 3600.0


def request_gpu_summary(samples: list[dict], start: float, end: float) -> dict:
    result = {}
    for index in (0, 1, 2):
        rows = [
            gpu
            for sample in samples
            if start <= sample["epoch_seconds"] <= end
            for gpu in sample.get("gpus", [])
            if gpu.get("index") == index
        ]
        powers = [(sample["epoch_seconds"], gpu["power_draw_w"]) for sample in samples for gpu in sample.get("gpus", []) if gpu.get("index") == index and start <= sample["epoch_seconds"] <= end]
        result[str(index)] = {
            "sample_count": len(rows),
            "max_memory_used_mib": max((row["memory_used_mib"] for row in rows), default=None),
            "max_utilization_gpu_percent": max((row["utilization_gpu_percent"] for row in rows), default=None),
            "max_power_draw_w": max((row["power_draw_w"] for row in rows), default=None),
            "mean_power_draw_w": (sum(row["power_draw_w"] for row in rows) / len(rows)) if rows else None,
            "max_temperature_c": max((row["temperature_c"] for row in rows), default=None),
            "min_clock_sm_mhz": min((row["clock_sm_mhz"] for row in rows), default=None),
            "max_clock_sm_mhz": max((row["clock_sm_mhz"] for row in rows), default=None),
            "energy_wh": energy(powers),
        }
    return result


def summarize_requests(cells: list[dict], sample_paths: list[Path]) -> None:
    all_samples = [sample for path in sample_paths for sample in load_samples(path)]
    for cell in cells:
        for request in cell.get("requests", []):
            start = request.get("request_started_epoch")
            end = request.get("request_finished_epoch")
            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                request["gpu_summary"] = request_gpu_summary(all_samples, start, end)
    write_json(ARTIFACT / "measurements.json", {"cells": cells, "sample_paths": [str(path) for path in sample_paths]})


def run_mode(mtp: bool) -> tuple[bool, list[dict], Path]:
    mode = "mtp3" if mtp else "non-mtp"
    command = server_command(mtp)
    env = server_environment()
    write_json(
        ARTIFACT / f"process-environment-{mode}.json",
        {
            "captured_at_utc": utc_now(),
            "mode": mode,
            "command": command,
            "environment": {key: env.get(key, "") for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "GGML_Q8_TURBO3_MMA_FUSED", "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS")},
        },
    )
    (ARTIFACT / f"exact-command-{mode}.txt").write_text(
        "environment:\n" + "\n".join(f"{key}={env.get(key, '')}" for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "GGML_Q8_TURBO3_MMA_FUSED", "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS")) + "\ncommand:\n" + " ".join(command) + "\n",
        encoding="utf-8",
    )
    stdout_path = ARTIFACT / f"server-{mode}.stdout-stderr.log"
    sample_path = ARTIFACT / f"gpu-samples-{mode}.jsonl"
    cells: list[dict] = []
    process = None
    log = None
    monitor_stop = threading.Event()
    monitor_thread = None
    try:
        log = stdout_path.open("wb")
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (ARTIFACT / f"server-{mode}.pid").write_text(str(process.pid) + "\n", encoding="utf-8")
        wait_ready(process)
        write_json(ARTIFACT / f"health-{mode}.json", {"status": 200, "models": get_json("/v1/models")[1], "captured_at_utc": utc_now()})
        # Do not query NVML while CUDA/NCCL is still initializing the ranks.
        monitor_thread = threading.Thread(target=monitor, args=(process.pid, sample_path, monitor_stop), daemon=True)
        monitor_thread.start()

        smoke_prompt = "Return only a Python function named clamp_int(value, low, high) that clamps an integer to the inclusive range [low, high]. Include a short type annotation and no markdown fences, explanation, or repeated text."
        smoke_tokens = tokenize(smoke_prompt, f"smoke-8k-{mode}-tokenize.json")
        smoke = request_cell("smoke-8k", 8192, smoke_prompt, smoke_tokens, "cold", True)
        smoke["mode"] = mode
        cells.append({"label": "smoke-8k", "mode": mode, "status": "passed" if all(smoke["correctness"].values()) else "failed", "requests": [smoke]})
        write_json(ARTIFACT / f"smoke-8k-{mode}-result.json", cells[-1])
        if cells[-1]["status"] != "passed":
            return False, cells, sample_path

        for target in DEPTHS:
            label = f"depth{target // 1000 if target != 239960 else 239960}k" if target != 239960 else "depth239960"
            prompt, prompt_tokens = build_prompt(target, label)
            cold = request_cell(label, target, prompt, prompt_tokens, "cold", False)
            warm = request_cell(label, target, prompt, prompt_tokens, "warm", False)
            item = {
                "label": label,
                "mode": mode,
                "target_token_depth": target,
                "actual_prompt_tokens": prompt_tokens,
                "status": "passed" if all(cold["correctness"].values()) and all(warm["correctness"].values()) else "failed",
                "requests": [cold, warm],
            }
            cells.append(item)
            write_json(ARTIFACT / f"{label}-{mode}-result.json", item)
            if item["status"] != "passed":
                break
        return all(item["status"] == "passed" for item in cells), cells, sample_path
    except Exception as exc:
        cells.append({"label": "mode-run", "mode": mode, "status": "failed", "error": repr(exc)})
        write_json(ARTIFACT / f"failure-{mode}.json", cells[-1])
        return False, cells, sample_path
    finally:
        monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=5)
        if process is not None:
            terminate(process)
            write_json(ARTIFACT / f"server-exit-{mode}.json", {"returncode": process.returncode, "captured_at_utc": utc_now()})
        if log is not None:
            log.close()
        after = {
            "captured_at_utc": utc_now(),
            "gpu_snapshot": gpu_snapshot(),
            "compute_apps": run_capture(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"], env=no_cuda_env()),
            "ports": run_capture(["ss", "-ltn"], env=no_cuda_env()),
            "candidate_processes": run_capture(["pgrep", "-af", "llama-server"], env=no_cuda_env()),
        }
        write_json(ARTIFACT / f"cleanup-{mode}.json", after)


def main() -> None:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    write_preflight()
    write_json(
        ARTIFACT / "runtime-metadata.json",
        {
            "runtime_root": str(RUNTIME_ROOT),
            "binary": str(BINARY),
            "binary_sha256": sha256(BINARY),
            "model": str(MODEL),
            "model_sha256": sha256(MODEL),
            "model_size_bytes": MODEL.stat().st_size,
            "vocab_map": str(VOCAB_MAP),
            "vocab_map_sha256": sha256(VOCAB_MAP),
            "version": run_capture([str(BINARY), "--version"], env=no_cuda_env()),
            "git_head": run_capture(["git", "-C", str(RUNTIME_ROOT), "log", "-1", "--format=%H %cI %s"], env=no_cuda_env()),
            "git_status": run_capture(["git", "-C", str(RUNTIME_ROOT), "status", "--short"], env=no_cuda_env()),
            "ldd": run_capture(["ldd", str(BINARY)], env=no_cuda_env()),
            "captured_at_utc": utc_now(),
        },
    )

    mtp_ok, mtp_cells, mtp_samples = run_mode(True)
    if mtp_ok:
        selected_mode = "mtp3"
        cells = mtp_cells
        sample_paths = [mtp_samples]
        mtp_decision = "MTP3 smoke passed in tensor mode; benchmarked existing MTP setup."
    else:
        non_mtp_ok, non_mtp_cells, non_mtp_samples = run_mode(False)
        selected_mode = "non-mtp"
        cells = mtp_cells + non_mtp_cells
        sample_paths = [mtp_samples, non_mtp_samples]
        mtp_decision = "MTP3 tensor-mode smoke failed or was incorrect; benchmarked supported non-MTP mode."
        if not non_mtp_ok:
            mtp_decision += " Non-MTP run also failed; no further cells attempted."
    summarize_requests(cells, sample_paths)
    summary = {
        "artifact": str(ARTIFACT),
        "completed_at_utc": utc_now(),
        "selected_mode": selected_mode,
        "mtp_decision": mtp_decision,
        "requested_configuration": {
            "physical_gpus": [1, 2],
            "cuda_visible_devices": "1,2",
            "model": str(MODEL),
            "ctx_size": CTX,
            "batch_size": 4096,
            "ubatch_size": 1024,
            "split_mode": "tensor",
            "tensor_split": "1,1",
            "fit": "off",
            "flash_attention": "on",
            "cache_k": "q8_0",
            "cache_v": "turbo3",
            "nccl_debug": "INFO",
        },
        "cells": cells,
    }
    write_json(ARTIFACT / "run-summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
