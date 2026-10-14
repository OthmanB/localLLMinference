#!/usr/bin/env python3
import csv
import hashlib
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path((os.environ['EXPERIMENT_ROOT'] + ''))
ARTIFACT = Path(sys.argv[1]).resolve()
SERVER = ROOT / "research/llamAmpere-v0.3-36a6bca81/build-sm86/bin/llama-server"
MODEL = ROOT / "research/qwen3.8-27b-atx-iq4_xs-m-gguf/Qwen3.8-27B-ATX-4-XS.gguf"
VOCAB_MAP = ROOT / "research/llamAmpere-v0.3-36a6bca81/docs/mtp-vocab/atx_65536.txt"
PORTS = {"gpu1": 18111, "gpu2": 18112}
VISIBLE = {"gpu1": "1", "gpu2": "2"}
TARGETS = [("064k", 64000), ("196k", 196000)]
MAX_TOKENS = 128
REQUEST_TIMEOUT = 900
SAMPLER_INTERVAL = 0.75
STOP_EVENT = threading.Event()
SERVERS = {}
REQUEST_SUMMARIES = []
PAIR_SUMMARIES = []


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_result(command, env=None, timeout=60):
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except Exception as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": repr(exc)}


def unmasked_env():
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def gpu_query_command():
    return [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current",
        "--format=csv,noheader,nounits",
    ]


def gpu_query():
    return command_result(gpu_query_command(), env=unmasked_env())


def compute_query(gpu):
    return command_result(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        env=unmasked_env(),
    )


def port_snapshot():
    return command_result(["ss", "-ltnp"], env=unmasked_env())


def process_snapshot(pids=None):
    if pids:
        pid_args = [str(pid) for pid in pids]
        command = ["ps", "-p", ",".join(pid_args), "-o", "pid=,ppid=,stat=,%cpu=,%mem=,rss=,vsz=,etime=,args="]
    else:
        command = ["ps", "-eo", "pid,ppid,user,stat,cmd"]
    return command_result(command, env=unmasked_env())


def host_snapshot():
    loadavg = Path("/proc/loadavg").read_text(encoding="utf-8").strip()
    meminfo = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meminfo[key] = value.strip()
    return {"loadavg": loadavg, "meminfo": meminfo}


def write_command_environment():
    common = [
        str(SERVER),
        "-m",
        str(MODEL),
        "-c",
        "245760",
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
        "--reasoning",
        "off",
        "--chat-template-kwargs",
        '{"enable_thinking":false}',
        "--device",
        "CUDA0",
    ]
    lines = [
        "artifact=%s" % ARTIFACT,
        "model=%s" % MODEL,
        "model_sha256=%s" % sha256_file(MODEL),
        "binary=%s" % SERVER,
        "binary_sha256=%s" % sha256_file(SERVER),
        "mtp_document=%s:59-65" % (ROOT / "research/llamAmpere-v0.3-36a6bca81/QWEN_AMPERE.md"),
        "mtp_gpu_verify=<unset> (documented default; no extra sampled-verification working set)",
        "",
    ]
    for name, port in PORTS.items():
        env = os.environ.copy()
        env.update(
            {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": VISIBLE[name],
                "GGML_Q8_TURBO3_MMA_FUSED": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        command = common + ["--host", "127.0.0.1", "--port", str(port)]
        lines.extend(
            [
                "[%s] physical_gpu=%s port=%d" % (name, VISIBLE[name], port),
                "CUDA_DEVICE_ORDER=%s" % env["CUDA_DEVICE_ORDER"],
                "CUDA_VISIBLE_DEVICES=%s" % env["CUDA_VISIBLE_DEVICES"],
                "GGML_Q8_TURBO3_MMA_FUSED=%s" % env["GGML_Q8_TURBO3_MMA_FUSED"],
                "PYTHONUNBUFFERED=%s" % env["PYTHONUNBUFFERED"],
                "LLAMA_MTP_GPU_VERIFY=<unset>",
                "LD_LIBRARY_PATH=%s" % env.get("LD_LIBRARY_PATH", "<unset>"),
                "PATH=%s" % env.get("PATH", "<unset>"),
                "command:",
                shlex.join(command),
                "",
            ]
        )
        SERVERS[name] = {"command": command, "env": env, "port": port}
    (ARTIFACT / "exact-command-environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def capture_preflight():
    write_json(ARTIFACT / "preflight-nvidia-smi.json", gpu_query())
    (ARTIFACT / "preflight-nvidia-smi.txt").write_text(
        gpu_query_command().__repr__() + "\n" + json.dumps(gpu_query(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for gpu in (0, 1, 2):
        result = compute_query(gpu)
        (ARTIFACT / ("preflight-compute-gpu%d.txt" % gpu)).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (ARTIFACT / "preflight-ports.txt").write_text(json.dumps(port_snapshot(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ARTIFACT / "preflight-processes.txt").write_text(json.dumps(process_snapshot(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_json(
        ARTIFACT / "preflight.json",
        {
            "captured_at_utc": utc_now(),
            "host": "127.0.0.1",
            "ports": PORTS,
            "physical_gpus": {"gpu1": 1, "gpu2": 2},
            "forbidden_physical_gpus": [0],
            "cuda_device_order": "PCI_BUS_ID",
            "server_visible_cuda_device": "CUDA0 per process",
            "systemd_used": False,
            "production_ports_used": False,
            "production_services_touched": False,
        },
    )


def runtime_metadata():
    git = command_result(["git", "-C", str(ROOT / "research/llamAmpere-v0.3-36a6bca81"), "log", "-1", "--format=%H %cI %s"])
    status = command_result(["git", "-C", str(ROOT / "research/llamAmpere-v0.3-36a6bca81"), "status", "--short"])
    metadata = {
        "captured_at_utc": utc_now(),
        "binary": str(SERVER),
        "binary_sha256": sha256_file(SERVER),
        "model": {"path": str(MODEL), "size_bytes": MODEL.stat().st_size, "sha256": sha256_file(MODEL)},
        "vocab_map": {"path": str(VOCAB_MAP), "size_bytes": VOCAB_MAP.stat().st_size, "sha256": sha256_file(VOCAB_MAP)},
        "version": command_result([str(SERVER), "--version"]),
        "ldd": command_result(["ldd", str(SERVER)]),
        "git": {"repo": str(ROOT / "research/llamAmpere-v0.3-36a6bca81"), "head": git, "status": status},
    }
    write_json(ARTIFACT / "runtime-metadata.json", metadata)


def start_servers():
    for name, server in SERVERS.items():
        instance_dir = ARTIFACT / name
        instance_dir.mkdir(parents=True, exist_ok=True)
        log_handle = (instance_dir / "server.stdout-stderr.log").open("wb")
        process = subprocess.Popen(
            server["command"],
            cwd=ROOT,
            env=server["env"],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        server["process"] = process
        server["log_handle"] = log_handle
        server["pid"] = process.pid
        (instance_dir / "server.pid").write_text(str(process.pid) + "\n", encoding="utf-8")
        write_json(
            instance_dir / "launch.json",
            {
                "started_at_utc": utc_now(),
                "pid": process.pid,
                "physical_gpu": int(VISIBLE[name]),
                "cuda_visible_devices": VISIBLE[name],
                "command": server["command"],
                "environment": {
                    "CUDA_DEVICE_ORDER": server["env"]["CUDA_DEVICE_ORDER"],
                    "CUDA_VISIBLE_DEVICES": server["env"]["CUDA_VISIBLE_DEVICES"],
                    "GGML_Q8_TURBO3_MMA_FUSED": server["env"]["GGML_Q8_TURBO3_MMA_FUSED"],
                    "PYTHONUNBUFFERED": server["env"]["PYTHONUNBUFFERED"],
                },
            },
        )


def get_json(base_url, path, payload, timeout=120):
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        base_url + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, dict(response.headers.items()), response.read()


def wait_ready(name):
    base_url = "http://127.0.0.1:%d" % SERVERS[name]["port"]
    poll_path = ARTIFACT / name / "health-poll.jsonl"
    deadline = time.monotonic() + 240
    with poll_path.open("w", encoding="utf-8") as handle:
        while time.monotonic() < deadline:
            item = {"captured_at_utc": utc_now(), "monotonic": time.monotonic()}
            try:
                with urllib.request.urlopen(base_url + "/health", timeout=10) as response:
                    item.update({"status": response.status, "body": response.read().decode("utf-8", "replace")})
                    ready = response.status == 200
            except Exception as exc:
                item.update({"status": None, "error": repr(exc)})
                ready = False
            handle.write(json.dumps(item, sort_keys=True) + "\n")
            handle.flush()
            if ready:
                write_json(ARTIFACT / name / "health-ready.json", item)
                return
            if SERVERS[name]["process"].poll() is not None:
                raise RuntimeError("%s server exited with %s" % (name, SERVERS[name]["process"].returncode))
            time.sleep(1)
    raise TimeoutError("timed out waiting for %s health" % name)


def process_environment_proof():
    proof = {}
    for name, server in SERVERS.items():
        pid = server.get("pid")
        entry = {"pid": pid, "physical_gpu": int(VISIBLE[name])}
        if pid:
            try:
                raw = Path("/proc/%d/environ" % pid).read_bytes()
                values = {}
                for part in raw.split(b"\0"):
                    if b"=" in part:
                        key, value = part.split(b"=", 1)
                        key = key.decode("utf-8", "replace")
                        if key in {"CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "GGML_Q8_TURBO3_MMA_FUSED", "PYTHONUNBUFFERED"}:
                            values[key] = value.decode("utf-8", "replace")
                entry["environment"] = values
                entry["cmdline"] = Path("/proc/%d/cmdline" % pid).read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
            except Exception as exc:
                entry["error"] = repr(exc)
        proof[name] = entry
    write_json(ARTIFACT / "process-environment-proof.json", proof)


def sampler():
    path = ARTIFACT / "gpu-samples.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        while not STOP_EVENT.is_set():
            record = {
                "captured_at_utc": utc_now(),
                "wall_time": time.time(),
                "monotonic": time.monotonic(),
                "gpu_query": gpu_query(),
                "compute_apps": {str(gpu): compute_query(gpu) for gpu in (0, 1, 2)},
                "processes": process_snapshot([server.get("pid") for server in SERVERS.values() if server.get("pid")]),
                "host": host_snapshot(),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            STOP_EVENT.wait(SAMPLER_INTERVAL)


def tokenize(name, text, label):
    base_url = "http://127.0.0.1:%d" % SERVERS[name]["port"]
    request = {"content": text}
    started = time.time()
    try:
        status, headers, body = get_json(base_url, "/tokenize", request, timeout=300)
        error = None
    except Exception as exc:
        status, headers, body, error = None, {}, b"", repr(exc)
    path = ARTIFACT / "prompts" / label / (name + "-tokenize.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / (name + "-tokenize.body")).write_bytes(body)
    write_json(
        path,
        {"status": status, "headers": headers, "error": error, "wall_seconds": time.time() - started, "request": request},
    )
    if error or status != 200:
        raise RuntimeError("tokenize failed for %s/%s: %s" % (name, label, error or status))
    return len(json.loads(body.decode("utf-8"))["tokens"])


def make_prompt(label, target):
    prefix = (
        "Synthetic independent dual-GPU serving workload. Read every record in order and retain the final instruction. "
        "The records are repetitive and include code-shaped and neutral text. Depth label: %s.\n\n" % label
    )
    units = [
        "Record class code: def normalize_value(value: int, offset: int) -> int: return value + offset.\n",
        "Record class neutral: The stable observation is blue cedar seven, with no external action.\n",
        "Record class code: if total < 0: total = 0; return min(total, 1000003).\n",
        "Record class neutral: Preserve ordering, preserve exact integers, and do not invent records.\n",
        "Record class code: result = {\"name\": \"sample\", \"enabled\": True, \"count\": 3}.\n",
        "Record class neutral: This sentence is a fixed synthetic datum for long-context measurement.\n",
        "Record class code: for item in values: checksum = (checksum * 31 + item) % 1000000007.\n",
        "Record class neutral: The expected final response is short and must not quote the records.\n",
    ]
    cycle = "".join(units)
    suffix = (
        "\nFinal instruction: after reading the complete context, output exactly one line with the label %s "
        "and the integer checksum 3817. Do not repeat the records.\n" % label
    )
    prefix_count = tokenize("gpu1", prefix, label + "-prefix")
    sample_count = tokenize("gpu1", prefix + cycle * 64, label + "-sample")
    cycle_tokens = max(1, (sample_count - prefix_count) // 64)
    cycles = max(1, (target - prefix_count) // cycle_tokens)
    text = prefix + cycle * cycles + suffix
    count = 0
    count1 = 0
    count2 = 0
    attempts = []
    for attempt in range(10):
        count1 = tokenize("gpu1", text, "%s-calibration-%d" % (label, attempt))
        count2 = tokenize("gpu2", text, "%s-calibration-%d" % (label, attempt))
        count = min(count1, count2)
        attempts.append({"attempt": attempt, "gpu1": count1, "gpu2": count2, "cycles": cycles})
        if abs(count - target) <= max(64, cycle_tokens // 2):
            break
        delta = target - count
        step = max(1, abs(delta) // cycle_tokens)
        cycles = max(1, cycles + (step if delta > 0 else -step))
        text = prefix + cycle * cycles + suffix
    prompt_dir = ARTIFACT / "prompts" / label
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "prompt.txt").write_text(text, encoding="utf-8")
    write_json(
        prompt_dir / "prompt-metadata.json",
        {
            "label": label,
            "target_tokens": target,
            "actual_tokens_gpu1": count1,
            "actual_tokens_gpu2": count2,
            "actual_tokens": count,
            "prefix_tokens": prefix_count,
            "cycle_tokens": cycle_tokens,
            "cycles": cycles,
            "calibration": attempts,
            "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        },
    )
    return text, count


def parse_stream_body(raw):
    parsed = []
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if line.startswith("data:"):
            value = line[5:].strip()
            if value and value != "[DONE]":
                try:
                    parsed.append(json.loads(value))
                except json.JSONDecodeError:
                    pass
        elif line.startswith("{"):
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return parsed


def stream_completion(name, label, phase, prompt, prompt_count):
    base_url = "http://127.0.0.1:%d" % SERVERS[name]["port"]
    request_payload = {
        "prompt": prompt,
        "n_predict": MAX_TOKENS,
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "seed": 1729,
        "ignore_eos": True,
        "stream": True,
        "cache_prompt": True,
    }
    request_dir = ARTIFACT / name / "requests" / ("depth%s-%s" % (label, phase))
    request_dir.mkdir(parents=True, exist_ok=True)
    write_json(request_dir / "request.json", request_payload)
    started_wall = time.time()
    started_mono = time.monotonic()
    first_byte_wall = None
    raw = bytearray()
    status = None
    headers = {}
    error = None
    try:
        body = json.dumps(request_payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            base_url + "/completion",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            status = response.status
            headers = dict(response.headers.items())
            while True:
                line = response.readline()
                if not line:
                    break
                if first_byte_wall is None:
                    first_byte_wall = time.time()
                raw.extend(line)
    except urllib.error.HTTPError as exc:
        status = exc.code
        headers = dict(exc.headers.items()) if exc.headers else {}
        raw.extend(exc.read())
        error = repr(exc)
    except Exception as exc:
        error = repr(exc)
    ended_wall = time.time()
    ended_mono = time.monotonic()
    raw_path = request_dir / "response.body"
    raw_path.write_bytes(bytes(raw))
    (request_dir / "headers.txt").write_text(
        "status=%s\nerror=%s\nstarted_wall=%.9f\nfirst_byte_wall=%s\nended_wall=%.9f\n%s"
        % (
            status,
            error,
            started_wall,
            "%.9f" % first_byte_wall if first_byte_wall is not None else "None",
            ended_wall,
            "".join("%s: %s\n" % item for item in sorted(headers.items())),
        ),
        encoding="utf-8",
    )
    chunks = parse_stream_body(bytes(raw))
    final = chunks[-1] if chunks else {}
    write_json(request_dir / "response.json", final)
    summary = {
        "name": name,
        "label": label,
        "phase": phase,
        "physical_gpu": int(VISIBLE[name]),
        "port": SERVERS[name]["port"],
        "status": status,
        "error": error,
        "prompt_count": prompt_count,
        "max_tokens": MAX_TOKENS,
        "started_wall": started_wall,
        "ended_wall": ended_wall,
        "started_monotonic": started_mono,
        "ended_monotonic": ended_mono,
        "wall_seconds": ended_wall - started_wall,
        "first_byte_wall": first_byte_wall,
        "ttft_ms": (first_byte_wall - started_wall) * 1000 if first_byte_wall is not None else None,
        "response_bytes": len(raw),
        "stream_chunks": len(chunks),
        "timings": final.get("timings", {}),
        "tokens_evaluated": final.get("tokens_evaluated"),
        "tokens_predicted": final.get("tokens_predicted"),
        "stop_type": final.get("stop_type"),
        "truncated": final.get("truncated"),
        "tokens_cached": final.get("tokens_cached"),
    }
    write_json(request_dir / "result.json", summary)
    REQUEST_SUMMARIES.append(summary)
    return summary


def run_pair(label, target, phase, prompt, prompt_count):
    pair_dir = ARTIFACT / "pairs" / ("depth%s-%s" % (label, phase))
    pair_dir.mkdir(parents=True, exist_ok=True)
    pair_start_wall = time.time()
    pair_start_mono = time.monotonic()
    gate = threading.Event()

    def worker(name):
        gate.wait()
        return stream_completion(name, label, phase, prompt, prompt_count)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {name: executor.submit(worker, name) for name in ("gpu1", "gpu2")}
        gate.set()
        results = {name: future.result() for name, future in futures.items()}
    pair_end_wall = time.time()
    pair_end_mono = time.monotonic()
    intervals = list(results.values())
    overlap_start = max(item["started_monotonic"] for item in intervals)
    overlap_end = min(item["ended_monotonic"] for item in intervals)
    overlap_seconds = max(0.0, overlap_end - overlap_start)
    individual = [item.get("timings", {}).get("predicted_n") or 0 for item in intervals]
    decode_intervals = []
    decode_seconds = []
    for item in intervals:
        predicted_ms = item.get("timings", {}).get("predicted_ms")
        if isinstance(predicted_ms, (int, float)):
            end = item["ended_monotonic"]
            start = end - predicted_ms / 1000.0
            decode_intervals.append((start, end))
            decode_seconds.append(predicted_ms / 1000.0)
    decode_window_seconds = 0.0
    if decode_intervals:
        decode_window_seconds = max(end for _, end in decode_intervals) - min(start for start, _ in decode_intervals)
    decode_work_seconds = sum(decode_seconds)
    pair = {
        "label": label,
        "target_tokens": target,
        "phase": phase,
        "pair_started_wall": pair_start_wall,
        "pair_ended_wall": pair_end_wall,
        "pair_wall_seconds": pair_end_wall - pair_start_wall,
        "request_overlap_seconds": overlap_seconds,
        "request_overlap_fraction_of_pair": overlap_seconds / max(0.000001, pair_end_wall - pair_start_wall),
        "aggregate_decode_tokens": sum(individual),
        "decode_window_seconds": decode_window_seconds or None,
        "decode_work_seconds": decode_work_seconds or None,
        "aggregate_decode_tok_s_decode_window": sum(individual) / decode_window_seconds if decode_window_seconds else None,
        "aggregate_decode_tok_s_work": sum(individual) / decode_work_seconds if decode_work_seconds else None,
        "aggregate_output_tok_s_pair_wall": sum(individual) / max(0.000001, pair_end_wall - pair_start_wall),
        "requests": results,
    }
    write_json(pair_dir / "pair-result.json", pair)
    PAIR_SUMMARIES.append(pair)
    return pair


def parse_gpu_rows(result):
    rows = []
    for line in result.get("stdout", "").splitlines():
        fields = next(csv.reader([line], skipinitialspace=True), [])
        if len(fields) < 12 or not fields[0].strip().isdigit():
            continue
        row: dict[str, object] = {"gpu": int(fields[0].strip())}
        keys = [
            "uuid",
            "name",
            "memory_used_mib",
            "memory_total_mib",
            "utilization_gpu_pct",
            "power_w",
            "temperature_c",
            "clock_sm_mhz",
            "clock_mem_mhz",
            "pcie_gen",
            "pcie_width",
        ]
        for key, value in zip(keys, fields[1:]):
            value = value.strip()
            try:
                row[key] = float(value)
            except ValueError:
                row[key] = value
        rows.append(row)
    return rows


def numeric_stats(rows, key):
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {"count": len(values), "min": min(values), "mean": sum(values) / len(values), "max": max(values)}


def samples():
    result = []
    path = ARTIFACT / "gpu-samples.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                item["gpu_rows"] = parse_gpu_rows(item.get("gpu_query", {}))
                result.append(item)
            except json.JSONDecodeError:
                pass
    return result


def energy_for_gpu(sample_rows, gpu_index, start, end, baseline_power=None):
    points = []
    for item in sample_rows:
        if not start <= item["monotonic"] <= end:
            continue
        for row in item.get("gpu_rows", []):
            if row.get("gpu") == gpu_index and isinstance(row.get("power_w"), (int, float)):
                power = row["power_w"]
                if baseline_power is not None:
                    power = max(0.0, power - baseline_power)
                points.append((item["monotonic"], power))
    points.sort()
    if len(points) < 2:
        return {"sample_points": len(points), "watt_hours": None, "method": "insufficient samples"}
    total_ws = sum((right_t - left_t) * (left_p + right_p) / 2.0 for (left_t, left_p), (right_t, right_p) in zip(points, points[1:]))
    return {"sample_points": len(points), "watt_hours": total_ws / 3600.0, "method": "trapezoid"}


def baseline_power(gpu_index):
    result = json.loads((ARTIFACT / "preflight-nvidia-smi.json").read_text(encoding="utf-8"))
    rows = parse_gpu_rows(result)
    for row in rows:
        if row.get("gpu") == gpu_index and isinstance(row.get("power_w"), (int, float)):
            return row["power_w"]
    return None


def build_measurements():
    sample_rows = samples()
    rows = []
    for pair in PAIR_SUMMARIES:
        requests = pair["requests"]
        for name, item in requests.items():
            timings = item.get("timings", {})
            gpu = item["physical_gpu"]
            in_window = [
                sample
                for sample in sample_rows
                if item["started_monotonic"] <= sample["monotonic"] <= item["ended_monotonic"]
            ]
            gpu_window = [row for sample in in_window for row in sample.get("gpu_rows", []) if row.get("gpu") == gpu]
            other = {}
            for other_gpu in (0, 1, 2):
                values = [row for sample in in_window for row in sample.get("gpu_rows", []) if row.get("gpu") == other_gpu]
                other[str(other_gpu)] = {
                    "max_memory_mib": numeric_stats(values, "memory_used_mib")["max"],
                    "max_utilization_pct": numeric_stats(values, "utilization_gpu_pct")["max"],
                    "pcie_link_gen": numeric_stats(values, "pcie_gen"),
                    "pcie_link_width": numeric_stats(values, "pcie_width"),
                }
            cpu_rows = []
            for sample in in_window:
                text = sample.get("processes", {}).get("stdout", "")
                for line in text.splitlines():
                    fields = line.split()
                    if len(fields) >= 8 and fields[0].isdigit():
                        try:
                            cpu_rows.append({"cpu_pct": float(fields[3]), "rss_kib": float(fields[5]), "vsz_kib": float(fields[6])})
                        except ValueError:
                            pass
            row = {
                "label": item["label"],
                "phase": item["phase"],
                "instance": name,
                "physical_gpu": gpu,
                "target_tokens": next(pair["target_tokens"] for pair in PAIR_SUMMARIES if pair is not None and pair["label"] == item["label"] and pair["phase"] == item["phase"]),
                "prompt_tokens": item["prompt_count"],
                "status": item["status"],
                "wall_seconds": item["wall_seconds"],
                "ttft_ms": item["ttft_ms"],
                "prefill_tokens": timings.get("prompt_n"),
                "prefill_ms": timings.get("prompt_ms"),
                "prefill_tok_s": timings.get("prompt_per_second"),
                "decode_tokens": timings.get("predicted_n"),
                "decode_ms": timings.get("predicted_ms"),
                "decode_tok_s": timings.get("predicted_per_second"),
                "cache_tokens": timings.get("cache_n"),
                "tokens_cached": item.get("tokens_cached"),
                "draft_tokens": timings.get("draft_n"),
                "draft_tokens_accepted": timings.get("draft_n_accepted"),
                "draft_acceptance": (timings.get("draft_n_accepted") / timings["draft_n"] if timings.get("draft_n") else None),
                "truncated": item.get("truncated"),
                "stop_type": item.get("stop_type"),
                "gpu_vram_mib": numeric_stats(gpu_window, "memory_used_mib"),
                "gpu_utilization_pct": numeric_stats(gpu_window, "utilization_gpu_pct"),
                "gpu_power_w": numeric_stats(gpu_window, "power_w"),
                "gpu_temperature_c": numeric_stats(gpu_window, "temperature_c"),
                "gpu_clock_sm_mhz": numeric_stats(gpu_window, "clock_sm_mhz"),
                "gpu_clock_mem_mhz": numeric_stats(gpu_window, "clock_mem_mhz"),
                "gpu_pcie_link_gen": numeric_stats(gpu_window, "pcie_gen"),
                "gpu_pcie_link_width": numeric_stats(gpu_window, "pcie_width"),
                "gpu_energy_wh": energy_for_gpu(sample_rows, gpu, item["started_monotonic"], item["ended_monotonic"]),
                "gpu_incremental_energy_wh": energy_for_gpu(sample_rows, gpu, item["started_monotonic"], item["ended_monotonic"], baseline_power(gpu)),
                "other_gpu_maxima": other,
                "server_cpu_pct": numeric_stats(cpu_rows, "cpu_pct"),
                "server_rss_kib": numeric_stats(cpu_rows, "rss_kib"),
                "sample_points": len(in_window),
            }
            rows.append(row)
    write_json(
        ARTIFACT / "measurements.json",
        {
            "requests": rows,
            "pairs": PAIR_SUMMARIES,
            "notes": [
                "TTFT is measured from request send to the first streamed response line.",
                "Energy is trapezoidal integration of nvidia-smi power samples; incremental energy subtracts preflight power.",
                "GPU indices are physical indices sampled with CUDA_VISIBLE_DEVICES removed.",
                "Aggregate decode throughput uses the elapsed window spanning per-request decode phases; pair-wall output rate is reported separately.",
                "This host exposes PCIe link generation and width but not tx/rx throughput fields, so direct PCIe contention is not observable.",
            ],
        },
    )
    fields = [
        "label", "phase", "instance", "physical_gpu", "target_tokens", "prompt_tokens", "status", "wall_seconds", "ttft_ms",
        "prefill_tokens", "prefill_ms", "prefill_tok_s", "decode_tokens", "decode_ms", "decode_tok_s", "cache_tokens", "tokens_cached",
        "draft_tokens", "draft_tokens_accepted", "draft_acceptance", "truncated", "stop_type", "gpu_vram_mib", "gpu_utilization_pct",
        "gpu_power_w", "gpu_temperature_c", "gpu_clock_sm_mhz", "gpu_clock_mem_mhz", "gpu_pcie_link_gen", "gpu_pcie_link_width",
        "gpu_energy_wh", "gpu_incremental_energy_wh", "other_gpu_maxima", "server_cpu_pct", "server_rss_kib", "sample_points",
    ]
    with (ARTIFACT / "measurements.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def cleanup():
    stop_records = {}
    for name, server in SERVERS.items():
        process = server.get("process")
        record = {"pid": server.get("pid"), "signal": "SIGTERM", "started_at_utc": utc_now()}
        if process and process.poll() is None:
            try:
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=45)
                record["returncode"] = process.returncode
            except subprocess.TimeoutExpired:
                record["signal"] = "SIGKILL"
                process.kill()
                process.wait(timeout=15)
                record["returncode"] = process.returncode
        elif process:
            record["returncode"] = process.returncode
        if server.get("log_handle"):
            server["log_handle"].close()
        stop_records[name] = record
        write_json(ARTIFACT / name / "stop-record.json", record)
    write_json(ARTIFACT / "stop-record.json", stop_records)
    verification = {
        "captured_at_utc": utc_now(),
        "ports": {str(port): port_snapshot() for port in PORTS.values()},
        "processes": process_snapshot(),
        "compute_apps": {str(gpu): compute_query(gpu) for gpu in (0, 1, 2)},
        "gpu_snapshot": gpu_query(),
        "server_pid_absent": {name: not bool(command_result(["ps", "-p", str(server.get("pid")), "-o", "pid="]).get("stdout", "").strip()) for name, server in SERVERS.items()},
        "port_absent": {str(port): ("127.0.0.1:%d" % port) not in port_snapshot().get("stdout", "") for port in PORTS.values()},
        "systemd_used": False,
        "production_interaction": False,
    }
    write_json(ARTIFACT / "cleanup-verification.json", verification)
    (ARTIFACT / "cleanup-verification.txt").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return verification


def gpu0_proof():
    sample_rows = samples()
    gpu0 = [row for sample in sample_rows for row in sample.get("gpu_rows", []) if row.get("gpu") == 0]
    compute = []
    for sample in sample_rows:
        compute.append(sample.get("compute_apps", {}).get("0", {}))
    proof = {
        "sample_count": len(sample_rows),
        "gpu0_memory": numeric_stats(gpu0, "memory_used_mib"),
        "gpu0_utilization": numeric_stats(gpu0, "utilization_gpu_pct"),
        "gpu0_compute_app_samples": compute,
        "gpu0_max_memory_mib": numeric_stats(gpu0, "memory_used_mib")["max"],
        "gpu0_max_utilization_pct": numeric_stats(gpu0, "utilization_gpu_pct")["max"],
        "gpu0_server_compute_app_absent_during_run": all(
            "llama" not in item.get("stdout", "").lower() for item in compute
        ),
    }
    write_json(ARTIFACT / "gpu0-proof.json", proof)
    return proof


def single_baseline(label, phase):
    path = ROOT / ("research/qwen3.8-27b-serving-alternatives-phase1-20260918T133922Z/%s-%s-summary.json" % ("depth" + label, phase))
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    timing = value.get("timings", {})
    return {
        "prefill_tok_s": timing.get("prompt_per_second"),
        "decode_tok_s": timing.get("predicted_per_second"),
        "wall_seconds": value.get("wall_seconds"),
        "source": str(path.relative_to(ROOT)),
    }


def write_gate(rows):
    comparisons = []
    for row in rows:
        baseline = single_baseline(row["label"], row["phase"])
        if row["phase"] != "cold" or not baseline:
            continue
        prefill_ratio = row["prefill_tok_s"] / baseline["prefill_tok_s"] if row.get("prefill_tok_s") and baseline.get("prefill_tok_s") else None
        decode_ratio = row["decode_tok_s"] / baseline["decode_tok_s"] if row.get("decode_tok_s") and baseline.get("decode_tok_s") else None
        wall_ratio = row["wall_seconds"] / baseline["wall_seconds"] if row.get("wall_seconds") and baseline.get("wall_seconds") else None
        comparisons.append(
            {
                "instance": row["instance"],
                "label": row["label"],
                "baseline": baseline,
                "dual": {"prefill_tok_s": row.get("prefill_tok_s"), "decode_tok_s": row.get("decode_tok_s"), "wall_seconds": row.get("wall_seconds")},
                "ratios": {"prefill": prefill_ratio, "decode": decode_ratio, "wall": wall_ratio},
                "close_at_15pct_threshold": bool(
                    prefill_ratio is not None and decode_ratio is not None and wall_ratio is not None
                    and prefill_ratio >= 0.85 and decode_ratio >= 0.85 and wall_ratio <= 1.25
                ),
            }
        )
    complete_64 = [item for item in comparisons if item["label"] == "064k"]
    complete_196 = [item for item in comparisons if item["label"] == "196k"]
    gate = bool(complete_64 and all(item["close_at_15pct_threshold"] for item in complete_64))
    if complete_196:
        gate = gate and all(item["close_at_15pct_threshold"] for item in complete_196)
    result = {
        "gate": "close_to_single_instance_performance" if gate else "not_close_to_single_instance_performance",
        "threshold": "each completed cold per-instance prefill/decode >= 85% of Phase 1 single-instance and wall <= 125%",
        "completed_cold_64k_instances": len(complete_64),
        "completed_cold_196k_instances": len(complete_196),
        "comparisons": comparisons,
        "partial_results_preserved": True,
    }
    write_json(ARTIFACT / "gate-result.json", result)
    lines = [
        "# llamAmpere Dual-Instance Topology Test",
        "",
        "Artifact: `%s`" % ARTIFACT.relative_to(ROOT),
        "",
        "Topology: GPU1 -> 127.0.0.1:18111; GPU2 -> 127.0.0.1:18112; GPU0 forbidden.",
        "",
        "Settings: `ctx=245760`, documented MTP flags, `parallel=1` per independent server, `n_predict=128`, `temperature=0`, thinking disabled.",
        "",
        "Gate: **%s**" % result["gate"],
        "",
        "Threshold: %s." % result["threshold"],
        "",
        "| Depth | Instance | Prefill ratio | Decode ratio | Wall ratio | Close |",
        "|---|---|---:|---:|---:|---|",
    ]
    for item in comparisons:
        ratios = item["ratios"]
        lines.append(
            "| %s | %s | %.3f | %.3f | %.3f | %s |"
            % (item["label"], item["instance"], ratios["prefill"], ratios["decode"], ratios["wall"], item["close_at_15pct_threshold"])
        )
    lines.extend(
        [
            "",
            "Completed cold 64k instances: `%d`; completed cold 196k instances: `%d`." % (len(complete_64), len(complete_196)),
            "",
            "Pair metrics distinguish decode-window throughput from end-to-end pair output throughput.",
            "",
            "Raw logs, streamed responses, prompts, GPU samples, exact commands, GPU0 proof, and cleanup verification are stored under this artifact.",
        ]
    )
    (ARTIFACT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def artifact_hashes():
    paths = sorted(path for path in ARTIFACT.rglob("*") if path.is_file() and path.name != "artifact-sha256.txt")
    (ARTIFACT / "artifact-file-list.txt").write_text("\n".join(str(path.relative_to(ARTIFACT)) for path in paths) + "\n", encoding="utf-8")
    lines = ["%s  %s" % (sha256_file(path), path.relative_to(ARTIFACT)) for path in paths]
    (ARTIFACT / "artifact-sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    write_command_environment()
    capture_preflight()
    runtime_metadata()
    sampler_thread = None
    rows = []
    verification = None
    try:
        start_servers()
        write_json(ARTIFACT / "server-pids.json", {name: server["pid"] for name, server in SERVERS.items()})
        for name in SERVERS:
            wait_ready(name)
        process_environment_proof()
        write_json(ARTIFACT / "post-start-gpu-processes.json", {str(gpu): compute_query(gpu) for gpu in (0, 1, 2)})
        sampler_thread = threading.Thread(target=sampler, name="gpu-sampler", daemon=True)
        sampler_thread.start()
        for label, target in TARGETS:
            prompt, count = make_prompt("dual-%s" % label, target)
            for phase in ("cold", "warm"):
                run_pair(label, target, phase, prompt, count)
    except Exception as exc:
        (ARTIFACT / "run-error.txt").write_text(repr(exc) + "\n", encoding="utf-8")
        raise
    finally:
        STOP_EVENT.set()
        if sampler_thread:
            sampler_thread.join(timeout=10)
        verification = cleanup()
    rows = build_measurements()
    proof = gpu0_proof()
    gate = write_gate(rows)
    write_json(ARTIFACT / "run-final.json", {"completed_at_utc": utc_now(), "gate": gate, "gpu0_proof": proof, "cleanup": verification})
    artifact_hashes()
    print(json.dumps({"artifact": str(ARTIFACT), "gate": gate, "cleanup": verification}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
