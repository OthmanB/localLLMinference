#!/usr/bin/env python3
import csv
import hashlib
import json
import os
import shlex
import signal
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path((os.environ['EXPERIMENT_ROOT'] + ''))
ARTIFACT = Path(sys.argv[1]).resolve()
SERVER = ROOT / "research/llamAmpere-v0.3-36a6bca81/build-sm86/bin/llama-server"
MODEL = ROOT / "research/qwen3.8-27b-atx-iq4_xs-m-gguf/Qwen3.8-27B-ATX-4-XS.gguf"
VOCAB_MAP = ROOT / "research/llamAmpere-v0.3-36a6bca81/docs/mtp-vocab/atx_65536.txt"
SOURCE_ARTIFACT = ROOT / "research/qwen3.8-27b-serving-alternatives-phase1-20260918T133922Z"
PORT = 18101
SAMPLER_INTERVAL = 1.0
REQUEST_TIMEOUT = 1800
SEED = 1729
TARGETS = [("196k", 196000), ("240k", 240000)]
CELLS = [
    (mtp, batch, depth, target)
    for mtp in (2, 3)
    for batch in (4096, 8192)
    for depth, target in TARGETS
]

STOP_SAMPLER = threading.Event()
ACTIVE_PID = None
ACTIVE_PID_LOCK = threading.Lock()
CURRENT_SERVER = None


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


def unmasked_env():
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


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


def gpu_query_command():
    return [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,uuid,pci.bus_id,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem",
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


def process_snapshot():
    return command_result(["ps", "-eo", "pid,ppid,user,stat,cmd"], env=unmasked_env())


def write_command_environment():
    helper = ARTIFACT / "phase1_workload_helper.py"
    analyzer = ARTIFACT / "analyze_phase1.py"
    shutil.copy2(SOURCE_ARTIFACT / "phase1_workload_helper.py", helper)
    shutil.copy2(SOURCE_ARTIFACT / "analyze_phase1.py", analyzer)
    helper.chmod(0o755)
    analyzer.chmod(0o755)
    lines = [
        "artifact=%s" % ARTIFACT,
        "source_helper=%s" % (SOURCE_ARTIFACT / "phase1_workload_helper.py"),
        "source_analyzer=%s" % (SOURCE_ARTIFACT / "analyze_phase1.py"),
        "model=%s" % MODEL,
        "model_sha256=%s" % sha256_file(MODEL),
        "binary=%s" % SERVER,
        "binary_sha256=%s" % sha256_file(SERVER),
        "vocab_map=%s" % VOCAB_MAP,
        "vocab_map_sha256=%s" % sha256_file(VOCAB_MAP),
        "physical_gpu=1",
        "server_visible_device=CUDA0 (mapped from physical GPU1)",
        "port=%d" % PORT,
        "systemd_used=False",
        "production_ports_used=False",
        "production_services_touched=False",
        "",
        "cell command template:",
        "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 GGML_Q8_TURBO3_MMA_FUSED=1",
        "<server> -m <model> -c 245760 -b <batch> -ub 1024 -t 8 -tb 8 -ngl 99 -fa on -ctk q8_0 -ctv turbo3 --parallel 1 --jinja --fit off --cache-prompt --cache-ram 8192 --ctx-checkpoints 24 --checkpoint-min-step 10240 --spec-type draft-mtp --spec-draft-n-max <mtp> --spec-draft-p-min 0 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 --spec-draft-vocab-map <vocab_map> --device CUDA0 --host 127.0.0.1 --port 18101 --log-file <cell server log>",
        "",
    ]
    for mtp, batch, depth, target in CELLS:
        label = "depth%s-mtp%d-b%d" % (depth, mtp, batch)
        command = server_command(mtp, batch, label)
        env = server_env()
        lines.extend(
            [
                "[%s] target_tokens=%d" % (label, target),
                "CUDA_DEVICE_ORDER=%s" % env["CUDA_DEVICE_ORDER"],
                "CUDA_VISIBLE_DEVICES=%s" % env["CUDA_VISIBLE_DEVICES"],
                "GGML_Q8_TURBO3_MMA_FUSED=%s" % env["GGML_Q8_TURBO3_MMA_FUSED"],
                "command:",
                shlex.join(command),
                "helper:",
                shlex.join([sys.executable, str(helper), str(ARTIFACT), label, str(target), str(SEED)]),
                "",
            ]
        )
    (ARTIFACT / "exact-command-environment.txt").write_text("\n".join(lines), encoding="utf-8")


def server_env():
    env = os.environ.copy()
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "1",
            "GGML_Q8_TURBO3_MMA_FUSED": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def server_command(mtp, batch, label):
    return [
        str(SERVER),
        "-m",
        str(MODEL),
        "-c",
        "245760",
        "-b",
        str(batch),
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
        str(mtp),
        "--spec-draft-p-min",
        "0",
        "--spec-draft-type-k",
        "q8_0",
        "--spec-draft-type-v",
        "q8_0",
        "--spec-draft-vocab-map",
        str(VOCAB_MAP),
        "--device",
        "CUDA0",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--log-file",
        str(ARTIFACT / ("server-%s.log" % label)),
    ]


def capture_preflight():
    (ARTIFACT / "preflight-gpu.txt").write_text(
        json.dumps(gpu_query(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (ARTIFACT / "preflight-gpu.csv").write_text(gpu_query().get("stdout", ""), encoding="utf-8")
    for gpu in (0, 1, 2):
        write_json(ARTIFACT / ("preflight-compute-gpu%d.json" % gpu), compute_query(gpu))
    write_json(ARTIFACT / "preflight-ports.json", port_snapshot())
    write_json(ARTIFACT / "preflight-processes.json", process_snapshot())
    write_json(
        ARTIFACT / "preflight.json",
        {
            "captured_at_utc": utc_now(),
            "physical_gpu": 1,
            "forbidden_physical_gpus": [0, 2],
            "cuda_device_order": "PCI_BUS_ID",
            "cuda_visible_devices": "1",
            "server_visible_cuda_device": "CUDA0",
            "port": PORT,
            "systemd_used": False,
            "production_ports_used": False,
            "production_services_touched": False,
            "cells_requested": len(CELLS),
        },
    )


def write_runtime_metadata():
    runtime_repo = SERVER.parents[2]
    git_head = command_result(["git", "-C", str(runtime_repo), "log", "-1", "--format=%H %cI %s"])
    git_status = command_result(["git", "-C", str(runtime_repo), "status", "--short"])
    write_json(
        ARTIFACT / "runtime-metadata.json",
        {
            "captured_at_utc": utc_now(),
            "binary": {"path": str(SERVER), "sha256": sha256_file(SERVER)},
            "model": {"path": str(MODEL), "size_bytes": MODEL.stat().st_size, "sha256": sha256_file(MODEL)},
            "vocab_map": {"path": str(VOCAB_MAP), "sha256": sha256_file(VOCAB_MAP)},
            "version": command_result([str(SERVER), "--version"]),
            "ldd": command_result(["ldd", str(SERVER)]),
            "git": {"repo": str(runtime_repo), "head": git_head, "status": git_status},
        },
    )


def write_sample(path, text):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        if text and not text.endswith("\n"):
            handle.write("\n")
        handle.flush()


def sampler():
    gpu_path = ARTIFACT / "gpu-samples.csv"
    host_path = ARTIFACT / "host-samples.log"
    while not STOP_SAMPLER.is_set():
        epoch = time.time()
        result = gpu_query()
        write_sample(gpu_path, "sample_epoch=%.9f\n%s" % (epoch, result.get("stdout", "")))
        with ACTIVE_PID_LOCK:
            pid = ACTIVE_PID
        if pid:
            ps = command_result(
                ["ps", "-p", str(pid), "-o", "pid=,ppid=,uid=,stat=,pcpu=,pmem=,rss=,vsz="],
                env=unmasked_env(),
            )
            mem = command_result(["free", "-b"], env=unmasked_env())
            write_sample(host_path, "sample_epoch=%.9f\n%s%s" % (epoch, ps.get("stdout", ""), mem.get("stdout", "")))
        STOP_SAMPLER.wait(SAMPLER_INTERVAL)


def write_baseline(path):
    result = gpu_query()
    path.write_text(result.get("stdout", ""), encoding="utf-8")


def wait_ready(process, label):
    poll_path = ARTIFACT / ("health-%s.log" % label)
    deadline = time.monotonic() + 300
    with poll_path.open("w", encoding="utf-8") as handle:
        while time.monotonic() < deadline:
            stamp = time.time()
            try:
                with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=10) as response:
                    body = response.read().decode("utf-8", "replace")
                    line = {"epoch": stamp, "status": response.status, "body": body}
                    ready = response.status == 200
            except Exception as exc:
                line = {"epoch": stamp, "status": None, "error": repr(exc)}
                ready = False
            handle.write(json.dumps(line, sort_keys=True) + "\n")
            handle.flush()
            if ready:
                write_json(ARTIFACT / ("health-%s-ready.json" % label), line)
                return line
            if process.poll() is not None:
                raise RuntimeError("server exited during startup with %s" % process.returncode)
            time.sleep(1)
    raise TimeoutError("timed out waiting for health for %s" % label)


def process_proof(pid, label):
    values = {}
    try:
        raw = Path("/proc/%d/environ" % pid).read_bytes()
        for part in raw.split(b"\0"):
            if b"=" in part:
                key, value = part.split(b"=", 1)
                key = key.decode("utf-8", "replace")
                if key in {"CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "GGML_Q8_TURBO3_MMA_FUSED", "PYTHONUNBUFFERED"}:
                    values[key] = value.decode("utf-8", "replace")
        cmdline = Path("/proc/%d/cmdline" % pid).read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except Exception as exc:
        values = {"error": repr(exc)}
        cmdline = ""
    write_json(ARTIFACT / ("process-environment-%s.json" % label), {"pid": pid, "environment": values, "cmdline": cmdline})


def launch_cell(mtp, batch, depth, target):
    label = "depth%s-mtp%d-b%d" % (depth, mtp, batch)
    cell_dir = ARTIFACT / "cells" / label
    cell_dir.mkdir(parents=True, exist_ok=True)
    command = server_command(mtp, batch, label)
    env = server_env()
    write_json(
        cell_dir / "cell-start.json",
        {
            "started_at_utc": utc_now(),
            "label": label,
            "target_token_depth": target,
            "mtp": mtp,
            "batch": batch,
            "ubatch": 1024,
            "ctx": 245760,
            "physical_gpu": 1,
            "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
            "command": command,
            "command_shell": shlex.join(command),
            "environment": {key: env[key] for key in ("CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "GGML_Q8_TURBO3_MMA_FUSED", "PYTHONUNBUFFERED")},
        },
    )
    output = (ARTIFACT / ("server-%s.stdout-stderr.log" % label)).open("wb")
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    output.flush()
    with ACTIVE_PID_LOCK:
        global ACTIVE_PID, CURRENT_SERVER
        ACTIVE_PID = process.pid
        CURRENT_SERVER = process
    (cell_dir / "server.pid").write_text(str(process.pid) + "\n", encoding="utf-8")
    process_proof(process.pid, label)
    return label, cell_dir, process, output


def stop_cell(label, cell_dir, process, output):
    record = {"label": label, "pid": process.pid, "signal": "SIGTERM", "stopped_at_utc": utc_now()}
    if process.poll() is None:
        try:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            record["signal"] = "SIGKILL"
            process.kill()
            process.wait(timeout=20)
    record["returncode"] = process.returncode
    output.close()
    with ACTIVE_PID_LOCK:
        global ACTIVE_PID, CURRENT_SERVER
        ACTIVE_PID = None
        CURRENT_SERVER = None
    write_json(cell_dir / "stop-record.json", record)
    write_json(cell_dir / "post-stop-ports.json", port_snapshot())
    write_json(cell_dir / "post-stop-processes.json", process_snapshot())
    write_json(cell_dir / "post-stop-compute-gpu1.json", compute_query(1))
    (cell_dir / "post-stop-gpu.csv").write_text(gpu_query().get("stdout", ""), encoding="utf-8")
    return record


def run_helper(label, target, cell_dir):
    command = [sys.executable, str(ARTIFACT / "phase1_workload_helper.py"), str(ARTIFACT), label, str(target), str(SEED)]
    write_json(cell_dir / "request-metadata.json", {"command": command, "command_shell": shlex.join(command), "label": label, "target_token_depth": target, "seed": SEED, "n_predict": 128, "temperature": 0.0, "stream": False, "cache_prompt": True})
    started = time.time()
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=REQUEST_TIMEOUT, check=False)
        error = None
    except Exception as exc:
        result = None
        error = repr(exc)
    stdout = result.stdout if result else ""
    stderr = result.stderr if result else ""
    (cell_dir / "helper.stdout.log").write_text(stdout, encoding="utf-8")
    (cell_dir / "helper.stderr.log").write_text(stderr, encoding="utf-8")
    record = {"label": label, "target_token_depth": target, "started_at_epoch": started, "ended_at_epoch": time.time(), "returncode": result.returncode if result else None, "error": error, "stdout": stdout, "stderr": stderr}
    write_json(cell_dir / "helper-result.json", record)
    write_json(cell_dir / "request-metadata-final.json", {**json.loads((cell_dir / "request-metadata.json").read_text()), "helper_returncode": record["returncode"], "helper_error": error, "summary_file": str(ARTIFACT / (label + "-summary.json"))})
    return record


def run_cell(mtp, batch, depth, target):
    label = "depth%s-mtp%d-b%d" % (depth, mtp, batch)
    cell_dir = ARTIFACT / "cells" / label
    cell = {"label": label, "mtp": mtp, "batch": batch, "depth": depth, "target_token_depth": target, "status": "starting", "started_at_utc": utc_now()}
    process = None
    output = None
    try:
        label, cell_dir, process, output = launch_cell(mtp, batch, depth, target)
        wait_ready(process, label)
        helper_result = run_helper(label, target, cell_dir)
        cell["helper_returncode"] = helper_result["returncode"]
        cell["status"] = "completed" if helper_result["returncode"] == 0 else "failed_request"
    except Exception as exc:
        cell["status"] = "failed_start_or_execution"
        cell["error"] = repr(exc)
        (cell_dir / "cell-error.txt").write_text(repr(exc) + "\n", encoding="utf-8")
    finally:
        if process is not None and output is not None:
            cell["stop"] = stop_cell(label, cell_dir, process, output)
        cell["ended_at_utc"] = utc_now()
        write_json(cell_dir / "cell-result.json", cell)
    return cell


def cleanup_verification():
    verification = {
        "captured_at_utc": utc_now(),
        "port": PORT,
        "port_absent": ("127.0.0.1:%d" % PORT) not in port_snapshot().get("stdout", ""),
        "llama_server_lines": [line for line in process_snapshot().get("stdout", "").splitlines() if "llama-server" in line],
        "compute_apps": {str(gpu): compute_query(gpu) for gpu in (0, 1, 2)},
        "gpu_snapshot": gpu_query(),
        "systemd_used": False,
        "production_interaction": False,
    }
    verification["candidate_server_absent"] = not verification["llama_server_lines"]
    verification["gpu1_compute_apps_absent"] = not bool(verification["compute_apps"]["1"].get("stdout", "").strip())
    write_json(ARTIFACT / "cleanup-verification.json", verification)
    (ARTIFACT / "cleanup-verification.txt").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ARTIFACT / "gpu-after-stop.csv").write_text(verification["gpu_snapshot"].get("stdout", ""), encoding="utf-8")
    return verification


def analyze():
    result = subprocess.run([sys.executable, str(ARTIFACT / "analyze_phase1.py"), str(ARTIFACT)], cwd=ROOT, capture_output=True, text=True, check=False)
    (ARTIFACT / "analyze-phase1.stdout.log").write_text(result.stdout, encoding="utf-8")
    (ARTIFACT / "analyze-phase1.stderr.log").write_text(result.stderr, encoding="utf-8")
    write_json(ARTIFACT / "analyze-result.json", {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    return result.returncode


def write_manifest(results, cleanup):
    write_json(
        ARTIFACT / "run-final.json",
        {
            "completed_at_utc": utc_now(),
            "artifact": str(ARTIFACT),
            "requested_cells": len(CELLS),
            "results": results,
            "cleanup": cleanup,
            "analysis_returncode": analyze(),
            "subset_policy": "8-cell controlled subset: MTP2 vs MTP3 at batch 4096 and 8192, at 196k and 240k; batch 6144 omitted by bounded-cost policy.",
            "warm_policy": "Each completed cell runs the existing phase1 helper's cold request followed by its warm/cache request on the same server instance.",
        },
    )


def artifact_hashes():
    files = sorted(path for path in ARTIFACT.rglob("*") if path.is_file() and path.name != "artifact-sha256.txt")
    (ARTIFACT / "artifact-file-list.txt").write_text("\n".join(str(path.relative_to(ARTIFACT)) for path in files) + "\n", encoding="utf-8")
    (ARTIFACT / "artifact-sha256.txt").write_text("\n".join("%s  %s" % (sha256_file(path), path.relative_to(ARTIFACT)) for path in files) + "\n", encoding="utf-8")


def main():
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    write_command_environment()
    capture_preflight()
    write_runtime_metadata()
    write_baseline(ARTIFACT / "gpu-before.csv")
    sampler_thread = threading.Thread(target=sampler, name="gpu-sampler", daemon=True)
    sampler_thread.start()
    results = []
    cleanup = None
    try:
        for mtp, batch, depth, target in CELLS:
            result = run_cell(mtp, batch, depth, target)
            results.append(result)
            write_json(ARTIFACT / "progress.json", {"completed_cells": results, "next_cell_index": len(results), "total_cells": len(CELLS)})
    finally:
        if CURRENT_SERVER is not None and CURRENT_SERVER.poll() is None:
            CURRENT_SERVER.send_signal(signal.SIGTERM)
        STOP_SAMPLER.set()
        sampler_thread.join(timeout=15)
        cleanup = cleanup_verification()
    write_manifest(results, cleanup)
    artifact_hashes()
    print(json.dumps({"artifact": str(ARTIFACT), "results": results, "cleanup": cleanup}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
