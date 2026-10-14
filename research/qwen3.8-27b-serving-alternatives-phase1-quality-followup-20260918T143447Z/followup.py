#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path((os.environ['EXPERIMENT_ROOT'] + ''))
ARTIFACT = Path(__file__).resolve().parent
SOURCE_ARTIFACT = ROOT / "research/qwen3.8-27b-serving-alternatives-phase1-quality-20260918T141239Z"
SOURCE_MANIFEST = SOURCE_ARTIFACT / "manifest.json"
QUALITY_GATE = SOURCE_ARTIFACT / "quality_gate.py"
TASK_IDS = ["T4-planning-implementation", "T8-code-generation-test"]
REQUESTED_PORT = 18102
MAX_TOKENS = 1024


def load_quality_gate():
    spec = importlib.util.spec_from_file_location("prior_quality_gate", QUALITY_GATE)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load the copied quality rubric")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def port_is_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def choose_port():
    if port_is_free(REQUESTED_PORT):
        return REQUESTED_PORT, "candidate-port"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1], "fallback-free-isolated-port"


def gpu1_snapshot(q):
    return q.run_capture(
        [
            "nvidia-smi",
            "-i",
            "1",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
        ],
        env=q.unmasked_env(),
    )


def gpu1_process_snapshot(q):
    result = q.run_capture(
        [
            "nvidia-smi",
            "-i",
            "1",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        env=q.unmasked_env(),
    )
    return "GPU 1\n%s%s" % (result["stdout"], result["stderr"])


def write_followup_manifest(source_manifest, port):
    tasks = [task for task in source_manifest["tasks"] if task["id"] in TASK_IDS]
    if [task["id"] for task in tasks] != TASK_IDS:
        raise RuntimeError("source manifest did not contain the requested tasks in order")
    sampling = dict(source_manifest["common_sampling"])
    sampling["n_predict"] = MAX_TOKENS
    manifest = {
        "manifest_version": source_manifest["manifest_version"],
        "created_at_utc": utc_now(),
        "purpose": "Focused llamAmpere quality follow-up for reviewer-marked tasks.",
        "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": hashlib.sha256(SOURCE_MANIFEST.read_bytes()).hexdigest(),
        "gpu_policy": {
            "atx_physical_gpus": [1],
            "forbidden_physical_gpus": [0],
            "one_server_at_a_time": True,
            "production_ports": "not used",
            "systemd": "not used",
        },
        "server": {
            "candidate_port": REQUESTED_PORT,
            "selected_port": port,
            "host": "127.0.0.1",
        },
        "common_sampling": sampling,
        "tasks": tasks,
    }
    write_json(ARTIFACT / "manifest.json", manifest)
    (ARTIFACT / "manifest.sha256").write_text(
        hashlib.sha256((ARTIFACT / "manifest.json").read_bytes()).hexdigest() + "  manifest.json\n",
        encoding="utf-8",
    )
    write_json(
        ARTIFACT / "prompt-identity.json",
        {
            "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
            "tasks": {
                task["id"]: {
                    "source_prompt_sha256": hashlib.sha256(task["prompt"].encode()).hexdigest(),
                    "followup_prompt_sha256": hashlib.sha256(task["prompt"].encode()).hexdigest(),
                    "identical": True,
                }
                for task in tasks
            },
        },
    )


def configure_quality_gate(q, port):
    q.ARTIFACT = ARTIFACT
    q.MANIFEST = ARTIFACT / "manifest.json"
    q.MAX_OUTPUT = MAX_TOKENS
    q.gpu_snapshot = lambda: gpu1_snapshot(q)
    q.gpu_process_snapshot = lambda: gpu1_process_snapshot(q)
    original_server_profile = q.server_profile

    def server_profile(variant):
        profile = original_server_profile(variant)
        if variant != "atx":
            raise RuntimeError("follow-up permits only the llamAmpere ATX variant")
        command = list(profile["command"])
        port_index = command.index("--port") + 1
        command[port_index] = str(port)
        profile["command"] = command
        profile["port"] = port
        profile["physical_gpus"] = [1]
        return profile

    q.server_profile = server_profile


def ps_for_pid(pid):
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid=,stat=,cmd="],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def write_cleanup_verification(q, port):
    stop_record = json.loads((ARTIFACT / "atx/stop-record.json").read_text(encoding="utf-8"))
    pid = stop_record.get("pid")
    port_listing = q.port_snapshot()
    process_listing = q.process_snapshot()
    gpu_listing = gpu1_process_snapshot(q)
    cleanup = {
        "captured_at_utc": utc_now(),
        "pid": pid,
        "pid_process_listing": ps_for_pid(pid) if pid else "",
        "server_pid_absent": not bool(ps_for_pid(pid)) if pid else False,
        "port": port,
        "port_absent_from_ss": ("127.0.0.1:%d" % port) not in port_listing,
        "gpu1_compute_apps_after_stop": gpu_listing,
        "gpu1_compute_apps_absent": not any(
            line.strip() and "No running processes found" not in line
            for line in gpu_listing.splitlines()[1:]
        ),
        "llama_server_lines_after_stop": [
            line for line in process_listing.splitlines() if "llama-server" in line
        ],
        "systemd_used": False,
        "production_interaction": False,
    }
    write_json(ARTIFACT / "cleanup-verification.json", cleanup)
    (ARTIFACT / "cleanup-verification.txt").write_text(
        "server_pid_absent=%s\nport_absent_from_ss=%s\ngpu1_compute_apps_absent=%s\n"
        "llama_server_lines_after_stop=%s\n"
        % (
            cleanup["server_pid_absent"],
            cleanup["port_absent_from_ss"],
            cleanup["gpu1_compute_apps_absent"],
            json.dumps(cleanup["llama_server_lines_after_stop"], ensure_ascii=True),
        ),
        encoding="utf-8",
    )
    return cleanup


def write_result_summary(source_manifest):
    results = json.loads((ARTIFACT / "atx/task-results.json").read_text(encoding="utf-8"))
    prior_atx = {
        item["task_id"]: item
        for item in json.loads(
            (SOURCE_ARTIFACT / "atx/task-results.json").read_text(encoding="utf-8")
        )
    }
    prior_reference = {
        item["task_id"]: item
        for item in json.loads(
            (SOURCE_ARTIFACT / "reference/task-results.json").read_text(encoding="utf-8")
        )
    }
    rows = []
    for task_id in TASK_IDS:
        result = next(item for item in results if item["task_id"] == task_id)
        followup_pass = (
            result["http_status"] == 200
            and result["rubric"]["automated_status"] == "pass"
            and result["truncated"] is False
        )
        rows.append(
            {
                "task_id": task_id,
                "followup": {
                    "automated_status": result["rubric"]["automated_status"],
                    "checks": result["rubric"]["checks"],
                    "finish_reason": result["finish_reason"],
                    "truncated": result["truncated"],
                    "http_status": result["http_status"],
                    "pass": followup_pass,
                },
                "prior_atx": {
                    "automated_status": prior_atx[task_id]["rubric"]["automated_status"],
                    "finish_reason": prior_atx[task_id]["finish_reason"],
                    "truncated": prior_atx[task_id]["truncated"],
                },
                "prior_reference": {
                    "automated_status": prior_reference[task_id]["rubric"]["automated_status"],
                    "finish_reason": prior_reference[task_id]["finish_reason"],
                    "truncated": prior_reference[task_id]["truncated"],
                },
            }
        )
    both_pass = all(row["followup"]["pass"] for row in rows)
    summary = {
        "created_at_utc": utc_now(),
        "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "sampling": source_manifest["common_sampling"],
        "max_tokens": MAX_TOKENS,
        "thinking_disabled": True,
        "normal_eos": True,
        "physical_gpus": [1],
        "tasks": rows,
        "both_tasks_pass": both_pass,
        "classification": "both_pass_no_regression" if both_pass else "followup_not_pass",
        "rubric_source": str(QUALITY_GATE.relative_to(ROOT)),
    }
    write_json(ARTIFACT / "followup-summary.json", summary)
    lines = [
        "# llamAmpere Quality Follow-up",
        "",
        "Source manifest: `%s`" % SOURCE_MANIFEST.relative_to(ROOT),
        "",
        "Settings: `temperature=0`, `seed=1729`, thinking disabled, normal EOS, `max_tokens=1024`, physical GPU 1.",
        "",
        "| Task | Prior ATX | Prior reference | Follow-up | Finish | Objective result |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        followup = row["followup"]
        lines.append(
            "| %s | %s | %s | %s | %s / truncated=%s | %s |"
            % (
                row["task_id"],
                row["prior_atx"]["automated_status"],
                row["prior_reference"]["automated_status"],
                followup["automated_status"],
                followup["finish_reason"],
                followup["truncated"],
                "PASS" if followup["pass"] else "NOT PASS",
            )
        )
    lines.extend(
        [
            "",
            "Conclusion: `%s`" % summary["classification"],
            "",
            "The objective rubric is the copied prior `quality_gate.py`; full request, response, server, GPU, and cleanup artifacts are in this directory.",
        ]
    )
    (ARTIFACT / "followup-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def write_artifact_hashes():
    files = sorted(path for path in ARTIFACT.rglob("*") if path.is_file())
    relative_names = [str(path.relative_to(ARTIFACT)) for path in files]
    (ARTIFACT / "artifact-file-list.txt").write_text("\n".join(relative_names) + "\n", encoding="utf-8")
    lines = []
    for path in files:
        if path.name == "artifact-sha256.txt":
            continue
        lines.append(
            "%s  %s" % (
                hashlib.sha256(path.read_bytes()).hexdigest(),
                path.relative_to(ARTIFACT),
            )
        )
    (ARTIFACT / "artifact-sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    port, port_selection = choose_port()
    write_json(
        ARTIFACT / "preflight.json",
        {
            "captured_at_utc": utc_now(),
            "port_selection": port_selection,
            "candidate_port": REQUESTED_PORT,
            "selected_port": port,
            "physical_gpu": 1,
            "forbidden_physical_gpu": 0,
            "systemd_used": False,
            "production_interaction": False,
        },
    )
    write_followup_manifest(source_manifest, port)
    q = load_quality_gate()
    configure_quality_gate(q, port)
    try:
        q.run_variant("atx")
    except Exception as exc:
        (ARTIFACT / "followup-run-error.txt").write_text(repr(exc) + "\n", encoding="utf-8")
        raise
    cleanup = write_cleanup_verification(q, port)
    summary = write_result_summary(json.loads((ARTIFACT / "manifest.json").read_text(encoding="utf-8")))
    write_artifact_hashes()
    print(json.dumps({"artifact": str(ARTIFACT), "cleanup": cleanup, "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
