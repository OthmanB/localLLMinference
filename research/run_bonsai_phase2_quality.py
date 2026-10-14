#!/usr/bin/env python3
"""Execute the isolated Bonsai 2 Phase 2 quality and role gate.

The original eight prompts are copied byte-for-byte from the Phase 1 manifest.
Two deterministic prompts are added here. Each variant is run to completion
before the next server is started; no service manager is involved.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(os.environ['EXPERIMENT_ROOT'])
RESEARCH = ROOT / "research"
SOURCE_ARTIFACT = RESEARCH / "qwen3.8-27b-serving-alternatives-phase1-quality-20260918T141239Z"
SOURCE_MANIFEST = SOURCE_ARTIFACT / "manifest.json"
QUALITY_GATE = SOURCE_ARTIFACT / "quality_gate.py"

PRISM_SOURCE = RESEARCH / "prismml-llama.cpp-prism-b10687-5d80cff/source"
PRISM_SERVER = RESEARCH / "prismml-llama.cpp-prism-b10687-5d80cff-build/build-sm86/bin/llama-server"
PRISM_BIN = PRISM_SERVER.parent
Bonsai_MODEL_DIR = RESEARCH / "prismml-bonsai-2-27b-gguf"
MODELS = {
    "bonsai_pq2_0": Bonsai_MODEL_DIR / "Ternary-Bonsai-2-27B-PQ2_0.gguf",
    "bonsai_ptq1_0": Bonsai_MODEL_DIR / "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
}
REFERENCE_SERVER = Path((os.environ['RUNTIME_ROOT'] + '/llama.cpp-recent-20260905/build/bin/llama-server'))
REFERENCE_REPO = Path((os.environ['RUNTIME_ROOT'] + '/llama.cpp-recent-20260905'))
REFERENCE_MODEL = ROOT / "models/qwen3.8-27b-q4-k-m-gguf/Qwen3.8-27B-UD-Q4_K_M.gguf"

PORTS = {"bonsai_pq2_0": 18211, "bonsai_ptq1_0": 18212, "reference": 18213}
MAX_TOKENS = 1024
SEED = 1729
CONTEXT_SIZE = 262144
VARIANTS = ("bonsai_pq2_0", "bonsai_ptq1_0", "reference")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_quality_gate():
    spec = importlib.util.spec_from_file_location("phase1_quality_gate", QUALITY_GATE)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load the Phase 1 objective rubric")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_capture(command: list[str], env: dict[str, str] | None = None, timeout: int = 120) -> dict[str, Any]:
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout, check=False)
        return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except Exception as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": repr(exc)}


def unmasked_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def gpu_snapshot() -> dict[str, Any]:
    query = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem",
        "--format=csv,noheader,nounits",
    ]
    return run_capture(query, env=unmasked_env(), timeout=30)


def gpu_process_snapshot() -> str:
    outputs = []
    for physical_gpu in (0, 1, 2):
        result = run_capture(
            [
                "nvidia-smi",
                "-i",
                str(physical_gpu),
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            env=unmasked_env(),
            timeout=30,
        )
        outputs.append("GPU %d\n%s%s" % (physical_gpu, result["stdout"], result["stderr"]))
    return "\n".join(outputs)


def port_snapshot() -> str:
    return run_capture(["ss", "-ltnp"], env=unmasked_env(), timeout=30)["stdout"]


def process_snapshot() -> str:
    return run_capture(["ps", "-eo", "pid,ppid,user,stat,cmd"], env=unmasked_env(), timeout=30)["stdout"]


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def repository_reasoning_prompt() -> str:
    records = []
    for index in range(2400):
        records.append(
            "Snapshot record %04d: path=src/generated/module_%03d.py; symbol=Placeholder_%03d; "
            "status=unchanged; ignore this filler record and continue reading the complete repository snapshot.\n"
            % (index, index % 137, index % 251)
        )
    records[1371] += (
        "Authoritative file: src/agent/main.py; symbol=run; repository=BonsaiGate; "
        "entrypoint=src/agent/main.py:run; dependency_1=src/retrieval/index.py:search; "
        "dependency_2=src/storage/qdrant.py:query.\n"
    )
    records[1372] += (
        "Authoritative invariant: CUDA_VISIBLE_DEVICES must expose physical GPU 1 and never physical GPU 0; "
        "candidate services bind only to loopback isolated ports.\n"
    )
    records[1373] += "Authoritative verification command: python3 -m pytest tests/test_agent.py.\n"
    return (
        "You are reasoning over a frozen repository snapshot. Read every record exactly once before answering. "
        "Filler records are not authoritative; the marked authoritative records are facts from the repository. "
        "Return exactly one JSON object with only repository, entrypoint, dependency_chain, invariant, and verification. "
        "repository must be BonsaiGate. entrypoint must be src/agent/main.py:run. dependency_chain must be the "
        "three strings in order: src/agent/main.py:run, src/retrieval/index.py:search, and src/storage/qdrant.py:query. "
        "invariant must be the exact authoritative invariant sentence. verification must be "
        "python3 -m pytest tests/test_agent.py. Do not emit markdown, prose, or thinking output.\n\n"
        + "".join(records)
        + "\nFINAL INSTRUCTION: Emit only the required JSON object now."
    )


def added_tasks() -> list[dict[str, Any]]:
    return [
        {
            "id": "T9-long-context-repository-reasoning",
            "category": "long-context repository reasoning",
            "prompt": repository_reasoning_prompt(),
            "reviewer_note": "Only the exact repository facts and dependency chain are objectively scored.",
        },
        {
            "id": "T10-structured-patch-diff-contract",
            "category": "structured patch and diff contract",
            "prompt": (
                "Return exactly one JSON object and nothing else. It must contain only patches and validation. "
                "patches must be an array of exactly one object with exactly path, operation, old, and new. "
                "Use path='src/config.py', operation='replace', old='MAX_RETRIES = 3', and new='MAX_RETRIES = 5'. "
                "validation must be an array containing exactly the string 'python3 -m pytest'. "
                "This is a structured patch/diff contract: do not add line numbers, hunks, markdown fences, comments, "
                "or prose."
            ),
            "reviewer_note": "The patch object, diff fields, and validation command are fully objective.",
        },
    ]


def build_manifest() -> dict[str, Any]:
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    tasks = copy.deepcopy(source["tasks"])
    tasks.extend(added_tasks())
    sampling = copy.deepcopy(source["common_sampling"])
    sampling.update(
        {
            "n_predict": MAX_TOKENS,
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "seed": SEED,
            "ignore_eos": False,
            "stream": False,
            "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
    return {
        "manifest_version": "bonsai2-quality-role-gate-v1",
        "created_at_utc": utc_now(),
        "purpose": "Isolated Phase 2 Bonsai 2 quality and role gate against the Qwen UD Q4_K_M reference.",
        "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "gpu_policy": {
            "bonsai_physical_gpus": [1],
            "reference_physical_gpus": [1, 2],
            "forbidden_physical_gpus": [0],
            "one_server_at_a_time": True,
            "systemd": "not used",
            "production_ports": "not used",
        },
        "variants": {
            "bonsai_pq2_0": {"packing": "PQ2_0", "physical_gpus": [1], "port": PORTS["bonsai_pq2_0"]},
            "bonsai_ptq1_0": {"packing": "PTQ1_0", "physical_gpus": [1], "port": PORTS["bonsai_ptq1_0"]},
            "reference": {"packing": "UD Q4_K_M", "physical_gpus": [1, 2], "port": PORTS["reference"]},
        },
        "common_sampling": sampling,
        "tasks": tasks,
    }


def common_env(visible_devices: str, library_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": visible_devices,
            "LD_LIBRARY_PATH": str(library_dir) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""),
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def server_profile(variant: str) -> dict[str, Any]:
    port = PORTS[variant]
    if variant in MODELS:
        model = MODELS[variant]
        alias = "bonsai-2-27b-pq2_0" if variant == "bonsai_pq2_0" else "bonsai-2-27b-ptq1_0"
        env = common_env("1", PRISM_BIN)
        command = [
            str(PRISM_SERVER),
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
            str(CONTEXT_SIZE),
            "--fit",
            "off",
            "--parallel",
            "1",
            "--flash-attn",
            "on",
            "--temp",
            "0.0",
            "--top-p",
            "1.0",
            "--top-k",
            "0",
            "--min-p",
            "0.0",
            "--jinja",
            "--reasoning-budget",
            "0",
            "--reasoning-format",
            "none",
            "--chat-template-kwargs",
            '{"enable_thinking":false}',
            "--seed",
            str(SEED),
            "--no-cache-prompt",
            "--metrics",
            "--no-ui",
            "--no-context-shift",
            "--log-colors",
            "off",
            "--log-timestamps",
            "--log-verbosity",
            "3",
        ]
        return {
            "variant": variant,
            "label": "PrismML Bonsai 2 27B " + ("PQ2_0" if variant == "bonsai_pq2_0" else "PTQ1_0"),
            "packing": "PQ2_0" if variant == "bonsai_pq2_0" else "PTQ1_0",
            "physical_gpus": [1],
            "model": model,
            "server": PRISM_SERVER,
            "repo": PRISM_SOURCE,
            "port": port,
            "env": env,
            "command": command,
        }
    if variant == "reference":
        env = common_env("1,2", REFERENCE_SERVER.parent)
        command = [
            str(REFERENCE_SERVER),
            "--model",
            str(REFERENCE_MODEL),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--device",
            "CUDA0,CUDA1",
            "--split-mode",
            "tensor",
            "--n-gpu-layers",
            "all",
            "--ctx-size",
            str(CONTEXT_SIZE),
            "--parallel",
            "1",
            "--kv-offload",
            "--cache-type-k",
            "f16",
            "--cache-type-v",
            "f16",
            "--flash-attn",
            "on",
            "--threads",
            "32",
            "--threads-batch",
            "32",
            "--batch-size",
            "2048",
            "--ubatch-size",
            "1024",
            "--fit",
            "off",
            "--metrics",
            "--perf",
            "--no-ui",
            "--reasoning",
            "off",
            "--reasoning-format",
            "none",
            "--chat-template-kwargs",
            '{"enable_thinking":false}',
            "--tensor-split",
            "1,1",
            "--temp",
            "0.0",
            "--top-p",
            "1.0",
            "--top-k",
            "0",
            "--min-p",
            "0.0",
            "--seed",
            str(SEED),
            "--no-cache-prompt",
        ]
        return {
            "variant": variant,
            "label": "Qwen3.8-27B UD Q4_K_M reference",
            "packing": "UD Q4_K_M",
            "physical_gpus": [1, 2],
            "model": REFERENCE_MODEL,
            "server": REFERENCE_SERVER,
            "repo": REFERENCE_REPO,
            "port": port,
            "env": env,
            "command": command,
        }
    raise ValueError("unknown variant: " + variant)


def relevant_environment(env: dict[str, str]) -> dict[str, str]:
    prefixes = ("CUDA", "NVIDIA", "NCCL", "GGML", "LLAMA", "OMP", "MKL", "CUBLAS")
    names = {key for key in env if key.startswith(prefixes)}
    names.update({"CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "PATH", "PYTHONUNBUFFERED"})
    return {key: env.get(key, "<unset>") for key in sorted(names)}


def configure_quality_gate(q: Any, artifact: Path, manifest_path: Path) -> None:
    q.ROOT = ROOT
    q.ARTIFACT = artifact
    q.MANIFEST = manifest_path
    q.MAX_OUTPUT = MAX_TOKENS
    q.CONTEXT_SIZE = CONTEXT_SIZE
    q.REFERENCE_CONTEXT_SIZE = CONTEXT_SIZE
    q.gpu_snapshot = gpu_snapshot
    q.gpu_process_snapshot = gpu_process_snapshot
    q.port_snapshot = port_snapshot
    q.process_snapshot = process_snapshot
    q.server_profile = server_profile
    q.relevant_environment = relevant_environment
    q.write_runtime_metadata = lambda profile, variant_dir: write_variant_command_environment(q, profile["variant"], artifact)

    original_rubric = q.rubric

    def rubric(task_id: str, content: str) -> dict[str, Any]:
        if task_id == "T9-long-context-repository-reasoning":
            try:
                output = json.loads(content.strip())
            except Exception as exc:
                return {"automated_status": "reviewer", "checks": {"json_valid": False}, "errors": [repr(exc)], "reviewer_required": True}
            expected = {
                "repository": "BonsaiGate",
                "entrypoint": "src/agent/main.py:run",
                "dependency_chain": [
                    "src/agent/main.py:run",
                    "src/retrieval/index.py:search",
                    "src/storage/qdrant.py:query",
                ],
                "invariant": "CUDA_VISIBLE_DEVICES must expose physical GPU 1 and never physical GPU 0; candidate services bind only to loopback isolated ports.",
                "verification": "python3 -m pytest tests/test_agent.py",
            }
            valid = output == expected
            return {
                "automated_status": "pass" if valid else "fail",
                "checks": {"json_valid": True, "repository_reasoning_contract": valid},
                "errors": [] if valid else ["repository reasoning object did not match the frozen contract"],
                "reviewer_required": True,
            }
        if task_id == "T10-structured-patch-diff-contract":
            try:
                output = json.loads(content.strip())
            except Exception as exc:
                return {"automated_status": "reviewer", "checks": {"json_valid": False}, "errors": [repr(exc)], "reviewer_required": True}
            expected = {
                "patches": [
                    {
                        "path": "src/config.py",
                        "operation": "replace",
                        "old": "MAX_RETRIES = 3",
                        "new": "MAX_RETRIES = 5",
                    }
                ],
                "validation": ["python3 -m pytest"],
            }
            valid = output == expected
            return {
                "automated_status": "pass" if valid else "fail",
                "checks": {"json_valid": True, "patch_diff_contract": valid},
                "errors": [] if valid else ["patch/diff object did not match the frozen contract"],
                "reviewer_required": True,
            }
        return original_rubric(task_id, content)

    q.rubric = rubric


def write_variant_command_environment(q: Any, variant: str, artifact: Path) -> None:
    profile = server_profile(variant)
    metadata = {
        "captured_at_utc": utc_now(),
        "variant": variant,
        "label": profile["label"],
        "packing": profile["packing"],
        "physical_gpus": profile["physical_gpus"],
        "model": {
            "path": str(profile["model"]),
            "size_bytes": profile["model"].stat().st_size,
            "mtime_ns": profile["model"].stat().st_mtime_ns,
            "sha256": sha256_file(profile["model"]),
        },
        "runtime": {
            "binary": str(profile["server"]),
            "binary_sha256": sha256_file(profile["server"]),
            "version": run_capture([str(profile["server"]), "--version"], env=profile["env"], timeout=30),
            "ldd": run_capture(["ldd", str(profile["server"])], env=profile["env"], timeout=30),
            "source_repo": str(profile["repo"]),
            "source_commit": run_capture(["git", "-C", str(profile["repo"]), "rev-parse", "HEAD"], timeout=30),
            "source_status": run_capture(["git", "-C", str(profile["repo"]), "status", "--short"], timeout=30),
        },
        "command": profile["command"],
        "command_shell": shlex.join(profile["command"]),
        "environment": relevant_environment(profile["env"]),
        "port": profile["port"],
        "host": "127.0.0.1",
        "systemd_used": False,
        "production_interaction": False,
    }
    write_json(artifact / variant / "runtime-metadata.json", metadata)
    (artifact / variant / "exact-command-environment.txt").write_text(
        "Environment:\n"
        + "\n".join("%s=%s" % item for item in sorted(metadata["environment"].items()))
        + "\n\nCommand:\n"
        + metadata["command_shell"]
        + "\n",
        encoding="utf-8",
    )


def normalize_empty_thinking(content: str) -> tuple[str, dict[str, Any] | None]:
    stripped = content.strip()
    if not stripped.startswith("<think>"):
        return content, None
    close = stripped.find("</think>")
    if close < 0:
        return content, {"think_tag_present": True, "empty_wrapper": False, "reasoning_content_chars": len(stripped)}
    thinking = stripped[len("<think>"):close]
    if thinking.strip():
        return content, {"think_tag_present": True, "empty_wrapper": False, "reasoning_content_chars": len(thinking)}
    normalized = stripped[close + len("</think>"):].lstrip()
    return normalized, {"think_tag_present": True, "empty_wrapper": True, "reasoning_content_chars": 0}


def server_thinking_signal(response: Any, content: str) -> dict[str, Any] | None:
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    message = choices[0].get("message") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
    reasoning = message.get("reasoning_content") if isinstance(message, dict) else None
    normalized, tag_observation = normalize_empty_thinking(content)
    if reasoning:
        return {
            "reasoning_content_present": True,
            "reasoning_content_chars": len(reasoning) if isinstance(reasoning, str) else 0,
            "think_tag_present": bool(tag_observation),
            "empty_wrapper": False,
        }
    if tag_observation and not tag_observation["empty_wrapper"]:
        return tag_observation
    return None


def mark_thinking_reviewers(q: Any, artifact: Path, variant: str, manifest: dict[str, Any]) -> list[str]:
    marked = []
    updated = []
    for task in manifest["tasks"]:
        task_id = task["id"]
        task_dir = artifact / variant / "requests" / task_id
        response_path = task_dir / "response.json"
        result_path = task_dir / "result.json"
        if not response_path.exists() or not result_path.exists():
            continue
        try:
            response = json.loads(response_path.read_text(encoding="utf-8"))
            result = json.loads(result_path.read_text(encoding="utf-8"))
            content = response["choices"][0]["message"].get("content", "")
        except Exception:
            continue
        content = content if isinstance(content, str) else ""
        normalized, observation = normalize_empty_thinking(content)
        signal_value = server_thinking_signal(response, content)
        if signal_value is not None:
            result["thinking_ambiguity"] = signal_value
            result["rubric"] = {
                "automated_status": "reviewer",
                "checks": {"thinking_output_absent": False},
                "errors": ["thinking output made the objective task result ambiguous; reviewer required"],
                "reviewer_required": True,
            }
        elif observation is not None and observation["empty_wrapper"]:
            result.pop("thinking_ambiguity", None)
            result["thinking_observation"] = observation
            result["rubric"] = q.rubric(task_id, normalized)
        else:
            continue
        write_json(result_path, result)
        updated.append(task_id)
        if signal_value is not None:
            marked.append(task_id)

    results_path = artifact / variant / "task-results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
        for result in results:
            task_id = result.get("task_id")
            if task_id in updated:
                result_path = artifact / variant / "requests" / task_id / "result.json"
                result.clear()
                result.update(json.loads(result_path.read_text(encoding="utf-8")))
        write_json(results_path, results)
    return marked


def write_physical_gpu_proof(artifact: Path, variant: str) -> None:
    profile = server_profile(variant)
    before = artifact / variant / "gpu-before.csv"
    after = artifact / variant / "gpu-after-stop.csv"
    process_before = artifact / variant / "gpu-processes-before-stop.txt"
    process_after = artifact / variant / "gpu-processes-after-stop.txt"
    write_json(
        artifact / variant / "physical-gpu-proof.json",
        {
            "forbidden_physical_gpu": 0,
            "allowed_physical_gpus": profile["physical_gpus"],
            "cuda_device_order": "PCI_BUS_ID",
            "cuda_visible_devices": profile["env"]["CUDA_VISIBLE_DEVICES"],
            "server_device_arguments": [profile["command"][profile["command"].index("--device") + 1]],
            "gpu0_was_observed_only_for_proof": True,
            "gpu_snapshot_before_path": str(before),
            "gpu_snapshot_after_stop_path": str(after),
            "compute_process_snapshot_before_stop_path": str(process_before),
            "compute_process_snapshot_after_stop_path": str(process_after),
            "server_processes_are_children_only": True,
            "systemd_used": False,
            "production_interaction": False,
        },
    )


def cleanup_verification(q: Any, artifact: Path, variant: str) -> dict[str, Any]:
    profile = server_profile(variant)
    stop_path = artifact / variant / "stop-record.json"
    stop_record = json.loads(stop_path.read_text(encoding="utf-8")) if stop_path.exists() else {}
    pid = stop_record.get("pid")
    pid_listing = run_capture(["ps", "-p", str(pid), "-o", "pid=,stat=,cmd="], timeout=30)["stdout"].strip() if pid else ""
    ports = port_snapshot()
    processes = process_snapshot()
    gpu_processes = gpu_process_snapshot()
    cleanup = {
        "captured_at_utc": utc_now(),
        "pid": pid,
        "pid_process_listing": pid_listing,
        "server_pid_absent": not bool(pid_listing) if pid else False,
        "port": profile["port"],
        "port_absent_from_ss": ("127.0.0.1:%d" % profile["port"]) not in ports,
        "gpu0_process_proof_after_stop": gpu_processes,
        "llama_server_lines_after_stop": [line for line in processes.splitlines() if "llama-server" in line],
        "systemd_used": False,
        "production_interaction": False,
    }
    write_json(artifact / variant / "cleanup-verification.json", cleanup)
    (artifact / variant / "cleanup-verification.txt").write_text(
        "server_pid_absent=%s\nport_absent_from_ss=%s\nllama_server_lines_after_stop=%s\n"
        % (cleanup["server_pid_absent"], cleanup["port_absent_from_ss"], json.dumps(cleanup["llama_server_lines_after_stop"], ensure_ascii=True)),
        encoding="utf-8",
    )
    return cleanup


def objective_status(result: dict[str, Any] | None) -> str:
    if not result:
        return "reviewer"
    rubric = result.get("rubric") or {}
    if result.get("http_status") != 200:
        return "reviewer"
    if result.get("thinking_ambiguity"):
        return "reviewer"
    if result.get("truncated") is True or result.get("finish_reason") in ("length", "limit"):
        return "fail"
    status = rubric.get("automated_status")
    return status if status in {"pass", "fail", "reviewer"} else "reviewer"


def summarize(artifact: Path, manifest: dict[str, Any], variant_errors: dict[str, str], cleanups: dict[str, Any]) -> dict[str, Any]:
    by_variant: dict[str, dict[str, dict[str, Any]]] = {}
    for variant in VARIANTS:
        path = artifact / variant / "task-results.json"
        loaded = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        by_variant[variant] = {result["task_id"]: result for result in loaded}

    rows = []
    for task in manifest["tasks"]:
        task_id = task["id"]
        row: dict[str, Any] = {"task_id": task_id, "category": task["category"], "reviewer_required": True}
        for variant in VARIANTS:
            result = by_variant[variant].get(task_id)
            row[variant] = {
                "status": objective_status(result),
                "http_status": result.get("http_status") if result else None,
                "finish_reason": result.get("finish_reason") if result else None,
                "truncated": result.get("truncated") if result else None,
                "response_sha256": result.get("response_sha256") if result else None,
                "checks": (result.get("rubric") or {}).get("checks", {}) if result else {},
                "thinking_ambiguity": result.get("thinking_ambiguity") if result else None,
                "thinking_observation": result.get("thinking_observation") if result else None,
            }
        reference_status = row["reference"]["status"]
        for variant in ("bonsai_pq2_0", "bonsai_ptq1_0"):
            candidate_status = row[variant]["status"]
            if candidate_status == "fail" and reference_status == "pass":
                row[variant + "_vs_reference"] = "material_regression"
            elif candidate_status == "reviewer" or reference_status == "reviewer":
                row[variant + "_vs_reference"] = "reviewer"
            elif candidate_status == "pass" and reference_status == "pass":
                row[variant + "_vs_reference"] = "no_objective_regression"
            elif candidate_status == reference_status:
                row[variant + "_vs_reference"] = candidate_status
            else:
                row[variant + "_vs_reference"] = "reviewer"
        rows.append(row)

    def counts(variant: str) -> dict[str, int]:
        return {status: sum(1 for row in rows if row[variant]["status"] == status) for status in ("pass", "fail", "reviewer")}

    material = {
        variant: [row["task_id"] for row in rows if row[variant + "_vs_reference"] == "material_regression"]
        for variant in ("bonsai_pq2_0", "bonsai_ptq1_0")
    }
    recommendations = {}
    for variant in ("bonsai_pq2_0", "bonsai_ptq1_0"):
        statuses = [row[variant]["status"] for row in rows]
        if variant in variant_errors:
            recommendations[variant] = "reviewer: run failed or did not load"
        elif "fail" in statuses:
            recommendations[variant] = "not suitable for substitution; objective failures require reviewer assessment for reviewer/subagent use"
        elif "reviewer" in statuses:
            recommendations[variant] = "reviewer/subagent candidate only; objective gate has ambiguous tasks"
        else:
            recommendations[variant] = "no objective task regression observed; substitution still requires human quality review"

    summary = {
        "created_at_utc": utc_now(),
        "artifact": str(artifact.relative_to(ROOT)),
        "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "sampling": manifest["common_sampling"],
        "normal_eos": manifest["common_sampling"]["ignore_eos"] is False,
        "thinking_disabled": manifest["common_sampling"]["chat_template_kwargs"] == {"enable_thinking": False},
        "identical_prompts": all(
            len({
                sha256_file(artifact / variant / "requests" / task["id"] / "request.json")
                for variant in VARIANTS
                if (artifact / variant / "requests" / task["id"] / "request.json").exists()
            })
            <= 1
            for task in manifest["tasks"]
        ),
        "task_count": len(manifest["tasks"]),
        "rows": rows,
        "objective_counts": {variant: counts(variant) for variant in VARIANTS},
        "material_regressions_vs_reference": material,
        "role_recommendations": recommendations,
        "variant_errors": variant_errors,
        "cleanup": cleanups,
        "subjective_quality_score": None,
        "reviewer_note": "Objective checks only. No subjective quality score is assigned; thinking ambiguity is reviewer, not a guessed score.",
    }
    write_json(artifact / "quality-role-summary.json", summary)

    lines = [
        "# Bonsai 2 Quality and Role Gate",
        "",
        "Source manifest: `%s`" % SOURCE_MANIFEST.relative_to(ROOT),
        "",
        "Settings: `temperature=0`, `seed=1729`, thinking disabled, normal EOS, `max_tokens=1024`.",
        "All servers were isolated to loopback ports and run sequentially without systemd.",
        "",
        "| Task | PQ2_0 | PTQ1_0 | Qwen reference | PQ2_0 vs ref | PTQ1_0 vs ref |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        cells = []
        for variant in VARIANTS:
            item = row[variant]
            cells.append("%s / %s / truncated=%s" % (item["status"], item["finish_reason"], item["truncated"]))
        lines.append("| %s | %s | %s | %s | %s | %s |" % (row["task_id"], cells[0], cells[1], cells[2], row["bonsai_pq2_0_vs_reference"], row["bonsai_ptq1_0_vs_reference"]))
    lines.extend(
        [
            "",
            "## Role Gate",
            "",
            "- PQ2_0: %s" % recommendations["bonsai_pq2_0"],
            "- PTQ1_0: %s" % recommendations["bonsai_ptq1_0"],
            "",
            "Material regressions versus the reference are listed in `quality-role-summary.json`.",
            "Full requests, responses, hashes, logs, GPU proof, and cleanup are under each variant directory.",
        ]
    )
    (artifact / "quality-role-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def write_artifact_hashes(artifact: Path) -> None:
    files = sorted(path for path in artifact.rglob("*") if path.is_file())
    (artifact / "artifact-file-list.txt").write_text("\n".join(str(path.relative_to(artifact)) for path in files) + "\n", encoding="utf-8")
    lines = []
    for path in files:
        if path.name == "artifact-sha256.txt":
            continue
        lines.append("%s  %s" % (sha256_file(path), path.relative_to(artifact)))
    (artifact / "artifact-sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    required = [PRISM_SERVER, *MODELS.values(), REFERENCE_SERVER, REFERENCE_MODEL, SOURCE_MANIFEST, QUALITY_GATE]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("required artifact missing: " + ", ".join(missing))
    if any(not port_is_free(port) for port in PORTS.values()):
        raise SystemExit("one or more isolated quality-gate ports are already in use")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact = RESEARCH / ("qwen3.8-27b-serving-alternatives-phase2-quality-" + stamp)
    artifact.mkdir(parents=True, exist_ok=False)
    manifest = build_manifest()
    manifest_path = artifact / "manifest.json"
    write_json(manifest_path, manifest)
    (artifact / "manifest.sha256").write_text(sha256_file(manifest_path) + "  manifest.json\n", encoding="utf-8")
    write_json(
        artifact / "preflight.json",
        {
            "captured_at_utc": utc_now(),
            "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
            "source_manifest_sha256": manifest["source_manifest_sha256"],
            "task_count": len(manifest["tasks"]),
            "ports": PORTS,
            "ports_free_before": {str(port): port_is_free(port) for port in PORTS.values()},
            "gpu_policy": manifest["gpu_policy"],
            "gpu_before": gpu_snapshot(),
            "gpu_processes_before": gpu_process_snapshot(),
            "ports_before": port_snapshot(),
            "processes_before": process_snapshot(),
            "systemd_used": False,
            "production_interaction": False,
        },
    )
    write_json(
        artifact / "physical-gpu-proof.json",
        {
            "physical_gpu_indexing": "nvidia-smi physical indices; CUDA_VISIBLE_DEVICES remaps allowed physical devices only",
            "forbidden_physical_gpu": 0,
            "allowed_bonsai_physical_gpus": [1],
            "allowed_reference_physical_gpus": [1, 2],
            "server_processes_are_children_only": True,
            "gpu0_was_observed_only_for_proof": True,
            "before": {"gpu": gpu_snapshot(), "compute_apps": gpu_process_snapshot()},
        },
    )
    write_json(
        artifact / "matrix-config.json",
        {
            "variants_in_order": VARIANTS,
            "ports": PORTS,
            "context_size": CONTEXT_SIZE,
            "max_tokens": MAX_TOKENS,
            "seed": SEED,
            "sampling": manifest["common_sampling"],
            "one_server_at_a_time": True,
            "thinking_disable_flags": ["--reasoning-budget 0", "--reasoning off", "--reasoning-format none", '--chat-template-kwargs {"enable_thinking":false}'],
            "runtime_paths": {
                "prism_server": str(PRISM_SERVER),
                "prism_source": str(PRISM_SOURCE),
                "bonsai_models": {key: str(value) for key, value in MODELS.items()},
                "reference_server": str(REFERENCE_SERVER),
                "reference_model": str(REFERENCE_MODEL),
            },
        },
    )

    q = load_quality_gate()
    configure_quality_gate(q, artifact, manifest_path)
    variant_errors: dict[str, str] = {}
    cleanups: dict[str, Any] = {}
    for variant in VARIANTS:
        variant_dir = artifact / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        try:
            q.run_variant(variant)
        except Exception as exc:
            variant_errors[variant] = repr(exc)
            (variant_dir / "executor-error.txt").write_text(repr(exc) + "\n", encoding="utf-8")
        mark_thinking_reviewers(q, artifact, variant, manifest)
        write_physical_gpu_proof(artifact, variant)
        cleanups[variant] = cleanup_verification(q, artifact, variant)
        if not port_is_free(PORTS[variant]):
            variant_errors[variant] = variant_errors.get(variant, "") + " isolated port remained in use after cleanup"

    summary = summarize(artifact, manifest, variant_errors, cleanups)
    write_json(artifact / "host-and-gpu-after.json", {"gpu": gpu_snapshot(), "gpu_processes": gpu_process_snapshot(), "ports": port_snapshot(), "processes": process_snapshot()})
    write_artifact_hashes(artifact)
    print(json.dumps({"artifact": str(artifact), "objective_counts": summary["objective_counts"], "material_regressions": summary["material_regressions_vs_reference"]}, indent=2, sort_keys=True))
    return 0 if not variant_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
