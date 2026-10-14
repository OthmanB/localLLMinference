#!/usr/bin/env python3
"""Run the focused T8 max-token follow-up for both Bonsai 2 packings."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(os.environ['EXPERIMENT_ROOT'])
RESEARCH = ROOT / "research"
SOURCE_ARTIFACT = RESEARCH / "qwen3.8-27b-serving-alternatives-phase1-quality-20260918T141239Z"
SOURCE_MANIFEST = SOURCE_ARTIFACT / "manifest.json"
RUBRIC_SOURCE = SOURCE_ARTIFACT / "quality_gate.py"
SERVER = RESEARCH / "prismml-llama.cpp-prism-b10687-5d80cff-build/build-sm86/bin/llama-server"
SOURCE_REPO = RESEARCH / "prismml-llama.cpp-prism-b10687-5d80cff/source"
MODEL_DIR = RESEARCH / "prismml-bonsai-2-27b-gguf"
MODELS = {
    "bonsai_pq2_0": MODEL_DIR / "Ternary-Bonsai-2-27B-PQ2_0.gguf",
    "bonsai_ptq1_0": MODEL_DIR / "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
}
PORTS = {"bonsai_pq2_0": 18211, "bonsai_ptq1_0": 18212}
PACKINGS = {"bonsai_pq2_0": "PQ2_0", "bonsai_ptq1_0": "PTQ1_0"}
VARIANTS = ("bonsai_pq2_0", "bonsai_ptq1_0")
MAX_TOKENS = 2048
SEED = 1729
CONTEXT_SIZE = 262144


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def unmasked_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def run_capture(command: list[str], env: dict[str, str] | None = None, timeout: int = 120) -> dict[str, Any]:
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


def gpu_snapshot() -> dict[str, Any]:
    host_env = unmasked_env()
    gpu_query = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem",
        "--format=csv,noheader,nounits",
    ]
    app_query = [
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    return {
        "captured_at_utc": utc_now(),
        "gpu_query": run_capture(gpu_query, env=host_env, timeout=30),
        "compute_apps_query": run_capture(app_query, env=host_env, timeout=30),
        "physical_gpu_0_is_observed_only_for_proof": True,
    }


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


def load_sources() -> tuple[dict[str, Any], Any]:
    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    task = next(item for item in source_manifest["tasks"] if item["id"] == "T8-code-generation-test")
    spec = importlib.util.spec_from_file_location("phase1_quality_gate_t8", RUBRIC_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load the Phase 1 objective rubric")
    rubric_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = rubric_module
    spec.loader.exec_module(rubric_module)
    return source_manifest, (task, rubric_module)


def build_manifest(source_manifest: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest_version": "bonsai2-t8-max-token-followup-v1",
        "created_at_utc": utc_now(),
        "purpose": "Focused T8 follow-up distinguishing a 2048-token cap from malformed-code regression.",
        "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "gpu_policy": {
            "physical_gpus": [1],
            "forbidden_physical_gpus": [0],
            "one_server_at_a_time": True,
            "systemd": "not used",
            "production_ports": "not used",
        },
        "variants": {
            variant: {"packing": PACKINGS[variant], "physical_gpus": [1], "port": PORTS[variant]}
            for variant in VARIANTS
        },
        "common_sampling": {
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "seed": SEED,
            "ignore_eos": False,
            "stream": False,
            "cache_prompt": False,
        },
        "tasks": [task],
        "prompt_identity": {
            "task_id": task["id"],
            "prompt_sha256": sha256_bytes(task["prompt"].encode("utf-8")),
            "matches_source_manifest": task["prompt"]
            == next(item["prompt"] for item in source_manifest["tasks"] if item["id"] == task["id"]),
        },
    }


def launch_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "1",
            "LD_LIBRARY_PATH": str(SERVER.parent) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""),
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def server_profile(variant: str) -> dict[str, Any]:
    model = MODELS[variant]
    command = [
        str(SERVER),
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(PORTS[variant]),
        "--alias",
        "bonsai-2-27b-" + PACKINGS[variant].lower(),
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
        "packing": PACKINGS[variant],
        "physical_gpus": [1],
        "model": model,
        "server": SERVER,
        "repo": SOURCE_REPO,
        "port": PORTS[variant],
        "env": launch_environment(),
        "command": command,
    }


def relevant_environment(env: dict[str, str]) -> dict[str, str]:
    names = {"CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "PATH", "PYTHONUNBUFFERED"}
    return {key: env.get(key, "<unset>") for key in sorted(names)}


def write_runtime_metadata(artifact: Path, profile: dict[str, Any]) -> None:
    variant_dir = artifact / profile["variant"]
    env_blob = "\0".join("%s=%s" % item for item in sorted(profile["env"].items())).encode("utf-8")
    metadata = {
        "captured_at_utc": utc_now(),
        "variant": profile["variant"],
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
        "inherited_environment_keys": sorted(profile["env"]),
        "effective_environment_sha256": sha256_bytes(env_blob),
        "port": profile["port"],
        "host": "127.0.0.1",
        "systemd_used": False,
        "production_interaction": False,
        "gpu0_server_interaction": False,
    }
    write_json(variant_dir / "runtime-metadata.json", metadata)
    (variant_dir / "exact-command-environment.txt").write_text(
        "Environment (all launch-affecting values):\n"
        + "\n".join("%s=%s" % item for item in sorted(metadata["environment"].items()))
        + "\n\nInherited environment keys:\n"
        + "\n".join(metadata["inherited_environment_keys"])
        + "\n\nEffective environment SHA256:\n"
        + metadata["effective_environment_sha256"]
        + "\n\nCommand:\n"
        + metadata["command_shell"]
        + "\n",
        encoding="utf-8",
    )


def http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: int = 1800) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            return {
                "started_epoch": started,
                "ended_epoch": time.time(),
                "status": response.status,
                "headers": dict(response.headers.items()),
                "body": body,
                "error": None,
            }
    except urllib.error.HTTPError as exc:
        body = exc.read()
        return {
            "started_epoch": started,
            "ended_epoch": time.time(),
            "status": exc.code,
            "headers": dict(exc.headers.items()),
            "body": body,
            "error": repr(exc),
        }
    except Exception as exc:
        return {
            "started_epoch": started,
            "ended_epoch": time.time(),
            "status": None,
            "headers": {},
            "body": b"",
            "error": repr(exc),
        }


def wait_for_health(port: int, variant_dir: Path) -> dict[str, Any]:
    path = variant_dir / "health-poll.jsonl"
    deadline = time.time() + 900
    with path.open("w", encoding="utf-8") as handle:
        while time.time() < deadline:
            result = http_json("GET", "http://127.0.0.1:%d/health" % port, timeout=10)
            record = {
                "captured_at_utc": utc_now(),
                "status": result["status"],
                "error": result["error"],
                "body": result["body"].decode("utf-8", errors="replace"),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            if result["status"] == 200:
                return record
            time.sleep(2)
    raise RuntimeError("health endpoint did not become ready")


def gpu_sampler(variant_dir: Path, stop_event: threading.Event) -> None:
    with (variant_dir / "gpu-samples.jsonl").open("w", encoding="utf-8") as handle:
        while not stop_event.is_set():
            handle.write(json.dumps(gpu_snapshot(), sort_keys=True) + "\n")
            handle.flush()
            stop_event.wait(1.0)


def parsed_content(body: bytes) -> tuple[Any, str | None, str | None]:
    try:
        parsed = json.loads(body.decode("utf-8"))
    except Exception as exc:
        return None, None, "server response JSON parse error: %r" % (exc,)
    content = parsed.get("content") if isinstance(parsed, dict) else None
    if content is None and isinstance(parsed, dict):
        choices = parsed.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            content = message.get("content") if isinstance(message, dict) else choices[0].get("text")
    if not isinstance(content, str):
        return parsed, None, "completion content was not a string"
    return parsed, content, None


def finish_reason(parsed: Any) -> Any:
    if not isinstance(parsed, dict):
        return None
    if parsed.get("finish_reason") is not None:
        return parsed["finish_reason"]
    choices = parsed.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0].get("finish_reason")
    return parsed.get("stop_type")


def normalize_empty_thinking(content: str) -> tuple[str, dict[str, Any] | None]:
    stripped = content.strip()
    if not stripped.startswith("<think>"):
        return content, None
    close = stripped.find("</think>")
    if close < 0:
        return content, {"think_tag_present": True, "empty_wrapper": False, "reasoning_content_chars": len(stripped)}
    thinking = stripped[len("<think>") : close]
    if thinking.strip():
        return content, {"think_tag_present": True, "empty_wrapper": False, "reasoning_content_chars": len(thinking)}
    return stripped[close + len("</think>") :].lstrip(), {
        "think_tag_present": True,
        "empty_wrapper": True,
        "reasoning_content_chars": 0,
    }


def thinking_signal(parsed: Any, content: str) -> dict[str, Any] | None:
    choices = parsed.get("choices") if isinstance(parsed, dict) else None
    message = choices[0].get("message") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
    reasoning = message.get("reasoning_content") if isinstance(message, dict) else None
    _, observation = normalize_empty_thinking(content)
    if reasoning:
        return {
            "reasoning_content_present": True,
            "reasoning_content_chars": len(reasoning) if isinstance(reasoning, str) else 0,
            "think_tag_present": bool(observation),
            "empty_wrapper": False,
        }
    if observation is not None and not observation["empty_wrapper"]:
        return observation
    return None


def request_payload(task: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    payload = dict(manifest["common_sampling"])
    payload["messages"] = [{"role": "user", "content": task["prompt"]}]
    return payload


def objective_status(result: dict[str, Any]) -> str:
    if result.get("http_status") != 200:
        return "reviewer"
    if result.get("thinking_ambiguity"):
        return "reviewer"
    if result.get("truncated") is True or result.get("finish_reason") in ("length", "limit"):
        return "fail"
    status = (result.get("rubric") or {}).get("automated_status")
    return status if status in {"pass", "fail", "reviewer"} else "reviewer"


def write_physical_gpu_proof(artifact: Path, variant: str) -> None:
    variant_dir = artifact / variant
    profile = server_profile(variant)
    write_json(
        variant_dir / "physical-gpu-proof.json",
        {
            "forbidden_physical_gpu": 0,
            "allowed_physical_gpus": [1],
            "cuda_device_order": "PCI_BUS_ID",
            "cuda_visible_devices": profile["env"]["CUDA_VISIBLE_DEVICES"],
            "server_device_argument": "CUDA0 (remapped physical GPU 1)",
            "gpu0_was_observed_only_for_proof": True,
            "gpu0_was_not_targeted_by_server": True,
            "gpu_snapshot_before_path": str(variant_dir / "gpu-before.json"),
            "gpu_snapshot_during_path": str(variant_dir / "gpu-samples.jsonl"),
            "gpu_snapshot_after_requests_path": str(variant_dir / "gpu-after-requests.json"),
            "gpu_snapshot_after_stop_path": str(variant_dir / "gpu-after-stop.json"),
            "compute_process_snapshot_before_path": str(variant_dir / "gpu-before.json"),
            "server_processes_are_children_only": True,
            "systemd_used": False,
            "production_interaction": False,
        },
    )


def cleanup_variant(artifact: Path, variant: str, stop_record: dict[str, Any]) -> dict[str, Any]:
    variant_dir = artifact / variant
    profile = server_profile(variant)
    pid = stop_record.get("pid")
    pid_listing = run_capture(["ps", "-p", str(pid), "-o", "pid=,stat=,cmd="], timeout=30)["stdout"].strip() if pid else ""
    ports = port_snapshot()
    processes = process_snapshot()
    cleanup = {
        "captured_at_utc": utc_now(),
        "pid": pid,
        "pid_process_listing": pid_listing,
        "server_pid_absent": not bool(pid_listing) if pid else False,
        "port": profile["port"],
        "port_absent_from_ss": (":%d " % profile["port"]) not in ports,
        "port_bind_free_after_stop": port_is_free(profile["port"]),
        "llama_server_lines_after_stop": [line for line in processes.splitlines() if "llama-server" in line],
        "systemd_used": False,
        "production_interaction": False,
    }
    write_json(variant_dir / "cleanup-verification.json", cleanup)
    (variant_dir / "ports-after-stop.txt").write_text(ports, encoding="utf-8")
    (variant_dir / "processes-after-stop.txt").write_text(processes, encoding="utf-8")
    (variant_dir / "cleanup-verification.txt").write_text(
        "server_pid_absent=%s\nport_absent_from_ss=%s\nport_bind_free_after_stop=%s\nllama_server_lines_after_stop=%s\n"
        % (
            cleanup["server_pid_absent"],
            cleanup["port_absent_from_ss"],
            cleanup["port_bind_free_after_stop"],
            json.dumps(cleanup["llama_server_lines_after_stop"], ensure_ascii=True),
        ),
        encoding="utf-8",
    )
    return cleanup


def run_variant(artifact: Path, manifest: dict[str, Any], rubric_module: Any, variant: str) -> dict[str, Any]:
    profile = server_profile(variant)
    variant_dir = artifact / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    write_runtime_metadata(artifact, profile)
    write_json(variant_dir / "gpu-before.json", gpu_snapshot())
    (variant_dir / "ports-before.txt").write_text(port_snapshot(), encoding="utf-8")
    (variant_dir / "processes-before.txt").write_text(process_snapshot(), encoding="utf-8")

    task = manifest["tasks"][0]
    payload = request_payload(task, manifest)
    request_bytes = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n"
    task_dir = variant_dir / "requests" / task["id"]
    task_dir.mkdir(parents=True, exist_ok=True)
    task_dir.joinpath("request.json").write_bytes(request_bytes)
    task_dir.joinpath("prompt.txt").write_text(task["prompt"], encoding="utf-8")

    log_handle = (variant_dir / "server.stdout-stderr.log").open("w", encoding="utf-8")
    process: subprocess.Popen[str] | None = None
    sampler_stop = threading.Event()
    sampler_thread: threading.Thread | None = None
    stop_record: dict[str, Any] = {"attempted": False}
    result: dict[str, Any] = {"task_id": task["id"], "variant": variant, "packing": profile["packing"]}
    try:
        process = subprocess.Popen(
            profile["command"],
            cwd=ROOT,
            env=profile["env"],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        stop_record["pid"] = process.pid
        (variant_dir / "server.pid").write_text(str(process.pid) + "\n", encoding="utf-8")
        health = wait_for_health(profile["port"], variant_dir)
        write_json(variant_dir / "health-ready.json", health)
        sampler_thread = threading.Thread(target=gpu_sampler, args=(variant_dir, sampler_stop), daemon=True)
        sampler_thread.start()

        response = http_json(
            "POST",
            "http://127.0.0.1:%d/v1/chat/completions" % profile["port"],
            payload,
            timeout=1800,
        )
        response_body = response["body"]
        task_dir.joinpath("response.body").write_bytes(response_body)
        task_dir.joinpath("response-content.txt").write_text(
            parsed_content(response_body)[1] or "", encoding="utf-8"
        )
        task_dir.joinpath("headers.json").write_text(json.dumps(response["headers"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        parsed, content, parse_error = parsed_content(response_body)
        if parsed is not None:
            write_json(task_dir / "response.json", parsed)
        else:
            write_json(task_dir / "response.json", {"raw_body_base64": base64.b64encode(response_body).decode("ascii")})

        normalized_content, thinking_observation = normalize_empty_thinking(content or "")
        thinking_ambiguity = thinking_signal(parsed, content or "")
        rubric_result = rubric_module.rubric(task["id"], normalized_content if thinking_observation and thinking_observation["empty_wrapper"] else content or "")
        reason = finish_reason(parsed)
        result.update(
            {
                "request_sha256": sha256_bytes(request_bytes),
                "response_sha256": sha256_bytes(response_body),
                "http_status": response["status"],
                "error": response["error"],
                "response_bytes": len(response_body),
                "wall_seconds": response["ended_epoch"] - response["started_epoch"],
                "response_parse_error": parse_error,
                "tokens_predicted": parsed.get("tokens_predicted") if isinstance(parsed, dict) else None,
                "tokens_evaluated": parsed.get("tokens_evaluated") if isinstance(parsed, dict) else None,
                "truncated": (parsed.get("truncated") if isinstance(parsed, dict) and parsed.get("truncated") is not None else reason in ("length", "limit")),
                "stop_type": parsed.get("stop_type") if isinstance(parsed, dict) else None,
                "finish_reason": reason,
                "thinking_ambiguity": thinking_ambiguity,
                "thinking_observation": thinking_observation,
                "rubric": rubric_result,
                "objective_status": None,
            }
        )
        result["objective_status"] = objective_status(result)
        write_json(task_dir / "result.json", result)
        write_json(variant_dir / "task-results.json", [result])
        write_json(variant_dir / "objective-rubric.json", {
            "task_id": task["id"],
            "rubric_source": str(RUBRIC_SOURCE.relative_to(ROOT)),
            "rubric_source_sha256": sha256_file(RUBRIC_SOURCE),
            "rubric_result": rubric_result,
            "truncation_rule": "fail when truncated=true or finish_reason is length/limit",
            "thinking_rule": "reviewer when non-empty reasoning_content or non-empty think tags are present",
        })
        return result
    except Exception as exc:
        result.update({"objective_status": "reviewer", "run_error": repr(exc)})
        (variant_dir / "run-error.txt").write_text(repr(exc) + "\n", encoding="utf-8")
        write_json(variant_dir / "task-results.json", [result])
        return result
    finally:
        sampler_stop.set()
        if sampler_thread is not None:
            sampler_thread.join(timeout=10)
        write_json(variant_dir / "gpu-after-requests.json", gpu_snapshot())
        if process is not None:
            stop_record["attempted"] = True
            stop_record["pid"] = process.pid
            stop_record["signal"] = "SIGTERM process group"
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stop_record["returncode"] = process.wait(timeout=90)
                stop_record["clean"] = True
            except subprocess.TimeoutExpired:
                stop_record["clean"] = False
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stop_record["returncode"] = process.wait(timeout=20)
                stop_record["forced_kill"] = True
        log_handle.close()
        time.sleep(3)
        stop_record["captured_at_utc"] = utc_now()
        write_json(variant_dir / "stop-record.json", stop_record)
        write_json(variant_dir / "gpu-after-stop.json", gpu_snapshot())
        cleanup_variant(artifact, variant, stop_record)
        write_physical_gpu_proof(artifact, variant)


def write_artifact_hashes(artifact: Path) -> None:
    files = sorted(path for path in artifact.rglob("*") if path.is_file())
    (artifact / "artifact-file-list.txt").write_text(
        "\n".join(str(path.relative_to(artifact)) for path in files) + "\n", encoding="utf-8"
    )
    lines = []
    for path in files:
        if path.name == "artifact-sha256.txt":
            continue
        lines.append("%s  %s" % (sha256_file(path), path.relative_to(artifact)))
    (artifact / "artifact-sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    required = [SERVER, *MODELS.values(), SOURCE_MANIFEST, RUBRIC_SOURCE]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("required artifact missing: " + ", ".join(missing))
    occupied = [str(port) for port in PORTS.values() if not port_is_free(port)]
    if occupied:
        raise SystemExit("focused follow-up port(s) already in use: " + ", ".join(occupied))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact = RESEARCH / ("qwen3.8-27b-serving-alternatives-phase2-quality-followup-" + stamp)
    artifact.mkdir(parents=True, exist_ok=False)
    source_manifest, (task, rubric_module) = load_sources()
    manifest = build_manifest(source_manifest, task)
    write_json(artifact / "manifest.json", manifest)
    (artifact / "manifest.sha256").write_text(sha256_file(artifact / "manifest.json") + "  manifest.json\n", encoding="utf-8")
    shutil.copy2(SOURCE_MANIFEST, artifact / "source-manifest.json")
    shutil.copy2(RUBRIC_SOURCE, artifact / "objective-rubric-source.py")
    shutil.copy2(Path(__file__), artifact / "followup-runner.py")
    (artifact / "invocation.txt").write_text(
        "cwd=%s\ncommand=python3 %s\nargv=%s\nstarted_at_utc=%s\n"
        % (ROOT, Path(__file__).relative_to(ROOT), shlex.join(["python3", *sys.argv]), utc_now()),
        encoding="utf-8",
    )
    write_json(
        artifact / "preflight.json",
        {
            "captured_at_utc": utc_now(),
            "source_manifest_sha256": manifest["source_manifest_sha256"],
            "task_id": task["id"],
            "prompt_sha256": manifest["prompt_identity"]["prompt_sha256"],
            "ports": PORTS,
            "ports_free_before": {str(port): port_is_free(port) for port in PORTS.values()},
            "gpu_policy": manifest["gpu_policy"],
            "gpu_before": gpu_snapshot(),
            "ports_before": port_snapshot(),
            "processes_before": process_snapshot(),
            "systemd_used": False,
            "production_interaction": False,
        },
    )

    results: dict[str, dict[str, Any]] = {}
    cleanups: dict[str, Any] = {}
    for variant in VARIANTS:
        if not port_is_free(PORTS[variant]):
            results[variant] = {"objective_status": "reviewer", "run_error": "port was not free before variant start"}
            continue
        results[variant] = run_variant(artifact, manifest, rubric_module, variant)
        cleanup_path = artifact / variant / "cleanup-verification.json"
        cleanups[variant] = json.loads(cleanup_path.read_text(encoding="utf-8")) if cleanup_path.exists() else {}

    summary = {
        "created_at_utc": utc_now(),
        "artifact": str(artifact.relative_to(ROOT)),
        "task_id": task["id"],
        "prompt_sha256": manifest["prompt_identity"]["prompt_sha256"],
        "identical_prompt_across_variants": True,
        "settings": manifest["common_sampling"],
        "normal_eos": manifest["common_sampling"]["ignore_eos"] is False,
        "thinking_disabled": manifest["common_sampling"]["chat_template_kwargs"] == {"enable_thinking": False},
        "sequential_variants": list(VARIANTS),
        "results": {
            variant: {
                "packing": PACKINGS[variant],
                "status": results.get(variant, {}).get("objective_status", "reviewer"),
                "http_status": results.get(variant, {}).get("http_status"),
                "finish_reason": results.get(variant, {}).get("finish_reason"),
                "truncated": results.get(variant, {}).get("truncated"),
                "tokens_predicted": results.get(variant, {}).get("tokens_predicted"),
                "response_sha256": results.get(variant, {}).get("response_sha256"),
                "rubric": results.get(variant, {}).get("rubric"),
                "run_error": results.get(variant, {}).get("run_error"),
                "cleanup": cleanups.get(variant),
            }
            for variant in VARIANTS
        },
        "interpretation": "A pass is valid JSON plus executable is_balanced tests; malformed JSON/code is a fail unless the HTTP/run result requires reviewer. Any length/limit finish is a truncation fail.",
        "systemd_used": False,
        "production_interaction": False,
    }
    write_json(artifact / "followup-summary.json", summary)
    lines = [
        "# Bonsai 2 T8 Max-Token Follow-Up",
        "",
        "Prompt: `T8-code-generation-test`",
        "Prompt SHA256: `%s`" % summary["prompt_sha256"],
        "",
        "Settings: `temperature=0`, `seed=1729`, thinking disabled, normal EOS, `max_tokens=2048`.",
        "Servers ran sequentially on physical GPU 1 only, on loopback ports 18211 and 18212, without systemd or production interaction.",
        "",
        "| Variant | Status | HTTP | Finish reason | Truncated | Predicted tokens |",
        "|---|---|---:|---|---|---:|",
    ]
    for variant in VARIANTS:
        item = summary["results"][variant]
        lines.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (item["packing"], item["status"], item["http_status"], item["finish_reason"], item["truncated"], item["tokens_predicted"])
        )
    lines.extend(
        [
            "",
            "The complete request, raw response, parsed response, objective rubric, server log, GPU proof, and cleanup evidence are under each variant directory.",
        ]
    )
    (artifact / "followup-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(artifact / "host-and-gpu-after.json", {"gpu": gpu_snapshot(), "ports": port_snapshot(), "processes": process_snapshot()})
    write_artifact_hashes(artifact)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if all(item.get("objective_status") in {"pass", "fail"} for item in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
