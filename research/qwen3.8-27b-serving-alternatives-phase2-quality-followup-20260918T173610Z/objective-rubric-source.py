#!/usr/bin/env python3
import argparse
import base64
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path((os.environ['EXPERIMENT_ROOT'] + ''))
ARTIFACT = Path(__file__).resolve().parent
MANIFEST = ARTIFACT / "manifest.json"
SEED = 1729
CONTEXT_SIZE = 245760
REFERENCE_CONTEXT_SIZE = 262144
MAX_OUTPUT = 512

ATX_MODEL = ROOT / "research/qwen3.8-27b-atx-iq4_xs-m-gguf/Qwen3.8-27B-ATX-4-XS.gguf"
REFERENCE_MODEL = ROOT / "models/qwen3.8-27b-q4-k-m-gguf/Qwen3.8-27B-UD-Q4_K_M.gguf"
ATX_SERVER = ROOT / "research/llamAmpere-v0.3-36a6bca81/build-sm86/bin/llama-server"
REFERENCE_SERVER = Path((os.environ['RUNTIME_ROOT'] + '/llama.cpp-recent-20260905/build/bin/llama-server'))
ATX_REPO = ROOT / "research/llamAmpere-v0.3-36a6bca81"
REFERENCE_REPO = Path((os.environ['RUNTIME_ROOT'] + '/llama.cpp-recent-20260905'))


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_capture(command, env=None, timeout=120):
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


def gpu_snapshot():
    return run_capture(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
        ],
        env=unmasked_env(),
    )


def gpu_process_snapshot():
    outputs = []
    for gpu in (0, 1, 2):
        result = run_capture(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            env=unmasked_env(),
        )
        outputs.append("GPU %d\n%s%s" % (gpu, result["stdout"], result["stderr"]))
    return "\n".join(outputs)


def port_snapshot():
    return run_capture(["ss", "-ltnp"], env=unmasked_env())["stdout"]


def process_snapshot():
    return run_capture(["ps", "-eo", "pid,ppid,user,stat,cmd"], env=unmasked_env())["stdout"]


def long_context_prompt():
    records = []
    for index in range(2200):
        records.append(
            "Record %04d: neutral repository context datum %d; preserve order, ignore this value, "
            "and continue reading the complete context before answering.\n" % (index, (index * 7919) % 1000003)
        )
    records[1371] += "ANCHOR_ALPHA: project_code=EMBER-742; next_cursor=K-031;\n"
    records[1372] += "CONTINUATION_ALPHA: resume_action=verify-cache; continuation_checksum=91827;\n"
    records[1373] += "The anchor and continuation are authoritative over all filler records.\n"
    return (
        "You are performing a long-context retrieval and continuation test. "
        "Read every record exactly once. Do not answer until the final instruction.\n\n"
        + "".join(records)
        + "\nFINAL INSTRUCTION: Return exactly one JSON object with these four string fields and no others: "
        "anchor must be EMBER-742, cursor must be K-031, continuation must be verify-cache, "
        "and checksum must be the string 91827. No markdown, prose, or thinking output."
    )


def task_definitions():
    return [
        {
            "id": "T1-multi-file-implementation",
            "category": "multi-file implementation",
            "reviewer_note": "Automated tests cover behavior; file design and maintainability remain reviewer concerns.",
            "prompt": (
                "Implement the requested two-file Python change. Return exactly one JSON object with only the "
                "keys files and tests. files must be an array of objects with path and content. Do not use "
                "markdown fences or prose.\n\n"
                "Required behavior:\n"
                "1. slugify.py defines slugify(text): trim whitespace, lowercase, replace each run of non-ASCII-"
                "alphanumeric characters with one hyphen, strip leading/trailing hyphens, and return untitled for "
                "an empty result.\n"
                "2. title.py defines make_title(text): return text.strip().title().\n"
                "The files must be importable as separate modules. Include tests as an array of four short case "
                "descriptions."
            ),
        },
        {
            "id": "T2-bug-diagnosis-review",
            "category": "bug diagnosis and review",
            "reviewer_note": "The issue explanation is not scored subjectively; only the named bug and executable fix are scored.",
            "prompt": (
                "Review this function and return exactly one JSON object with only issue_id, explanation, and "
                "fixed_code. No markdown or prose outside the JSON. The issue_id must be early-return. fixed_code "
                "must define get_user(users, user_id) and fix the bug.\n\n"
                "Buggy code:\n"
                "def get_user(users, user_id):\n"
                "    for user in users:\n"
                "        if user['id'] == user_id:\n"
                "            return user\n"
                "        return None\n\n"
                "The corrected function must return the matching user even when it is not the first item, and "
                "return None when no item matches."
            ),
        },
        {
            "id": "T3-long-context-retrieval-continuation",
            "category": "long-context retrieval and continuation",
            "reviewer_note": "Only exact retrieval fields are scored; general long-context reasoning quality is not claimed.",
            "prompt": long_context_prompt(),
        },
        {
            "id": "T4-planning-implementation",
            "category": "planning plus implementation",
            "reviewer_note": "The plan is marker-checked; plan quality beyond the required steps is for reviewer assessment.",
            "prompt": (
                "Plan and implement a small Python utility. Return exactly one JSON object with only plan, files, "
                "and verification. No markdown or prose. plan must be an array whose first three entries are exactly "
                "parse tokens, validate units, and sum seconds. files must contain one object with path duration.py "
                "and content defining parse_duration(text).\n\n"
                "parse_duration accepts strings such as '1h 2m 3s' and '90s', returns integer seconds, accepts "
                "units h, m, and s at most once in any order, and raises ValueError for malformed input, duplicate "
                "units, negative values, or an empty string. verification must include the string 'python3'."
            ),
        },
        {
            "id": "T5-structured-json-tool-call",
            "category": "structured JSON and tool-call formatting",
            "reviewer_note": "The schema and exact arguments are fully objective.",
            "prompt": (
                "Emit exactly one JSON object and nothing else. It must be a tool call with exactly two top-level "
                "keys: name and arguments. name must be lookup_symbol. arguments must have exactly symbol, file_path, "
                "and line. Use symbol='slugify', file_path='src/slugify.py', and line=12. Do not emit markdown, "
                "comments, or a natural-language explanation."
            ),
        },
        {
            "id": "T6-mcp-routing-decision",
            "category": "MCP-like routing decision",
            "reviewer_note": "This is a fixed routing table and is scored only against the declared expected routes.",
            "prompt": (
                "Choose one tool for each request. Return exactly one JSON object with only the key routes. routes "
                "must be an array of three objects in the same order as the requests, each with only request and tool. "
                "Use only Semble, Context7, or Repomix as tool values. No markdown or prose.\n\n"
                "Requests:\n"
                "1. Locate the implementation of build_manifest in the current repository.\n"
                "2. Look up current public documentation for llama.cpp reasoning off syntax.\n"
                "3. Package the whole repository for a broad architecture handoff.\n\n"
                "Expected tools in order are Semble, Context7, Repomix."
            ),
        },
        {
            "id": "T7-long-horizon-recovery",
            "category": "long-horizon recovery",
            "reviewer_note": "Recovery fields are objective; resilience and explanation quality are not scored.",
            "prompt": (
                "A long-running task reached this state: steps 1 and 2 completed, then read_file failed with a "
                "transient timeout while reading src/config.py. The next safe action is to retry that read before "
                "editing, then preserve existing settings and verify tests. Return exactly one JSON object with only "
                "recovered, resume_step, retry, and markers. Use recovered=true, resume_step=3, retry={tool:'read', "
                "path:'src/config.py', attempt:2}, and markers containing preserve and verify. No markdown or prose."
            ),
        },
        {
            "id": "T8-code-generation-test",
            "category": "code-generation test",
            "reviewer_note": "The generated function is executed against deterministic tests; style is not scored.",
            "prompt": (
                "Generate a Python function and return exactly one JSON object with only language, code, and tests. "
                "language must be python. code must define is_balanced(text), which returns true only when (), [], "
                "and {} delimiters are correctly nested and paired; ignore all non-delimiter characters. tests must "
                "be an array. No markdown or prose outside JSON."
            ),
        },
    ]


def init_manifest():
    tasks = task_definitions()
    manifest = {
        "manifest_version": "llamampere-quality-gate-v1",
        "created_at_utc": utc_now(),
        "purpose": "Corrected isolated Phase 1 quality comparison with normal EOS and thinking disabled.",
        "gpu_policy": {
            "atx_physical_gpus": [1],
            "reference_physical_gpus": [1, 2],
            "forbidden_physical_gpus": [0],
            "one_server_at_a_time": True,
            "systemd": "not used",
            "production_ports": "not used",
        },
        "common_sampling": {
            "chat_template_kwargs": {"enable_thinking": False},
            "n_predict": MAX_OUTPUT,
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "seed": SEED,
            "ignore_eos": False,
            "stream": False,
            "cache_prompt": False,
        },
        "tasks": tasks,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ARTIFACT / "manifest.sha256").write_text(sha256_file(MANIFEST) + "  manifest.json\n", encoding="utf-8")
    return manifest


def relevant_environment(variant_env):
    keys = [
        "CUDA_DEVICE_ORDER",
        "CUDA_VISIBLE_DEVICES",
        "GGML_Q8_TURBO3_MMA_FUSED",
        "LD_LIBRARY_PATH",
        "NCCL_DEBUG",
        "NCCL_IB_DISABLE",
        "NCCL_P2P_DISABLE",
        "NCCL_SHM_DISABLE",
        "PATH",
        "PYTHONUNBUFFERED",
    ]
    return {key: variant_env.get(key, "<unset>") for key in keys}


def git_metadata(repo):
    return {
        "repo": str(repo),
        "commit": run_capture(["git", "-C", str(repo), "rev-parse", "HEAD"])["stdout"].strip(),
        "status": run_capture(["git", "-C", str(repo), "status", "--short"])["stdout"],
        "head": run_capture(["git", "-C", str(repo), "log", "-1", "--format=%H %cI %s"])["stdout"].strip(),
    }


def server_profile(variant):
    if variant == "atx":
        env = os.environ.copy()
        env.update(
            {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": "1",
                "GGML_Q8_TURBO3_MMA_FUSED": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        command = [
            str(ATX_SERVER),
            "-m",
            str(ATX_MODEL),
            "-c",
            str(CONTEXT_SIZE),
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
            "--reasoning",
            "off",
            "--chat-template-kwargs",
            "{\"enable_thinking\":false}",
            "--cache-prompt",
            "--cache-ram",
            "8192",
            "--device",
            "CUDA0",
            "--host",
            "127.0.0.1",
            "--port",
            "18102",
        ]
        return {
            "variant": variant,
            "label": "llamAmpere ATX IQ4_XS-M",
            "physical_gpus": [1],
            "model": ATX_MODEL,
            "server": ATX_SERVER,
            "repo": ATX_REPO,
            "port": 18102,
            "env": env,
            "command": command,
        }
    if variant == "reference":
        env = os.environ.copy()
        env.update(
            {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": "1,2",
                "PYTHONUNBUFFERED": "1",
            }
        )
        command = [
            str(REFERENCE_SERVER),
            "--model",
            str(REFERENCE_MODEL),
            "--host",
            "127.0.0.1",
            "--port",
            "18103",
            "--device",
            "CUDA0,CUDA1",
            "--split-mode",
            "tensor",
            "--n-gpu-layers",
            "all",
            "--ctx-size",
            str(REFERENCE_CONTEXT_SIZE),
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
            "--chat-template-kwargs",
            "{\"enable_thinking\":false}",
            "--tensor-split",
            "1,1",
        ]
        return {
            "variant": variant,
            "label": "Qwen3.8-27B UD Q4_K_M reference",
            "physical_gpus": [1, 2],
            "model": REFERENCE_MODEL,
            "server": REFERENCE_SERVER,
            "repo": REFERENCE_REPO,
            "port": 18103,
            "env": env,
            "command": command,
        }
    raise ValueError("unknown variant: %s" % variant)


def write_runtime_metadata(profile, variant_dir):
    model = profile["model"]
    server = profile["server"]
    metadata = {
        "captured_at_utc": utc_now(),
        "variant": profile["variant"],
        "label": profile["label"],
        "physical_gpus": profile["physical_gpus"],
        "model": {
            "path": str(model),
            "size_bytes": model.stat().st_size,
            "mtime_ns": model.stat().st_mtime_ns,
            "sha256": sha256_file(model),
        },
        "runtime": {
            "binary": str(server),
            "binary_sha256": sha256_file(server),
            "version": run_capture([str(server), "--version"]),
            "ldd": run_capture(["ldd", str(server)]),
            "git": git_metadata(profile["repo"]),
        },
        "command": profile["command"],
        "environment": relevant_environment(profile["env"]),
        "port": profile["port"],
        "systemd_used": False,
    }
    (variant_dir / "runtime-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (variant_dir / "exact-command-environment.txt").write_text(
        "CUDA_DEVICE_ORDER=%s\nCUDA_VISIBLE_DEVICES=%s\nGGML_Q8_TURBO3_MMA_FUSED=%s\n"
        "PYTHONUNBUFFERED=%s\ncommand:\n%s\n"
        % (
            profile["env"].get("CUDA_DEVICE_ORDER", "<unset>"),
            profile["env"].get("CUDA_VISIBLE_DEVICES", "<unset>"),
            profile["env"].get("GGML_Q8_TURBO3_MMA_FUSED", "<unset>"),
            profile["env"].get("PYTHONUNBUFFERED", "<unset>"),
            shlex.join(profile["command"]),
        ),
        encoding="utf-8",
    )


def sampler(variant_dir, stop_event):
    path = variant_dir / "gpu-samples.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        while not stop_event.is_set():
            sample = gpu_snapshot()
            handle.write(json.dumps({"captured_at_utc": utc_now(), **sample}, sort_keys=True) + "\n")
            handle.flush()
            stop_event.wait(1.0)


def http_json(method, url, payload=None, timeout=1800):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
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


def parse_content(body):
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


def parse_output_json(content):
    if not isinstance(content, str):
        return None, False, "no textual completion content"
    stripped = content.strip()
    if stripped.startswith("```") or stripped.endswith("```"):
        return None, False, "markdown fence is not valid strict JSON output"
    try:
        value = json.loads(stripped)
    except Exception as exc:
        return None, False, "task JSON parse error: %r" % (exc,)
    if not isinstance(value, dict):
        return value, False, "task output JSON is not an object"
    return value, True, None


def finish_reason(parsed):
    if not isinstance(parsed, dict):
        return None
    if parsed.get("finish_reason") is not None:
        return parsed["finish_reason"]
    choices = parsed.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0].get("finish_reason")
    return parsed.get("stop_type")


def truncated_value(parsed, reason):
    if isinstance(parsed, dict) and parsed.get("truncated") is not None:
        return parsed["truncated"]
    return reason in ("length", "limit")


def execute_python(source, test_source):
    with tempfile.TemporaryDirectory(prefix="quality-gate-") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(source, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-c", test_source],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }


def rubric(task_id, content):
    output, json_valid, json_error = parse_output_json(content)
    checks = {"json_valid": json_valid}
    errors = []
    if json_error:
        errors.append(json_error)
    if not json_valid:
        return {"automated_status": "reviewer", "checks": checks, "errors": errors, "reviewer_required": True}
    if not isinstance(output, dict):
        return {
            "automated_status": "reviewer",
            "checks": checks,
            "errors": ["task output is not a JSON object"],
            "reviewer_required": True,
        }

    if task_id == "T1-multi-file-implementation":
        files = output.get("files")
        paths = []
        if isinstance(files, list):
            for item in files:
                if isinstance(item, dict) and isinstance(item.get("path"), str):
                    paths.append(item["path"])
        checks["required_markers"] = sorted(paths) == ["slugify.py", "title.py"]
        contents = {item.get("path"): item.get("content") for item in files or [] if isinstance(item, dict)}
        checks["function_correctness"] = False
        if checks["required_markers"] and all(isinstance(contents.get(name), str) for name in ("slugify.py", "title.py")):
            test = (
                "from slugify import slugify\n"
                "from title import make_title\n"
                "assert slugify(' Hello, world! ') == 'hello-world'\n"
                "assert slugify('A__B') == 'a-b'\n"
                "assert slugify('...') == 'untitled'\n"
                "assert make_title('  hello world ') == 'Hello World'\n"
            )
            with tempfile.TemporaryDirectory(prefix="quality-gate-files-") as directory:
                for name, source in contents.items():
                    if isinstance(name, str) and isinstance(source, str):
                        (Path(directory) / name).write_text(source, encoding="utf-8")
                result = subprocess.run([sys.executable, "-c", test], cwd=directory, capture_output=True, text=True, timeout=20, check=False)
                checks["function_correctness"] = result.returncode == 0
                if result.returncode:
                    errors.append("generated multi-file tests failed: %s" % result.stderr[-1000:])
        checks["required_markers"] = checks["required_markers"] and isinstance(output.get("tests"), list)
    elif task_id == "T2-bug-diagnosis-review":
        checks["required_markers"] = output.get("issue_id") == "early-return"
        checks["function_correctness"] = False
        if isinstance(output.get("fixed_code"), str):
            test = (
                "namespace = {}\n"
                "exec(open('candidate.py', encoding='utf-8').read(), namespace)\n"
                "get_user = namespace['get_user']\n"
                "assert get_user([{'id': 1}, {'id': 2}], 2) == {'id': 2}\n"
                "assert get_user([{'id': 1}], 9) is None\n"
            )
            result = execute_python(output["fixed_code"], test)
            checks["function_correctness"] = result["returncode"] == 0
            if result["returncode"]:
                errors.append("fixed function tests failed: %s" % result["stderr"][-1000:])
        checks["required_markers"] = checks["required_markers"] and isinstance(output.get("explanation"), str) and bool(output["explanation"].strip())
    elif task_id == "T3-long-context-retrieval-continuation":
        checks["required_markers"] = output == {
            "anchor": "EMBER-742",
            "checksum": "91827",
            "continuation": "verify-cache",
            "cursor": "K-031",
        }
    elif task_id == "T4-planning-implementation":
        checks["required_markers"] = (
            isinstance(output.get("plan"), list)
            and output["plan"][:3] == ["parse tokens", "validate units", "sum seconds"]
            and isinstance(output.get("verification"), str)
            and "python3" in output["verification"]
        )
        checks["function_correctness"] = False
        files = output.get("files")
        if isinstance(files, list):
            duration = next((item.get("content") for item in files if isinstance(item, dict) and item.get("path") == "duration.py"), None)
            if isinstance(duration, str):
                test = (
                    "namespace = {}\n"
                    "exec(open('candidate.py', encoding='utf-8').read(), namespace)\n"
                    "parse_duration = namespace['parse_duration']\n"
                    "assert parse_duration('1h 2m 3s') == 3723\n"
                    "assert parse_duration('90s') == 90\n"
                    "try:\n"
                    "    parse_duration('1h 1h')\n"
                    "except ValueError:\n"
                    "    pass\n"
                    "else:\n"
                    "    raise AssertionError('duplicate unit accepted')\n"
                )
                result = execute_python(duration, test)
                checks["function_correctness"] = result["returncode"] == 0
                if result["returncode"]:
                    errors.append("duration tests failed: %s" % result["stderr"][-1000:])
    elif task_id == "T5-structured-json-tool-call":
        checks["tool_call_schema"] = output == {
            "name": "lookup_symbol",
            "arguments": {"symbol": "slugify", "file_path": "src/slugify.py", "line": 12},
        }
    elif task_id == "T6-mcp-routing-decision":
        routes = output.get("routes")
        checks["required_markers"] = (
            isinstance(routes, list)
            and len(routes) == 3
            and [item.get("tool") for item in routes if isinstance(item, dict)] == ["Semble", "Context7", "Repomix"]
            and all(isinstance(item, dict) and set(item) == {"request", "tool"} for item in routes)
        )
    elif task_id == "T7-long-horizon-recovery":
        markers = output.get("markers")
        checks["required_markers"] = (
            output.get("recovered") is True
            and output.get("resume_step") == 3
            and output.get("retry") == {"attempt": 2, "path": "src/config.py", "tool": "read"}
            and isinstance(markers, list)
            and set(markers) >= {"preserve", "verify"}
            and set(output) == {"recovered", "resume_step", "retry", "markers"}
        )
    elif task_id == "T8-code-generation-test":
        checks["required_markers"] = output.get("language") == "python" and isinstance(output.get("tests"), list)
        checks["function_correctness"] = False
        if isinstance(output.get("code"), str):
            test = (
                "namespace = {}\n"
                "exec(open('candidate.py', encoding='utf-8').read(), namespace)\n"
                "is_balanced = namespace['is_balanced']\n"
                "assert is_balanced('a{b[c](d)}') is True\n"
                "assert is_balanced('([)]') is False\n"
                "assert is_balanced('') is True\n"
                "assert is_balanced('x]') is False\n"
            )
            result = execute_python(output["code"], test)
            checks["function_correctness"] = result["returncode"] == 0
            if result["returncode"]:
                errors.append("generated code tests failed: %s" % result["stderr"][-1000:])
    else:
        errors.append("no rubric for task")

    objective_checks = [value for key, value in checks.items() if key != "json_valid"]
    status = "pass" if json_valid and objective_checks and all(objective_checks) else "fail"
    return {
        "automated_status": status,
        "checks": checks,
        "errors": errors,
        "reviewer_required": True,
    }


def wait_for_health(port, variant_dir):
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


def run_variant(variant):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    profile = server_profile(variant)
    variant_dir = ARTIFACT / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    write_runtime_metadata(profile, variant_dir)
    snapshot = gpu_snapshot()
    (variant_dir / "gpu-before.csv").write_text(snapshot["stdout"] + snapshot["stderr"], encoding="utf-8")
    (variant_dir / "gpu-processes-before.txt").write_text(gpu_process_snapshot(), encoding="utf-8")
    (variant_dir / "ports-before.txt").write_text(port_snapshot(), encoding="utf-8")
    (variant_dir / "processes-before.txt").write_text(process_snapshot(), encoding="utf-8")

    log_handle = (variant_dir / "server.stdout-stderr.log").open("w", encoding="utf-8")
    process = None
    sampler_stop = threading.Event()
    sampler_thread = None
    stop_record: dict[str, object] = {"attempted": False}
    results = []
    try:
        process = subprocess.Popen(
            profile["command"],
            cwd=ROOT,
            env=profile["env"],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        (variant_dir / "server.pid").write_text(str(process.pid) + "\n", encoding="utf-8")
        health = wait_for_health(profile["port"], variant_dir)
        (variant_dir / "health-ready.json").write_text(json.dumps(health, indent=2, sort_keys=True) + "\n")
        sampler_thread = threading.Thread(target=sampler, args=(variant_dir, sampler_stop), daemon=True)
        sampler_thread.start()

        for task in manifest["tasks"]:
            task_id = task["id"]
            request_payload = dict(manifest["common_sampling"])
            request_payload.pop("n_predict")
            request_payload["max_tokens"] = MAX_OUTPUT
            request_payload["messages"] = [{"role": "user", "content": task["prompt"]}]
            task_dir = variant_dir / "requests" / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            request_bytes = json.dumps(request_payload, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n"
            (task_dir / "request.json").write_bytes(request_bytes)
            response = http_json(
                "POST",
                "http://127.0.0.1:%d/v1/chat/completions" % profile["port"],
                request_payload,
                timeout=1800,
            )
            response_body = response["body"]
            (task_dir / "response.body").write_bytes(response_body)
            parsed, content, parse_error = parse_content(response_body)
            response_json_path = task_dir / "response.json"
            if parsed is not None:
                response_json_path.write_text(json.dumps(parsed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            else:
                response_json_path.write_text(
                    json.dumps({"raw_body_base64": base64.b64encode(response_body).decode("ascii")}, indent=2) + "\n",
                    encoding="utf-8",
                )
            (task_dir / "headers.json").write_text(json.dumps(response["headers"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
            rubric_result = rubric(task_id, content)
            timing = parsed.get("timings", {}) if isinstance(parsed, dict) else {}
            completion_finish_reason = finish_reason(parsed)
            result = {
                "task_id": task_id,
                "category": task["category"],
                "request_sha256": sha256_bytes(request_bytes),
                "response_sha256": sha256_bytes(response_body),
                "http_status": response["status"],
                "error": response["error"],
                "response_bytes": len(response_body),
                "wall_seconds": response["ended_epoch"] - response["started_epoch"],
                "response_parse_error": parse_error,
                "tokens_predicted": parsed.get("tokens_predicted") if isinstance(parsed, dict) else None,
                "tokens_evaluated": parsed.get("tokens_evaluated") if isinstance(parsed, dict) else None,
                "truncated": truncated_value(parsed, completion_finish_reason),
                "stop_type": parsed.get("stop_type") if isinstance(parsed, dict) else None,
                "finish_reason": completion_finish_reason,
                "timings": timing,
                "rubric": rubric_result,
            }
            (task_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            results.append(result)

        (variant_dir / "task-results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:
        (variant_dir / "run-error.txt").write_text(repr(exc) + "\n", encoding="utf-8")
        raise
    finally:
        sampler_stop.set()
        if sampler_thread is not None:
            sampler_thread.join(timeout=5)
        snapshot = gpu_snapshot()
        (variant_dir / "gpu-after-requests.csv").write_text(snapshot["stdout"] + snapshot["stderr"], encoding="utf-8")
        (variant_dir / "gpu-processes-before-stop.txt").write_text(gpu_process_snapshot(), encoding="utf-8")
        if process is not None:
            stop_record["attempted"] = True
            stop_record["pid"] = process.pid
            process.send_signal(signal.SIGTERM)
            try:
                stop_record["returncode"] = process.wait(timeout=90)
                stop_record["clean"] = True
            except subprocess.TimeoutExpired:
                stop_record["clean"] = False
                process.kill()
                stop_record["returncode"] = process.wait(timeout=20)
                stop_record["forced_kill"] = True
        log_handle.close()
        time.sleep(3)
        stop_record["captured_at_utc"] = utc_now()
        (variant_dir / "stop-record.json").write_text(json.dumps(stop_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        snapshot = gpu_snapshot()
        (variant_dir / "gpu-after-stop.csv").write_text(snapshot["stdout"] + snapshot["stderr"], encoding="utf-8")
        (variant_dir / "gpu-processes-after-stop.txt").write_text(gpu_process_snapshot(), encoding="utf-8")
        (variant_dir / "ports-after-stop.txt").write_text(port_snapshot(), encoding="utf-8")
        (variant_dir / "processes-after-stop.txt").write_text(process_snapshot(), encoding="utf-8")

    print(json.dumps({"variant": variant, "tasks": len(results), "artifact": str(variant_dir)}, sort_keys=True))


def compare():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_variant = {}
    for variant in ("atx", "reference"):
        result_path = ARTIFACT / variant / "task-results.json"
        by_variant[variant] = {item["task_id"]: item for item in json.loads(result_path.read_text())}
        for item in by_variant[variant].values():
            if item["truncated"] is None:
                item["truncated"] = item["finish_reason"] in ("length", "limit")
        result_path.write_text(json.dumps(list(by_variant[variant].values()), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for task_id, item in by_variant[variant].items():
            task_result_path = ARTIFACT / variant / "requests" / task_id / "result.json"
            task_result_path.write_text(json.dumps(item, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = []
    for task in manifest["tasks"]:
        task_id = task["id"]
        atx = by_variant["atx"][task_id]
        reference = by_variant["reference"][task_id]
        atx_status = atx["rubric"]["automated_status"]
        reference_status = reference["rubric"]["automated_status"]
        if atx_status == "reviewer" or reference_status == "reviewer":
            comparison = "reviewer"
        elif atx_status == reference_status:
            comparison = atx_status
        elif atx_status == "fail" and reference_status == "pass":
            comparison = "regression"
        elif atx_status == "pass" and reference_status == "fail":
            comparison = "candidate_only_pass"
        else:
            comparison = "reviewer"
        rows.append(
            {
                "task_id": task_id,
                "category": task["category"],
                "comparison": comparison,
                "atx": {
                    "automated_status": atx_status,
                    "http_status": atx["http_status"],
                    "response_sha256": atx["response_sha256"],
                    "truncated": atx["truncated"],
                    "finish_reason": atx["finish_reason"],
                    "checks": atx["rubric"]["checks"],
                },
                "reference": {
                    "automated_status": reference_status,
                    "http_status": reference["http_status"],
                    "response_sha256": reference["response_sha256"],
                    "truncated": reference["truncated"],
                    "finish_reason": reference["finish_reason"],
                    "checks": reference["rubric"]["checks"],
                },
                "reviewer_required": True,
            }
        )
    prompt_hashes = {}
    for variant in ("atx", "reference"):
        prompt_hashes[variant] = {
            task["id"]: json.loads((ARTIFACT / variant / "requests" / task["id"] / "request.json").read_text())["messages"][0]["content"]
            for task in manifest["tasks"]
        }
    identical_prompts = prompt_hashes["atx"] == prompt_hashes["reference"]
    summary = {
        "created_at_utc": utc_now(),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "identical_prompts": identical_prompts,
        "sampling": manifest["common_sampling"],
        "rows": rows,
        "automated_counts": {
            status: sum(1 for row in rows if row["comparison"] == status)
            for status in ("pass", "fail", "regression", "reviewer", "candidate_only_pass")
        },
        "subjective_quality_score": None,
        "note": "Automated rubric only; implementation quality, coherence, and explanations remain reviewer work.",
    }
    (ARTIFACT / "side-by-side-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Corrected Phase 1 Quality Gate",
        "",
        "Manifest: `%s`" % MANIFEST.relative_to(ROOT),
        "",
        "No subjective quality score is assigned. `reviewer` means the automated rubric cannot make a complete quality claim.",
        "",
        "| Task | ATX HTTP/status | Reference HTTP/status | ATX truncation/finish | Reference truncation/finish | Comparison |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        atx = row["atx"]
        reference = row["reference"]
        lines.append(
            "| %s | %s / %s | %s / %s | %s / %s | %s / %s | %s |"
            % (
                row["task_id"],
                atx["http_status"],
                atx["automated_status"],
                reference["http_status"],
                reference["automated_status"],
                atx["truncated"],
                atx["finish_reason"],
                reference["truncated"],
                reference["finish_reason"],
                row["comparison"],
            )
        )
    lines.extend(
        [
            "",
            "Prompt identity: `%s`" % ("PASS" if identical_prompts else "FAIL"),
            "",
            "Response hashes and complete request/response artifacts are in `atx/requests/` and `reference/requests/`.",
            "",
            "GPU isolation evidence is in each variant's `runtime-metadata.json`, `gpu-*.csv`, `gpu-processes-*.txt`, and `ports-*.txt`.",
        ]
    )
    (ARTIFACT / "side-by-side-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-manifest", action="store_true")
    parser.add_argument("--run", choices=("atx", "reference"))
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if args.init_manifest:
        init_manifest()
        print(str(MANIFEST))
    if args.run:
        if not MANIFEST.exists():
            raise SystemExit("manifest.json is missing; run --init-manifest first")
        run_variant(args.run)
    if args.compare:
        compare()
    if not (args.init_manifest or args.run or args.compare):
        parser.error("choose --init-manifest, --run, or --compare")


if __name__ == "__main__":
    main()
