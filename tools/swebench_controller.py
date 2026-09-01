#!/usr/bin/env python3
"""Durable paired SWE-bench controller for the Q4/Q5 comparison."""

import argparse
import collections
import contextlib
import fcntl
import hashlib
import json
import os
import signal
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from datasets import load_dataset

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(os.environ.get("SWEBENCH_ROOT", Path(__file__).resolve().parents[1]))
TOOLS = ROOT / "tools"
VENV = Path(os.environ.get("SWEBENCH_VENV", sys.prefix))
MINI_EXTRA = VENV / "bin/mini-extra"
STOCK_CONFIG = VENV / "lib/python3.12/site-packages/minisweagent/config/benchmarks/swebench.yaml"
RUNTIME_CONFIG = Path(os.environ.get("SWEBENCH_RUNTIME_CONFIG", ROOT / "swebench/configs/runtime-stock.yaml"))
COMMON_CONFIG = ROOT / "swebench/configs/common.yaml"
OVERLAYS = {"q4": ROOT / "swebench/configs/q4-128k.yaml", "q5": ROOT / "swebench/configs/q5-128k.yaml"}
GOLD_MARKER = ROOT / "swebench/gold/verified40.validation.json"
MAX_INFRA_RETRIES = 1
HEARTBEAT_SECONDS = 30
STOP_REQUESTED = False
ACTIVE_PROCESSES: dict[str, subprocess.Popen] = {}
ACTIVE_PROCESSES_LOCK = threading.Lock()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = manifest.get("manifest_sha256")
    if not expected:
        raise RuntimeError("manifest has no manifest_sha256")
    body = dict(manifest)
    body.pop("manifest_sha256")
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    actual = hashlib.sha256(canonical).hexdigest()
    if actual != expected:
        raise RuntimeError(f"manifest hash mismatch: expected {expected}, computed {actual}")
    if len(manifest.get("tasks", [])) != 40:
        raise RuntimeError("manifest must contain exactly 40 tasks")
    if len({task["instance_id"] for task in manifest["tasks"]}) != 40:
        raise RuntimeError("manifest contains duplicate task IDs")
    return manifest


def connect_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS run_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tasks (
            model TEXT NOT NULL,
            task_order INTEGER NOT NULL,
            task_id TEXT NOT NULL,
            state TEXT NOT NULL,
            attempt INTEGER NOT NULL DEFAULT 0,
            started_at TEXT,
            heartbeat_at TEXT,
            finished_at TEXT,
            exit_status TEXT,
            error_class TEXT,
            trajectory_path TEXT,
            stdout_path TEXT,
            submission_chars INTEGER,
            agent_calls INTEGER,
            max_context_tokens INTEGER,
            finish_reasons TEXT,
            wall_seconds REAL,
            PRIMARY KEY (model, task_id)
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            model TEXT,
            task_id TEXT,
            event TEXT NOT NULL,
            detail TEXT
        );
        """
    )
    return connection


def record_event(connection: sqlite3.Connection, event: str, model: str | None = None, task_id: str | None = None, detail: str = "") -> None:
    connection.execute(
        "INSERT INTO events(created_at, model, task_id, event, detail) VALUES (?, ?, ?, ?, ?)",
        (utc_now(), model, task_id, event, detail),
    )
    connection.commit()


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def init_run(manifest_path: Path, run_dir: Path, task_ids: list[str] | None = None) -> None:
    manifest = load_manifest(manifest_path)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"run directory is not empty: {run_dir}")
    selected_tasks = manifest["tasks"]
    if task_ids:
        task_by_id = {task["instance_id"]: task for task in manifest["tasks"]}
        missing = sorted(set(task_ids) - task_by_id.keys())
        if missing or len(set(task_ids)) != len(task_ids):
            raise RuntimeError(f"unknown task IDs: {missing}")
        selected_tasks = [task_by_id[task_id] for task_id in task_ids]
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, run_dir / "manifest.json")
    (run_dir / "configs").mkdir()
    if not STOCK_CONFIG.is_file():
        raise RuntimeError(f"stock mini-SWE config is missing: {STOCK_CONFIG}")
    if not RUNTIME_CONFIG.is_file():
        raise RuntimeError(f"runtime config is missing: {RUNTIME_CONFIG}")
    shutil.copy2(STOCK_CONFIG, run_dir / "configs/swebench-stock.yaml")
    shutil.copy2(RUNTIME_CONFIG, run_dir / "configs/runtime.yaml")
    for model, overlay in OVERLAYS.items():
        shutil.copy2(overlay, run_dir / f"configs/{model}-128k.yaml")

    connection = connect_db(run_dir / "ledger.sqlite3")
    connection.executemany(
        "INSERT INTO run_meta(key, value) VALUES (?, ?)",
        [
            ("manifest_sha256", manifest["manifest_sha256"]),
            ("created_at", utc_now()),
            ("controller_version", "2"),
            ("stock_config_sha256", file_sha256(STOCK_CONFIG)),
            ("runtime_config_sha256", file_sha256(RUNTIME_CONFIG)),
            ("selected_task_ids", json.dumps([task["instance_id"] for task in selected_tasks])),
        ],
    )
    task_rows = []
    for model in ("q4", "q5"):
        task_rows.extend(
            (
                model,
                selection_order,
                task["instance_id"],
                "queued",
            )
            for selection_order, task in enumerate(selected_tasks)
        )
    connection.executemany(
        "INSERT INTO tasks(model, task_order, task_id, state) VALUES (?, ?, ?, ?)", task_rows
    )
    record_event(connection, "run_initialized", detail=manifest["manifest_sha256"])
    connection.close()
    print(f"initialized {run_dir}")
    print(f"manifest_sha256 {manifest['manifest_sha256']}")


def get_run_manifest(run_dir: Path) -> dict:
    return load_manifest(run_dir / "manifest.json")


def endpoint_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"accept": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def preflight(run_dir: Path, check_endpoints: bool = True) -> None:
    manifest = get_run_manifest(run_dir)
    failures: list[str] = []

    try:
        cached_dataset = load_dataset(manifest["dataset"]["name"], split=manifest["dataset"]["split"])
        if cached_dataset._fingerprint != manifest["dataset"]["fingerprint"]:
            failures.append(
                f"cached dataset fingerprint mismatch: {cached_dataset._fingerprint} != {manifest['dataset']['fingerprint']}"
            )
        cached_rows = {row["instance_id"]: row for row in cached_dataset}
        for task in manifest["tasks"]:
            if cached_rows.get(task["instance_id"], {}).get("base_commit") != task["base_commit"]:
                failures.append(f"cached dataset base commit mismatch: {task['instance_id']}")
    except Exception as error:
        failures.append(f"frozen dataset cache unavailable: {error}")

    try:
        gold_marker = json.loads(GOLD_MARKER.read_text(encoding="utf-8"))
        if gold_marker.get("manifest_sha256") != manifest["manifest_sha256"]:
            failures.append("gold validation belongs to a different manifest")
        if gold_marker.get("validated_tasks") != 40:
            failures.append("gold validation does not cover all 40 tasks")
    except (OSError, json.JSONDecodeError, TypeError):
        failures.append(f"gold validation marker is missing: {GOLD_MARKER}")

    if shutil.which("docker") is None:
        failures.append("docker executable is missing")
    else:
        try:
            subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=30)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            failures.append(f"docker daemon unavailable: {error}")

    for model in ("q4", "q5"):
        spec = manifest["models"][model]
        model_path = Path(spec["path"])
        if not model_path.is_file():
            failures.append(f"{model} model file missing: {model_path}")
        elif file_sha256(model_path) != spec["sha256"]:
            failures.append(f"{model} model hash mismatch")
        if check_endpoints:
            try:
                health = endpoint_json(f"{spec['endpoint'][:-3]}/health")
                if health.get("status") != "ok":
                    failures.append(f"{model} health is not ok: {health}")
                models = endpoint_json(f"{spec['endpoint']}/models")
                model_ids = {item.get("id") for item in models.get("data", [])}
                if spec["served_model"] not in model_ids:
                    failures.append(f"{model} endpoint does not serve {spec['served_model']}: {model_ids}")
            except (OSError, urllib.error.URLError, ValueError) as error:
                failures.append(f"{model} endpoint unavailable: {error}")

    try:
        stock_path = run_dir / "configs/swebench-stock.yaml"
        runtime_path = run_dir / "configs/runtime.yaml"
        if stock_path.is_file() and runtime_path.is_file():
            stock = yaml.safe_load(stock_path.read_text(encoding="utf-8"))
            runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
            if "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt" not in stock["agent"]["instance_template"]:
                failures.append("stock config does not contain the patch-emitting submission protocol")
            if stock["agent"]["step_limit"] != 250:
                failures.append("stock config step_limit is not 250")
            if runtime["environment"]["run_args"] != ["--rm", "--network", "none"]:
                failures.append("runtime config does not enforce --network none")
            if runtime["environment"]["timeout"] != 600:
                failures.append("runtime config command timeout is not 600 seconds")
            if runtime["model"]["model_kwargs"]["max_tokens"] != 8192:
                failures.append("runtime config max_tokens is not 8192")
            if runtime["model"]["model_kwargs"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is not False:
                failures.append("runtime config does not disable Qwen thinking")
            if runtime["agent"]["wall_time_limit_seconds"] != 7200:
                failures.append("runtime config task timeout is not 7200 seconds")
        else:
            common = yaml.safe_load((run_dir / "configs/common.yaml").read_text(encoding="utf-8"))
            if common["environment"]["run_args"] != ["--rm", "--network", "none"]:
                failures.append("common config does not enforce --network none")
            if common["environment"]["timeout"] != 600:
                failures.append("common config command timeout is not 600 seconds")
            if common["model"]["model_kwargs"]["max_tokens"] != 8192:
                failures.append("common config max_tokens is not 8192")
            if common["model"]["model_kwargs"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is not False:
                failures.append("common config does not disable Qwen thinking")
            if common["agent"]["wall_time_limit_seconds"] != 7200:
                failures.append("common config task timeout is not 7200 seconds")
        for model in ("q4", "q5"):
            overlay = yaml.safe_load((run_dir / f"configs/{model}-128k.yaml").read_text(encoding="utf-8"))
            expected_model = manifest["models"][model]["served_model"]
            expected_endpoint = manifest["models"][model]["endpoint"]
            if overlay["model"]["model_name"] != f"hosted_vllm/{expected_model}":
                failures.append(f"{model} config model name does not match manifest")
            if overlay["model"]["model_kwargs"]["api_base"] != expected_endpoint:
                failures.append(f"{model} config endpoint does not match manifest")
    except (OSError, KeyError, TypeError, yaml.YAMLError) as error:
        failures.append(f"config validation failed: {error}")

    if failures:
        print("PREFLIGHT FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("PREFLIGHT OK")
    print(f"manifest_sha256 {manifest['manifest_sha256']}")
    print("docker: available")
    if check_endpoints:
        print("endpoints: q4=ok q5=ok")
    else:
        print("endpoints: deferred")
    print("network: none")


def reconcile(connection: sqlite3.Connection) -> None:
    now = utc_now()
    rows = connection.execute("SELECT model, task_id FROM tasks WHERE state = 'running'").fetchall()
    connection.execute(
        "UPDATE tasks SET state = 'interrupted', finished_at = ?, error_class = 'controller_restart' WHERE state = 'running'",
        (now,),
    )
    for row in rows:
        record_event(connection, "task_reconciled", row["model"], row["task_id"], "running -> interrupted")
    connection.commit()


def claim_task(connection: sqlite3.Connection, model: str) -> sqlite3.Row | None:
    connection.execute("BEGIN IMMEDIATE")
    row = connection.execute(
        "SELECT * FROM tasks WHERE model = ? AND state IN ('queued', 'interrupted') ORDER BY task_order LIMIT 1",
        (model,),
    ).fetchone()
    if row is None:
        connection.commit()
        return None
    now = utc_now()
    connection.execute(
        "UPDATE tasks SET state = 'running', attempt = attempt + 1, started_at = ?, heartbeat_at = ?, finished_at = NULL, error_class = NULL WHERE model = ? AND task_id = ?",
        (now, now, model, row["task_id"]),
    )
    record_event(connection, "task_started", model, row["task_id"], f"attempt={row['attempt'] + 1}")
    connection.commit()
    return connection.execute(
        "SELECT * FROM tasks WHERE model = ? AND task_id = ?", (model, row["task_id"])
    ).fetchone()


def update_heartbeat(connection: sqlite3.Connection, model: str, task_id: str) -> None:
    connection.execute(
        "UPDATE tasks SET heartbeat_at = ? WHERE model = ? AND task_id = ?", (utc_now(), model, task_id)
    )
    connection.commit()


def classify_failure(
    returncode: int | None,
    output: str,
    timed_out: bool = False,
    exit_status: str = "",
) -> tuple[str, bool]:
    if timed_out:
        return "task_timeout", False
    if exit_status == "LimitsExceeded":
        return "step_limit", False
    if exit_status == "TimeExceeded":
        return "task_timeout", False
    if exit_status == "Submitted":
        return "submission_protocol_error", False
    lower = output.lower()
    retryable_patterns = (
        "cannot connect to the docker daemon",
        "is the docker daemon running",
        "no such file or directory: 'docker'",
        "connection refused",
        "apiconnectionerror",
        "cuda out of memory",
        "server process died",
        "no space left on device",
        "input/output error",
        "context length",
        "maximum context",
        "prompt is too long",
        "exceeds the available context",
    )
    if any(pattern in lower for pattern in retryable_patterns):
        if "docker" in lower:
            return "infra_docker", True
        if "cuda" in lower or "server process" in lower or "connection refused" in lower:
            return "infra_model_server", True
        if "space" in lower or "input/output" in lower:
            return "infra_storage", True
        if "context" in lower or "prompt" in lower:
            return "context_exceeded", False
        return "infra_transport", True
    if returncode is not None and returncode != 0:
        return "agent_process", False
    return "agent_no_submission", False


def terminate_process(process: subprocess.Popen) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def request_stop(_signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    with ACTIVE_PROCESSES_LOCK:
        for process in ACTIVE_PROCESSES.values():
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)


def trajectory_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    info = data.get("info", {})
    messages = data.get("messages", [])
    finish_reasons = collections.Counter()
    max_context = 0
    completion_tokens = 0
    agent_calls = 0
    for message in messages:
        if message.get("role") == "assistant":
            agent_calls += 1
        response = message.get("extra", {}).get("response", {})
        if not isinstance(response, dict):
            continue
        usage = response.get("usage", {}) or {}
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        generated = usage.get("completion_tokens", 0) or 0
        max_context = max(max_context, prompt_tokens + generated)
        completion_tokens += generated
        choices = response.get("choices", []) or []
        for choice in choices:
            if choice.get("finish_reason"):
                finish_reasons[choice["finish_reason"]] += 1
    submission = info.get("submission") or ""
    return {
        "exit_status": info.get("exit_status", ""),
        "submission": submission,
        "submission_chars": len(submission),
        "agent_calls": agent_calls,
        "max_context_tokens": max_context,
        "completion_tokens": completion_tokens,
        "finish_reasons": dict(finish_reasons),
    }


def write_predictions(connection: sqlite3.Connection, run_dir: Path, model: str) -> None:
    predictions = []
    for row in connection.execute(
        "SELECT * FROM tasks WHERE model = ? AND state IN ('completed', 'failed', 'infra_failed') ORDER BY task_order",
        (model,),
    ):
        patch = ""
        trajectory = row["trajectory_path"]
        if trajectory and Path(trajectory).is_file():
            with contextlib.suppress(OSError, json.JSONDecodeError):
                patch = json.loads(Path(trajectory).read_text(encoding="utf-8")).get("info", {}).get("submission", "") or ""
        predictions.append(
            {
                "model_name_or_path": model,
                "instance_id": row["task_id"],
                "model_patch": patch,
            }
        )
    atomic_write(run_dir / model / "preds.jsonl", "\n".join(json.dumps(row) for row in predictions) + ("\n" if predictions else ""))


def execute_task(connection: sqlite3.Connection, run_dir: Path, model: str, task: sqlite3.Row) -> None:
    task_id = task["task_id"]
    task_dir = run_dir / model / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = task_dir / f"{task_id}.traj.json"
    stdout_path = task_dir / "agent.log"
    config_paths = [
        run_dir / "configs/swebench-stock.yaml",
        run_dir / "configs/runtime.yaml",
        run_dir / f"configs/{model}-128k.yaml",
    ]
    if not all(path.is_file() for path in config_paths):
        config_paths = [run_dir / "configs/common.yaml", run_dir / f"configs/{model}-128k.yaml"]
    command = [
        str(MINI_EXTRA),
        "swebench-single",
        "--subset",
        "verified",
        "--split",
        "test",
        "--instance",
        task_id,
    ]
    for config_path in config_paths:
        command.extend(["--config", str(config_path)])
    command.extend([
        "--output",
        str(trajectory_path),
        "--yolo",
        "--exit-immediately",
        "--cost-limit",
        "0",
    ])
    started = time.monotonic()
    timed_out = False
    returncode = None
    with stdout_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=run_dir,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={
                **os.environ,
                "MSWEA_COST_TRACKING": "ignore_errors",
                "TQDM_DISABLE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
            },
        )
        process_key = f"{model}:{task_id}"
        with ACTIVE_PROCESSES_LOCK:
            ACTIVE_PROCESSES[process_key] = process
        try:
            while True:
                returncode = process.poll()
                if returncode is not None:
                    break
                if STOP_REQUESTED:
                    terminate_process(process)
                    returncode = process.returncode
                    break
                if time.monotonic() - started >= 7200:
                    timed_out = True
                    terminate_process(process)
                    returncode = process.returncode
                    break
                time.sleep(HEARTBEAT_SECONDS)
                update_heartbeat(connection, model, task_id)
        finally:
            with ACTIVE_PROCESSES_LOCK:
                ACTIVE_PROCESSES.pop(process_key, None)

    output = stdout_path.read_text(encoding="utf-8", errors="replace")
    metrics = None
    if trajectory_path.is_file():
        with contextlib.suppress(OSError, json.JSONDecodeError, KeyError):
            metrics = trajectory_metrics(trajectory_path)
    if STOP_REQUESTED:
        state = "interrupted"
        error_class = "controller_stop"
    elif metrics and metrics["exit_status"] == "Submitted" and metrics["submission"]:
        state = "completed"
        error_class = None
    else:
        error_class, retryable = classify_failure(
            returncode,
            output,
            timed_out,
            metrics["exit_status"] if metrics else "",
        )
        if retryable and task["attempt"] <= MAX_INFRA_RETRIES:
            state = "queued"
        elif retryable:
            state = "infra_failed"
        else:
            state = "failed"

    wall_seconds = time.monotonic() - started
    connection.execute(
        """UPDATE tasks SET state = ?, finished_at = ?, heartbeat_at = ?, exit_status = ?, error_class = ?,
           trajectory_path = ?, stdout_path = ?, submission_chars = ?, agent_calls = ?, max_context_tokens = ?,
           finish_reasons = ?, wall_seconds = ? WHERE model = ? AND task_id = ?""",
        (
            state,
            utc_now(),
            utc_now(),
            metrics["exit_status"] if metrics else str(returncode),
            error_class,
            str(trajectory_path) if trajectory_path.is_file() else None,
            str(stdout_path),
            metrics["submission_chars"] if metrics else 0,
            metrics["agent_calls"] if metrics else 0,
            metrics["max_context_tokens"] if metrics else 0,
            json.dumps(metrics["finish_reasons"], sort_keys=True) if metrics else "{}",
            wall_seconds,
            model,
            task_id,
        ),
    )
    record_event(connection, "task_finished", model, task_id, f"state={state} error={error_class}")
    connection.commit()
    write_predictions(connection, run_dir, model)
    print(f"{model} {task_id} {state} {wall_seconds:.1f}s", flush=True)


def worker(run_dir: Path, model: str) -> None:
    connection = connect_db(run_dir / "ledger.sqlite3")
    while not STOP_REQUESTED:
        task = claim_task(connection, model)
        if task is None:
            break
        execute_task(connection, run_dir, model, task)
    connection.close()


@contextlib.contextmanager
def controller_lock(run_dir: Path):
    lock_path = run_dir / "controller.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def run_both(run_dir: Path) -> None:
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    with controller_lock(run_dir):
        preflight(run_dir)
        connection = connect_db(run_dir / "ledger.sqlite3")
        reconcile(connection)
        connection.close()
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(worker, run_dir, model) for model in ("q4", "q5")]
            for future in futures:
                future.result()
    if STOP_REQUESTED:
        print("paired run interrupted")
    else:
        print("paired run complete")


def status(run_dir: Path) -> None:
    connection = connect_db(run_dir / "ledger.sqlite3")
    print(f"run_dir {run_dir}")
    for model in ("q4", "q5"):
        rows = connection.execute("SELECT state, COUNT(*) AS count FROM tasks WHERE model = ? GROUP BY state", (model,)).fetchall()
        counts = {row["state"]: row["count"] for row in rows}
        print(f"{model} {json.dumps(counts, sort_keys=True)}")
        current = connection.execute(
            "SELECT task_id, state, attempt, heartbeat_at FROM tasks WHERE model = ? AND state = 'running' ORDER BY task_order",
            (model,),
        ).fetchall()
        for row in current:
            print(f"{model} running task={row['task_id']} attempt={row['attempt']} heartbeat={row['heartbeat_at']}")
    connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--manifest", type=Path, required=True)
    init_parser.add_argument("--run-dir", type=Path, required=True)
    init_parser.add_argument("--task-id", action="append", default=[])

    for name in ("preflight", "run-both", "status"):
        command_parser = subparsers.add_parser(name)
        command_parser.add_argument("--run-dir", type=Path, required=True)
        if name == "preflight":
            command_parser.add_argument("--skip-endpoints", action="store_true")

    args = parser.parse_args()
    if args.command == "init":
        init_run(args.manifest, args.run_dir, args.task_id)
    elif args.command == "preflight":
        preflight(args.run_dir, check_endpoints=not args.skip_endpoints)
    elif args.command == "run-both":
        run_both(args.run_dir)
    elif args.command == "status":
        status(args.run_dir)


if __name__ == "__main__":
    main()
