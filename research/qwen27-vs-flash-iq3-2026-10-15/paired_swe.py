#!/usr/bin/env python3
"""Paired, restartable SWE generation and official evaluation. No service control.

Ledger/metrics design adapted from tools/swebench_controller.py; that helper is
snapshotted for provenance, never modified or executed.
"""
import argparse
import concurrent.futures
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

HERE = Path(__file__).resolve().parent
SMOKE = ["pallets__flask-5014", "astropy__astropy-13579"]
TERMINAL = {"submitted", "failed", "infra_failed", "deadline"}
STOP = threading.Event()
MUTEX = threading.RLock()
OFFLINE = {"HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
           "HF_HUB_DISABLE_TELEMETRY": "1", "MSWEA_COST_TRACKING": "ignore_errors",
           "LITELLM_LOCAL_MODEL_COST_MAP": "True", "TQDM_DISABLE": "1", "PYTHONDONTWRITEBYTECODE": "1"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def load(path):
    return json.loads(Path(path).read_text())


def local_path(value):
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(HERE) or path == HERE:
        raise ValueError(f"run directory must be below {HERE}")
    return path


@contextlib.contextmanager
def lock(run):
    with (run / "controller.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def save(run, ledger):
    # Caller holds MUTEX; a single atomic document commits all model states.
    ledger["heartbeat_epoch"] = time.time()
    atomic(run / "ledger.json", ledger)
    summary = summarize(ledger)
    atomic(run / "status.json", summary)
    for model, result in summary["models"].items():
        atomic(run / model / "summary.json", result)


def summarize(ledger):
    result = {"phase": ledger["phase"], "generation_deadline_epoch": ledger.get("generation_deadline_epoch"),
              "envelope_deadline_epoch": ledger.get("envelope_deadline_epoch"),
              "heartbeat_epoch": ledger.get("heartbeat_epoch"), "models": {}}
    for model, tasks in ledger["tasks"].items():
        counts = {}
        for task in tasks.values():
            counts[task["state"]] = counts.get(task["state"], 0) + 1
        resolved = sum(t.get("evaluation", {}).get("resolved") is True for t in tasks.values())
        evaluated = sum(isinstance(t.get("evaluation", {}).get("resolved"), bool) for t in tasks.values())
        attempts = [a for t in tasks.values() for a in t["attempts"]]
        result["models"][model] = {
            "denominator": len(tasks), "states": counts, "official_resolved": resolved,
            "official_evaluated": evaluated, "resolved_fraction_all_tasks": resolved / len(tasks),
            "not_officially_evaluated": len(tasks) - evaluated,
            "attempt_output_tokens": sum(a.get("metrics", {}).get("completion_tokens", 0) for a in attempts),
            "attempt_calls": sum(a.get("metrics", {}).get("backend_requests", a.get("metrics", {}).get("agent_calls", 0)) for a in attempts),
            "attempt_agent_calls": sum(a.get("metrics", {}).get("agent_calls", 0) for a in attempts),
            "attempt_wall_seconds": sum(a.get("wall_seconds", 0) for a in attempts),
            "max_context_tokens": max([a.get("metrics", {}).get("max_context_tokens", 0) for a in attempts] or [0]),
            "active": [i for i, t in tasks.items() if t["state"] == "running"],
        }
    return result


def verify(run):
    for name, expected in load(run / "immutable-sha256.json").items():
        if digest(run / name) != expected:
            raise RuntimeError(f"immutable artifact changed: {name}")
    for name in ("paired_swe.py", "instrumented_model.py", "official_eval.py"):
        if digest(HERE / name) != digest(run / "frozen" / name):
            raise RuntimeError(f"controller source differs from initialized run: {name}; restore frozen source before resume")
    config = load(run / "config.json")
    for path, expected in config["installed_source_hashes"].items():
        if digest(path) != expected:
            raise RuntimeError(f"installed dependency source changed: {path}")
    for package, version in config["versions"].items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError(f"dependency changed: {package}")
    return config


def init(args):
    import yaml
    from datasets import load_dataset
    import minisweagent
    from minisweagent.config import builtin_config_dir
    import swebench.harness.run_evaluation as official
    manifest_path = args.manifest.resolve()
    manifest = load(manifest_path)
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    actual = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if actual != manifest.get("manifest_sha256"):
        raise ValueError("frozen manifest hash mismatch")
    tasks = manifest["tasks"]
    ids = [t["instance_id"] for t in tasks]
    if len(ids) != 40 or len(set(ids)) != 40 or not set(SMOKE) <= set(ids):
        raise ValueError("expected frozen 40 unique tasks including both smoke tasks")
    dataset = load_dataset(manifest["dataset"]["name"], split=manifest["dataset"]["split"])
    if dataset._fingerprint != manifest["dataset"]["fingerprint"]:
        raise ValueError("offline dataset fingerprint differs from frozen manifest")
    cached = {r["instance_id"]: r for r in dataset}
    safe_tasks = []
    for t in tasks:
        row = cached[t["instance_id"]]
        if row["base_commit"] != t["base_commit"]:
            raise ValueError(f"base commit mismatch: {t['instance_id']}")
        safe_tasks.append({**t, "problem_statement": row["problem_statement"]})
    evaluator = load(args.evaluator)
    if {r["instance_id"] for r in evaluator} != set(ids):
        raise ValueError("evaluator must match all 40 frozen tasks")
    for row in evaluator:
        if row["base_commit"] != cached[row["instance_id"]]["base_commit"]:
            raise ValueError("evaluator base commit mismatch")
    rest = sorted(set(ids) - set(SMOKE))
    random.Random(20261015).shuffle(rest)
    order = SMOKE + rest
    run = local_path(args.run_dir)
    if run.exists() and any(run.iterdir()):
        raise ValueError("init requires an empty run directory")
    run.mkdir(parents=True, exist_ok=True)
    (run / "frozen").mkdir()
    stock = builtin_config_dir / "benchmarks/swebench.yaml"
    if "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt" not in stock.read_text():
        raise ValueError("stock submission instructions missing")
    copies = {"source-manifest.json": manifest_path, "evaluator.json": args.evaluator,
              "stock.yaml": stock, "existing-controller.py": args.helper,
              "paired_swe.py": Path(__file__), "instrumented_model.py": HERE / "instrumented_model.py",
              "official_eval.py": HERE / "official_eval.py"}
    for name, src in copies.items():
        shutil.copyfile(src, run / "frozen" / name)
    atomic(run / "frozen/tasks.json", safe_tasks)
    atomic(run / "frozen/order.json", {"seed": 20261015, "smoke_first": SMOKE, "order": order})
    versions = {p: importlib.metadata.version(p) for p in ("mini-swe-agent", "swebench", "datasets", "litellm")}
    expected = {"mini-swe-agent": "2.4.6", "swebench": "5.0.2", "datasets": "5.0.1"}
    if any(versions[p] != v for p, v in expected.items()):
        raise ValueError(f"unexpected installed versions: {versions}")
    assert official.__file__ and minisweagent.__file__
    sources = [Path(official.__file__), *Path(official.__file__).parent.rglob("*.py"),
               *Path(minisweagent.__file__).parent.rglob("*.py")]
    models = {
        "qwen": {"endpoint": args.qwen_endpoint, "alias": args.qwen_alias, "backend": args.qwen_backend},
        "flash": {"endpoint": args.flash_endpoint, "alias": args.flash_alias, "backend": args.flash_backend},
    }
    if models["qwen"]["endpoint"].rstrip("/") == models["flash"]["endpoint"].rstrip("/"):
        raise ValueError("primary models require distinct endpoints")
    config = {"models": models, "label": args.label, "python": str(Path(sys.executable).absolute()),
              "mini_extra": str(Path(sys.executable).parent / "mini-extra"), "order": order,
              "dataset": manifest["dataset"], "versions": versions,
              "installed_source_hashes": {str(p): digest(p) for p in sources},
              "generation_seconds": 24 * 3600, "envelope_seconds": 30 * 3600,
              "task_seconds": 1800, "infra_retries": 1, "context_tokens": 262144,
              "note": "Frozen original manifest model/runtime fields are historical; config.json overrides them."}
    atomic(run / "config.json", config)
    for model, spec in models.items():
        (run / model).mkdir()
        overlay = {
            "agent": {"step_limit": 150, "cost_limit": 0, "wall_time_limit_seconds": 1800},
            "environment": {"timeout": 300, "container_timeout": "1900s", "forward_env": [],
                            "run_args": ["--rm", "--pull=never", "--network", "none", "--cpus", "2", "--memory", "6g"]},
            "model": {"model_class": "instrumented_model.TimedModel", "cost_tracking": "ignore_errors",
                      "model_name": f"{spec['backend']}/{spec['alias']}",
                      "model_kwargs": {"api_base": spec["endpoint"], "api_key": "local",
                                       "drop_params": False, "parallel_tool_calls": False,
                                       "stream": False,
                                       "max_tokens": 8192, "temperature": 0.7, "top_p": 0.8,
                                       "presence_penalty": 1.5, "reasoning_effort": "none",
                                       "timeout": 300, "num_retries": 0,
                                       "extra_body": {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}}},
        }
        (run / "frozen" / f"{model}.yaml").write_text(yaml.safe_dump(overlay))
    immutable = {str(p.relative_to(run)): digest(p) for p in (run / "frozen").iterdir()}
    immutable["config.json"] = digest(run / "config.json")
    atomic(run / "immutable-sha256.json", immutable)
    for name in immutable:
        (run / name).chmod(0o444)
    ledger = {"phase": "initialized", "created_epoch": time.time(), "smoke_gate": False, "evaluations": [],
              "tasks": {m: {i: {"state": "queued", "attempts": []} for i in order} for m in models}}
    save(run, ledger)
    print(json.dumps({"initialized": str(run), "offline_dataset_fingerprint": dataset._fingerprint, "versions": versions}))


def metrics(path):
    data = load(path) if path.exists() else {}
    info = data.get("info", {})
    # v2 stores the exit in info, but tolerate message-level exit metadata.
    for message in data.get("messages", []):
        if message.get("role") == "exit":
            info = {**info, **message.get("extra", {})}
    result = {"exit_status": info.get("exit_status", ""), "submission": info.get("submission") or "",
              "agent_calls": 0, "completion_tokens": 0, "max_context_tokens": 0,
              "max_prompt_tokens": 0, "tool_calls": 0, "request_latency_seconds": [], "usage_missing_calls": 0}
    for msg in data.get("messages", []):
        extra = msg.get("extra", {})
        if msg.get("role") != "assistant" and not isinstance(extra.get("response"), dict):
            continue
        result["agent_calls"] += 1
        result["tool_calls"] += len(extra.get("actions", []))
        if "request_latency_seconds" in extra:
            result["request_latency_seconds"].append(extra["request_latency_seconds"])
        response = extra.get("response", {})
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        if not usage:
            result["usage_missing_calls"] += 1
        prompt, output = usage.get("prompt_tokens", 0) or 0, usage.get("completion_tokens", 0) or 0
        result["completion_tokens"] += output
        result["max_context_tokens"] = max(result["max_context_tokens"], prompt + output)
        result["max_prompt_tokens"] = max(result["max_prompt_tokens"], prompt)
    request_log = path.parent / "requests.jsonl"
    if request_log.exists():
        requests = [json.loads(line) for line in request_log.read_text().splitlines() if line.strip()]
        result["backend_requests"] = len(requests)
        result["request_latency_seconds"] = [r["latency_seconds"] for r in requests]
        result["request_errors"] = [r for r in requests if r.get("error")]
        result["completion_tokens"] = sum((r.get("usage") or {}).get("completion_tokens", 0) or 0 for r in requests)
        result["usage_missing_calls"] = sum(not r.get("usage") for r in requests)
        for record in requests:
            usage = record.get("usage") or {}
            prompt = usage.get("prompt_tokens", 0) or 0
            output = usage.get("completion_tokens", 0) or 0
            result["max_prompt_tokens"] = max(result["max_prompt_tokens"], prompt)
            result["max_context_tokens"] = max(result["max_context_tokens"], prompt + output)
    return result


def classify(exit_status, output, timed_out=False, interrupted=False):
    if interrupted:
        return "interrupted", "controller_interrupted"
    if timed_out or exit_status == "TimeExceeded":
        return "failed", "task_timeout"
    if exit_status in {"LimitsExceeded", "FormatError", "RepeatedFormatError", "Submitted",
                       "UnsupportedParamsError", "BadRequestError", "AuthenticationError"}:
        return "failed", exit_status
    text = (exit_status + " " + output).lower()
    # Context/protocol errors are model outcomes, never infrastructure retries.
    if any(s in text for s in ("contextwindow", "context length", "maximum context", "prompt is too long")):
        return "failed", "context_exceeded"
    if any(s in text for s in ("connection refused", "apiconnectionerror", "cannot connect to the docker daemon",
                               "no space left on device", "input/output error", "cuda out of memory")):
        return "infra_retry", "infrastructure"
    return "failed", exit_status or "process_or_submission_failure"


def terminate(process):
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def docker_cleanup(label=None, eval_id=None):
    # Only exact per-attempt labels or this fresh official run-id's names.
    query = ["docker", "ps", "-aq"]
    query += ["--filter", f"label=paired-swe.attempt={label}"] if label else ["--filter", f"label=paired-swe.eval={eval_id}"]
    found = subprocess.run(query, capture_output=True, text=True, timeout=20, check=True)
    for container in found.stdout.split():
        subprocess.run(["docker", "rm", "-f", container], capture_output=True, timeout=30, check=True)


def process_identity(pid):
    try:
        # Field 22, after the parenthesized comm field; resilient to spaces in comm.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def recover_process(record):
    pid = record.get("process_pid")
    if not pid or not record.get("process_identity") or process_identity(pid) != record["process_identity"]:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGTERM)
    until = time.monotonic() + 10
    while process_identity(pid) == record["process_identity"] and time.monotonic() < until:
        time.sleep(0.1)
    if process_identity(pid) == record["process_identity"]:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)


def supervise(command, directory, log, deadline, heartbeat, cleanup, record):
    started = time.monotonic()
    process = None
    try:
        with log.open("w") as output:
            process = subprocess.Popen(command, cwd=directory, env={**os.environ, **OFFLINE,
                                       "PYTHONPATH": str(directory / "frozen"),
                                       "PAIRED_SWE_REQUEST_LOG": str(log.parent / "requests.jsonl")}, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            with MUTEX:
                record.update(process_pid=process.pid, process_identity=process_identity(process.pid))
                heartbeat()
            next_heartbeat = time.monotonic() + 30
            while process.poll() is None:
                if time.monotonic() >= next_heartbeat:
                    heartbeat()
                    next_heartbeat = time.monotonic() + 30
                if STOP.is_set() or time.time() >= deadline:
                    terminate(process)
                    break
                STOP.wait(2)
            return process.returncode, time.monotonic() - started, time.time() >= deadline
    finally:
        if process is not None:
            terminate(process)
        cleanup()


def predictions(run, ledger, model, ids, path):
    rows = []
    for iid in ids:
        task = ledger["tasks"][model][iid]
        patch = task.get("patch", "") if task["state"] == "submitted" else ""
        rows.append({"model_name_or_path": model, "instance_id": iid, "model_patch": patch})
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def worker(run, config, ledger, model, ids):
    for iid in ids:
        while not STOP.is_set():
            with MUTEX:
                task = ledger["tasks"][model][iid]
                if task["state"] in TERMINAL:
                    break
                if time.time() >= ledger["generation_deadline_epoch"]:
                    task["state"] = "deadline"
                    save(run, ledger)
                    break
                attempt_id = uuid.uuid4().hex
                folder = run / model / iid / attempt_id
                folder.mkdir(parents=True)
                attempt = {"id": attempt_id, "started_epoch": time.time(), "directory": str(folder.relative_to(run))}
                task["attempts"].append(attempt)
                task["state"] = "running"
                save(run, ledger)
            import yaml
            overlay = load_yaml(run / "frozen" / f"{model}.yaml")
            overlay["environment"]["run_args"] += ["--label", f"paired-swe.attempt={attempt_id}"]
            (folder / "attempt.yaml").write_text(yaml.safe_dump(overlay))
            trajectory = folder / "trajectory.json"
            command = [config["mini_extra"], "swebench-single", "--subset", config["dataset"]["name"],
                       "--split", config["dataset"]["split"], "--instance", iid,
                       "--config", str(run / "frozen/stock.yaml"), "--config", str(folder / "attempt.yaml"),
                       "--output", str(trajectory), "--yolo", "--exit-immediately", "--cost-limit", "0"]
            atomic(folder / "command.json", command)

            def beat():
                with MUTEX:
                    attempt["heartbeat_epoch"] = time.time()
                    save(run, ledger)

            try:
                code, wall, timed = supervise(command, run, folder / "agent.log",
                    min(time.time() + config["task_seconds"], ledger["generation_deadline_epoch"]), beat,
                    lambda: docker_cleanup(label=attempt_id), attempt)
                measured = metrics(trajectory)
                patch = measured.pop("submission", "")
                state, reason = classify(measured.get("exit_status", ""), (folder / "agent.log").read_text(errors="replace"), timed, STOP.is_set())
                if not STOP.is_set() and not timed and code == 0 and measured.get("exit_status") == "Submitted" and patch.strip():
                    state, reason = "submitted", None
                with MUTEX:
                    attempt.update(returncode=code, wall_seconds=wall, metrics=measured,
                                   finished_epoch=time.time(), outcome=state, reason=reason)
                    infra_attempts = sum(a.get("outcome") == "infra_retry" for a in task["attempts"])
                    task["state"] = "infra_failed" if state == "infra_retry" and infra_attempts > config["infra_retries"] else state
                    if task["state"] == "submitted":
                        task["patch"] = patch
                    predictions(run, ledger, model, config["order"], run / model / "predictions.jsonl")
                    save(run, ledger)
            except Exception as exc:
                with MUTEX:
                    attempt.update(outcome="interrupted", error=repr(exc), finished_epoch=time.time())
                    task["state"] = "interrupted"
                    save(run, ledger)
                STOP.set()
                raise
            if task["state"] == "interrupted":
                break


def load_yaml(path):
    import yaml
    return yaml.safe_load(path.read_text())


def evaluate(run, config, ledger, ids, stage):
    # Sequential model harnesses: max_workers=2 TOTAL, no generation concurrently.
    for model in config["models"]:
        pending = [i for i in ids if ledger["tasks"][model][i]["state"] == "submitted"
                   and "evaluation" not in ledger["tasks"][model][i]]
        if not pending:
            continue
        if STOP.is_set() or time.time() >= ledger["envelope_deadline_epoch"]:
            return False
        run_id = "paired-" + uuid.uuid4().hex
        folder = run / model / "evaluation" / run_id
        folder.mkdir(parents=True)
        pred = folder / "predictions.jsonl"
        predictions(run, ledger, model, pending, pred)
        record = {"run_id": run_id, "model": model, "stage": stage, "ids": pending,
                  "directory": str(folder.relative_to(run)), "started_epoch": time.time()}
        ledger["evaluations"].append(record)
        save(run, ledger)
        command = [config["python"], str(run / "frozen/official_eval.py"), "--dataset", str(run / "frozen/evaluator.json"),
                   "--predictions", str(pred), "--run-id", run_id, "--ids", *pending]
        atomic(folder / "command.json", command)

        def beat():
            record["heartbeat_epoch"] = time.time()
            save(run, ledger)

        code, wall, timed = supervise(command, folder, folder / "harness.log", ledger["envelope_deadline_epoch"],
                                     beat, lambda: docker_cleanup(eval_id=run_id), record)
        record.update(returncode=code, wall_seconds=wall, timed_out=timed, finished_epoch=time.time())
        missing = []
        for iid in pending:
            report = folder / "logs/run_evaluation" / run_id / model / iid / "report.json"
            if not report.exists():
                missing.append(iid)
                continue
            entry = load(report).get(iid, {})
            if not isinstance(entry.get("resolved"), bool):
                missing.append(iid)
                continue
            ledger["tasks"][model][iid]["evaluation"] = {"resolved": entry["resolved"], "run_id": run_id,
                                                         "report": str(report.relative_to(run))}
        record["missing_reports"] = missing
        save(run, ledger)
        if missing or code != 0 or STOP.is_set():
            return False  # Explicit resume retries ONLY missing infrastructure reports, fresh run id.
    return True


def generate_pair(run, config, ledger, ids):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, run, config, ledger, model, ids) for model in config["models"]]
        for future in futures:
            future.result()


def run_all(args):
    if not args.swe_phase:
        raise ValueError("--swe-phase required: reserve endpoints exclusively for this SWE phase")
    run = local_path(args.run_dir)
    with lock(run):
        config = verify(run)
        ledger = load(run / "ledger.json")
        if ledger["phase"] == "complete":
            print(json.dumps(summarize(ledger), indent=2))
            return
        if ledger["phase"] != "initialized" and not args.resume:
            raise ValueError("existing run requires --resume")
        now = time.time()
        ledger.setdefault("generation_deadline_epoch", now + config["generation_seconds"])
        ledger.setdefault("envelope_deadline_epoch", now + config["envelope_seconds"])
        # Deadlines survive pauses/restarts. Clean only this run's recorded attempts.
        for tasks in ledger["tasks"].values():
            for task in tasks.values():
                if task["state"] in {"running", "interrupted"}:
                    if task["attempts"]:
                        recover_process(task["attempts"][-1])
                        docker_cleanup(label=task["attempts"][-1]["id"])
                    task["state"] = "interrupted"
        for record in ledger["evaluations"]:
            if "finished_epoch" not in record:
                recover_process(record)
                docker_cleanup(eval_id=record["run_id"])
        if not ledger["smoke_gate"]:
            ledger["phase"] = "smoke_generation"
            save(run, ledger)
            generate_pair(run, config, ledger, SMOKE)
            if STOP.is_set():
                return
            ledger["phase"] = "smoke_evaluation"
            save(run, ledger)
            if not evaluate(run, config, ledger, SMOKE, "smoke"):
                raise RuntimeError("smoke evaluation incomplete; inspect harness logs, then --resume")
            for model in config["models"]:
                smoke = [ledger["tasks"][model][i] for i in SMOKE]
                if any(t["state"] not in {"submitted", "failed"} for t in smoke):
                    raise RuntimeError(f"{model}: smoke infrastructure/deadline failure")
                if any(not t["attempts"] or t["attempts"][-1].get("metrics", {}).get("tool_calls", 0) == 0
                       or t["attempts"][-1].get("metrics", {}).get("exit_status") in
                       {"FormatError", "RepeatedFormatError", "UnsupportedParamsError", "AuthenticationError", "BadRequestError"} for t in smoke):
                    raise RuntimeError(f"{model}: broken smoke protocol; create a separately labelled run after fixing")
            ledger["smoke_gate"] = True
        if getattr(args, "smoke_only", False):
            ledger["phase"] = "smoke_complete"
            save(run, ledger)
            return
        ledger["phase"] = "generation"
        save(run, ledger)
        generate_pair(run, config, ledger, config["order"][2:])
        if STOP.is_set():
            return
        ledger["phase"] = "evaluation"
        save(run, ledger)
        if not evaluate(run, config, ledger, config["order"], "full"):
            raise RuntimeError("evaluation incomplete; inspect logs; resume missing reports before envelope deadline")
        ledger["phase"] = "complete"
        save(run, ledger)


def main():
    os.environ.update(OFFLINE)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init", help="Offline cache validation and immutable snapshot; no inference or containers")
    p.add_argument("--run-dir", required=True)
    project = HERE.parent.parent
    p.add_argument("--manifest", type=Path, default=Path(os.getenv("SWEBENCH_MANIFEST", project / "swebench/manifest-q4-q5-40.json")))
    p.add_argument("--evaluator", type=Path, default=Path(os.getenv("SWEBENCH_EVALUATOR", project / "swebench/gold/verified40-evaluator.json")))
    p.add_argument("--helper", type=Path, default=project / "tools/swebench_controller.py")
    p.add_argument("--label", default="primary-single-instance-native262k")
    for model, endpoint, alias in [("qwen", "http://127.0.0.1:8080/v1", "qwen3.8-27b-atx-iq4xs-m-262144-gpu1"),
                                   ("flash", "http://127.0.0.1:18144/v1", "qwen3.8-flash-next-iq3_s")]:
        p.add_argument(f"--{model}-endpoint", default=os.getenv(f"{model.upper()}_ENDPOINT", endpoint))
        p.add_argument(f"--{model}-alias", default=os.getenv(f"{model.upper()}_ALIAS", alias))
        p.add_argument(f"--{model}-backend", default="hosted_vllm")
    p = sub.add_parser("run", help="Smoke, official sanity evaluation, generation, then official evaluation")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--swe-phase", action="store_true", help="Confirm no other HTTP workload/probes use these endpoints")
    p.add_argument("--smoke-only", action="store_true", help="Validate two paired tasks and official evaluation, then pause")
    p = sub.add_parser("status", help="Read ledger only")
    p.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    if args.command == "init":
        init(args)
    elif args.command == "status":
        print(json.dumps(summarize(load(local_path(args.run_dir) / "ledger.json")), indent=2))
    else:
        signal.signal(signal.SIGTERM, lambda *_: STOP.set())
        signal.signal(signal.SIGINT, lambda *_: STOP.set())
        run_all(args)


if __name__ == "__main__":
    main()
