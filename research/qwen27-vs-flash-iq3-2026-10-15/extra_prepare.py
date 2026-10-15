#!/usr/bin/env python3
"""Freeze a disjoint metadata-stratified cohort; prepare images without inference."""
import argparse
from collections import Counter
import contextlib
import fcntl
from fractions import Fraction
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
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
COHORT = HERE / "extra-cohort"
ORIGINAL = PROJECT / "swebench/manifest-q4-q5-40.json"
PROTECTED = ("paired_swe.py", "instrumented_model.py", "official_eval.py")
SEED = 20261016
UNIT = "qwen27-flash-extra-images-20261015"
STOP = False


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def body_hash(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path, body, mode=0o600):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as f:
        os.fchmod(f.fileno(), mode)
        json.dump(body, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def under_here(path):
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to(HERE) or path == HERE:
        raise ValueError("output must be below the research directory")
    return path


def selection(rows, original, seed=SEED):
    """Only ID, repo, input length and original quotas can influence selection."""
    excluded = {t["instance_id"] for t in original["tasks"]}
    if len(excluded) != 40 or len(original["tasks"]) != 40:
        raise ValueError("original cohort must contain exactly 40 unique IDs")
    eligible = [r for r in rows if r["instance_id"] not in excluded]
    if len({r["instance_id"] for r in eligible}) != len(eligible):
        raise ValueError("duplicate eligible IDs")
    available = Counter(r["repo"] for r in eligible)
    weights = original["selection"]["quotas"]
    quotas = {repo: min(count, available[repo]) for repo, count in sorted(weights.items())}
    if sum(available[r] for r in weights) < 40:
        raise ValueError("not enough eligible tasks within original repository frame")
    while sum(quotas.values()) < 40:
        candidates = [r for r in quotas if quotas[r] < available[r]]
        repo = min(candidates, key=lambda r: (-Fraction(weights[r], quotas[r] + 1), r))
        quotas[repo] += 1
    if sum(quotas.values()) != 40:
        raise ValueError("expected positive exact 40-task allocation")
    selected = []
    for repo, count in sorted(quotas.items()):
        if not count:
            continue
        candidates = sorted((r for r in eligible if r["repo"] == repo),
                            key=lambda r: (len(r["problem_statement"]), r["instance_id"]))
        rng = random.Random(f"{seed}:{repo}")
        for index in range(count):
            low, high = index * len(candidates) // count, (index + 1) * len(candidates) // count
            selected.append(candidates[rng.randrange(low, high)])
    selected.sort(key=lambda r: (r["repo"], len(r["problem_statement"]), r["instance_id"]))
    ids = {r["instance_id"] for r in selected}
    assert len(selected) == len(ids) == 40 and not ids & excluded
    smoke_order = sorted(selected, key=lambda r: hashlib.sha256(f"{seed}:smoke:{r['instance_id']}".encode()).hexdigest())
    first = smoke_order[0]
    second = next(r for r in smoke_order if r["repo"] != first["repo"])
    return selected, quotas, dict(sorted(available.items())), [first["instance_id"], second["instance_id"]]


def dataset_metadata(dataset, name):
    return {"name": name, "split": "test", "fingerprint": dataset._fingerprint,
            "rows": len(dataset), "version": str(dataset.info.version),
            "hub_sha": Path(dataset.cache_files[0]["filename"]).parent.name}


def freeze(args):
    os.environ.update(HF_DATASETS_OFFLINE="1", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    from datasets import load_dataset
    cohort = under_here(args.cohort)
    if cohort.exists() and (cohort / "manifest.json").exists():
        verify(cohort)
        print(json.dumps({"already_frozen": True, "manifest_sha256": read(cohort / "manifest.json")["manifest_sha256"]}))
        return
    original = read(args.original)
    if body_hash({k: v for k, v in original.items() if k != "manifest_sha256"}) != original["manifest_sha256"]:
        raise ValueError("original manifest checksum mismatch")
    verified = load_dataset(original["dataset"]["name"], split="test")
    if len(verified) != 500 or verified._fingerprint != original["dataset"]["fingerprint"]:
        raise ValueError("expected frozen Verified500 cache")
    # Only input metadata is supplied to selection; no outcome, patch or image access.
    metadata = [{k: r[k] for k in ("instance_id", "repo", "base_commit", "problem_statement", "version", "created_at")} for r in verified]
    selected, quotas, available, smoke = selection(metadata, original)
    canonical = load_dataset(args.canonical_dataset, split="test")
    cohort.mkdir(parents=True, exist_ok=True, mode=0o700)
    cohort.chmod(0o700)
    manifest = {"schema_version": 1, "dataset": dataset_metadata(verified, original["dataset"]["name"]),
                "canonical_dataset": dataset_metadata(canonical, args.canonical_dataset),
                "original_manifest_sha256": original["manifest_sha256"],
                "selection": {"seed": SEED, "task_count_per_model": 40,
                    "method": "original repo quotas capped by unseen availability; weighted deterministic redistribution; seeded uniform sampling within equal rank input-length strata",
                    "original_quotas": original["selection"]["quotas"], "quotas": quotas,
                    "eligible_repo_counts": available, "eligible_total": sum(available.values()),
                    "original_overlap_count": 0, "smoke_ids": smoke,
                    "order_seed": 20261015, "version_counts": dict(sorted(Counter(f"{r['repo']}:{r['version']}" for r in selected).items()))},
                "tasks": [{"order": i, "instance_id": r["instance_id"], "repo": r["repo"],
                           "base_commit": r["base_commit"], "problem_statement_chars": len(r["problem_statement"]),
                           "version": r["version"], "created_at": r["created_at"]} for i, r in enumerate(selected)]}
    manifest["manifest_sha256"] = body_hash(manifest)
    # Commit the choice BEFORE gold crosschecks or any Docker inspect/pull.
    write(cohort / "manifest.json", manifest, 0o444)
    write(cohort / "original-ids.json", sorted(t["instance_id"] for t in original["tasks"]), 0o444)
    policy = {"schema_version": 1, "seed": SEED, "manifest_sha256": manifest["manifest_sha256"],
              "source_hashes": {name: sha(HERE / name) for name in (*PROTECTED, "extra_prepare.py", "extra_swe.py")},
              "original_manifest_file_sha256": sha(args.original), "smoke_ids": smoke,
              "count": 40, "max_steps": 150, "task_wall_seconds": 1800,
              "max_tokens": 8192, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
              "presence_penalty": 1.5, "reasoning_effort": "none", "enable_thinking": False,
              "deadline_policy": "inherit exact generation/envelope epochs from original primary ledger; never extend",
              "image_policy": "canonical image tags; no substitution; sequential pulls; root free >=25GiB before each; no pruning"}
    primary = read(args.primary_run / "ledger.json")
    policy["primary_deadlines"] = {key: primary[key] for key in ("generation_deadline_epoch", "envelope_deadline_epoch")}
    write(cohort / "policy.json", policy, 0o444)
    try:
        by_id = {r["instance_id"]: r for r in canonical}
        verified_by_id = {r["instance_id"]: r for r in verified}
        evaluator, images = [], []
        for task in manifest["tasks"]:
            iid = task["instance_id"]
            row, gold = verified_by_id[iid], by_id[iid]
            for field in ("repo", "base_commit", "patch", "test_patch", "problem_statement", "version"):
                if row[field] != gold[field]:
                    raise ValueError(f"canonical crosscheck failed: {iid}/{field}")
            for field in ("image", "eval_script", "log_parser", "eval_type"):
                if not gold.get(field):
                    raise ValueError(f"canonical evaluator field absent: {iid}/{field}")
            expected = f"swebench/sweb.eval.x86_64.{iid.replace('__', '_1776_')}:latest".lower()
            if gold["image"].removeprefix("docker.io/") != expected:
                raise ValueError(f"canonical image is incompatible with stock mini-agent: {iid}")
            evaluator.append(gold)
            images.append({"instance_id": iid, "image": gold["image"]})
        write(cohort / "canonical-evaluator.HOST-ONLY.json", evaluator, 0o400)
        write(cohort / "images.json", images, 0o444)
        seal = {"files": {name: sha(cohort / name) for name in ("manifest.json", "original-ids.json", "policy.json", "canonical-evaluator.HOST-ONLY.json", "images.json")},
                "frozen_epoch": time.time(), "datasets_version": importlib.metadata.version("datasets")}
        write(cohort / "seal.HOST-ONLY.json", seal, 0o400)
    except Exception as exc:
        write(cohort / "crosscheck-failure.json", {"error": repr(exc), "choice_preserved": True})
        raise
    print(json.dumps({"selected": 40, "overlap": 0, "quotas": quotas, "smoke": smoke, "manifest_sha256": manifest["manifest_sha256"]}, indent=2))


def verify(cohort):
    cohort = under_here(cohort)
    for name, expected in read(cohort / "seal.HOST-ONLY.json")["files"].items():
        if sha(cohort / name) != expected:
            raise RuntimeError(f"cohort artifact changed: {name}")
    policy, manifest = read(cohort / "policy.json"), read(cohort / "manifest.json")
    for name, expected in policy["source_hashes"].items():
        if sha(HERE / name) != expected:
            raise RuntimeError(f"source hash guard failed: {name}")
    if body_hash({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest["manifest_sha256"]:
        raise RuntimeError("manifest hash invalid")
    ids = {t["instance_id"] for t in manifest["tasks"]}
    if len(ids) != 40 or len(manifest["tasks"]) != 40 or ids & set(read(cohort / "original-ids.json")):
        raise RuntimeError("cohort is not exactly 40 disjoint tasks")
    smoke = manifest["selection"]["smoke_ids"]
    repos = {t["instance_id"]: t["repo"] for t in manifest["tasks"]}
    if len(smoke) != 2 or len(set(smoke)) != 2 or not set(smoke) <= ids or repos[smoke[0]] == repos[smoke[1]]:
        raise RuntimeError("invalid different-repository smoke IDs")
    if policy["manifest_sha256"] != manifest["manifest_sha256"] or policy["smoke_ids"] != smoke or policy["count"] != 40:
        raise RuntimeError("cohort policy and manifest disagree")
    return manifest, policy


def inspect(image):
    result = subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True, timeout=30)
    if result.returncode:
        if "no such image" in result.stderr.lower() or "no such object" in result.stderr.lower():
            return None
        raise RuntimeError(f"Docker inspect failed: {result.stderr.strip()}")
    info = json.loads(result.stdout)[0]
    return {"image_id": info["Id"], "repo_digests": info.get("RepoDigests", []), "size_bytes": info.get("Size")}


def images_cached(cohort):
    verify(cohort)
    missing = [item["image"] for item in read(cohort / "images.json") if inspect(item["image"]) is None]
    if missing:
        raise RuntimeError(f"{len(missing)}/40 images missing; run image preparation first")


def pull_one(image, log, heartbeat, deadline):
    process = subprocess.Popen(["docker", "pull", image], stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    start = time.monotonic()
    try:
        while process.poll() is None:
            heartbeat(process.pid)
            if STOP or time.monotonic() - start >= 3600 or time.time() >= deadline:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                raise RuntimeError("image pull cancelled or exceeded one-hour timeout")
            time.sleep(5)
        if process.returncode:
            raise RuntimeError(f"docker pull exited {process.returncode}")
    finally:
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=20)


def prepare(args):
    cohort = under_here(args.cohort)
    _, policy = verify(cohort)
    deadline = policy["primary_deadlines"]["generation_deadline_epoch"]
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (cohort / "preparation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = cohort / "preparation-status.json"
        state = read(path) if path.exists() else {"images": {}, "attempts": []}
        state.update(state="scanning", worker_pid=os.getpid(), started_epoch=time.time())
        for key in ("error", "failed_epoch", "finished_epoch", "pull_pid"):
            state.pop(key, None)
        items = read(cohort / "images.json")

        def update():
            state["heartbeat_epoch"] = time.time()
            state["cached_count"] = sum(i.get("state") == "cached" for i in state["images"].values())
            state["total"] = len(items)
            write(path, state)

        try:
            # Inspect all first so progress includes existing cache, but never use it for selection.
            for item in items:
                info = inspect(item["image"])
                state["images"][item["instance_id"]] = {**item, "state": "cached" if info else "missing", **(info or {})}
                update()
            for item in items:
                entry = state["images"][item["instance_id"]]
                if STOP:
                    raise RuntimeError("preparation interrupted")
                if entry["state"] == "cached":
                    continue
                if time.time() >= deadline:
                    state.update(state="deadline", finished_epoch=time.time())
                    update()
                    return
                free = shutil.disk_usage("/").free
                state["root_free_bytes"] = free
                if free < 25 * 1024**3:
                    state["state"] = "blocked_space"
                    raise RuntimeError("root filesystem free space below 25GiB; selection unchanged")
                state.update(state="pulling", current_instance=item["instance_id"])
                attempt = {**item, "started_epoch": time.time(), "root_free_bytes_before": free}
                state["attempts"].append(attempt)
                update()
                print(json.dumps({"pull": item["instance_id"], "root_free_gib": round(free / 1024**3, 2)}), flush=True)

                def beat(pid):
                    state["pull_pid"] = pid
                    update()

                with (cohort / "pull-output.log").open("a") as log:
                    pull_one(item["image"], log, beat, deadline)
                info = inspect(item["image"])
                if not info:
                    raise RuntimeError("pull returned success but image is absent")
                entry.update(state="cached", **info)
                attempt["finished_epoch"] = time.time()
                state.pop("pull_pid", None)
                update()
            state.update(state="ready", finished_epoch=time.time(), current_instance=None)
            update()
            print(json.dumps({"state": "ready", "cached": state["cached_count"]}), flush=True)
        except Exception as exc:
            if time.time() >= deadline:
                state.update(state="deadline", error=repr(exc), finished_epoch=time.time())
                update()
                return
            if state["state"] != "blocked_space":
                state["state"] = "interrupted" if STOP else "failed"
            state.update(error=repr(exc), failed_epoch=time.time())
            update()
            print(json.dumps({"state": state["state"], "error": repr(exc), "choice_preserved": True}), flush=True)
            raise


def stop(*_):
    global STOP
    STOP = True


def launch(args):
    cohort = under_here(args.cohort)
    verify(cohort)
    for name in ("preparation.log", "pull-output.log"):
        (cohort / name).touch(mode=0o600, exist_ok=True)
        (cohort / name).chmod(0o600)
    command = ["systemd-run", "--user", f"--unit={UNIT}", "--description=Disjoint SWE image preparation only",
               "--property=Type=exec", "--property=Restart=on-failure", "--property=RestartSec=300",
               "--property=KillMode=control-group", "--property=TimeoutStopSec=30", "--property=Nice=10",
               "--property=IOSchedulingClass=idle", "--property=UMask=0077",
               f"--property=WorkingDirectory={HERE}", f"--property=StandardOutput=append:{cohort / 'preparation.log'}",
               f"--property=StandardError=append:{cohort / 'preparation.log'}", "--setenv=PYTHONDONTWRITEBYTECODE=1",
               sys.executable, "-B", str(HERE / "extra_prepare.py"), "prepare", "--cohort", str(cohort)]
    write(cohort / "preparation-launch.HOST-ONLY.json", {"command": command, "epoch": time.time()})
    subprocess.run(command, check=True, timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "prepare", "launch", "status", "check-images"):
        p = sub.add_parser(name)
        p.add_argument("--cohort", type=Path, default=COHORT)
        if name == "freeze":
            p.add_argument("--original", type=Path, default=Path(os.getenv("SWEBENCH_MANIFEST", ORIGINAL)))
            p.add_argument("--canonical-dataset", default="SWE-bench/SWE-bench")
            p.add_argument("--primary-run", type=Path, default=HERE / "runs/primary-single-instance")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args)
    elif args.command == "prepare":
        prepare(args)
    elif args.command == "launch":
        launch(args)
    elif args.command == "check-images":
        images_cached(args.cohort)
        print("All 40 images cached; cohort/source hashes verified")
    else:
        verify(args.cohort)
        path = args.cohort / "preparation-status.json"
        state = read(path) if path.exists() else {"state": "not_started"}
        print(json.dumps({k: v for k, v in state.items() if k not in {"images", "attempts"}}, indent=2))


if __name__ == "__main__":
    main()
