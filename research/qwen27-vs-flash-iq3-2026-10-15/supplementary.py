#!/usr/bin/env python3
"""Paired supplementary benchmarks. Existing endpoints only; no server management.

See SUPPLEMENTARY.md for run contracts, timing denominators and isolation limits.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import uuid
from typing import Any

from supplementary_http import payload, stream_chat
from supplementary_fixtures import CODE_TASK, fixture, grade, local_tokenizer

SCHEMA = 1
MODELS = {
    "qwen": dict(base_url="http://127.0.0.1:8080/v1", model="qwen3.8-27b-atx-iq4xs-m-262144-gpu1",
                 hardware="GPU1: one Qwen 27B replica", quantization="ATX IQ4_XS M"),
    "qwen2": dict(base_url="http://127.0.0.1:8081/v1", model="qwen3.8-27b-atx-iq4xs-m-262144-gpu2",
                  hardware="GPU2: second Qwen 27B replica", quantization="ATX IQ4_XS M"),
    "flash": dict(base_url="http://127.0.0.1:18144/v1", model="qwen3.8-flash-next-iq3_s",
                  hardware="GPU0: single Flash replica, server parallel=1 (operator supplied)", quantization="IQ3_S"),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def load(path):
    return json.loads(path.read_text())


def distribution(values):
    values = [v for v in values if isinstance(v, (float, int)) and not isinstance(v, bool)]
    return dict(n=len(values), median=statistics.median(values) if values else None,
                min=min(values) if values else None, max=max(values) if values else None)


class Run:
    def __init__(self, args):
        self.args = args
        self.root = args.run_dir.resolve() / args.phase
        self.root.mkdir(parents=True, exist_ok=True)
        self.private = self.root / "private"
        self.private.mkdir(mode=0o700, exist_ok=True)
        self.private.chmod(0o700)
        self.end = time.monotonic() + args.overall_seconds
        self.lock = (self.root / ".lock").open("a")
        import fcntl
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.models = {k: dict(v, base_url=getattr(args, k + "_url"), model=getattr(args, k + "_model"))
                       for k, v in MODELS.items()}
        config: dict[str, Any] = dict(schema=SCHEMA, phase=args.phase, models=self.models,
                      body_tokens=args.body_tokens, repeats=args.repeats,
                      max_tokens=4096 if args.phase == "evalplus" else args.max_tokens,
                      qwen_cache_prompt_false=args.qwen_cache_prompt_false,
                      dataset_version="default", seed=args.seed,
                      thinking="reasoning_effort=none; enable_thinking=false",
                      prompt_template_version=1)
        config["source_sha256"] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                   for name in ("supplementary.py", "supplementary_http.py", "supplementary_fixtures.py")}
        if args.phase in ("speed", "doc-quality"):
            directory = os.environ.get("STRATA_TOKENIZER")
            if directory:
                config["tokenizer_sha256"] = {name: hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest()
                                               for name in ("vocab.json", "merges.txt", "token_type.json")}
        path = self.root / "manifest.json"
        if path.exists() and load(path) != config:
            raise ValueError("Resume manifest differs; choose another run-dir")
        save(path, config)
        self.status("running", invocation_seconds=args.overall_seconds)

    def status(self, stage, **extra):
        row = dict(schema=SCHEMA, phase=self.args.phase, stage=stage,
                   utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   remaining_seconds=max(0, self.end - time.monotonic()), **extra)
        save(self.root / "status.json", row)
        with (self.root / "progress.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def remaining(self):
        return self.end - time.monotonic()

    def request(self, key, backend, messages, extra=None, max_tokens=None):
        path = self.private / "records" / (key + ".json")
        model = self.models[backend]
        cache = False if self.args.qwen_cache_prompt_false and backend.startswith("qwen") else None
        body = payload(model["model"], messages, max_tokens or self.args.max_tokens, cache)
        contract_hash = digest(dict(base_url=model["base_url"], payload=body))
        if path.exists():
            result = load(path)
            if result["request_hash"] != contract_hash:
                raise ValueError(f"Request contract changed for {key}")
            return result
        raw_path = self.private / "raw" / (key + ".sse")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        result = stream_chat(model["base_url"], body, self.end, raw_path, self.args.request_seconds)
        result.update(key=key, backend=backend, request_hash=contract_hash,
                      messages_hash=digest(messages), model=model, **(extra or {}))
        save(path, result)
        return result


def paired_messages(tag, text):
    # Random tag is the FIRST system content, before the otherwise common prefix.
    return [dict(role="system", content=f"{tag}\nYou are a helpful coding assistant. Answer without thinking tags."),
            dict(role="user", content=text)]


def public_row(row):
    return {k: v for k, v in row.items() if k not in ("content", "reasoning_content")}


def speed_summary(rows, planned):
    metrics = ("ttft_s", "wall_s", "engine_prefill_tps", "engine_decode_tps", "output_tokens_per_wall_s")
    groups = {}
    for row in rows:
        group = f"{row['kind']}/{row['target_body_tokens']}/{row['backend']}"
        groups.setdefault(group, []).append(row)
    summary = {}
    for key, values in groups.items():
        valid = [r for r in values if not r["error"]]
        summary[key] = dict(attempted=len(values), outcomes=dict(Counter(r["outcome"] for r in values)),
                            cached_observations=dict(Counter(r["cache_observation"] for r in values)),
                            quality_correct=sum(r.get("grading", {}).get("all_correct", False) for r in valid),
                            usage_counts={field: distribution(r["usage"].get(field) for r in values)
                                          for field in ("prompt_tokens", "completion_tokens", "total_tokens")},
                            cached_prompt_tokens=distribution(r["cached_prompt_tokens"] for r in values),
                            context_headroom_invalid=sum(not r.get("context_headroom_valid", False) for r in values),
                            all_responses={m: distribution(r.get(m) for r in valid) for m in metrics},
                            zero_reported_cache={m: distribution(r.get(m) for r in valid if r["cache_observation"] == "zero_reported") for m in metrics})
    pairs = []
    for pair_id in sorted({r["pair_id"] for r in rows}):
        pair = {r["backend"]: r for r in rows if r["pair_id"] == pair_id}
        if set(pair) == {"qwen", "flash"}:
            q, f = pair["qwen"], pair["flash"]
            pairs.append(dict(pair_id=pair_id, identical_messages=q["messages_hash"] == f["messages_hash"],
                              both_zero_reported_cache=all(r["cache_observation"] == "zero_reported" for r in pair.values()),
                              outcomes={k: r["outcome"] for k, r in pair.items()},
                              flash_minus_qwen={m: f[m] - q[m] if not q["error"] and not f["error"] and q.get(m) is not None and f.get(m) is not None else None for m in metrics}))
    return dict(planned_requests=planned, attempted_requests=len(rows), groups=summary, pairs=pairs,
                paired_delta_distributions={m: distribution(p["flash_minus_qwen"][m] for p in pairs) for m in metrics},
                denominator_note="All attempts retained. Timing distributions require no transport error; caps remain labelled. Only zero_reported_cache distributions claim zero measured reuse; unknown is not uncached.")


def speed(run):
    tok = local_tokenizer()
    kinds = ["documents"] if run.args.phase == "doc-quality" else ["documents", "code"]
    rows = []
    planned = len(kinds) * len(run.args.body_tokens) * run.args.repeats * 2
    complete = True
    for target in run.args.body_tokens:
        for repeat in range(run.args.repeats):
            for kind in kinds:
                pair_id = f"{kind}-{target}-r{repeat}"
                path = run.private / "fixtures" / (pair_id + ".json")
                if path.exists():
                    data = load(path)
                else:
                    if run.remaining() <= 0:
                        complete = False
                        continue
                    data = fixture(tok, target, run.args.seed + repeat, uuid.uuid4().hex, kind)
                    save(path, data)
                messages = paired_messages(data["tag"], data["text"])
                extra = {k: v for k, v in data.items() if k not in ("text", "expected")}
                extra["pair_id"] = pair_id
                if run.remaining() <= 0:
                    # Collect completed records without manufacturing unattempted requests.
                    for backend in ("qwen", "flash"):
                        record = run.private / "records" / f"{pair_id}-{backend}.json"
                        if record.exists():
                            rows.append(load(record))
                    complete = False
                    continue
                run.status("pair", pair_id=pair_id)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    jobs = [pool.submit(run.request, pair_id + "-" + b, b, messages, extra) for b in ("qwen", "flash")]
                    for job in jobs:
                        row = job.result()
                        if kind == "documents":
                            row["grading"] = grade(row["content"], data["expected"])
                            # Output cap/transport and answer grading are independent axes.
                        row["context_headroom_valid"] = (row["usage"].get("prompt_tokens", 262145) + run.args.max_tokens <= 262144)
                        save(run.private / "records" / (row["key"] + ".json"), row)
                        rows.append(row)
                save(run.root / "summary.json", speed_summary(rows, planned))
    save(run.root / "requests.json", [public_row(r) for r in rows])
    save(run.root / "summary.json", speed_summary(rows, planned))
    run.status("complete" if complete and len(rows) == planned else "deadline", attempted=len(rows), planned=planned)
    return 0 if complete and len(rows) == planned else 2


def capacity_group(rows, expected=2) -> dict[str, Any]:
    tokens = [r["usage"].get("completion_tokens") for r in rows]
    span = max(r["ended_monotonic"] for r in rows) - min(r["started_monotonic"] for r in rows)
    valid = len(rows) == expected and all(not r["error"] for r in rows) and all(n is not None for n in tokens)
    return dict(requests=[public_row(r) for r in rows], request_count=len(rows),
                group_wall_s=span, aggregate_output_tokens=sum(n for n in tokens if n is not None),
                aggregate_output_tokens_per_wall_s=sum(tokens) / span if valid and span > 0 else None,
                aggregate_requests_per_wall_s=len(rows) / span if valid and span > 0 else None,
                per_request_wall_s=distribution(r["wall_s"] for r in rows),
                aggregate_valid=valid)


def capacity(run):
    rounds = []
    for repeat in range(run.args.repeats):
        path = run.root / f"round-{repeat}.json"
        if path.exists():
            rounds.append(load(path))
            continue
        if run.remaining() <= 0:
            break
        fixture_path = run.private / "fixtures" / f"capacity-{repeat}.json"
        if fixture_path.exists():
            tags = load(fixture_path)
        else:
            tags = [uuid.uuid4().hex, uuid.uuid4().hex]
            save(fixture_path, tags)
        keys = [(b, slot, f"capacity-r{repeat}-{b}-{slot}") for slot in range(2)
                for b in (("qwen" if slot == 0 else "qwen2"), "flash")]
        resumed = any((run.private / "records" / (key + ".json")).exists() for _, _, key in keys)
        run.status("capacity_round", repeat=repeat)
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = [pool.submit(run.request, key, b, paired_messages(tags[slot], CODE_TASK),
                                dict(slot=slot, repeat=repeat), 2048) for b, slot, key in keys]
            rows = [job.result() for job in jobs]
        result: dict[str, Any] = dict(repeat=repeat, label="hardware allocation capacity, not model accuracy",
                      qwen_hardware="2 replicas across GPU1 + GPU2; 2 concurrent requests",
                      flash_hardware="1 replica on GPU0, server parallel=1; 2 concurrently submitted queued requests",
                      resumed_partial_round=resumed,
                      qwen=capacity_group([r for r in rows if r["backend"].startswith("qwen")]),
                      flash=capacity_group([r for r in rows if r["backend"] == "flash"]))
        if resumed:
            for name in ("qwen", "flash"):
                result[name].update(aggregate_valid=False, group_wall_s=None,
                                    aggregate_output_tokens_per_wall_s=None, aggregate_requests_per_wall_s=None)
        save(path, result)
        rounds.append(result)
    save(run.root / "summary.json", dict(rounds=rounds,
         aggregate_rates={b: distribution(r[b]["aggregate_output_tokens_per_wall_s"] for r in rounds) for b in ("qwen", "flash")},
         note="Aggregate denominator is group makespan (including queue + prefill). Per-request metrics use each request's own wall. Interrupted/resumed partial rounds have no aggregate rate; no mixed-invocation makespan."))
    run.status("complete" if len(rounds) == run.args.repeats else "deadline")
    return 0 if len(rounds) == run.args.repeats else 2


def evalplus_prompt(task):
    return ("Please provide a self-contained Python script that solves the following problem "
            "in a markdown code block. Preserve the requested function name and signature.\n```python\n"
            + task["prompt"].strip() + "\n```")


def evaluator_limits(seconds):
    """Local subprocess limits; official evaluator supplies its per-sample guard."""
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (1024, 1024))
    cpu = max(1, int(seconds))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))


def official_score(result, task_ids):
    values = result.get("eval", {})
    if set(values) != set(task_ids) or any(len(v) != 1 for v in values.values()):
        raise ValueError("Official result coverage must be exactly one sample per planned task")
    rows = [values[t][0] for t in task_ids]
    return dict(denominator=len(task_ids),
                base_pass_at_1=sum(r["base_status"] == "pass" for r in rows) / len(rows),
                plus_pass_at_1=sum(r["base_status"] == r["plus_status"] == "pass" for r in rows) / len(rows))


def evaluate_samples(run, dataset, backend, samples, task_ids, dataset_hash):
    folder = samples.parent
    status_path = folder / "evaluation.json"
    result_path = folder / "samples_eval_results.json"
    samples_hash = hashlib.sha256(samples.read_bytes()).hexdigest()
    if status_path.exists():
        previous = load(status_path)
        if previous.get("samples_sha256") == samples_hash and previous.get("status") == "complete":
            return previous
    if run.remaining() <= 0:
        return dict(status="deadline", denominator=len(task_ids))
    work = folder / "work"
    work.mkdir(exist_ok=True)
    tmp = work / "tmp"
    tmp.mkdir(exist_ok=True)
    command = [run.args.evalplus_python, "-m", "evalplus.evaluate", "--dataset", dataset,
               "--samples", str(samples), "--parallel", "2"]
    seconds = min(run.remaining(), run.args.evaluation_seconds)
    info: dict[str, Any] = dict(status="running", command=command, dataset=dataset, backend=backend,
                samples_sha256=samples_hash, dataset_hash=dataset_hash, parallel=2,
                isolation="local subprocess; no container/security boundary; official EvalPlus untrusted_check guard; cwd/TMPDIR restricted to run work; core disabled, CPU time bounded, nofile=1024; network and host filesystem not isolated",
                timeout_seconds=seconds)
    save(status_path, info)
    env = dict(os.environ, TMPDIR=str(tmp), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    start = time.monotonic()
    # New process group: timeout kills evaluator AND generated-code descendants.
    with (folder / "evaluation.log").open("ab") as log:
        process = subprocess.Popen(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True,
                                   preexec_fn=lambda: evaluator_limits(seconds))
        try:
            code = process.wait(timeout=seconds)
            info.update(returncode=code, status="complete" if code == 0 else "evaluation_infrastructure")
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            info.update(returncode=process.returncode, status="deadline")
    info["wall_s"] = time.monotonic() - start
    if info["status"] == "complete":
        try:
            result = load(result_path)
            if result.get("hash") != dataset_hash:
                raise ValueError("official dataset hash mismatch")
            info.update(official_score(result, task_ids))
        except Exception as exc:
            info.update(status="evaluation_infrastructure", error=str(exc))
    save(status_path, info)
    return info


def evalplus(run):
    data_module = importlib.import_module("evalplus.data")
    get_human_eval_plus = data_module.get_human_eval_plus
    get_human_eval_plus_hash = data_module.get_human_eval_plus_hash
    get_mbpp_plus = data_module.get_mbpp_plus
    get_mbpp_plus_hash = data_module.get_mbpp_plus_hash
    sanitize = importlib.import_module("evalplus.sanitize").sanitize
    version = importlib.metadata.version("evalplus")
    if version != "0.3.1":
        raise ValueError(f"Expected EvalPlus 0.3.1; got {version}")
    summary = {}
    for dataset, getter, hasher, expected_count in (
        ("humaneval", get_human_eval_plus, get_human_eval_plus_hash, 164),
        ("mbpp", get_mbpp_plus, get_mbpp_plus_hash, 378)):
        # Refuse dataset downloads: these loaders must resolve an existing cache.
        from unittest.mock import patch
        def no_download(*args, **kwargs):
            raise RuntimeError("Dataset not cached; downloads disabled")
        with patch("wget.download", no_download):
            tasks = getter()
            dataset_hash = hasher()
        if len(tasks) != expected_count:
            raise ValueError(f"Expected {expected_count} {dataset} tasks, found {len(tasks)}")
        version_module = importlib.import_module("evalplus.data." + dataset)
        dataset_version = getattr(version_module, "HUMANEVAL_PLUS_VERSION" if dataset == "humaneval" else "MBPP_PLUS_VERSION")
        override = os.environ.get("HUMANEVAL_OVERRIDE_PATH" if dataset == "humaneval" else "MBPP_OVERRIDE_PATH")
        metadata = dict(dataset=dataset, dataset_hash=dataset_hash,
                        dataset_version="override (hash authoritative)" if override else dataset_version,
                        evalplus_version=version, task_count=len(tasks),
                        task_prompt_hash=digest({k: dict(prompt=v["prompt"], entry_point=v["entry_point"]) for k, v in tasks.items()}),
                        generation=dict(temperature=0, max_tokens=4096, thinking=False, top_p=1),
                        sanitization="evalplus.sanitize.sanitize(raw, entrypoint=task['entry_point'])",
                        evaluator_python=run.args.evalplus_python)
        data_path = run.root / dataset / "dataset.json"
        if data_path.exists() and load(data_path) != metadata:
            raise ValueError("Dataset or generation contract changed on resume")
        save(data_path, metadata)
        for index, (task_id, task) in enumerate(tasks.items()):
            key = f"{dataset}-{task_id.replace('/', '_')}"
            tag_path = run.private / "fixtures" / (key + ".json")
            if not tag_path.exists():
                if run.remaining() <= 0:
                    break
                save(tag_path, dict(tag=uuid.uuid4().hex, task_id=task_id, prompt=evalplus_prompt(task)))
            data = load(tag_path)
            messages = paired_messages(data["tag"], data["prompt"])
            if run.remaining() <= 0:
                break
            run.status("generation", dataset=dataset, task_id=task_id, index=index, denominator=len(tasks))
            # A task barrier ensures serial generation per backend, two backends concurrently.
            with ThreadPoolExecutor(max_workers=2) as pool:
                jobs = [pool.submit(run.request, key + "-" + b, b, messages,
                                    dict(task_id=task_id, dataset=dataset, dataset_hash=dataset_hash), 4096)
                        for b in ("qwen", "flash")]
                for job in jobs:
                    job.result()
        summary[dataset] = {}
        for backend in ("qwen", "flash"):
            folder = run.private / "evalplus" / dataset / backend
            folder.mkdir(parents=True, exist_ok=True)
            samples = []
            records = []
            for task_id, task in tasks.items():
                path = run.private / "records" / f"{dataset}-{task_id.replace('/', '_')}-{backend}.json"
                if not path.exists():
                    continue
                row = load(path)
                # Transport failures get an empty solution, never omitted or retried.
                try:
                    solution = sanitize(row["content"], entrypoint=task["entry_point"]) if not row["error"] else ""
                    sanitize_error = None
                except Exception as exc:
                    solution, sanitize_error = "", f"{type(exc).__name__}: {exc}"
                samples.append(dict(task_id=task_id, solution=solution))
                records.append(dict(**public_row(row), sanitized_sha256=digest(solution),
                                    sanitized_chars=len(solution), sanitize_error=sanitize_error))
            sample_path = folder / "samples.jsonl"
            sample_text = "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples)
            if not sample_path.exists() or sample_path.read_text() != sample_text:
                # No official cached result can survive a changed sample artifact.
                if (folder / "samples_eval_results.json").exists():
                    raise ValueError("Samples changed after evaluation; use a new run directory")
                sample_path.write_text(sample_text)
            save(folder / "metadata.json", records)
            score: dict[str, Any] = dict(status="generation_incomplete", denominator=len(tasks), attempted=len(records),
                         outcomes=dict(Counter(r["outcome"] for r in records)),
                         sanitizer_failures=sum(bool(r["sanitize_error"]) for r in records))
            if len(samples) == len(tasks) and not run.args.generate_only:
                score["official"] = evaluate_samples(run, dataset, backend, sample_path, list(tasks), dataset_hash)
                score["status"] = score["official"]["status"]
            elif len(samples) == len(tasks):
                score["status"] = "generated"
            summary[dataset][backend] = score
        save(run.root / "summary.json", summary)
        if run.remaining() <= 0:
            break
    desired = "generated" if run.args.generate_only else "complete"
    complete = len(summary) == 2 and all(v["status"] == desired for ds in summary.values() for v in ds.values())
    run.status("complete" if complete else "incomplete", datasets=summary)
    return 0 if complete else 2


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("speed", "doc-quality", "capacity", "evalplus"))
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--overall-seconds", type=float, default=108000, help="Per-invocation overall deadline; resume grants a new budget")
    p.add_argument("--request-seconds", type=float, default=1800)
    p.add_argument("--evaluation-seconds", type=float, default=7200)
    p.add_argument("--body-tokens", type=int, nargs="+", default=[4096, 32768, 131072, 245760])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=2715)
    p.add_argument("--qwen-cache-prompt-false", action="store_true", help="Opt-in only if endpoint supports cache_prompt=false; no automatic retry")
    p.add_argument("--evalplus-python", default=os.environ.get("EVALPLUS_PYTHON", sys.executable))
    p.add_argument("--generate-only", action="store_true", help="Resume without this flag to perform official evaluation")
    for key, value in MODELS.items():
        p.add_argument(f"--{key}-url", default=os.environ.get(key.upper() + "_URL", value["base_url"]))
        p.add_argument(f"--{key}-model", default=os.environ.get(key.upper() + "_MODEL", value["model"]))
    return p


def main():
    os.umask(0o077)
    p = parser()
    args = p.parse_args()
    if min(args.overall_seconds, args.request_seconds, args.evaluation_seconds, args.repeats, args.max_tokens, *args.body_tokens) <= 0:
        p.error("budgets, repeats and token counts must be positive")
    if args.phase in ("speed", "doc-quality") and max(args.body_tokens) + args.max_tokens + 1024 > 262144:
        p.error("body + output + 1024 template headroom must fit Qwen context=262144")
    run = None
    try:
        run = Run(args)
        return {"speed": speed, "doc-quality": speed, "capacity": capacity, "evalplus": evalplus}[args.phase](run)
    except BaseException as exc:
        if run:
            run.status("interrupted" if isinstance(exc, KeyboardInterrupt) else "infrastructure", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    sys.exit(main())
