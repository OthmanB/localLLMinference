#!/usr/bin/env python3
"""Offline, allowlisted reporting of two fixed paired SWE cohorts (stdlib only).

Never imports a runner or reads evaluator datasets, trajectories, or patches on
disk. Ledger patch presence is inspected only to distinguish no-patch failures;
no patch contents or free-text ledger fields are copied into output.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

HERE = Path(__file__).resolve().parent
MODELS = ("qwen", "flash")
BOOTSTRAP_SEED = 20261016
STATES = {"queued", "running", "submitted", "failed", "infra_failed",
          "infra_retry", "deadline", "interrupted", "not_initialized", "missing"}


def load(path):
    # Reject duplicate object keys, including duplicated task IDs in a ledger.
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key rejected")
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(path, expected_count=40):
    data = load(path)
    body = {k: v for k, v in data.items() if k != "manifest_sha256"}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != data.get("manifest_sha256"):
        raise ValueError("manifest body checksum mismatch")
    tasks = data["tasks"]
    ids = [t["instance_id"] for t in tasks]
    if len(ids) != expected_count or len(set(ids)) != len(ids):
        raise ValueError("cohort must contain exact unique task IDs")
    # Public task identity is restricted to the benchmark's ID/repository syntax.
    import re
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-[0-9]+", i) for i in ids):
        raise ValueError("invalid benchmark task ID")
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", t["repo"]) for t in tasks):
        raise ValueError("invalid benchmark repository")
    seed = data["selection"].get("seed")
    if type(seed) is not int:
        raise ValueError("selection seed must be an integer")
    return data, {"file_sha256": sha(path), "body_sha256": digest, "selection_seed": seed}


def enforce_disjoint(cohorts):
    seen = set()
    for tasks in cohorts:
        ids = [t["instance_id"] for t in tasks]
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError("duplicate or overlapping cohort task IDs rejected")
        seen.update(ids)


def wilson(successes, n):
    if not 0 <= successes <= n or n <= 0:
        raise ValueError("invalid Wilson counts")
    z = 1.959963984540054
    p = successes / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [max(0.0, center-half), min(1.0, center+half)]


def mcnemar(qwen_only, flash_only):
    """Exact two-sided conditional Binomial(n_discordant, .5), doubled tail."""
    n = qwen_only + flash_only
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(qwen_only, flash_only)+1)) / 2**n)


def percentile(values, p):
    pos = (len(values)-1)*p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return values[lo] + (values[hi]-values[lo])*(pos-lo)


def paired_bootstrap(rows, samples=20000, seed=BOOTSTRAP_SEED):
    """Percentile interval, fixed-size resampling within cohort/repository cells.

    Singleton cells are held fixed. If all cells are singletons no bootstrap
    interval is identifiable; return None rather than fabricate variability.
    """
    if samples < 1 or not rows or any(r["qwen"] is None or r["flash"] is None for r in rows):
        raise ValueError("bootstrap requires determinate pairs and positive samples")
    strata = defaultdict(list)
    for row in rows:
        strata[(row["cohort"], row["repo"])].append(int(row["qwen"])-int(row["flash"]))
    cells = [strata[key] for key in sorted(strata)]
    if not any(len(cell) > 1 for cell in cells):
        return None
    rng = random.Random(seed)
    draws = sorted(sum(sum(cell[rng.randrange(len(cell))] for _ in cell) for cell in cells)/len(rows)
                   for _ in range(samples))
    return [percentile(draws, .025), percentile(draws, .975)]


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def task_view(task):
    if task is None:
        task = {"state": "missing", "attempts": []}
    state = task.get("state")
    state = state if state in STATES else "unknown"
    attempts = task.get("attempts", [])
    if not isinstance(attempts, list):
        raise ValueError("attempts must be a list")
    resolved = task.get("evaluation", {}).get("resolved")
    official = type(resolved) is bool and state == "submitted"
    no_patch_failure = state == "failed" and not task.get("patch", "").strip()
    outcome = resolved if official else False if no_patch_failure else None
    category = ("official_success" if resolved else "official_patch_failure") if official else (
        "generation_no_patch_failure" if no_patch_failure else
        "deployment_infrastructure_failure" if state == "infra_failed" else
        "pending_official_evaluation" if state == "submitted" else "unknown_or_pending")
    metrics = {}
    for label, key in (("wall_seconds", None), ("output_tokens", "completion_tokens"),
                       ("backend_calls", "backend_requests"), ("agent_calls", "agent_calls")):
        values = []
        for attempt in attempts:
            measured = attempt.get("metrics", {})
            value = attempt.get("wall_seconds") if key is None else measured.get(key)
            if key == "backend_requests" and value is None:
                value = measured.get("agent_calls")
            if number(value):
                values.append(value)
        metrics[label] = {"total_observed": sum(values), "attempts_measured": len(values)}
    return {"state": state, "attempted": bool(attempts), "attempt_count": len(attempts),
            "official_scored": official, "outcome": outcome, "category": category,
            "infra_attempts": sum(a.get("outcome") == "infra_retry" for a in attempts),
            "metrics": metrics}


def model_summary(views):
    n = len(views)
    states = Counter(v["state"] for v in views)
    categories = Counter(v["category"] for v in views)
    known = sum(v["outcome"] is not None for v in views)
    successes = sum(v["outcome"] is True for v in views)
    metrics = {}
    for label in ("wall_seconds", "output_tokens", "backend_calls", "agent_calls"):
        measured_tasks = [v["metrics"][label]["total_observed"] for v in views
                          if v["attempt_count"] and v["metrics"][label]["attempts_measured"] == v["attempt_count"]]
        metrics[label] = {
            "total_observed": sum(v["metrics"][label]["total_observed"] for v in views),
            "attempts_measured": sum(v["metrics"][label]["attempts_measured"] for v in views),
            "tasks_with_all_attempts_measured": len(measured_tasks),
            "mean_task_total_if_all_attempts_measured": statistics.mean(measured_tasks) if measured_tasks else None,
            "median_task_total_if_all_attempts_measured": statistics.median(measured_tasks) if measured_tasks else None,
        }
    return {"denominator": n, "states": dict(sorted(states.items())), "categories": dict(sorted(categories.items())),
            "attempted": sum(v["attempted"] for v in views), "submitted": states["submitted"],
            "failed_state": states["failed"],
            "failed_generation_no_patch": categories["generation_no_patch_failure"],
            "infra_failed": states["infra_failed"], "infra_retry_attempts": sum(v["infra_attempts"] for v in views),
            "attempts": sum(v["attempt_count"] for v in views),
            "official_scored": sum(v["official_scored"] for v in views),
            "official_scored_coverage": sum(v["official_scored"] for v in views)/n,
            "submitted_missing_official_bool": sum(v["state"] == "submitted" and not v["official_scored"] for v in views),
            "determinate": known, "unknown": n-known, "successes_observed": successes,
            "success_fraction_lower_bound": successes/n, "success_fraction_upper_bound": (successes+n-known)/n,
            "complete_success_score": successes/n if known == n else None,
            "wilson95": wilson(successes, n) if known == n else None,
            "deployment_submission_fraction": states["submitted"]/n, "metrics": metrics}


def summarize(rows, samples):
    models = {m: model_summary([r["models"][m] for r in rows]) for m in MODELS}
    pairs = Counter({k: 0 for k in ("both_resolved", "qwen_only", "flash_only", "both_unresolved", "unknown_pairs")})
    paired = []
    for row in rows:
        q, f = (row["models"][m]["outcome"] for m in MODELS)
        key = "unknown_pairs" if q is None or f is None else (
            "both_resolved" if q and f else "qwen_only" if q else "flash_only" if f else "both_unresolved")
        pairs[key] += 1
        paired.append({"cohort": row["cohort"], "repo": row["repo"], "qwen": q, "flash": f})
    complete = pairs["unknown_pairs"] == 0
    paired_metrics = {}
    for label in ("wall_seconds", "output_tokens", "backend_calls", "agent_calls"):
        observed = []
        for row in rows:
            views = [row["models"][m] for m in MODELS]
            if all(v["attempt_count"] and v["state"] in {"submitted", "failed", "infra_failed"}
                   and v["metrics"][label]["attempts_measured"] == v["attempt_count"] for v in views):
                observed.append(views[0]["metrics"][label]["total_observed"] - views[1]["metrics"][label]["total_observed"])
        paired_metrics[label] = {"terminal_pairs_measured": len(observed),
                                 "mean_qwen_minus_flash": statistics.mean(observed) if observed else None}
    return {"tasks": len(rows), "models": models, "pairs": dict(pairs), "all_outcomes_determinate": complete,
            "paired_difference_qwen_minus_flash": (pairs["qwen_only"]-pairs["flash_only"])/len(rows) if complete else None,
            "paired_difference95": paired_bootstrap(paired, samples) if complete else None,
            "mcnemar_exact_two_sided_p": mcnemar(pairs["qwen_only"], pairs["flash_only"]) if complete else None,
            "paired_resource_comparison": paired_metrics}


def read_cohort(root, name, run_name, source_manifest):
    data, provenance = manifest(root / source_manifest)
    run = root / "runs" / run_name
    frozen = run / "frozen/source-manifest.json"
    if frozen.exists() and sha(frozen) != provenance["file_sha256"]:
        raise ValueError("run manifest does not match selected cohort")
    path = run / "ledger.json"
    ledger = load(path) if path.exists() else None
    ids = {t["instance_id"] for t in data["tasks"]}
    if ledger is not None:
        if set(ledger["tasks"]) != set(MODELS) or any(set(ledger["tasks"][m]) != ids for m in MODELS):
            raise ValueError("ledger must pair exactly the manifest IDs for both models")
    rows = []
    for task in data["tasks"]:
        iid = task["instance_id"]
        rows.append({"instance_id": iid, "repo": task["repo"], "cohort": name,
                     "models": {m: task_view(ledger["tasks"][m][iid] if ledger else
                                 {"state": "not_initialized", "attempts": []}) for m in MODELS}})
    records = ledger.get("evaluations", []) if ledger else []
    status = {"initialized": ledger is not None, "controller_complete": bool(ledger and ledger.get("phase") == "complete"),
              "evaluation_records": len(records),
              "evaluation_records_unfinished": sum("finished_epoch" not in r for r in records),
              "evaluation_records_failed_or_timed_out": sum(r.get("returncode") not in (None, 0) or r.get("timed_out") is True for r in records)}
    return data, provenance, rows, status


def protocol_check(root):
    """Compare frozen instruction/sampling files and allowlisted config limits.

    File hashes avoid parsing/copying endpoint-bearing YAML. Historical runtime
    fields in selection manifests are deliberately not used as protocol evidence.
    """
    runs = [root / "runs" / name for name in ("primary-single-instance", "additional40-single-instance")]
    files = ("frozen/stock.yaml", "frozen/qwen.yaml", "frozen/flash.yaml", "frozen/paired_swe.py",
             "frozen/instrumented_model.py", "frozen/official_eval.py")
    comparisons = {}
    for name in files:
        hashes = [sha(run/name) if (run/name).exists() else None for run in runs]
        comparisons[name.split("/")[-1]] = {"original_sha256": hashes[0], "additional_sha256": hashes[1],
                                              "equal": hashes[0] == hashes[1] if all(hashes) else None}
    keys = ("generation_seconds", "envelope_seconds", "task_seconds", "infra_retries", "context_tokens", "versions")
    configs = [load(run/"config.json") if (run/"config.json").exists() else {} for run in runs]
    config_equal = {key: configs[0][key] == configs[1][key] if all(key in c for c in configs) else None for key in keys}
    ledgers = [load(run/"ledger.json") if (run/"ledger.json").exists() else {} for run in runs]
    deadlines = {key: ledgers[0][key] == ledgers[1][key] if all(key in l for l in ledgers) else None
                 for key in ("generation_deadline_epoch", "envelope_deadline_epoch")}
    checks = [v["equal"] for v in comparisons.values()] + list(config_equal.values()) + list(deadlines.values())
    return {"verified_same": all(v is True for v in checks), "frozen_files": comparisons,
            "config_fields_equal": config_equal, "inherited_deadlines_equal": deadlines,
            "mismatch_count": sum(v is False for v in checks), "unverified_count": sum(v is None for v in checks)}


def build_report(root, samples=20000):
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    definitions = (("original40", "primary-single-instance", "runs/primary-single-instance/frozen/source-manifest.json"),
                   ("additional40", "additional40-single-instance", "extra-cohort/manifest.json"))
    cohorts, all_rows, manifests, statuses = {}, [], [], {}
    for name, run, source in definitions:
        data, provenance, rows, status = read_cohort(root, name, run, source)
        manifests.append(data)
        all_rows.extend(rows)
        cohorts[name] = {"manifest": provenance, **summarize(rows, samples)}
        statuses[name] = status
    enforce_disjoint([m["tasks"] for m in manifests])
    if manifests[1]["selection"]["seed"] != 20261016 or manifests[1].get("original_manifest_sha256") != manifests[0]["manifest_sha256"]:
        raise ValueError("additional cohort seed or original manifest reference mismatch")
    cohorts["combined80"] = summarize(all_rows, samples)
    protocol = protocol_check(root)
    blockers = []
    if not cohorts["combined80"]["all_outcomes_determinate"]:
        blockers.append("unknown_task_outcomes")
    if any(cohorts["combined80"]["models"][m]["submitted_missing_official_bool"] for m in MODELS):
        blockers.append("submitted_patches_missing_official_bool")
    if any(not s["controller_complete"] or s["evaluation_records_unfinished"] for s in statuses.values()):
        blockers.append("controller_or_evaluator_incomplete")
    if not protocol["verified_same"]:
        blockers.append("resource_instructions_sampling_or_deadlines_unverified_or_mismatched")
    return {"schema_version": 1, "report_kind": "unpublished_snapshot", "final_claim": False,
            "generated_utc": datetime.now(timezone.utc).isoformat(), "cohorts": cohorts,
            "evaluator_status": statuses, "protocol": protocol,
            "integration_gate": {"ready_for_complete_comparison": not blockers, "blocking_conditions": blockers,
                                 "thresholds": {"unknown_outcomes": 0, "submitted_without_official_bool": 0,
                                                "unfinished_evaluations": 0, "protocol_mismatches_or_unverified": 0,
                                                "controllers_complete": 2}},
            "statistics": {"bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_samples": samples,
                           "confidence_level": .95, "bootstrap_method": "paired percentile; fixed cohort/repository stratum sizes; singleton strata fixed; no CI when all singleton",
                           "mcnemar_method": "exact two-sided doubled conditional Binomial(discordant pairs, 0.5) tail; zero discordance p=1"},
            "task_progress": all_rows,
            "interpretation": [
                "Progress snapshot, not a final accuracy or publication claim. Sources: selected manifests and atomic task ledgers, not status summaries.",
                "Success means official resolved=True. Submitted patches need official bool. Terminal failed generation without a patch is a determinate protocol failure, not an official score.",
                "Infrastructure exhaustion, deadlines, running, queued, interrupted and unscored submitted patches remain unknown; never imputed False. Unknown pairs are excluded from the four determinate pair cells.",
                "Full-cohort success scores, Wilson intervals, paired difference intervals and McNemar p values are withheld until all relevant task outcomes are determinate. Observed lower/upper bounds are not accuracy estimates.",
                "Deployment submission availability includes infrastructure/deadline effects. Official patch failures and no-patch generation failures are separate categories; neither isolates intrinsic code quality from this deployment and agent protocol.",
                "Wilson intervals assume independent Bernoulli task outcomes with a common probability as a descriptive approximation. Bootstrap assumes exchangeable paired tasks within repository/cohort cells; singleton cells provide no estimated variation. Degenerate intervals can underestimate uncertainty.",
                "Fixed input-metadata-stratified selection is not a simple random sample of Verified500; single agent seed. No population-generalization probability guarantee. Repository/cohort cells approximate the finer input-length selection strata.",
                "Increasing 40 to 80 gives rough SE ratio sqrt(40/80)=0.707 (about 29% reduction) under comparable independent variance, not a guaranteed winner or realized precision gain.",
                "All attempt costs include retries. Missing measurements remain missing; observed totals can be incomplete, especially running attempts. Backend calls fall back to agent_calls when backend_requests is absent. Paired resource means use terminal pairs with every attempt measured.",
                "Frozen stock instructions, model sampling overlays, controller/evaluator implementations, config limits and inherited deadlines are compared. Missing additional-run files leave protocol unverified. Historical selection-manifest runtime settings are not active settings.",
            ]}


def markdown(report):
    cohorts = report["cohorts"]
    names = ("original40", "additional40", "combined80")
    lines = ["# Paired SWE cohort comparison — unpublished snapshot", "", "**Not a final accuracy or publication claim.**", "",
             "| Measure | Original 40 | Additional 40 | Combined 80 |", "|---|---:|---:|---:|"]
    def row(label, values):
        lines.append("| " + label + " | " + " | ".join(str(v) for v in values) + " |")
    row("Tasks per model", [cohorts[n]["tasks"] for n in names])
    for model in MODELS:
        for label, key in (("Attempted", "attempted"), ("Submitted", "submitted"), ("Failed generation, no patch", "failed_generation_no_patch"),
                           ("Failed state", "failed_state"), ("Infra exhausted", "infra_failed"), ("Infra retry attempts", "infra_retry_attempts"),
                           ("Official scored", "official_scored"), ("Official scored coverage", "official_scored_coverage"),
                           ("Submitted without official bool", "submitted_missing_official_bool"), ("Determinate", "determinate"),
                           ("Unknown/pending", "unknown"), ("Observed successes", "successes_observed"),
                           ("Deployment submission fraction", "deployment_submission_fraction"),
                           ("Success fraction lower bound (progress)", "success_fraction_lower_bound"),
                           ("Success fraction upper bound (progress)", "success_fraction_upper_bound"),
                           ("Complete-cohort score", "complete_success_score"), ("Wilson 95% CI", "wilson95")):
            row(f"{model}: {label}", [format_value(cohorts[n]["models"][model][key]) for n in names])
        for metric in ("wall_seconds", "output_tokens", "backend_calls", "agent_calls"):
            row(f"{model}: {metric} observed total", [format_value(cohorts[n]["models"][model]["metrics"][metric]["total_observed"]) for n in names])
            row(f"{model}: {metric} tasks fully measured", [cohorts[n]["models"][model]["metrics"][metric]["tasks_with_all_attempts_measured"] for n in names])
            row(f"{model}: {metric} mean task total (fully measured)", [format_value(cohorts[n]["models"][model]["metrics"][metric]["mean_task_total_if_all_attempts_measured"]) for n in names])
        row(f"{model}: official patch failures", [cohorts[n]["models"][model]["categories"].get("official_patch_failure", 0) for n in names])
    for key in ("both_resolved", "qwen_only", "flash_only", "both_unresolved", "unknown_pairs"):
        row(f"Pairs: {key}", [cohorts[n]["pairs"][key] for n in names])
    for label, key in (("Paired difference (Qwen − Flash)", "paired_difference_qwen_minus_flash"),
                       ("Paired bootstrap 95% CI", "paired_difference95"), ("Exact McNemar p", "mcnemar_exact_two_sided_p")):
        row(label, [format_value(cohorts[n][key]) for n in names])
    for metric in ("wall_seconds", "output_tokens", "backend_calls", "agent_calls"):
        row(f"Paired {metric}: mean Qwen − Flash", [format_value(cohorts[n]["paired_resource_comparison"][metric]["mean_qwen_minus_flash"]) for n in names])
        row(f"Paired {metric}: terminal pairs measured", [cohorts[n]["paired_resource_comparison"][metric]["terminal_pairs_measured"] for n in names])
    lines += ["", "## Evaluator and integration status", "", "Complete-comparison gate: **" + ("ready" if report["integration_gate"]["ready_for_complete_comparison"] else "BLOCKED / incomplete") + "**.",
              "Thresholds: zero unknown outcomes, zero submitted patches without official bool, zero unfinished evaluations, both controllers complete, and all frozen-protocol comparisons verified equal."]
    for condition in report["integration_gate"]["blocking_conditions"]:
        lines.append("- " + condition)
    for name, status in report["evaluator_status"].items():
        lines.append(f"- {name}: initialized={status['initialized']}; controller_complete={status['controller_complete']}; evaluation records={status['evaluation_records']}; unfinished={status['evaluation_records_unfinished']}; failed/timed-out historical records={status['evaluation_records_failed_or_timed_out']} (a later retry may supersede these).")
    lines += ["", "## Selection provenance", "", "Zero overlap enforced; exact identical manifest IDs paired across models."]
    for name in names[:2]:
        p = cohorts[name]["manifest"]
        lines.append(f"- {name}: selection seed `{p['selection_seed']}`; manifest body SHA-256 `{p['body_sha256']}`; file SHA-256 `{p['file_sha256']}`.")
    lines += ["", "## Protocol and statistical interpretation", "", f"Bootstrap: {report['statistics']['bootstrap_samples']} replicates; seed {BOOTSTRAP_SEED}; paired percentile 95% interval within repository/cohort strata. Exact McNemar uses the doubled conditional binomial tail.",
              f"Frozen protocol verified equal: {report['protocol']['verified_same']}; mismatches={report['protocol']['mismatch_count']}; unverified checks={report['protocol']['unverified_count']}."]
    lines.extend("- " + note for note in report["interpretation"])
    return "\n".join(lines) + "\n"


def format_value(value):
    if value is None:
        return "withheld / unavailable"
    if isinstance(value, list):
        return "[" + ", ".join(f"{v:.4f}" for v in value) + "]"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=HERE)
    parser.add_argument("--output-dir", type=Path, default=HERE / "combined-report")
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output == HERE or not output.is_relative_to(HERE):
        parser.error("output directory must be below this reporting study directory")
    # Avoid touching any live run or frozen cohort even via a symlink.
    relative = output.relative_to(HERE)
    if relative.parts[0] in {"runs", "extra-cohort", "campaign", "offline-validation", "offline-validation-final"}:
        parser.error("output directory must be a separate reporting directory")
    try:
        report = build_report(args.study_root.expanduser().resolve(), args.bootstrap_samples)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Report blocked: {type(exc).__name__}; check input schema, manifests and ledger pairing.\n")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.chmod(0o700)
    for name, content in (("combined-swe.json", json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n"),
                          ("COMBINED_SWE.md", markdown(report))):
        target = output / name
        if target.is_symlink():
            parser.error("report destination must not be a symlink")
        temporary = output / (name + ".tmp")
        if temporary.exists() or temporary.is_symlink():
            parser.error("temporary report destination already exists")
        with temporary.open("x") as stream:
            stream.write(content)
        temporary.chmod(0o600)
        temporary.replace(target)
    print(json.dumps({"outputs": ["COMBINED_SWE.md", "combined-swe.json"], "report_kind": report["report_kind"],
                      "integration_gate": report["integration_gate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
