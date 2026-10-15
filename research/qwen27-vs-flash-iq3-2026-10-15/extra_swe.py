#!/usr/bin/env python3
"""Separate additional-40 CLI; import unchanged paired controller with new smoke IDs."""
import argparse
import contextlib
import os
from pathlib import Path
import signal
import time

import extra_prepare as prep

HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE / "runs/additional40-single-instance"
PRIMARY = HERE / "runs/primary-single-instance"


def deadlines(primary):
    ledger = prep.read(primary / "ledger.json")
    result = {key: ledger[key] for key in ("generation_deadline_epoch", "envelope_deadline_epoch")}
    if not all(isinstance(v, (int, float)) for v in result.values()):
        raise ValueError("primary deadlines have not been established")
    if result["envelope_deadline_epoch"] <= result["generation_deadline_epoch"]:
        raise ValueError("invalid primary deadline ordering")
    return result


def controller(cohort):
    # Guards run before importing controller code. Only this process's SMOKE
    # global is changed; no file, config, ledger or primary process is modified.
    manifest, policy = prep.verify(cohort)
    import paired_swe as swe
    os.environ.update(swe.OFFLINE)
    swe.SMOKE = list(manifest["selection"]["smoke_ids"])
    return swe, manifest, policy


def init(args):
    import shutil
    swe, manifest, policy = controller(args.cohort)
    run = prep.under_here(args.run_dir)
    primary = Path(args.primary_run).expanduser().resolve()
    if run == primary or run.is_relative_to(primary):
        raise ValueError("additional run must be separate from primary")
    inherited = deadlines(primary)
    if inherited != policy["primary_deadlines"]:
        raise ValueError("primary deadlines differ from frozen cohort policy")
    if prep.sha(primary / "frozen/source-manifest.json") != policy["original_manifest_file_sha256"]:
        raise ValueError("primary run source manifest differs from excluded original cohort")
    if time.time() >= inherited["generation_deadline_epoch"]:
        raise ValueError("original generation deadline already passed")
    # Must precede even creating a run directory; no automatic image pulling here.
    prep.images_cached(args.cohort)
    args.manifest = args.cohort / "manifest.json"
    args.evaluator = args.cohort / "canonical-evaluator.HOST-ONLY.json"
    args.helper = prep.PROJECT / "tools/swebench_controller.py"
    swe.init(args)
    # Freeze wrapper/policy and inherited epochs BEFORE allowing any run command.
    frozen = run / "frozen"
    extras = {"extra_swe.py": HERE / "extra_swe.py", "extra_prepare.py": HERE / "extra_prepare.py",
              "extra-policy.json": args.cohort / "policy.json", "extra-seal.HOST-ONLY.json": args.cohort / "seal.HOST-ONLY.json"}
    for name, source in extras.items():
        shutil.copyfile(source, frozen / name)
        (frozen / name).chmod(0o444)
    prep.write(frozen / "extra-deadlines.HOST-ONLY.json", inherited, 0o400)
    hashes = prep.read(run / "immutable-sha256.json")
    for name in (*extras, "extra-deadlines.HOST-ONLY.json"):
        hashes[f"frozen/{name}"] = prep.sha(frozen / name)
    prep.write(run / "immutable-sha256.json", hashes, 0o444)
    ledger = prep.read(run / "ledger.json")
    ledger.update(inherited)
    ledger["extra_cohort"] = {"manifest_sha256": manifest["manifest_sha256"], "policy_sha256": prep.sha(args.cohort / "policy.json"),
                              "wrapper_sha256": prep.sha(HERE / "extra_swe.py"), "primary_run": str(primary),
                              "separate_cohort": True, "original_overlap": 0}
    swe.save(run, ledger)
    prep.write(run / "extra-ready.HOST-ONLY.json", {"manifest_sha256": manifest["manifest_sha256"], **inherited}, 0o400)
    print("Additional cohort initialized with original absolute deadlines; no generation launched")


def verify_run(run, cohort, primary):
    swe, manifest, policy = controller(cohort)
    ready = prep.read(run / "extra-ready.HOST-ONLY.json")
    ledger = prep.read(run / "ledger.json")
    expected = prep.read(run / "frozen/extra-deadlines.HOST-ONLY.json")
    current_primary = deadlines(primary)
    for key, value in expected.items():
        if ledger.get(key) != value or ready.get(key) != value or current_primary[key] != value:
            raise RuntimeError("original absolute deadline mismatch; refusing extension")
    for name in ("extra_swe.py", "extra_prepare.py"):
        if prep.sha(HERE / name) != prep.sha(run / "frozen" / name):
            raise RuntimeError(f"wrapper source hash mismatch: {name}")
    if ready["manifest_sha256"] != manifest["manifest_sha256"]:
        raise RuntimeError("run belongs to a different additional cohort")
    swe.verify(run)
    return swe, ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "run", "status"):
        p = sub.add_parser(name)
        p.add_argument("--cohort", type=Path, default=prep.COHORT)
        p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
        p.add_argument("--primary-run", type=Path, default=PRIMARY)
        if name == "init":
            p.add_argument("--label", default="additional40-single-instance-native262k")
            for model, port, alias in [("qwen", 8080, "qwen3.8-27b-atx-iq4xs-m-262144-gpu1"),
                                        ("flash", 18144, "qwen3.8-flash-next-iq3_s")]:
                p.add_argument(f"--{model}-endpoint", default=os.getenv(f"{model.upper()}_ENDPOINT", f"http://127.0.0.1:{port}/v1"))
                p.add_argument(f"--{model}-alias", default=os.getenv(f"{model.upper()}_ALIAS", alias))
                p.add_argument(f"--{model}-backend", default="hosted_vllm")
        if name == "run":
            p.add_argument("--swe-phase", action="store_true")
            p.add_argument("--resume", action="store_true")
            p.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    args.run_dir = prep.under_here(args.run_dir)
    args.cohort = prep.under_here(args.cohort)
    if args.command == "init":
        init(args)
    else:
        swe, ledger = verify_run(args.run_dir, args.cohort, args.primary_run)
        if args.command == "status":
            import json
            print(json.dumps(swe.summarize(ledger), indent=2))
        else:
            if not args.swe_phase:
                parser.error("--swe-phase required; main orchestrator must exclusively reserve both endpoints")
            primary = prep.read(args.primary_run / "ledger.json")
            if primary["phase"] != "complete":
                raise RuntimeError("primary must finish official evaluation before additional generation")
            prep.images_cached(args.cohort)
            swe.STOP.clear()
            signal.signal(signal.SIGTERM, lambda *_: swe.STOP.set())
            signal.signal(signal.SIGINT, lambda *_: swe.STOP.set())
            swe.run_all(args)


if __name__ == "__main__":
    main()
