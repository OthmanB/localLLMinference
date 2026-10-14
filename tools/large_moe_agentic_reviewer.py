#!/usr/bin/env python3
"""Direct Docker reviewer screen for the three-GPU large-MoE probe."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path((os.environ['EXPERIMENT_ROOT'] + ''))
sys.path.insert(0, str(ROOT / "tools"))

import mistral_agentic_reviewer as reviewer  # noqa: E402
import mistral_context_sweep as sweep  # noqa: E402


MODEL = "deepseek-review"
PORT = 8090
ALIAS = "deepseek-v4-flash-0731-iq3-xxs"
MODEL_STORE = Path(os.environ.get("MODEL_STORE", ROOT / "models"))
MODEL_PATH = MODEL_STORE / "deepseek-v4-flash-0731-iq3-xxs-gguf/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf"

sweep.MODELS[MODEL] = {
    "gpu": "0,1,2",
    "port": PORT,
    "alias": ALIAS,
    "path": MODEL_PATH,
}
reviewer.MODEL = MODEL


def start_server(ctx: int, n_cpu_moe: int, tensor_split: str, log_path: Path) -> subprocess.Popen:
    args = [
        str(reviewer.sweep.LLAMA),
        "--model", str(MODEL_PATH),
        "--alias", ALIAS,
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--ctx-size", str(ctx),
        "--parallel", "1",
        "--no-cache-prompt",
        "--n-gpu-layers", "99",
        "--fit", "off",
        "--tensor-split", tensor_split,
        "--n-cpu-moe", str(n_cpu_moe),
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--flash-attn", "on",
        "--threads", "32",
        "--threads-batch", "32",
        "--batch-size", "2048",
        "--ubatch-size", "512",
        "--metrics",
        "--perf",
        "--reasoning", "auto",
        "--log-colors", "off",
        "-lv", "4",
    ]
    log = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        args,
        stdout=log,
        stderr=subprocess.STDOUT,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "0,1,2",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "0",
        },
        start_new_session=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct repository-aware DeepSeek reviewer diagnostic")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="*", default=["case1", "case2", "case3", "case4"])
    parser.add_argument("--ctx", type=int, default=131072)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--n-cpu-moe", type=int, default=24)
    parser.add_argument("--tensor-split", default="0.5,1,1")
    parser.add_argument("--startup-timeout", type=int, default=1800)
    args = parser.parse_args()

    if not MODEL_PATH.is_file():
        raise SystemExit(f"model not found: {MODEL_PATH}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_cases = json.loads(reviewer.CASES_PATH.read_text(encoding="utf-8"))
    by_id = {case["id"]: case for case in all_cases}
    missing = [case_id for case_id in args.cases if case_id not in by_id or case_id not in reviewer.IMAGES]
    if missing:
        raise SystemExit(f"unknown or unsupported case IDs: {missing}")

    server_log = args.output.parent / "deepseek-review.server.log"
    process = start_server(args.ctx, args.n_cpu_moe, args.tensor_split, server_log)
    results: list[dict] = []
    try:
        sweep.wait_ready(PORT, process, args.startup_timeout)
        if not reviewer.tool_smoke(PORT, ALIAS, args.output.parent / "tool-smoke.json"):
            raise RuntimeError("DeepSeek bash tool-call smoke test failed")
        for case_id in args.cases:
            result = reviewer.run_case(by_id[case_id], args.output.parent, ALIAS, args.max_steps)
            result.update({
                "model": ALIAS,
                "context_tokens": args.ctx,
                "n_cpu_moe": args.n_cpu_moe,
                "tensor_split": args.tensor_split,
            })
            results.append(result)
            args.output.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
            print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        reviewer.cleanup_container("deepseek-review-unused")
        reviewer.sweep.terminate(process)


if __name__ == "__main__":
    main()
