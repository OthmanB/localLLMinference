#!/usr/bin/env python3
"""Phase 4: blind reviewer A/B.

Task: Mistral IQ3_XXS and IQ3_S review a fixed set of REAL, genuinely-flawed
proposed fixes (Qwen's unresolved SWE-bench patches). An independent scorer
compares the reviews with a fixed gold-derived key. The metric is valid
objections found and missed, not code-writing skill. The reviewer prompt is
identical for every case and model (blind model identity); none of the reviewers
is told which model is which or that the patch is flawed.

Mistral reviewers run at their maximum reasoning effort via per-request
chat_template_kwargs (reasoning_effort="high"; the template only accepts
none|high). Qwen requests remain opt-in diagnostics only and are not used for
supervision or the primary Phase 4 score.

Mistral reviewers are spawned on GPU 1+2 (reuse the Phase 2/3 server base, but a
short 32k context is enough for these ~2-5k-token review prompts and avoids the
128k premature-EOS regime). Qwen is the production server on 127.0.0.1:8080
(read-only inference requests; the server is left running).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mistral_context_sweep as sweep  # noqa: E402  (reuse tested helpers)

LLAMA = sweep.LLAMA
MISTRAL_CTX = 32768            # short context; these review prompts are ~2-5k tokens
REVIEW_EFFORT = "high"         # max effort for both families (Qwen maps high->xhigh)
REVIEW_TEMP = 0.8
REVIEW_MAX_TOKENS = 16384      # safety cap; Qwen xhigh reasoning is ~8k+ tokens, needs headroom
QWEN_PORT = 8080
QWEN_MODEL = "qwen3.8-27b-q4-tensor262k"

SYSTEM_PROMPT = (
    "You are a meticulous, adversarial senior software engineer performing a code "
    "review of a proposed bug fix. You hunt for concrete, specific defects: real bugs, "
    "missing code paths, unhandled edge cases, incomplete or incorrect logic, and "
    "regressions. You are skeptical and thorough, and you reason carefully before "
    "concluding. You cite the exact code and explain the consequence of each problem."
)


def build_user(problem: str, patch: str) -> str:
    return (
        "A developer proposed the following patch to fix a bug. Review it adversarially "
        "and critically.\n\n"
        f"=== BUG REPORT ===\n{problem}\n=== END BUG REPORT ===\n\n"
        f"=== PROPOSED PATCH (unified diff) ===\n{patch}\n=== END PROPOSED PATCH ===\n\n"
        "Task: identify every CONCRETE problem with this proposed patch that would (a) "
        "prevent it from correctly and COMPLETELY fixing the bug, or (b) introduce a new "
        "bug or regression. For each objection, state: the specific code/location, why it "
        "is wrong or insufficient, and the consequence. Rank your objections by severity. "
        "If, after careful review, you believe the patch is fully correct and complete, say "
        "so explicitly and list the specific things you verified. Report only real "
        "objections on the patch itself; do not restate the bug report."
    )


def review_payload(alias: str, case: dict, temp: float, max_tokens: int) -> dict:
    return {
        "model": alias,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user(case["problem"], case["candidate_patch"])},
        ],
        "max_tokens": max_tokens,
        "temperature": temp,
        "stream": False,
        "chat_template_kwargs": {"reasoning_effort": REVIEW_EFFORT},
    }


def call_reviewer(port: int, alias: str, case: dict, temp: float, seed: int | None,
                  max_tokens: int, timeout: int) -> dict:
    payload = review_payload(alias, case, temp, max_tokens)
    if seed is not None:
        payload["seed"] = seed
    t0 = time.monotonic()
    resp = sweep.http_json(f"http://127.0.0.1:{port}/v1/chat/completions", payload, timeout=timeout)
    wall = time.monotonic() - t0
    msg = resp["choices"][0].get("message", {})
    usage = resp.get("usage", {}) or {}
    return {
        "case_id": case["id"],
        "reasoning": msg.get("reasoning_content") or "",
        "answer": msg.get("content") or "",
        "finish_reason": resp["choices"][0].get("finish_reason"),
        "wall_seconds": round(wall, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_chars": len(msg.get("reasoning_content") or ""),
        "answer_chars": len(msg.get("content") or ""),
    }


def start_mistral_server(spec: dict, ctx: int, tensor_split: str, verbosity: int,
                         log_path: Path) -> subprocess.Popen:
    log = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        [
            str(LLAMA),
            "--model", str(spec["path"]),
            "--alias", spec["alias"],
            "--host", "127.0.0.1",
            "--port", str(spec["port"]),
            "--ctx-size", str(ctx),
            "--parallel", "1",
            "--no-cache-prompt",
            "--n-gpu-layers", sweep.N_GPU_LAYERS,
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
            "--flash-attn", "on",
            "--threads", sweep.THREADS,
            "--threads-batch", sweep.THREADS,
            "--batch-size", sweep.BATCH_SIZE,
            "--ubatch-size", sweep.UBATCH_SIZE,
            "--fit", "off",
            "--tensor-split", tensor_split,
            "--reasoning", "auto",
            "--metrics",
            "--perf",
            "-lv", str(verbosity),
        ],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": spec["gpu"],
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "0",
        },
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def run_mistral(model: str, cases: list[dict], outdir: Path, temp: float, seed: int | None,
                max_tokens: int, request_timeout: int, startup_timeout: int,
                tensor_split: str, verbosity: int) -> list[dict]:
    spec = sweep.MODELS[model]
    port = spec["port"]
    quant = "IQ3_XXS" if model == "mistral-xxs" else "IQ3_S"
    log_path = outdir / f"{model}.server.log"
    results: list[dict] = []
    proc = start_mistral_server(spec, MISTRAL_CTX, tensor_split, verbosity, log_path)
    try:
        sweep.wait_ready(port, proc, startup_timeout)
        res = sweep.check_residency(log_path)
        print(f"[{model}] residency: fully={res.get('fully_resident')} "
              f"offloaded={res.get('offloaded_layers')}/{res.get('total_layers')}", flush=True)
        sweep.warmup(port, spec["alias"])
        for c in cases:
            out = call_reviewer(port, spec["alias"], c, temp, seed, max_tokens, request_timeout)
            out.update({"model": model, "quant": quant, "port": port, "ctx": MISTRAL_CTX,
                        "seed": seed, "temperature": temp, "reasoning_effort": REVIEW_EFFORT})
            results.append(out)
            tag = f"{model}-{c['id']}"
            (outdir / f"{tag}.reasoning.txt").write_text(out["reasoning"], encoding="utf-8")
            (outdir / f"{tag}.answer.txt").write_text(out["answer"], encoding="utf-8")
            (outdir / f"{tag}.full.txt").write_text(
                f"[REASONING] ({out['reasoning_chars']} chars)\n{out['reasoning']}\n\n"
                f"[ANSWER] ({out['answer_chars']} chars)\n{out['answer']}\n",
                encoding="utf-8",
            )
            print(f"[{model}] {c['id']}: finish={out['finish_reason']} "
                  f"comp={out['completion_tokens']} reason_ch={out['reasoning_chars']} "
                  f"answer_ch={out['answer_chars']} wall={out['wall_seconds']}s", flush=True)
    except (OSError, RuntimeError, TimeoutError) as exc:
        print(f"[{model}] ERROR: {type(exc).__name__}: {exc}", flush=True)
        for c in cases:
            if not any(r["case_id"] == c["id"] for r in results):
                results.append({"case_id": c["id"], "model": model, "quant": quant,
                                "error": f"{type(exc).__name__}: {exc}"})
    finally:
        sweep.terminate(proc)
    return results


def run_qwen(cases: list[dict], outdir: Path, temp: float, seed: int | None,
             max_tokens: int, request_timeout: int) -> list[dict]:
    results: list[dict] = []
    for c in cases:
        out = call_reviewer(QWEN_PORT, QWEN_MODEL, c, temp, seed, max_tokens, request_timeout)
        out.update({"model": "qwen", "quant": "Q4_K_M (production, GPU0)", "port": QWEN_PORT,
                    "ctx": None, "seed": seed, "temperature": temp,
                    "reasoning_effort": f"{REVIEW_EFFORT} (->xhigh)"})
        results.append(out)
        tag = f"qwen-{c['id']}"
        (outdir / f"{tag}.reasoning.txt").write_text(out["reasoning"], encoding="utf-8")
        (outdir / f"{tag}.answer.txt").write_text(out["answer"], encoding="utf-8")
        (outdir / f"{tag}.full.txt").write_text(
            f"[REASONING] ({out['reasoning_chars']} chars)\n{out['reasoning']}\n\n"
            f"[ANSWER] ({out['answer_chars']} chars)\n{out['answer']}\n",
            encoding="utf-8",
        )
        print(f"[qwen] {c['id']}: finish={out['finish_reason']} "
              f"comp={out['completion_tokens']} reason_ch={out['reasoning_chars']} "
              f"answer_ch={out['answer_chars']} wall={out['wall_seconds']}s", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 blind reviewer A/B")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True, help="artifact dir")
    parser.add_argument("--reviewers", type=str, nargs="*",
                        default=["mistral-xxs", "mistral-s"])
    parser.add_argument("--temp", type=float, default=REVIEW_TEMP)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-tokens", type=int, default=REVIEW_MAX_TOKENS)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--startup-timeout", type=int, default=360)
    parser.add_argument("--tensor-split", type=str, default="1,1")
    parser.add_argument("--verbosity", type=int, default=4)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cases = json.load(open(args.cases, encoding="utf-8"))
    results: list[dict] = []
    for reviewer in args.reviewers:
        if reviewer in ("mistral-xxs", "mistral-s"):
            results.extend(run_mistral(reviewer, cases, args.outdir, args.temp, args.seed,
                                       args.max_tokens, args.request_timeout,
                                       args.startup_timeout, args.tensor_split, args.verbosity))
        elif reviewer == "qwen":
            results.extend(run_qwen(cases, args.outdir, args.temp, args.seed,
                                    args.max_tokens, args.request_timeout))
        else:
            raise SystemExit(f"unknown reviewer: {reviewer}")
        sweep.save_results(args.output, results)
    print(f"\nWrote {args.output} with {len(results)} reviews", flush=True)


if __name__ == "__main__":
    main()
