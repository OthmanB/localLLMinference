#!/usr/bin/env python3
"""Phase 3: review-mode point for Mistral-Small-4-119B-2603 (IQ3 quants).

Answers: "at 128k context with reasoning_effort=high, does the model finish a
generation in bounded wall time, and what does it produce?" This is the only
generation-quality measurement of the plan; it is separate from decode
throughput (Phase 2).

Design (locked from the plan):
  * Same 128k deterministic content as the Phase 2 128k point (reuses the
    Phase 2 harness prompt builder + constant chat-template overhead).
  * reasoning_effort="high" via chat_template_kwargs (the Mistral template only
    accepts "none" or "high"; anything else raises). This injects the
    [MODEL_SETTINGS] block and makes the model emit [THINK]...[/THINK]
    reasoning, which llama.cpp's template-based parser splits into
    message.reasoning_content.
  * Natural EOS: NO ignore_eos, NO forced decode length. Only a generous safety
    max_tokens cap so a runaway on the repetitive probe text cannot run forever.
  * Residency rule (same as Phase 2): the point must be fully GPU-resident or it
    is not scored.
  * Full reasoning + answer text is saved per quant as the Phase 4 A/B artifact.

The Phase 2 server base is reused verbatim (same flags) plus `--reasoning auto`;
verbosity is 4 (as in Phase 2) so the startup log exceeds the 8 KiB stdio buffer
and the residency lines are flushed to disk for check_residency().
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mistral_context_sweep as sweep  # noqa: E402  (reuse tested helpers)

LLAMA = sweep.LLAMA
REVIEW_CTX = 131072            # 128k context
REVIEW_EFFORT = "high"         # template only accepts "none" | "high"
REVIEW_TEMP = 0.7              # Mistral's recommended temperature for reasoning_effort=high
REVIEW_RESERVE = 16384         # context reserved for generation (reasoning + answer)
WALL_BUDGET_S = 1200           # ~20 min stop condition from the plan
#
# NOTE on prompt sizing (confirmed design decision):
#   The Phase 2 128k point used target = ctx - 128 - overhead - 128, which leaves
#   only ~272 tokens for output (fine for a forced 128-token decode). Phase 3 uses
#   NATURAL EOS, so the model needs real room to emit reasoning + answer. We keep
#   the same 128k CONTEXT and the same deterministic content generator, but size the
#   prompt to reserve REVIEW_RESERVE tokens for generation:
#       target_content = ctx - overhead - REVIEW_RESERVE
#   The safety max_tokens equals the reserve, so natural EOS is expected well before
#   either bound.


def start_review_server(spec: dict, ctx: int, tensor_split: str, verbosity: int,
                        log_path: Path, prompt_log_dir: Path | None = None) -> subprocess.Popen:
    """Spawn llama-server with the Phase 2 base args + --reasoning auto."""
    log = log_path.open("w", encoding="utf-8")
    args = [
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
    ]
    if prompt_log_dir is not None:
        prompt_log_dir.mkdir(parents=True, exist_ok=True)
        args.extend(["--log-prompts-dir", str(prompt_log_dir)])
    return subprocess.Popen(
        args,
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


def wait_resident(log_path: Path, timeout: int = 60) -> dict:
    """Poll check_residency() until the startup log has flushed the offload
    line (guards against stdio full-buffering) or the deadline elapses."""
    deadline = time.monotonic() + timeout
    res = sweep.check_residency(log_path)
    while time.monotonic() < deadline and res.get("offloaded_layers") is None:
        time.sleep(1)
        res = sweep.check_residency(log_path)
    return res


def run_review_point(model: str, ctx: int, output_dir: Path,
                      request_timeout: int, startup_timeout: int,
                      tensor_split: str, verbosity: int, reserve: int,
                      seed: int | None = None, tag: str = "run",
                      temperature: float = REVIEW_TEMP,
                      reasoning_format: str | None = None,
                      raw_artifacts: bool = False) -> dict:
    spec = sweep.MODELS[model]
    port = spec["port"]
    gpu_ids = [int(x) for x in spec["gpu"].split(",") if x.strip()]
    log_path = output_dir / f"{model}-{tag}.server.log"
    result = {
        "phase": 3,
        "mode": "review",
        "model": model,
        "tag": tag,
        "alias": spec["alias"],
        "quant": "IQ3_XXS" if model == "mistral-xxs" else "IQ3_S",
        "context_tokens": ctx,
        "reasoning_effort": REVIEW_EFFORT,
        "gpu": spec["gpu"],
        "port": port,
        "tensor_split": tensor_split,
        "generation_reserve_tokens": reserve,
        "safety_max_tokens": reserve,
        "wall_budget_s": WALL_BUDGET_S,
        "started_at": sweep.now(),
        "status": "error",
        "fully_resident": None,
    }
    prompt_log_dir = output_dir / f"{model}-{tag}.prompts" if raw_artifacts else None
    process = start_review_server(spec, ctx, tensor_split, verbosity, log_path, prompt_log_dir)
    samples: list[dict] = []
    sampler: threading.Thread | None = None
    stop_sampling = threading.Event()

    def sample() -> None:
        while not stop_sampling.wait(1):
            try:
                snap = sweep.gpu_snapshot(gpu_ids)
                snap.update(sweep.host_snapshot())
                snap["process_rss_mib"] = sweep.proc_rss_mib(process.pid)
                samples.append(snap)
            except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError):
                pass

    try:
        sweep.wait_ready(port, process, startup_timeout)
        res = wait_resident(log_path)
        result["residency"] = res
        result["fully_resident"] = res.get("fully_resident")
        if not res.get("fully_resident"):
            result["status"] = "not_fully_resident"
            return result
        warm_err = sweep.warmup(port, spec["alias"])
        if warm_err:
            result["warmup_error"] = warm_err
        overhead = sweep.measure_template_overhead(port, spec["alias"])
        target_content = ctx - overhead - reserve
        content, content_tokens = sweep.build_prompt(port, model, ctx, target_content)
        result["template_overhead_tokens"] = overhead
        result["prompt_target_tokens"] = target_content
        result["prompt_content_tokens"] = content_tokens
        result["prompt_tokens_reported"] = None

        payload = {
            "model": spec["alias"],
            "messages": [{"role": "user", "content": content}],
            "max_tokens": reserve,
            "temperature": temperature,
            "stream": False,
            "chat_template_kwargs": {"reasoning_effort": REVIEW_EFFORT},
        }
        if reasoning_format is not None:
            payload["reasoning_format"] = reasoning_format
        if seed is not None:
            payload["seed"] = seed
        result["seed"] = seed
        result["temperature"] = temperature
        result["reasoning_format"] = reasoning_format or "server-default"
        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        t0 = time.monotonic()
        response = sweep.http_json(
            f"http://127.0.0.1:{port}/v1/chat/completions", payload, timeout=request_timeout
        )
        wall = time.monotonic() - t0

        choice = response["choices"][0]
        msg = choice.get("message", {})
        reasoning = msg.get("reasoning_content") or ""
        answer = msg.get("content") or ""
        usage = response.get("usage", {}) or {}
        timings = response.get("timings", {}) or {}
        finish = choice.get("finish_reason")

        result.update({
            "status": "passed",
            "finished": finish == "stop",
            "finish_reason": finish,
            "wall_seconds": round(wall, 2),
            "within_wall_budget": bool(wall <= WALL_BUDGET_S),
            "reasoning_tokens_est": sweep.tokenize(port, reasoning) if reasoning else 0,
            "answer_tokens_est": sweep.tokenize(port, answer) if answer else 0,
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "reasoning_chars": len(reasoning),
            "answer_chars": len(answer),
            "timings": timings,
        })
        result["decode_tok_s"] = (
            timings.get("predicted_n", 0) / (timings.get("predicted_ms", 0) / 1000)
            if timings.get("predicted_ms") else None
        )

        if raw_artifacts:
            request_path = output_dir / f"{model}-{tag}.request.json"
            response_path = output_dir / f"{model}-{tag}.response.json"
            request_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            response_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
            result["request_artifact"] = str(request_path)
            result["response_artifact"] = str(response_path)
            result["prompt_log_dir"] = str(prompt_log_dir)

        (output_dir / f"{model}-{tag}.reasoning.txt").write_text(reasoning, encoding="utf-8")
        (output_dir / f"{model}-{tag}.answer.txt").write_text(answer, encoding="utf-8")
        (output_dir / f"{model}-{tag}.full.txt").write_text(
            f"[REASONING] ({len(reasoning)} chars)\n{reasoning}\n\n"
            f"[ANSWER] ({len(answer)} chars)\n{answer}\n",
            encoding="utf-8",
        )
    except (OSError, RuntimeError, TimeoutError) as exc:
        if result["fully_resident"] is None:
            result["status"] = sweep._classify_startup_failure(log_path, exc)
        else:
            result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        stop_sampling.set()
        if sampler is not None:
            sampler.join(timeout=3)
        if samples:
            result["sample_count"] = len(samples)
            result["max_power_draw_w"] = max(s["max_power_draw_w"] for s in samples)
            result["max_temperature_c"] = max(s["max_temperature_c"] for s in samples)
            result["max_memory_used_mib_total"] = max(s["memory_used_mib_total"] for s in samples)
            per_card_peak = {}
            for idx in gpu_ids:
                vals = [c["memory_used_mib"] for s in samples for c in s["cards"] if c["index"] == idx]
                per_card_peak[str(idx)] = max(vals) if vals else None
            result["peak_memory_used_mib_per_card"] = per_card_peak
            gens = [c["pcie_gen_current"] for s in samples for c in s["cards"]]
            widths = [c["pcie_width_current"] for s in samples for c in s["cards"]]
            result["pcie_gen_under_load"] = {"max": max(gens), "min": min(gens)} if gens else None
            result["pcie_width_under_load"] = {"max": max(widths), "min": min(widths)} if widths else None
        sweep.terminate(process)
        if prompt_log_dir is not None:
            result["prompt_log_files"] = [
                str(path) for path in sorted(prompt_log_dir.iterdir()) if path.is_file()
            ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 review-mode point (128k, reasoning_effort=high, natural EOS)")
    parser.add_argument("--output", type=Path, required=True, help="results json path")
    parser.add_argument("--models", type=str, nargs="*", default=["mistral-xxs", "mistral-s"])
    parser.add_argument("--ctx", type=int, default=REVIEW_CTX)
    parser.add_argument("--reserve", type=int, default=REVIEW_RESERVE,
                        help="context tokens reserved for generation (reasoning + answer)")
    parser.add_argument("--seed", type=int, default=None,
                        help="fixed sampling seed for reproducibility (default: server random)")
    parser.add_argument("--tag", type=str, default=None,
                        help="label for artifact files (default: seed<N> if --seed set, else 'run')")
    parser.add_argument("--temp", type=float, default=REVIEW_TEMP,
                        help=f"sampling temperature (default {REVIEW_TEMP} = official high-reasoning recommendation)")
    parser.add_argument("--reasoning-format", type=str, default=None,
                        choices=["none", "deepseek", "deepseek-legacy"],
                        help="override llama.cpp reasoning parsing for this request")
    parser.add_argument("--raw-artifacts", action="store_true",
                        help="save the exact request/response JSON and llama-server rendered-prompt logs")
    parser.add_argument("--request-timeout", type=int, default=1500)
    parser.add_argument("--startup-timeout", type=int, default=360)
    parser.add_argument("--tensor-split", type=str, default="1,1")
    parser.add_argument("--verbosity", type=int, default=4)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    tag = args.tag or (f"seed{args.seed}" if args.seed is not None else "run")
    if args.temp != REVIEW_TEMP:
        tag = f"{tag}-t{args.temp}"
    results: list[dict] = []
    for model in args.models:
        result = run_review_point(
            model=model,
            ctx=args.ctx,
            output_dir=args.output.parent,
            request_timeout=args.request_timeout,
            startup_timeout=args.startup_timeout,
            tensor_split=args.tensor_split,
            verbosity=args.verbosity,
            reserve=args.reserve,
            seed=args.seed,
            tag=tag,
            temperature=args.temp,
            reasoning_format=args.reasoning_format,
            raw_artifacts=args.raw_artifacts,
        )
        results.append(result)
        sweep.save_results(args.output, results)
        print(json.dumps({k: result.get(k) for k in
                          ("model", "status", "fully_resident", "finished", "finish_reason",
                           "wall_seconds", "completion_tokens", "reasoning_tokens_est",
                           "answer_tokens_est", "max_temperature_c")}, sort_keys=True), flush=True)
        print(f"[{model}] status={result.get('status')} wall={result.get('wall_seconds')}s "
              f"finish={result.get('finish_reason')}", flush=True)


if __name__ == "__main__":
    main()
