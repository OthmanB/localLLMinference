# Qwen3.8-27B RTX 5090 Phase F Stage 0 B0 Baseline

Date: 2026-09-20
Status: complete for the two primary three-session decode points. No candidate
is quality-approved or routed; llama.cpp Q4 remains the live service.

## Purpose

Establish the no-speculative, eager vLLM TP2 baseline used to evaluate native
MTP. The target is three already-populated, distinct long-context sessions, not
only a capacity test with three simultaneous cold prefills.

Configuration:

```text
calibrated ModelOpt NVFP4/FP8-KV export, config SHA-256
731d9ea745aa5c171822e4e64b4a863ae3269620a43ca2bec8a1e42c13a89b68
vLLM 0.29.0, TP2 on GPUs 0,1, max-model-len 262144, max-num-seqs 3
--attention-config {"backend":"TRITON_ATTN","use_trtllm_attention":false}
--enforce-eager
VLLM_USE_FLASHINFER_SAMPLER=0
FP8 E4M3 KV, 500 W cap on each GPU, 85 C defensive stop
```

## Harness Changes

`tools/rtx5090_phase_d.py` now records vLLM metric snapshots before and after a
run, typed MTP/graph controls, SSE TTFT/p50/p95 ITL, and the known forced output
count when a stream ends by length. Re-tokenizing only visible SSE content was
found to undercount generated tokens, so it is retained only as a diagnostic
estimate and is not used for decode throughput.

The new `--decode-only` mode performs the following sequence:

1. Build three distinct long prefixes and warm each in the vLLM prefix cache.
2. Confirm the warm requests stop normally and save a metrics snapshot.
3. Submit three simultaneous 4,096-token, `ignore_eos` streams that share their
   respective warmed prefixes.
4. Use the known 4,096 generated tokens and time from first token to completion
   for each user's decode rate.

The 250k C=3 final repeat reports 747,936 local prefix-cache-hit tokens. The
short TTFTs below are therefore cached-context decode measurements, not cold
prefill measurements.

## Primary B0 Results

Each point has three guarded repeats, three users per repeat, and 4,096 forced
completion tokens per user. All 18 decode streams ended by `length`; no request
error, OOM, or thermal stop occurred.

| Prompt per user | Median per-user decode | Observed per-user range | Aggregate decode | Median TTFT | Median client p95 ITL | Median wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 196,017 tokens | 16.33 tok/s | 16.00-16.71 tok/s | 48.98 tok/s | 1.24 s | 0.121 s | 252.38 s |
| 250,017 tokens | 16.37 tok/s | 16.09-16.54 tok/s | 49.11 tok/s | 2.27 s | 0.120 s | 252.61 s |

The 250k result is not slower than 196k within this repeat variation. The
steady three-way decode profile is approximately 49 aggregate tok/s and 16.3
tok/s per user; use these values, not a single-user or cold-arrival decode
figure, as the MTP comparison baseline.

## Cold Arrival Evidence

The same engine was also measured with three simultaneous uncached 512-token
long requests. This is intentionally a different workload and must not be used
as the decode-only score.

| Prompt per user | C=3 wall range | Observed TTFT sequence |
| --- | ---: | --- |
| 196,017 tokens | 309.61-320.15 s | 90.93-98.22 s, 184.36-193.56 s, 277.57-288.39 s |
| 250,017 tokens | 436.34-438.08 s | 131.28-133.49 s, 268.20-269.65 s, 405.25-407.17 s |

The queueing is expected while the scheduler processes three long prefills. It
explains why the previous 512-token streams showed very low apparent decode for
the first two users. Cold-prefill and active-session decode must remain separate
metrics in later MTP comparisons.

## Correctness and Safety

- The first B0 full-suite run passed normal EOS, forced `lookup_symbol` tool
  calling, one 250k request, C=3 196k capacity, prefill/decode interference,
  and a 300-second C=3 sustained load (24/24 requests; 12,288 completion
  tokens; 795 telemetry samples; no thermal stop).
- Fresh 249,041-token buried-fact retrieval returned
  `phase-d-retrieval-raven-73` exactly in 130.55 s.
- Every window verified reference rollback before the candidate started and
  restored `llama-qwen3.8-q4-native.service` plus
  `ai-qwen3.8-q4-native-monitor.service` afterward.
- After the final run, both services are active. The unrelated GPU-0 RAG PID
  2598220 remains present; no unrelated process was stopped.

## MTP Gate Derived From B0

For MTP1 to advance, it must retain all correctness and safety results and meet
both primary thresholds across three warmed repeats:

| Prompt per user | Minimum median per-user decode | Maximum median p95 ITL |
| --- | ---: | ---: |
| 196k | 17.96 tok/s | 0.133 s |
| 250k | 18.01 tok/s | 0.132 s |

These are the plan's +10% decode and no-more-than-10% p95 ITL regression gates,
calculated from B0. MTP acceptance metrics will be captured from the same raw
Prometheus snapshots. The bounded GSM8K failure remains unresolved, so passing
this performance gate does not authorize a canary.

## Artifacts

```text
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-f-b0-eager-triton-196k-c3-vllm-2026-09-20T103645Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-f-b0-eager-triton-196k-c3-r{1,2,3}-vllm-2026-09-20T*/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-f-b0-eager-triton-250k-c3-r{1,2,3}-vllm-2026-09-20T*/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-f-b0-eager-triton-retrieval-vllm-2026-09-20T114416Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-f-b0-eager-triton-decode-196k-c3-r{1,2,3}-vllm-2026-09-20T*/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-f-b0-eager-triton-decode-250k-c3-r{1,2,3}-vllm-2026-09-20T*/
```

The brace expressions above are documentation shorthand, not literal paths.
Each run directory contains the guarded maintenance log, candidate server log,
raw metrics snapshots, request result, and one-second telemetry samples.
