# Qwen3.8-27B RTX 5090 Phase F Stage 1 MTP1 Result

Date: 2026-09-20
Status: rejected. Native MTP1 preserved the tested correctness contracts but
cannot serve the three warmed long-context sessions. MTP2 and MTP3 will not be
run. The live llama.cpp Q4 service remains the only routed model.

## Candidate

The candidate differed from B0 only by the allowlisted native speculative
setting below. It retained vLLM 0.29.0, TP2 on GPUs 0 and 1, FP8 E4M3 KV,
eager execution, `TRITON_ATTN`, the disabled FlashInfer sampler, 500 W caps,
and the 85 C defensive stop.

```text
--spec-method mtp --spec-tokens 1
```

The server log confirms `Qwen3_5MTP`, `SpeculativeConfig(method='mtp',
num_spec_tokens=1)`, eager execution, Triton attention, and PYNCCL fallback.
It also records vLLM's MTP-specific default of 2,048 scheduled tokens and
hybrid-model dense prefix-cache retention.

## Preflight And Correctness

The 8k one-session preflight completed 22 request outcomes successfully:
normal EOS, the forced `lookup_symbol` tool call, one single-context request,
one capacity request, two interference streams, and 16 sustained requests over
300 seconds. Normal EOS and tool response hashes exactly match B0. The
249,041-token retrieval separately returned `phase-d-retrieval-raven-73`
exactly in 138.78 seconds.

No request error, OOM, thermal stop, or nonzero candidate exit code occurred.
Peak temperature was 50 C on GPU 0 and 44 C on GPU 1. Minimum free VRAM was
2,978 MiB and 3,831 MiB respectively.

The runner emits an `EngineDeadError` after it sends SIGTERM and vLLM force
kills EngineCore during teardown. The same shutdown-only diagnostic appears in
the B0 decode log; it followed completed requests and exit code zero, so it is
not counted as a candidate request failure.

## One-Session Long Decode

Both one-session decode probes completed by `length` with 4,096 forced output
tokens. They are operational observations, not the primary three-user score.

| Prompt per user | Decode | TTFT | Client p95 ITL | Draft acceptance |
| --- | ---: | ---: | ---: | ---: |
| 196k | 14.17 tok/s | 2.53 s | 0.166 s | 1,993 / 2,102 (94.81%) |
| 250k | 11.38 tok/s | 3.07 s | 0.183 s | 1,991 / 2,104 (94.63%) |

## Three-Session Gate Failure

The first 196k C=3 run completed all warm and scoring streams normally, but it
did not retain any warmed prefix for the scoring requests. The warm phase
computed 588,078 prompt tokens. The subsequent three streams computed a second
588,078 prompt tokens: `local_compute` increased by 588,078 while
`local_cache_hit` and `prefix_cache_hits_total` remained zero. Thus this is not
a valid warmed-decode measurement and its aggregate wall-time quotient must not
be compared with B0's cached-context aggregate decode.

The observed per-stream results nevertheless fail the primary gate by a wide
margin:

| Stream | Decode | TTFT | Client p95 ITL |
| --- | ---: | ---: | ---: |
| decode-0 | 8.92 tok/s | 98.67 s | 0.860 s |
| decode-1 | 14.06 tok/s | 298.58 s | 0.145 s |
| decode-2 | 11.04 tok/s | 199.98 s | 0.596 s |
| Observed median | 11.04 tok/s | 199.98 s | 0.596 s |

The observed median is 32.37% below B0's 16.33 tok/s and 38.51% below the
MTP advance minimum of 17.96 tok/s. Its 0.596 s p95 ITL exceeds the 0.133 s
maximum by 348.15%. Speculative acceptance itself was high at 5,986 accepted
of 6,301 drafted tokens (95.00%), so acceptance cannot establish usable
three-session performance when the requested context is recomputed.

This run stayed within safety limits: GPU 0 reached 81 C, GPU 1 reached 61 C,
minimum free VRAM was 3,178 MiB and 4,015 MiB respectively, all requests ended
by `length`, candidate exit code was zero, and no thermal stop occurred.

The measurement establishes cache non-reuse and the resulting gate failure. It
does not establish the underlying cause. Cache-key mismatch, hybrid retention
behavior, eviction, and MTP/runtime interactions remain hypotheses.

## Decision

MTP1 fails the 196k primary C=3 requirement before a valid three-repeat matrix
can be formed. Per the Stage 1 stop rule, do not run MTP2 or MTP3, C=3 250k
repeats, four-session capacity, or the additional MTP interference/sustained
matrix. No MTP profile advances to Stage 2 and no optimization result is
authorized for routing.

The harness validation was narrowed so `--decode-concurrency` is checked only
for `--decode-only` runs. Previously its unused default of three incorrectly
rejected the one-session preflight before the guarded candidate started.

## Artifacts

```text
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-f-m1-mtp1-preflight-8k-c1-eager-triton-vllm-2026-09-20T135502Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-f-m1-mtp1-retrieval-eager-triton-vllm-2026-09-20T140440Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-f-m1-mtp1-eager-triton-decode-196k-c1-vllm-2026-09-20T140957Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-f-m1-mtp1-eager-triton-decode-250k-c1-vllm-2026-09-20T141818Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-f-m1-mtp1-eager-triton-decode-196k-c3-r1-vllm-2026-09-20T142834Z/
```

Each directory contains the guarded maintenance log, server log, raw
Prometheus snapshots, result summary, and one-second telemetry samples.
