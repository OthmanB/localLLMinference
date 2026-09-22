# Qwen3.8-27B RTX 5090 Phase F Stage 1.5 MTP Cache Diagnostic

Date: 2026-09-21
Status: completed. The C=3 cache miss is not a universal MTP prefix-cache
failure, but native MTP1 remains rejected on the pinned eager runtime because
it loses a cache-neutral 196k C=1 decode comparison.

## Question

The first Stage 1 MTP1 C=3 score recomputed all three 196k warmed prefixes.
That invalidated its comparison with B0's warmed score, but did not establish
whether MTP itself disabled hybrid prefix caching or whether its raw decode was
useful without caching. This diagnostic separates those questions without
expanding the MTP matrix.

Every run used the guarded maintenance wrapper, TP2 on physical GPUs 0 and 1,
the calibrated export, FP8 E4M3 KV, eager execution, Triton attention, the
disabled FlashInfer sampler, 500 W power limits, an 85 C stop, loopback only,
and the preserved unrelated GPU-0 RAG PID. The reference llama.cpp service and
monitor were restored after each run.

## 128k Identical-Prompt Cache Reproducer

Each run sent the exact same approximately 128k prompt twice with prefix
caching explicitly enabled, `max-num-seqs=1`, and a 32-token reply. The
counter deltas below are from the interval containing only the second request.

| Runtime | Speculation | Block size | Second-request prefix-cache hits | Second-request local-cache tokens | First request | Second request |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| pinned vLLM 0.29.0 | off | 1,568 | 127,008 | 127,008 | 46.54 s | 1.08 s |
| pinned vLLM 0.29.0 | MTP1 | 1,584 | 125,136 | 125,136 | 54.58 s | 2.12 s |
| isolated vLLM 0.29.1rc1.dev438+g01f1f58f1 | off | 1,568 | 127,008 | 127,008 | 52.15 s | 1.09 s |
| isolated vLLM 0.29.1rc1.dev438+g01f1f58f1 | MTP1 | 1,584 | 125,136 | 125,136 | 55.06 s | 2.04 s |

All four pairs returned the expected deterministic text. In both MTP1 runs,
the effective cache configuration reports `enable_prefix_caching=True` and
Mamba `align` mode. The newer runtime used the default
`prefix_cache_retention_interval=0`; the pinned MTP1 run resolved it to
`None`. The outcome is positive cache reuse in both cases, regardless of that
difference.

Thus, this host does not reproduce a universal MTP-plus-hybrid cache-disablement
failure. The earlier 196k C=3 zero-hit result remains real, but its cause is
narrowed to workload/concurrency/retention geometry or another C=3 interaction.
This diagnostic did not alter block sizing, retention, scheduling, or model
weights, so it cannot identify which of those factors caused the C=3 miss.

## Cache-Neutral Raw Decode

To test MTP independently of cross-request reuse, the same 196k C=1 workload
used explicit `--no-enable-prefix-caching`, no warm request, and a forced
4,096-token output. Both requests computed all 196,026 API prompt tokens and
recorded zero local-cache-hit tokens.

| Setting | Decode | TTFT | Client p50 ITL | Client p95 ITL |
| --- | ---: | ---: | ---: | ---: |
| B0, no prefix cache | 16.53 tok/s | 93.90 s | 0.059 s | 0.120 s |
| MTP1, no prefix cache | 14.08 tok/s | 100.66 s | 0.137 s | 0.147 s |

MTP1 is 14.83% slower than B0, with a 21.97% worse p95 ITL. It drafted 2,118
tokens and accepted 1,977 (93.34%), confirming useful proposal quality but not
useful end-to-end decode performance in this eager configuration. Both runs
ended by `length`, had zero thermal stops, and exited successfully.

## Isolated Runtime Check

The newer runtime was installed in
`/home/michel/LLMs-tests/.phase-f1p5-vllm-nightly` from the official nightly
wheel index. It is separate from the pinned
`/home/michel/LLMs-tests/.phase2-vllm-venv` and uses CUDA 13.2/Torch 2.13.0.
The two 128k B0/MTP1 cache probes above both passed. This is a cache-behavior
sanity check only, not a new performance baseline or a promotion candidate.

An official nightly Docker image was also pinned at
`sha256:f29125bcc6d276ed38a67c9dfd4e51ccbaba09ad92847a0913c0df38d9c05c71`,
but could not access host GPUs because the Docker daemon has neither an NVIDIA
runtime nor a configured CDI GPU vendor. No Docker, NVIDIA, driver, or service
configuration was changed to work around that unrelated infrastructure gap.

## Decision

The Stage 1 MTP1 rejection stands, but its rationale is refined:

1. The initial C=3 cache-non-reuse observation is valid but is not evidence
   that MTP categorically disables prefix caching on this host.
2. MTP1 loses the controlled cache-neutral C=1 decode comparison despite high
   acceptance. It therefore fails the Stage 1 prerequisite for MTP2/MTP3 even
   without relying on the invalid warmed C=3 comparison.
3. Do not run MTP2, MTP3, CUDA graphs, or another long C=3 MTP matrix on the
   pinned runtime. A longer C=3 cache investigation would not change the
   negative raw-decode result and would not create a Stage 2 winner.
4. Future runtime research may reproduce the C=3 condition in an isolated
   newer build and inspect retention, scheduler ordering, and cache geometry,
   but only after establishing a non-regressing B0/MTP profile in that runtime.

## Artifacts

```text
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-f1p5-cache128-b0-vllm-2026-09-20T220711Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-f1p5-cache128-mtp1-vllm-2026-09-20T220959Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-f1p5-rawdecode196k-b0-vllm-2026-09-20T221313Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-f1p5-rawdecode196k-mtp1-vllm-2026-09-20T222104Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-f1p5-nightly-cache128-b0-vllm-2026-09-20T224103Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-f1p5-nightly-cache128-mtp1-vllm-2026-09-20T224416Z/
```

Each candidate directory includes the complete command, server log, raw
Prometheus snapshots, request result, GPU/host telemetry, and guarded
maintenance log.
