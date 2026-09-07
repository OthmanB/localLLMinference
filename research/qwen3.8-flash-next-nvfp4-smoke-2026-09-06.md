# Qwen3.8 Flash-Next NVFP4 FreeToken Smoke Test

Date: 2026-09-06

## Scope

This is a guarded, single-request 4k FreeToken viability test for
`nvidia/Qwen3.8-Flash-Next-NVFP4`. It is not a production or quality-promotion
run.

- Runtime: isolated FreeToken source at `af71ba43206e124f5ff6419b47ee36c6e9981078`.
- Runtime patch: recognize ModelOpt `MIXED_PRECISION` Qwen4Exp expert entries as
  NVFP4, while preserving the checkpoint's BF16 dense, attention, and head
  weights. The Qwen4Exp configuration and weight tests passed 31/31.
- Service: `freetoken-flash-next-candidate.service`, local port 1901.
- GPU: CUDA0 only, RTX 3090, 300 W cap. Qwen4Exp supports TP=1 only, so GPU1
  and GPU2 were not used.
- Request controls: one running request, 4,096-token QSA KV pool, 512-token
  prefill chunks, 128 output-token cap, thinking off, temperature zero.
- MoE: hybrid CPU/GPU; 512 expert-cache slots, Triton NVFP4, serial expert load,
  disk-backed PLE, and disabled two-buffer prefill overlap.

## Startup

The initial 128-slot configuration failed before serving because Qwen4Exp
requires at least one cache slot per expert. The corrected 512-slot service
loaded successfully.

- QSA sparse attention selected; page size resolved to 64.
- Disk PLE used io_uring with O_DIRECT; the 47.7 GiB PLE table was not pinned
  in host RAM.
- FreeToken loaded 48 layers of 512 NVFP4 experts and started 31 pinned AVX2 CPU
  MoE workers.
- Calibrated hybrid policy fetched 13.6% of decode expert misses over PCIe and
  computed the remainder on CPU.
- KV allocation was 0.10 GiB. GPU free memory was 11.19 GiB after CUDA graph
  capture.

## Results

| Probe | Prompt / output | Result |
|---|---:|---|
| Arithmetic | 29 / 4 tokens | `17 * 19` returned `323` exactly. |
| Generation warm-up | 1,926 / 127 tokens | Repeated the requested phrase and stopped at the output limit; 53.68 s wall time. Scheduler decode settled at 27.27 tok/s. |
| Unique long prompt | 3,790 / 99 tokens | Completed in 92.34 s with zero cached prompt tokens. Scheduler prefill settled near 46.7 tok/s and decode reached 23.20 tok/s. The model declined the intentionally repetitive output request, so this is a runtime measurement rather than a format-quality pass. |
| Long-context retrieval | 3,669 / 22 tokens | Completed in 88.82 s and returned `ALPHA-739, BRAVO-284, CHARLIE-615` exactly. |

The first prefill chunk is consistently slower while request state is admitted;
later 512-token chunks sustain approximately 46.4-46.8 tok/s. The scheduler
reported zero cached tokens for both unique long prompts.

## Resource Gate

- Service cgroup current memory: 129.64 GiB.
- Service cgroup peak memory: 132.44 GiB.
- Service cgroup swap: 0 B; host swap: 0 B throughout.
- Observed idle GPU0 allocation after testing: 12,942 MiB at the 300 W cap.

The process fit physically, but its host-memory margin is narrow. Do not add a
second large host-bank workload, increase context, enable pinned PLE, or promote
this candidate before a longer stability and task-quality evaluation.

## Decision

The 4k single-GPU-plus-CPU hybrid smoke test passed. This historical result is
superseded by the guarded 192k and 256k runs; the production Qwen service
remains unchanged and stopped.
