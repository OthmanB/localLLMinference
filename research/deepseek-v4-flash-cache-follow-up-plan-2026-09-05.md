# DeepSeek-V4-Flash Cache Follow-Up Plan

Date: 2026-09-05

## Goal

Improve the isolated MoE-cache runtime beyond the current warmed-cache result
without sacrificing output correctness, physical-VRAM operation, or populated
long-context capacity. The immediate target is 30 tok/s warm decode at 128k;
the secondary target is restoring at least 25 tok/s at 512k.

This is an experiment plan. It does not authorize replacing the production
runtime or starting the Qwen service.

## Current Reference

- Runtime: `moe-cache-v2-pr` commit `e3096b046`, built separately with CUDA,
  NCCL, SM86, Flash Attention, and CUDA Graphs.
- Target: DeepSeek-V4-Flash-0731 `UD-IQ3_XXS` on three RTX 3090s.
- Baseline placement: expert layers `0-7` on CUDA0, `8-16` on CUDA1, `17-25`
  on CUDA2, all other expert layers on CPU.
- Common target settings: 12 inference and batch threads, Q8 K/V, Flash
  Attention, `--parallel 1`, `--fit off`, physical VRAM only, and
  `--no-cache-prompt` for scoring.
- Cache settings: `--moe-cache 16000 --no-repack` with
  `GGML_CUDA_MOE_CACHE_RESERVE_MB=1024`.

| Fully populated context | Warm decode tok/s | Prefill tok/s | Actual cache capacity MiB, CUDA0/1/2 |
|---:|---:|---:|---|
| 128k | 27.7022 | 70.0668 | 1,282 / 1,701 / 1,605 |
| 256k | 26.5615 | 66.3797 | 1,108 / 1,293 / 1,217 |
| 512k | 24.1073 | 57.2680 | 760 / 725 / 693 |

At 8k, three warmed 1,024-token scores averaged 28.2667 tok/s with a sample
standard deviation of 0.0099 tok/s. Do not call a screening change an
improvement unless the repeated median exceeds both this noise floor and the
expected thermal/run variation.

## Non-Negotiable Controls

- Keep `llama-qwen3.8-q4-192k.service` stopped and verify `SwapTotal=0` before
  every scored run.
- Keep the current target model, binary commit, GPU ordering, 12 threads,
  cache types, batch sizes, tensor split, and output gate fixed unless the
  experiment explicitly changes one of them.
- Do not set `GGML_CUDA_ENABLE_UNIFIED_MEMORY`; its presence enables UVM in
  llama.cpp.
- Start a fresh server for every score. Use a unique, recorded `XDG_CACHE_HOME`
  for every arm so no host-side state can prewarm a supposedly cold run.
- Require arithmetic, capital, and exact-echo checks before scoring. For every
  promoted candidate, run 20 additional sequential checks and record server RSS
  plus GPU memory before and after.
- Use three 256-token warmups followed by a measured 1,024-token decode for
  warm scoring. Record cold 256-token decode separately.
- Confirm a cache arm by log evidence, not its command line: `[moe-cache]
  enabled`, nonzero pool allocations, hit rates, fills, and zero fill/dispatch
  failures are mandatory.
- Record actual pool capacity, not only the requested 16,000 MiB cap.

## Priority Order

| Priority | Experiment | Screening success | Stop condition |
|---:|---|---|---|
| 1 | Static placement for larger dynamic pools | >=1.0 tok/s warm gain at 8k | Cache does not enable, output fails, or no gain after repeats |
| 2 | VRAM reserve sweep | >=1.0 tok/s warm gain or >=25 tok/s at 512k | OOM, pool instability, output failure, or no gain |
| 3 | PCIe diagnostic trace | Explains misses/fills without performance regression | Telemetry unavailable or no actionable correlation |
| 4 | CPU DSpark compatibility recheck | Server starts, >=40% acceptance, and better end-to-end decode | Loader/scheduler failure, acceptance <40%, or long-output failure |

GPU power-limit testing is intentionally excluded. Observed warm-decode power
draw was far below the configured caps, so changing caps is not a credible
first-order lever.

## 1. Static Expert Placement Sweep

The cache can use VRAM freed by moving complete resident expert layers to CPU.
Do not move dense tensors or change the GPU set. Keep the 16,000 MiB cache cap
and 1,024 MiB reserve initially.

Test the following exact expert rules. The fallback CPU rule is last in every
override list.

| Name | CUDA0 expert layers | CUDA1 expert layers | CUDA2 expert layers | Static total |
|---|---|---|---|---:|
| Reference | 0-7 | 8-16 | 17-25 | 26 |
| `7/8/8` | 0-6 | 8-15 | 17-24 | 23 |
| `6/7/7` | 0-5 | 8-14 | 17-23 | 20 |

This moves layers `7, 16, 25` to CPU in `7/8/8`, then moves `6, 15, 24` as
well in `6/7/7`. It is a controlled capacity trade: all additional CPU-resident
layers can become cache candidates, but cache misses may be slower.

1. Run one cold and one warmed 8k screen for each placement.
2. Record decode and prefill, actual per-GPU cache capacity, final hit rate,
   GPU memory, host memory, swap, and cache failures.
3. Repeat the two highest warm-decode candidates three times.
4. If a new placement wins, inspect per-layer routing/caching log statistics.
   Retain static layers with low cache value or high miss cost; do not infer
   entropy solely from layer index.
5. Promote only the best stable placement to the reserve sweep.

## 2. Reserve Sweep

Apply this only to the placement selected above. Keep `--moe-cache 16000`; the
request is already capped by free VRAM.

| Reserve MiB | Purpose |
|---:|---|
| 1,024 | Existing safe reference |
| 768 | Moderate additional dynamic pool capacity |
| 512 | Lowest allowed reserve for this plan |

1. Screen each reserve at 8k with cold and warmed scores.
2. Reject a setting immediately on CUDA allocation failure, growing GPU usage
   over the 20-request stability sequence, output failure, swap activity, or a
   cache fill/dispatch failure.
3. Repeat the best two reserves three times.
4. Advance the best reserve/placement pair to a fully populated 128k run.
5. If its populated 128k decode is >=25 tok/s, run the fully populated 512k
   gate. Use request timeouts of at least 7,200 seconds for 256k and 14,400
   seconds for 512k; the earlier fixed 1,800-second client timeout is invalid
   for those prefills.

Do not reduce the reserve below 512 MiB in this campaign. Capacity beyond that
point is not worth risking failed allocation or an unstable long-running cache.

## 3. PCIe Telemetry Diagnostic

This is observational, not a tuning arm. Capture per-GPU PCIe RX/TX throughput
at one-second intervals throughout a warmed 1,024-token decode for the current
reference and for the winning cache configuration. Confirm the counter names on
this driver before adding them to the harness.

Interpretation:

- Low, flat RX/TX while decode remains slow points away from PCIe as the steady
  warm-path bottleneck and toward GPU small-matvec latency plus remaining CPU
  misses/RAM bandwidth.
- Bursty or sustained high traffic aligned with low hit rates, fills, or decode
  stalls makes GPU0 Gen3 x4 a plausible cache-miss/fill limiter.
- Do not use low aggregate CPU utilization to dismiss RAM-bandwidth pressure;
  a small number of active threads can saturate memory channels.

Record hit rate and pool capacity alongside traffic. PCIe bandwidth alone is
not an explanatory metric.

## 4. CPU DSpark Recheck

CPU DSpark is conditional because both available artifacts previously failed
before readiness on the stock isolated runtime:

- Legacy `deepseek_v4_flash_dspark_draft` GGUF was unknown to the loader.
- Standardized `dflash` GGUF aborted with `output.weight` on CUDA2 unable to
  run operation `NONE`, for both CPU-only and GPU0 draft placement.

Do not treat this as a simple flag sweep. First validate the cache fork's
loader and scheduler with the checksum-verified standardized artifact:

```text
--spec-draft-model <Q2_K-Q8_0-dflash.gguf>
--spec-type draft-dflash
--spec-draft-n-max 3
--spec-draft-p-min 0
--spec-draft-device none
--spec-draft-ngl 0
--spec-draft-type-k q8_0
--spec-draft-type-v q8_0
```

If a future compatible binary recognizes the legacy DSpark architecture, use
the requested `--spec-type draft-dspark` form instead. Do not mislabel the
standardized `dflash` artifact as `draft-dspark`.

1. First run a no-score startup and output-gate check at 8k with the winning
   cache placement/reserve. Stop on the previous scheduler assertion.
2. If it serves, run a 256-token screen and record drafted tokens, accepted
   tokens, verification steps, acceptance fraction, and end-to-end decode rate.
3. Require >=40% acceptance and a faster end-to-end rate than the non-speculative
   cache baseline before testing `--spec-draft-n-max 2, 3, 4`.
4. For the best value, test at least one 3,072-token decode and a 20-request
   stability sequence. Reject output corruption, a long-generation failure,
   growing VRAM/RSS, zero-swap violations, or any regression against the
   non-speculative cache arm.

CPU DSpark needs no separate draft-model VRAM budget. That is precisely why it
must be validated against the cache candidate, but it may still contend with
CPU-resident cache misses for RAM bandwidth.

## Reporting and Decision Rules

- Create one result directory per arm under
  `research/large-moe-3gpu-probe-2026-09-04/moe-cache-v2-follow-up/`.
- Include the complete server command, environment overrides, actual cache
  pools, PCIe telemetry, raw results, and server log in each arm directory.
- Update the existing cache experiment record after each completed phase; do
  not overwrite prior baselines.
- The best non-speculative result should be expressed as cold decode, warmed
  decode, populated 128k decode, populated 512k decode, and cache capacity.
- Do not deploy this fork solely on benchmark results. Promotion requires a
  broader representative coding/task suite in addition to the existing narrow
  deterministic gates.

## Sources

- [MoE-cache RFC and measurements](https://github.com/ggml-org/llama.cpp/discussions/24528)
- [Current speculative-decoding documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
- [Reported DeepSeek-V4 multi-GPU issue](https://github.com/ggml-org/llama.cpp/issues/26554)
- `large-moe-3gpu-probe-2026-09-04/recent-build-moe-cache-v2.md`
- `large-moe-3gpu-probe-2026-09-04/recent-build-dspark.md`
