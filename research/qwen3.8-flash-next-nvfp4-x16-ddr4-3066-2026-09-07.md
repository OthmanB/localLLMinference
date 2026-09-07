# Qwen3.8 Flash-Next NVFP4 x16 and DDR4-3066 Validation

Date: 2026-09-07

## Hardware Change

GPU0 negotiated PCIe Gen3 x16 (`8.0 GT/s`, width `16`) through the riser. The
eight DIMMs were configured at DDR4-3066 MT/s. The prior long-context baseline
was recorded under the user-reported PCIe x8 and DDR4-2933 configuration.

## Recalibration

The prior hybrid profile was measured before the PCIe change and selected a
13.6% PCIe-fetch split. It made the first x16 decode result non-representative,
despite its successful 262k capacity and retrieval result.

`ft bench bw --dtype nvfp4 --gpu 0 --cpu-threads 31` measured:

| Metric | Result |
|---|---:|
| CPU STREAM read | 52.5 GB/s |
| PCIe linear H2D | 13.1 GB/s |
| CPU MoE decode | 45.9 GB/s |
| PCIe expert gather | 13.1 GB/s |
| Overlapped CPU MoE / PCIe gather | 39.4 / 11.0 GB/s |
| Automatic PCIe-fetch split | 21.8% |

The per-GPU profile is stored at
`research/qwen3.8-flash-next-nvfp4-2026-09-06/cache/freetoken/benchbw/GPU-8c1e5d04-43a0-dba7-ebf6-c623bfa2f325.json`.

## Validated 0.90 Configuration

- GPU0, PCIe Gen3 x16, 300 W cap.
- 262,144-token QSA KV pool.
- 1,664 NVFP4 LRU expert-cache slots.
- Eight GDN recurrent-state slots.
- Hybrid CPU/GPU MoE with the recalibrated 21.8% PCIe fetch fraction.
- Disk-backed PLE, serial expert loading, one request, 512-token prefill chunks,
  thinking off for the probe, and `MemorySwapMax=0` in the permanent unit.

The 128 additional expert slots consume about 0.33 GiB of remaining cache
budget while retaining the existing 90% memory-ratio safety reserve.

## Native-Context Result

The final fresh probe used 261,945 prompt tokens plus 127 generated tokens.
Every scheduler prefill chunk reported zero cached prompt tokens; the returned
UUID marker was exact and the first completion content.

| Metric | Result |
|---|---:|
| Wall time | 2,687.54 s |
| Steady prefill | 97.7 tok/s |
| Settled decode | 35.40 tok/s |
| Peak GPU allocation | 22,182 MiB |
| Cgroup memory peak | 131.77 GiB |
| Host and cgroup swap | 0 B |

The prefill result is approximately twice the earlier x8 baseline of 48.9 tok/s.
The final decode result exceeds the earlier 28.19 tok/s baseline and the 28
tok/s promotion gate. The gains are from the combined x16 link, DDR4-3066
configuration, retuned hybrid fetch split, and slightly larger expert cache;
this run does not isolate their individual contributions.

## Validated 0.97 Cache Expansion

The final candidate raises the FreeToken free-VRAM ratio to 97% and expands the
MoE cache to 2,344 slots while keeping the 262,144-token KV pool and eight GDN
slots. It uses 14,067,609,664 bytes of configured cache pools against a
14,077,746,872-byte runtime cache budget.

Two 4k probes passed with exact first markers and zero swap. The second warmed
request completed in 43.88 seconds, with 96.95 tok/s estimated prefill and
34.4 tok/s observed by the probe's coarse stats sampler. Both short requests
peaked at 24,000 MiB GPU allocation, leaving 576 MiB free.

The fresh native-context probe used 261,952 prompt tokens plus 127 generated
tokens, returned `FT-E6847FB3A8554645B751CBFA8E9AAD01` first and exactly, and
reported zero cached prompt tokens in every scheduler prefill chunk.

| Metric | 0.97 result |
|---|---:|
| Wall time | 2,574.37 s |
| Sustained prefill | ~102.0 tok/s |
| Settled scheduler decode | 39.61 tok/s |
| Peak GPU allocation | 24,000 MiB |
| Physical GPU headroom | 576 MiB |
| Peak cgroup memory | 131.52 GiB |
| Host and cgroup swap | 0 B |

The 0.97 full-context result improves prefill from 97.7 to about 102.0 tok/s
and settled decode from 35.40 to 39.61 tok/s relative to the validated 0.90
configuration. The full KV pool is preallocated, so this probe exercises the
intended operational peak rather than an idle-only memory estimate.

## Decision

Promote the 0.97 GPU0 FreeToken service as the permanent 262k model. The prior
Q4 service must remain disabled because both services require GPU0. This result
establishes runtime capacity, populated-context throughput, marker retrieval,
and the deliberate 576 MiB GPU headroom for the single-request profile.
