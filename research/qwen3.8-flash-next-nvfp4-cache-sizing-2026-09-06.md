# Qwen3.8 Flash-Next NVFP4 Cache Sizing

Date: 2026-09-06

## Scope

This follow-up sizes the FreeToken runtime caches for a single-request 256k
candidate on GPU0. It uses the live engine's measured tensor sizes and runtime
cache rebuilds rather than estimates from checkpoint metadata.

## Live Unit Costs

The 4k candidate reported a 12,328,429,158-byte cache budget and these unit
costs through `/v1/cache/status`:

| Pool | Unit cost |
|---|---:|
| QSA KV | 25,344 bytes/token |
| NVFP4 MoE cache | 2,772,480 bytes/expert slot |
| GDN recurrent state | 115,642,376 bytes/usable slot |

The model requires at least 512 MoE slots and the single-request hybrid-radix
configuration requires at least four usable recurrent-state slots. Startup
currently allocates eight recurrent-state slots.

## Capacity Validation

Runtime rebuilds succeeded for both of these geometries:

| KV capacity | MoE slots | GDN slots | Idle GPU allocation |
|---:|---:|---:|---:|
| 262,144 tokens | 512 | 8 | 18,956 MiB |
| 262,144 tokens | 1,536 | 8 | 21,698 MiB |

The selected 256k geometry leaves 2,878 MiB of physical GPU memory free. Its
configured pools consume about 11.02 GiB of the 11.48 GiB FreeToken cache
budget, leaving about 0.47 GiB inside that budget in addition to the runtime's
separate 10% graph/activation reserve.

## Decode Screening

Each point used the same 2,172-token prompt and 127-token deterministic output.
The first request after each rebuild warmed the LRU; the second request supplied
the steady scheduler rate.

| MoE slots | Steady decode | Request wall time |
|---:|---:|---:|
| 512 | 23.23 tok/s | 16.47 s |
| 1,024 | 25.46 tok/s | 15.75 s |
| 1,536 | 27.52 tok/s | 15.62 s |

The wall time includes a 60-token uncached prompt suffix and therefore is not a
decode-only metric. Scheduler decode throughput is the selection metric.

## Selected Configuration

The installed and active candidate service requests:

- 262,144-token sequence and KV capacity.
- 1,536 unified NVFP4 expert slots.
- 512-token prefill chunks until a guarded long-context run establishes
  activation headroom.
- Prefill copy overlap, which is FreeToken's default and is valid once the MoE
  cache has at least 1,024 slots.
- D2D reuse of cache-resident prefill experts when the runtime copy path supports
  it.

The installed systemd unit and active 4k process were not changed. A restart is
required before testing prompts above 4,096 tokens.

## Validated 0.97 Expansion

The subsequent x16/DDR4-3066 validation used 1,664 slots at the same 90%
free-VRAM ratio and passed the 262k marker probe. The final geometry retains the
262,144-token KV pool and eight GDN slots, but raises the ratio to 97% and the
MoE cache to 2,344 slots. Its 14,067,609,664-byte pool allocation fits within
the 14,077,746,872-byte FreeToken cache budget and peaked at 24,000 MiB GPU
allocation, retaining 576 MiB physical headroom. The short and native-context
benchmarks completed with zero swap, exact marker retrieval, 102.0 tok/s
prefill, and 39.61 tok/s settled decode.
