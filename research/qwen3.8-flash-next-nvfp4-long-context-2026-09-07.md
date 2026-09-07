# Qwen3.8 Flash-Next NVFP4 Long-Context Validation

Date: 2026-09-07

> Superseded for production selection by
> `qwen3.8-flash-next-nvfp4-x16-ddr4-3066-2026-09-07.md`, which records the
> validated PCIe x16, DDR4-3066, 0.97-ratio, 2,344-slot configuration.

## Active Configuration

`freetoken-flash-next-candidate.service` runs on GPU0 with one request at a
time. The active cache geometry is:

| Pool | Allocation |
|---|---:|
| QSA KV | 4,096 pages x 64 tokens = 262,144 tokens |
| NVFP4 MoE LRU | 1,536 slots |
| GDN recurrent state | 8 usable slots |

Other relevant settings are hybrid CPU/GPU MoE, Triton NVFP4, serial expert
loading, disk-backed PLE, 512-token prefill chunks, 90% memory ratio, prefill
overlap, and D2D reuse of resident prefill experts.

## Method

Each probe built a fresh prompt containing a UUID marker, disabled thinking,
used temperature zero, and requested 128 output tokens. The marker was placed
only at the start and end of the prompt and the model was instructed to return
it first. Every logged 512-token prefill chunk reported zero cached prompt
tokens. The client captured GPU, cgroup, swap, and server telemetry throughout.

## Results

| Target context | Prompt / output tokens | Wall time | Steady prefill | Settled decode | Marker |
|---:|---:|---:|---:|---:|---|
| 196,608 | 196,414 / 127 | 4,059.50 s | 48.64 tok/s | 28.08 tok/s | exact, first |
| 262,144 | 261,943 / 127 | 5,364.86 s | 48.91 tok/s | 28.19 tok/s | exact, first |

The first decode scheduler sample is an admission transition and is not used as
a throughput result. The short `/v1/stats` rate can similarly spike at the
final partial prefill chunk, so the stable scheduler chunk rates above are the
authoritative prefill measurement.

The 256k completion emitted EOS immediately after the requested marker. Because
the probe deliberately sets `ignore_eos`, subsequent protocol tokens appear in
the captured output. They are after the exact first marker and do not affect the
retrieval result.

## Resource Gate

| Metric | 192k | 256k |
|---|---:|---:|
| Peak GPU allocation | 21,840 MiB | 21,842 MiB |
| Highest sampled cgroup memory | 130.47 GiB | 130.35 GiB |
| Cgroup memory peak for this service lifetime | 132.05 GiB | 132.05 GiB |
| Cgroup and host swap | 0 B | 0 B |

The 256k run filled the KV pool to 100%, reached 100% sampled GPU utilization,
and stayed below the 300 W GPU0 cap. Its host-memory margin is narrow. Do not
run another large host-bank workload alongside it, enable pinned PLE, or raise
the request concurrency without a new memory validation.

## Decision

The candidate meets the long-context gate through the checkpoint's 262,144-token
limit: populated, uncached 192k and 256k prompts retrieve their unique marker,
decode above 15 tok/s, and complete without swap. Keep it isolated from the
unchanged production Qwen service. This is a runtime viability result, not a
broad task-quality or multi-request production promotion.
