# Qwen3.8-27B Q4 v0.4.0 Runtime A/B

Date: 2026-09-06

## Scope

This compares the established Qwen3.8-27B Q4_K_M GPU0 service settings with
an isolated official llama.cpp v0.4.0 build. The production service was kept
stopped and unchanged. GPU0 used its configured 300 W limit.

- Model: `models/qwen3.8-27b-q4-k-m-gguf/Qwen3.8-27B-UD-Q4_K_M.gguf`.
- GPU: CUDA0 only; Q8 K/V GPU cache, Flash Attention, 32 target and batch
  threads, batch 2048, ubatch 512, one slot, and no UVM.
- Service context: 196,608 tokens.
- Workload: fresh server, no prompt-cache reuse, 128 output tokens,
  temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5, thinking off.
- Established runtime: 0.3.0-dev build 1, commit `c841aee`.
- Candidate runtime: official v0.4.0-dev build 10809, commit `5266f24da`.
- All-quant FA runtime: the same official source, built separately with
  `GGML_CUDA_FA_ALL_QUANTS=ON`, CUDA 13.2, SM86, NCCL, CUDA Graphs, and Flash
  Attention enabled.

## Result

| Runtime and mode | Context | Prefill tok/s | Decode tok/s | Peak VRAM | Verdict |
|---|---:|---:|---:|---:|---|
| Established 0.3.0, base | 8k | 1250.81 | 37.38 | 15,568 MiB | Control |
| Official v0.4.0, base | 8k | 1250.19 | 37.26 | 15,570 MiB | Neutral |
| Official v0.4.0, native MTP n=3 | 8k | 1149.06 | 38.77 | 16,608 MiB | +4.1% decode; slower prefill |
| Official v0.4.0, base | 192k | 629.99 | 16.56 | 22,736 MiB | Prefill gain; decode loses control |
| Official v0.4.0, native MTP n=3 | 192k | 581.36 | 5.21 | 24,124 MiB | Reject |
| Established 0.3.0, base, same session | 192k | 574.38 | 16.80 | 22,736 MiB | Decode control |

The same-session established-runtime control removes the apparent v0.4.0
decode improvement: v0.4.0 is 1.4% slower on decode, although it improves
prefill by 9.7%. The prior 300 W 192k record for the established runtime was
555.00 tok/s prefill and 15.43 tok/s decode; it was historical rather than a
controlled comparison.

## All-Quant Flash Attention KV Cache

The standard v0.4.0 build has `GGML_CUDA_FA_ALL_QUANTS=OFF`, so an isolated
all-quant FA build was used to validate lower-bit GPU KV cache modes. Each
result used a fresh 196,216-token prompt, 128 generated tokens, no prompt
cache reuse, thinking off, and no UVM.

| K/V cache | Context | Prefill tok/s | Decode tok/s | Peak VRAM | Verdict |
|---|---:|---:|---:|---:|---|
| Q8_0/Q4_0 | 8k | 1260.80 | 36.97 | 15,570 MiB | Valid |
| F16/Q4_0 | 8k | 1261.69 | 37.31 | 15,698 MiB | Valid |
| Q4_0/Q4_0 | 8k | 1248.31 | 36.80 | 15,442 MiB | Valid |
| Q8_0/Q4_0 | 192k | 631.49 | 15.56 | 21,200 MiB | Reject: slower decode |
| F16/Q4_0 | 192k | 630.13 | 16.00 | 23,760 MiB | Reject: slower decode, little headroom |
| Q4_0/Q4_0 | 192k | 623.98 | 15.74 | 19,664 MiB | Reject: slower decode |
| F16/Q4_0, MTP n=2 | 192k | 549.83 | 1.04 | 24,124 MiB | Reject |

F16/Q4_0 was the fastest lower-bit base mode at 192k, but was still 4.8%
slower than the established Q8_0/Q8_0 runtime. Its MTP n=2 run drafted 108
tokens and accepted 72 (66.7%), but decode collapsed to 1.04 tok/s. Lower-bit
KV cache modes are useful only where their VRAM savings are required; they do
not improve this populated-192k service.

## Native MTP

Qwen3.8-27B Q4 supports llama.cpp native `draft-mtp`; it is not an MoE expert
cache and was tested separately with `--spec-draft-n-max 3`.

At 8k it drafted 195 tokens and accepted 61, producing a small decode gain.
At 192k it drafted 188 tokens and accepted 63 but consumed nearly all GPU0
VRAM and reduced decode by 68.5%. Do not enable native MTP in the 192k service.
It may be useful only for a separately measured short-context profile.

## Decision

Keep the established 0.3.0 Q8_0/Q8_0 runtime for the 192k service. Do not
enable MTP or lower-bit KV cache modes, and do not stage a v0.4.0 service
change: none improves the target populated-192k decode rate. The v0.4.0 base
runtime remains available if its prefill improvement later warrants a separate
agent-quality evaluation. Raw all-quant results are under
`research/qwen3.8-27b-v0.4.0-fa-all-2026-09-06/`.
