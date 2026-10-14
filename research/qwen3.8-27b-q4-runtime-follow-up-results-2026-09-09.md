# Qwen3.8-27B Runtime Follow-Up Results

Date: 2026-09-09

## Result

The workload has three distinct optima:

- Tensor split with F16 K/V is the preferred llama.cpp profile for one
  long-running coding/reasoning agent.
- Layer split with Q8 K/V remains the capacity and prompt-ingestion profile,
  including the only configuration proven to fit two native 262k slots.
- A clean SGLang comparison remains blocked by quantization-loader correctness,
  not by an architectural verdict against SGLang.

No production service, routing, or model placement changed.

## Long-Generation Crossover

Fresh 196,612-token prompts with 4,096 forced generated tokens validated the
prediction from the 128-token campaign.

| Profile | Prefill tok/s | Decode tok/s | Wall s | GPU2 max C | GPU2 graphics clock MHz |
|---|---:|---:|---:|---:|---:|
| Layer, Q8 K/V | 1073.20 | 18.85 | 401.17 | 77 | 1200-1920 |
| Tensor, F16 K/V | 816.06 | 36.52 | 353.72 | 87 | 1545-1860 |

Tensor completes 47.44 seconds sooner, an 11.8% end-to-end reduction. Its
extra prefill time is recovered after about 2,300 generated tokens; this
4,096-token result is a direct validation rather than an extrapolation. GPU2
held at least 1545 MHz throughout the tensor request, so its 87 C peak did not
cause a sustained graphics-clock collapse.

Artifact:
`research/qwen3.8-27b-q4-196k-long-output-20260909T102855Z/results.json`.

## Tensor Moderate Concurrency

Tensor split was tested with a 262,144-token shared pool, two 131,072-token
slots, 128,000 plus 64,000 input tokens, and 1,024 forced output tokens each.

| Stream | Prompt tok/s | Decode tok/s | Request s |
|---|---:|---:|---:|
| 128k | 939.55 | 11.72 | 225.13 |
| 64k | 333.76 | 30.27 | 225.84 |

Aggregate decode was 23.25 tok/s. Both streams made substantial progress; this
is materially different from layer split's 0.36-2.42 tok/s starved stream in
the large simultaneous-prefill tests. Peak VRAM was 16,424 MiB per GPU, GPU2
peaked at 78 C, and its graphics clock stayed between 1695 and 1920 MHz.

This is a promising moderate-concurrency result, not a claim that tensor can
fit the strict 524k/two-native-262k requirement. F16 K/V prevents that.

Artifact:
`research/qwen3.8-27b-q4-tensor-parallel2-moderate-20260909T105636Z/results.json`.

## SGLang Native-INT4 Gate

The public 21.02 GB `cyankiwi/Qwen3.8-27B-AWQ-INT4` checkpoint was downloaded
to `models/qwen3.8-27b-awq-int4`. It is a native Qwen3.5 Transformers model,
not a GGUF. SGLang 0.5.16 TP=2 loaded it as `compressed-tensors` and started an
8k server, including chunked prefill and a unified radix cache.

The quality smoke failed: the instruction "Reply with the word ready" generated
32 repeated `!` tokens. The server log also reports missing packed
linear-attention parameters. This is a numerical correctness failure, so its
196k and concurrent throughput tests were intentionally not run.

The alternative public `RedHatAI/Qwen3.8-27B-INT4` uses the same
`pack-quantized`/`compressed-tensors` format and hybrid linear-attention layout,
so downloading another 21 GB checkpoint would not be an independent gate.
The exact Q4_K_M GGUF remains separately blocked by the unsupported `qwen35`
GGUF architecture.

## Next Gate

Revisit SGLang when a release validates Qwen3.8 hybrid linear attention with a
correct native quantized checkpoint. Then run 196k single-agent and 196k+64k
concurrent comparisons, followed by coding-quality evaluation against Q4_K_M.
Until then, select layer or tensor by workload and retain the existing
non-production status.
