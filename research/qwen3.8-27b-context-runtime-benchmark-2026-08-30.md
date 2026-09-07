# Qwen3.8-27B one-RTX-3090 context benchmark

Date: 2026-08-30

## Result

The 200 W cap was a material limiter for GPU-resident Qwen inference. At 250 W, the best tested
supervisor profile, Qwen3.8-27B Q5 through llama.cpp with a 64k Q8 GPU KV cache, sustained 17.01
tok/s at a populated 64k position while leaving GPU 1 unused. The equivalent 200 W result was
10.26 tok/s.

The Q4 follow-up saves 3,307,069,440 bytes versus Q5 and fits the native 262,144-token context
with Q8 GPU KV. It sustained 10.10 tok/s at a populated 224k position and 7.70 tok/s at 262k.
At matched 32k and 64k contexts and 250 W, Q4 decoded at 24.93 and 20.20 tok/s versus Q5 at
20.21 and 17.29 tok/s. With the corrected thinking-disabled protocol, Q4 scored 92.1% HumanEval+
pass@1 versus 90.2% for Q5.

FreeToken NVFP4 is faster through 32k, sustaining 27.5-31.5 tok/s at 250 W. Its normal FP32
recurrent-state profile cannot allocate 48k on a 24 GiB card, but BF16 recurrent state plus disabled
CUDA graphs can run a populated 48k request at 25.4-26.1 tok/s. That profile leaves almost no VRAM
reserve and 64k remains unavailable. The native 262k host-KV llama.cpp profile fits easily but is
too slow for an interactive supervisor.

| Runtime | 8k decode, 200 W / 250 W | 16k decode, 200 W / 250 W | 32k decode, 200 W / 250 W | 48k | 64k decode, 200 W / 250 W |
|---|---:|---:|---:|---|---:|
| llama.cpp Q5, Q8 GPU KV | 15.41 / 24.14 | Not run | 12.53 / 20.59 | Not run | 10.26 / 17.01 |
| FreeToken NVFP4, BF16 GPU KV | 18.8-19.1 / 29.0-31.5 | 17.8-18.1 / 28.6-30.2 | 16.5-17.0 / 27.5-27.8 | See live-resize results below | 64k unavailable |
| llama.cpp Q5, Q8 host KV | 5.39 / 5.34 | Not run | 3.19 / 3.14 | Not run | 2.01 / 2.03 |

The requested threshold of 64k at more than 10 tok/s is met by llama.cpp GPU KV at both caps; 250
W gives a 65.8% improvement at this populated context.

The extended 250 W sweep establishes a practical GPU-KV limit well below the model's native 262k
context. Decode remains useful through 128k at 13.03 tok/s, drops to 5.71 tok/s at 160k, then
collapses to 0.93 tok/s at 176k and 0.44 tok/s at 192k. A 224k prompt prefills successfully, so
the observed limit is a decode-saturation cliff, not a startup or prefill OOM.

## Test system

- CPU: Threadripper 3970X, 32 physical cores, AVX2.
- RAM: 141 GiB total.
- GPUs: 2 x RTX 3090, 24 GiB each; the original sweep used GPU 0, while the Q4/Q5 follow-up used
  GPU 0 for Q4 and GPU 1 for Q5.
- GPU caps: GPU 0 was 200 W for the first series and 250 W for the second; the Q4/Q5 follow-up ran
  both GPUs at 250 W.
- Driver: 595.84.
- CUDA toolkit/runtime: 13.2; llama-server links to `libcudart.so.13` from CUDA 13.2.
- PCIe: GPU 0 reported Gen3 x8 under/after tested work.
- llama.cpp: `c841aeeb8bb2fe417038dadfa9b007cf1a9ef950`, static internal build.
- FreeToken: 0.1.2.

## Checkpoints

### llama.cpp

- Repository: `unsloth/Qwen3.8-27B-GGUF`.
- Revision: `4ca720788d1e01f1bff70c033e0d0028fd02e502`.
- File: `Qwen3.8-27B-UD-Q5_K_M.gguf`.
- Size: 19,771,509,664 bytes, 18.41 GiB.
- Follow-up file: `Qwen3.8-27B-UD-Q4_K_M.gguf`.
- Follow-up size: 16,464,440,224 bytes, 15.33 GiB.
- Follow-up SHA-256: `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`.

### FreeToken

- Repository: `RadixArk/Qwen3.8-27B-NVFP4`.
- Revision: `319f741cce68d7914884900c138a1fbb70a42f30`.
- Safetensors: 21,921,697,280 bytes, 20.42 GiB.
- Layout: NVFP4 MLP/lm_head, FP8 attention, BF16 ancillary tensors; text-only in FreeToken.

These quantizations are not identical, so this is an operational runtime comparison rather than a
controlled kernel comparison.

## Method

- One visible GPU and one server slot.
- Unique repeated-token prompts prevented prefix-cache reuse.
- llama.cpp generated exactly 128 tokens with EOS ignored; server-returned timings are reported.
- FreeToken used xhigh reasoning to consume 127 tokens; scheduler decode lines provide the range.
- All prompts included chat-template overhead, so actual counts are shown.
- Each populated-context point is one run. No confidence interval is available.
- The original single-GPU series left GPU 1 idle. The Q4/Q5 follow-up deliberately used both cards
  as independent TP=1 servers; quality comparisons are direct, while any throughput comparison
  must account for the two cards' separate power and thermal conditions.
- The 250 W GPU-KV and FreeToken repeats used the same checkpoint, context, prompt shape, output
  length, and server settings as their successful 200 W points. The earlier host-KV points used
  short completions and are qualified separately.

## llama.cpp Q5 with GPU KV

Launch controls:

```text
--ctx-size 65536
--n-gpu-layers 99
--kv-offload
--cache-type-k q8_0
--cache-type-v q8_0
--flash-attn on
--parallel 1
--fit off
```

| Prompt tokens | Completion tokens | Prompt tok/s, 200 W / 250 W | Decode tok/s, 200 W / 250 W | Wall time, 200 W / 250 W |
|---:|---:|---:|---:|---:|
| 8,019 | 128 | 716.34 / 1,042.11 | 15.41 / 24.14 | 19.46 s / 12.98 s |
| 32,019 | 128 | 623.58 / 933.51 | 12.53 / 20.59 | 62.08 s / 41.04 s |
| 64,019 | 128 | 525.11 / 791.15 | 10.26 / 17.01 | 135.37 s / 89.55 s |

VRAM was 20,540 MiB after startup and 20,938 MiB after the populated 64k run at both caps. GPU 1
remained at 9 MiB. A sampled 250 W 32k run held GPU 0 at 248-250 W and reached 71 C, confirming
the higher cap was actually exercised. GPU 1 remained near 21 W.

### Extended 250 W GPU-KV sweep

All entries below use Q5, Q8 GPU KV, Flash Attention on, one GPU, and 128 forced output tokens
unless noted. `nvidia-smi` reported 21,692 MiB at 96k, 22,716 MiB at 128k, 23,868 MiB at 160k, and
24,124 MiB at 176k and above during service initialization.

| Allocated context | Prompt tokens | Prompt tok/s | Decode tok/s | Wall time | Result |
|---:|---:|---:|---:|---:|---|
| 65,536 | 64,019 | 791.15 | 17.01 | 89.55 s | Passed |
| 98,304 | 96,019 | 684.07 | 14.85 | 148.99 s | Passed |
| 131,072 | 128,019 | 589.49 | 13.03 | 227.00 s | Passed |
| 163,840 | 160,019 | 515.67 | 5.71 | 332.66 s | Passed |
| 180,224 | 176,019 | 472.14 | 0.93 | 509.95 s | Passed, impractical decode |
| 196,608 | 192,019 | 425.30 | 0.44 | 739.55 s | Passed, impractical decode |
| 229,376 | 224,018 | 296.12 | N/A, one-token probe | 756.65 s | Prefill passed; no OOM |

The non-linear transition starts between 128k and 160k, then sharpens after the initialization
allocation reaches 24,124 MiB at 176k. That correlation does not establish root cause; a kernel
profile would be needed to distinguish a near-VRAM-limit fallback from a long-context decode-path
issue. The 224k one-token probe proves that the full prompt can populate its configured range, but
does not supply a valid decode rate because the server reports no timed generation interval for one
 token. Native 262k GPU-KV was not attempted: the 176k/192k rates already make it non-operational
 for interactive use.

## Q4 follow-up

The Q4_K_M follow-up used the same one-slot llama.cpp controls as Q5, with Q8 K/V, Flash Attention,
GPU 0 at 250 W, and a fresh server allocation for each point. The 262k prompt was trimmed slightly
so 128 completion tokens fit inside the 262,144-token slot.

### Q4 Q8 GPU-KV capacity

| Allocated context | Prompt tokens | Prompt tok/s | Decode tok/s | Wall time | Startup VRAM |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 32,059 | 990.42 | 24.93 | 37.48 s | 16,316 MiB |
| 65,536 | 64,059 | 842.20 | 20.20 | 82.40 s | 17,468 MiB |
| 131,072 | 128,059 | 612.46 | 14.67 | 217.81 s | 19,644 MiB |
| 163,840 | 160,059 | 534.94 | 12.73 | 309.27 s | 20,668 MiB |
| 196,608 | 192,059 | 476.09 | 11.26 | 414.80 s | 21,820 MiB |
| 229,376 | 224,059 | 429.71 | 10.10 | 534.10 s | 22,844 MiB |
| 262,144 | 261,959 | 380.13 | 7.70 | 705.74 s | 23,996 MiB |

Q4 therefore reaches the native context limit with Q8 GPU KV, but 224k is the largest tested point
near the requested 10 tok/s interactive threshold. At 262k, only about 580 MiB of VRAM remained
after startup and decode fell to 7.70 tok/s.

### Matched Q4/Q5 timing

These are fresh 250 W runs on separate cards: Q4 on GPU 0 and Q5 on GPU 1. Both used the same
32,059- or 64,059-token prompt, 128 forced output tokens, and Q8 K/V.

| Runtime | Context | Prompt tok/s | Decode tok/s | Wall time | Startup VRAM |
|---|---:|---:|---:|---:|---:|
| Q4_K_M | 32k | 990.42 | 24.93 | 37.48 s | 16,316 MiB |
| Q5_K_M | 32k | 906.23 | 20.21 | 42.18 s | 19,524 MiB |
| Q4_K_M | 64k | 842.20 | 20.20 | 82.40 s | 17,468 MiB |
| Q5_K_M | 64k | 790.37 | 17.29 | 88.43 s | 20,548 MiB |

Q4 was 23.4% faster than Q5 at 32k and 16.8% faster at 64k in this one-run comparison. The
throughput result is directional because the servers were on different cards, but both cards were
set to 250 W.

### Quality and behavior

The first EvalPlus 0.3.1 HumanEval+ run used its OpenAI provider defaults. That provider defaults to
`max_new_tokens=768` and sends no Qwen thinking-control field. It produced many empty or truncated
solutions and is invalid as a capability measurement; the results are retained only as a protocol
diagnostic.

The corrected run used `tools/evalplus_qwen_codegen.py` with the same EvalPlus v0.1.10 dataset,
greedy single-sample prompts, `max_tokens=4096`, `temperature=0`, `top_p=0.95`, and
`chat_template_kwargs={"enable_thinking": false}`. Docker was unavailable, so evaluation used
EvalPlus's local execution runner. Both corrected runs had 164/164 `stop` responses, zero empty
samples, and zero reasoning-content fields.

| Runtime | Base HumanEval pass@1 | HumanEval+ pass@1 | Samples | Validity |
|---|---:|---:|---|---|
| Q4_K_M, original run | 45.7% | 44.5% | `quality-results/q4/...jsonl` | Invalid: 71 empty raw outputs; 768-token cap |
| Q5_K_M, original run | 47.6% | 47.0% | `quality-results/q5-250w/...jsonl` | Invalid: 66 empty raw outputs; 768-token cap |
| Q4_K_M, corrected, GPU 0, 250 W | 98.2% | 92.1% | `quality-results/q4-corrected/...jsonl` | Valid |
| Q5_K_M, corrected, GPU 1, 250 W | 96.3% | 90.2% | `quality-results/q5-corrected/...jsonl` | Valid |

The corrected Q4 result is 1.9 percentage points above Q5 on HumanEval+. This reverses the invalid
first-run conclusion and is consistent with the expected Qwen3.8 quality range. No SWE-bench Verified
subset or full evaluation was started: the worktree has no SWE-bench runner or agent scaffold, and
Docker is unavailable for the standard isolated execution path. A formal Q4 quality threshold is
still not required for the next timing decision because Q4 is at least as strong on this screen.

The corrected metadata audit recorded Q4 completion lengths from 53 to 1,826 tokens (median 225)
and Q5 lengths from 53 to 2,170 tokens (median 218), with no `finish_reason=length`, no empty
responses, and no missing generated content caused by the output cap.

The llama.cpp function-call suite, run non-streaming with its five scenarios, passed 5/5 for both
Q4 and Q5. A custom deterministic 32k retrieval smoke test placed an exact `TARGET:` needle at the
beginning, middle, and end of approximately 29.8k-token documents; both Q4 and Q5 returned all
three values correctly (3/3 each). This is a smoke test, not a standardized long-context benchmark.

The existing function-call script cannot directly score FreeToken because it omits the required
`model` field and consequently returned HTTP 422 for all five cases. A direct FreeToken BMI tool
call and follow-up round trip succeeded, so the result is recorded as an API-harness mismatch, not
as a model 0/5 quality score.

## FreeToken NVFP4

Common controls:

```text
--max-running-requests 1
--num-tokens CONTEXT
--max-seq-len-override CONTEXT
--nvfp4-backend triton
--cache-type radix
--cuda-graph-max-bs 1
```

| KV allocation | Prompt tokens | Prefill chunk | Result, 200 W / 250 W | Decode, 200 W / 250 W |
|---:|---:|---:|---|---:|
| 8,192 | 8,059 | 8,192 | Passed in 18.45 s / 12.89 s wall | 18.76-19.11 / 29.03-31.52 tok/s |
| 16,384 | 16,059 | 8,192 | Passed in 31.10 s / 21.96 s wall | 17.82-18.06 / 28.64-30.21 tok/s |
| 32,768 | 32,059 | 8,192 | Prefill OOM; worker exited | No result |
| 32,768 | 32,059 | 2,048 | Passed in 55.72 s / 39.89 s wall | 16.46-16.99 / 27.47-27.80 tok/s |
| 49,152 | N/A | 2,048 | Startup OOM at 200 W; not rerun at 250 W | N/A |
| 65,536 | N/A | 2,048 | Startup OOM at 200 W; not rerun at 250 W | N/A |

The 32k service used 23,504 MiB VRAM and retained only 0.61 GiB after graph capture. An 8k prefill
chunk then failed on a 32 MiB allocation. Reducing the chunk to 2k made the full request pass. The
same 250 W run held GPU 0 at 248-249 W, reached 70 C, and improved warmed prefill to approximately
836-1,000 tok/s from approximately 605-724 tok/s at 200 W.

The original direct 48k and 64k startup attempts failed before serving. Their 3.0/4.0 GiB KV pools
left insufficient room for FreeToken's recurrent-state allocation. The later live-resize experiment
shows that BF16 recurrent state can make 48k allocatable, but not reliably usable with graphs enabled
and not allocatable at 64k. This remains a capacity limit of the current dense checkpoint and TP=1
implementation, not the long-prefill issue discussed below.

### 32k FreeToken runtime cross-check

The existing FreeToken NVFP4 checkpoint was run separately at 32k with the same repeated-token
prompt shape and GPU 0 at 250 W. It completed in 39.70 s for 32,059 prompt tokens and 127 generated
tokens; the scheduler decode range was approximately 27.5-28.4 tok/s. This is a runtime reference
only: FreeToken cannot load the GGUF Q4_K_M checkpoint, so it is not a Q4 quality result.

### FreeToken live cache resize and BF16 SSM experiment

The daemon cache controls were exercised while idle, using GPU 0 at 250 W, one request, 2,048-token
prefill chunks, the Triton NVFP4 backend, and a 128-token xhigh reasoning request. `mamba=5` means
five recurrent-state slots; the model's `dtype=torch.bfloat16` log entry is the model activation
dtype, while `FREETOKEN_MAMBA_SSM_DTYPE` controls the recurrent-state pool.

| Profile and operation | Cache allocation | VRAM / reserve | Result |
|---|---:|---:|---|
| FP32 SSM, initial startup | 32k KV + 8 slots | 23,504 MiB / 0.61 GiB | Passed; existing 32k baseline |
| FP32 SSM, live `mamba=5` | 32k KV + 5 slots | 23,128 MiB observed | Passed; service remained healthy |
| FP32 SSM, live KV 48k | 48k KV + 5 slots | 3.00 GiB requested / 2.84 GiB budget | Rejected safely; old 32k cache retained |
| FP32 SSM, live KV 64k | 64k KV + 5 slots | 4.00 GiB requested / 2.84 GiB budget | Rejected safely; old 32k cache retained |
| BF16 SSM, initial startup | 32k KV + 8 slots | 22,864 MiB; 74.8 MiB per slot | Passed; 32k request took 39.877 s, 27.6-28.2 tok/s |
| BF16 SSM, live `mamba=5` | 32k KV + 5 slots | 0.37 GiB state pool | Passed; service remained healthy |
| BF16 SSM, live KV 48k, graphs enabled | 48k KV + 5 slots | 23,688 MiB after rebuild | Allocation passed; populated 48k request crashed on a 52 MiB scratch allocation with 58.56 MiB free |
| BF16 SSM, live KV 64k, graphs enabled | 64k KV + 5 slots | 4.00 GiB requested / 3.26 GiB budget | Rejected safely; old 32k cache retained |
| BF16 SSM, graph 0 startup | 32k KV + 8 slots | 1.26 GiB free after initialization | Passed; log explicitly reports CUDA graph disabled |
| BF16 SSM, graph 0, live KV 48k | 48k KV + 5 slots | 3.00 GiB KV + 0.37 GiB state | Passed; populated 48k request took 67.580 s, 25.4-26.1 tok/s, 23,948-24,124 MiB sampled (24,124 MiB peak) |
| BF16 SSM, graph 0, live KV 64k | 64k KV + 5 slots | 4.00 GiB requested / 3.26 GiB budget | Rejected safely; old 48k cache retained |
| BF16 SSM, naive 64k startup | 64k KV, naive cache | 4.00 GiB KV; 24.56 MiB free | Failed during startup on a 256 MiB FlashInfer workspace allocation; engine exited |

The BF16 SSM pool is approximately half the FP32 size, reducing each slot from 146.8 MiB to
74.8 MiB. This is enough to make a 48k radix allocation possible, but CUDA graph capture leaves
insufficient transient workspace for the populated request. Disabling graphs makes 48k usable in
this single-run test, at the cost of approximately 67.6 s wall time and almost no production VRAM
reserve. The 64k preflight still rejects because the 4.00 GiB KV pool alone exceeds the remaining
cache budget once recurrent state is included. Naive cache mode does not improve capacity and was
not promoted because it also failed before serving and would lose radix prefix reuse.

## Native 262k host-KV reference

llama.cpp successfully allocated Q8 host KV at the model's native 262,144-token limit while Q5
weights remained on GPU 0. Because `--no-kv-offload` also runs attention on CPU, decode degraded
with populated context:

| Prompt tokens | Prompt tok/s, 200 W / 250 W | Decode tok/s, 200 W / 250 W |
|---:|---:|---:|
| 8,016 | 596.76 / 627.76 | 5.39 / 5.34 |
| 32,016 | 535.82 / 559.73 | 3.19 / 3.14 |
| 64,016 | 438.63 / 477.48 | 2.01 / 2.03 |

This profile is a capacity fallback, not the preferred supervisor. The 200 W reference used only
two or three generated tokens, while the 250 W series forced 128; the near-identical decode values
are directional rather than a high-confidence A/B result. They are nevertheless consistent with
the expected CPU-attention bottleneck: 250 W improved prompt throughput modestly but did not improve
decode materially.

## Power-cap result

GPU 0 at 200 W was materially power-limited for GPU-resident Qwen inference. Relative to the 200 W
series, 250 W improved llama.cpp GPU-KV decode by 56.7% at 8k, 64.3% at 32k, and 65.8% at 64k. It
improved FreeToken decode by approximately 57-66% through its standard tested 32k profile. The
BF16 graph-disabled 48k diagnostic also reached 246-249 W during decode, but is not a production
profile. The 32k sampled runs reached the 250 W cap continuously, ruling out a measurement that
merely changed the configured limit without changing delivered power.

This is a one-card finding. It does not justify raising all three future RTX 3090s to 250 W: that
would violate the existing 1,100 W room-server power budget under representative CPU load. GPU 0
and GPU 1 were both temporarily set to 250 W for the isolated Q4/Q5 comparison; the normal policy
keeps GPU 0 at 250 W and GPU 1 at 200 W.

## Issue review

- llama.cpp issue 27623 remains open, but its reporter retracted the claimed >80k decode cliff on
  2026-08-26. The original low number divided generated tokens by total request time, including
  long prompt prefill. Current server-side timings did not reproduce the cliff.
- A separate 2 x RTX 3090 result in that issue used Q4_K_XL across both cards and measured 33.97
  tok/s at 65,536 and 22.02 tok/s at 262,144. It is not comparable to this one-card Q5 test,
  whether at 200 W or 250 W.
- FreeToken issue 87 is closed. Its crash came from setting one 262k prefill chunk. The confirmed
  workaround is 8k/16k chunked prefill, including successful 200k tests. This report used at most
  8k, then 2k where transient VRAM required it.
- FreeToken issue 141 remains open; advanced KV compression is not currently available.

## Decision

1. Prefer Q4_K_M with Q8 GPU KV for the supervisor: it measured 20.20 tok/s at 64k and 10.10
   tok/s at 224k, while corrected HumanEval+ pass@1 was 92.1% versus Q5's 90.2%.
2. Use Q5_K_M with Q8 GPU KV only when its higher-precision quantization is specifically required;
   its corrected HumanEval+ result did not exceed Q4 in this run.
3. Treat Q4 at native 262k as a capacity mode, not an interactive default: it passed at 7.70 tok/s
   with only about 580 MiB of VRAM reserve.
4. Use FreeToken NVFP4 with GPU 0 at 250 W for 8k-16k latency-critical work. Its 32k profile works only with 2k chunks
   and has too little VRAM reserve for a production default. BF16 SSM plus `--graph 0` reaches 48k
   at 25.4-26.1 tok/s, but its near-full-VRAM footprint makes it diagnostic rather than a promoted
   default.
5. Do not use the host-KV 262k profile for interactive work.
6. The persistent service is currently configured for 250 W on both GPUs for this comparison. Keep
   GPU 1 at 250 W only while the second-card workload is intentional; reassess permanent dual-250 W
   operation against the room power budget before adding further GPU workloads.
7. Run a representative SWE-bench Verified subset only after a runner/agent scaffold and isolated
   execution environment are available. Keep the full SWE-bench evaluation gated on that subset.

## Sources

1. [llama.cpp issue 27623](https://github.com/ggml-org/llama.cpp/issues/27623)
2. [FreeToken issue 87](https://github.com/FlashML-org/FreeToken/issues/87)
3. [FreeToken issue 141](https://github.com/FlashML-org/FreeToken/issues/141)
4. [llama.cpp Qwen3.8 RTX 3090 discussion](https://github.com/ggml-org/llama.cpp/discussions/27164)
5. [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)
6. [Unsloth Qwen3.8 GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
7. [FreeToken CLI](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md)
