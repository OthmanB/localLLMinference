# Qwen3.8-27B Q4 Dual-GPU Layer Results

Date: 2026-09-09

Status: Superseded for promotion purposes by
`qwen3.8-27b-q4-multigpu-runtime-evaluation-results-2026-09-09.md`. This
earlier campaign ran before the UVM environment issue was identified and before
the required tensor, SGLang, and strict two-slot evaluations.

## Result

Qwen3.8-27B Q4_K_M ran successfully on GPU1+GPU2 with llama.cpp layer split,
Q8 K/V, Flash Attention, a 262,144-token allocation, and one server slot.
The profile materially improves native-context decode and prefill while keeping
total measured GPU plus CPU request energy approximately equal to the one-GPU
native-context control because the request completes much sooner.

| Prompt | Profile | Prefill tok/s | Decode tok/s | Wall s | GPU+CPU energy | Result |
|---:|---|---:|---:|---:|---:|---|
| 4k | GPU1, 262k allocation | 1236.87 | 38.06 | 6.71 | 2.05 kJ | Passed |
| 4k | GPU1+GPU2 layer, 262k allocation | 1643.07 | 38.99 | 5.81 | 3.29 kJ | Passed |
| 196k | GPU1, 196k allocation | 567.20 | 16.02 | 354.57 | 135.77 kJ | Passed |
| 196k | GPU1, 262k allocation | 566.99 | 8.01 | 363.37 | 137.83 kJ | Passed |
| 196k | GPU1+GPU2 layer, 262k allocation | 939.67 | 18.65 | 216.72 | 138.59 kJ | Passed |
| 262k | GPU1, 262k allocation | 453.44 | 10.44 | 590.39 | 224.39 kJ | Passed |
| 262k | GPU1+GPU2 layer, 262k allocation | 770.22 | 15.58 | 348.92 | 223.95 kJ | Passed |

At 196k, layer split improved prefill by 65.7%, decode by 132.7%, and request
wall time by 40.4% relative to the one-GPU server allocated to 262k. At native
262k, it improved prefill by 69.9%, decode by 49.2%, and wall time by 40.9%,
with 0.2% less sampled total request energy.

The short 4k point improves prefill by 32.8% but leaves decode effectively
unchanged (38.99 versus 38.06 tok/s); using two GPUs costs more energy for a
short request. The long-context results are the reason to consider the profile.

The clean deployed-allocation control confirms the candidate is faster even
after giving the single-GPU server its normal 196,608-token allocation. Against
that control, the 262k layer profile improved prefill by 65.7%, decode by
16.5%, and wall time by 38.9%, with 2.1% more sampled GPU plus CPU request
energy. The layer request included 385 more prompt tokens, a 0.2% difference
that slightly understates the candidate's speed advantage.

## Capacity and Thermals

The native layer point peak allocation was 13,764 MiB on GPU1 and 14,914 MiB
on GPU2, leaving approximately 10,812 MiB and 9,662 MiB respectively. This is
substantially safer than the one-GPU native point at 24,126 MiB of 24,576 MiB.

The clean 196k one-GPU control peaked at 22,728 MiB, leaving 1,848 MiB. The
262k layer candidate at a 196k prompt peaked at 13,250 MiB on GPU1 and 14,400
MiB on GPU2. The candidate therefore supplies native 262k capacity without
the single-GPU allocation cliff.

The room was approximately 28-30 C without air conditioning. GPU1 peaked at
76 C and GPU2 at 87 C during the native layer request. Record the latter for a
cooler-room repeat or production fan/thermal validation; it is not treated as a
campaign failure under the observed ambient conditions.

## Controls

- One-GPU baseline: physical GPU1 with `CUDA_VISIBLE_DEVICES=1`.
- Candidate: physical GPU1+GPU2 with `CUDA_VISIBLE_DEVICES=1,2`,
  `--device CUDA0,CUDA1`, `--split-mode layer`, and `--tensor-split 1,1`.
- Both profiles used the existing Q4 GGUF, `--ctx-size 262144`, `--parallel 1`,
  all GPU layers, Q8 K/V, Flash Attention, 32 generation/batch CPU threads,
  batch size 2048, microbatch size 512, no unified memory, and `--fit off`.
- The fresh deployed control used the same one-GPU settings with
  `--ctx-size 196608` and a 196,224-token target prompt.
- Prompts were deterministic, fresh-server requests with thinking disabled,
  EOS ignored, and 128 forced output tokens.
- Flash, Q4, and Muse were stopped before launch and were originally inactive,
  so the wrapper left all three inactive after successful completion. Gateway
  and OpenCode routing were not changed.

## Deployed 196k Comparison

The fresh GPU1-only `196608` baseline completed successfully at 02:36:42 UTC.
It confirms the benchmark server initialized one slot with a 196,608-token
context and ran exclusively with `CUDA_VISIBLE_DEVICES=1`. Its result artifact
is:

`research/qwen3.8-27b-q4-dual-gpu-layer-20260909T023018Z/results.json`

Its populated 196k point completed with exit status zero, no prompt-cache
reuse, 337 NVIDIA samples, and the expected 196,227 prompt tokens.

## Historical Recommendation (Withdrawn)

The recommendation below is withdrawn. The physical-VRAM follow-up confirms
that layer split is the only strict-capacity candidate, but its long-context
two-slot scheduler starves one active request. SGLang also cannot load the
exact GGUF. Do not promote from this historical result.

No production service, routing, Muse placement, or Flash disposition changed
as part of either evaluation.

## Superseded Deferred Work

Tensor, SGLang, and strict two-slot evaluation are now complete. See
`qwen3.8-27b-q4-multigpu-runtime-evaluation-results-2026-09-09.md` for their
physical-VRAM results and the current non-promotion conclusion.
