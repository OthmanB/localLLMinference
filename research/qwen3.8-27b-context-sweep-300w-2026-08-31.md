# Qwen3.8-27B Long-Context Speed Sweep

Date: 2026-08-31

## Result

The sweep used a fresh llama.cpp server allocation for every point, 300 W GPU
limits, Q8 GPU KV, Flash Attention, one server slot, and thinking disabled.

| Model | Context | Prompt tokens | Prefill tok/s | Decode tok/s | Request wall | Max VRAM | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| Q4 | 192k / 196608 | 196227 | 555.00 | **15.43** | 362.5 s | 22728 MiB | Passed |
| Q4 | native / 262144 | 261765 | 448.01 | **10.16** | 597.7 s | 24124 MiB | Passed |
| Q5 | 192k / 196608 | 196227 | 467.25 | **0.36** | 775.8 s | 24126 MiB | Passed, impractical |
| Q5 | native / 262144 | 261761 | 244.10 | **0.17** | >1800 s | 24126 MiB | Partial timeout |

The Q5 native-context server allocated successfully and completed the full
261761-token prefill. It generated 118 tokens at approximately 0.17 tok/s before
the 1800-second request ceiling cancelled it. This was not an OOM or startup
failure; the decode path is operationally unusable at this context.

## Controls

- Q4 used GPU 0; Q5 used GPU 1.
- `--ctx-size` was `196608` and `262144`.
- `--n-gpu-layers 99`, `--kv-offload`, `--cache-type-k q8_0`,
  `--cache-type-v q8_0`, `--flash-attn on`, `--parallel 1`, and `--fit off`.
- Unified memory was disabled.
- Each request asked for 128 generated tokens with EOS ignored.
- Sampling matched the final SWE-bench run: temperature 0.7, top-p 0.8,
  top-k 20, presence penalty 1.5.
- `chat_template_kwargs.enable_thinking` was explicitly `false` and
  `reasoning_format` was `none`.
- Prompt content was generated close to the requested context size, leaving a
  384-token margin for the completion and chat-template overhead.
- Every point used a fresh server, so no prior request prefix cache was reused.

## Context Scaling

Q4 remains usable at native context with GPU KV. Its decode rate falls from
15.43 tok/s at 192k to 10.16 tok/s at 262k, a 34.1% reduction. Native Q4 uses
24124 MiB of the 24576 MiB card, leaving approximately 452 MiB before runtime
variation.

Q5 is already below practical interactive speed at 192k. Increasing to native
262k reduces the observed rate to approximately 0.17 tok/s, with the card using
24126 MiB and approximately 450 MiB remaining. Increasing the power limit does
not remove this long-context decode cliff.

## Comparison With 250 W Reference

The earlier 250 W measurements reported Q4 at 11.26 tok/s for 192k and 7.70
tok/s for native 262k. The new 300 W results are 37.0% and 32.0% higher,
respectively. The earlier Q5 192k result was 0.44 tok/s; the new 0.36 tok/s
result is not faster, indicating that Q5 is dominated by the near-capacity
long-context path rather than raw compute power. These cross-run comparisons
are directional because prompt construction and request policy were not
identical in every detail.

## Relation To SWE-bench Quality

The SWE-bench run used thinking disabled and resolved 22/40 tasks for both Q4
and Q5. The approximately 66% result cited separately used medium thinking, so
it is not a directly matched quality comparison. The speed sweep also keeps
thinking disabled and therefore describes the deployed SWE-bench policy, not
the medium-thinking quality configuration.

## Artifacts

- Results JSON: `qwen3.8-27b-context-sweep-300w-2026-08-31.json`
- Q4 192k server log: `q4-196608.server.log`
- Q4 native server log: `q4-262144.server.log`
- Q5 192k server log: `q5-196608.server.log`
- Q5 native server log: `q5-262144.server.log`

All temporary sweep servers have been stopped after the measurements; both GPUs
are idle.
