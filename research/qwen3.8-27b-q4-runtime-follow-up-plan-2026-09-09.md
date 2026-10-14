# Qwen3.8-27B Runtime Follow-Up Plan

Date: 2026-09-09

## Objective

Separate the three runtime optima exposed by the physical-VRAM evaluation:

- Layer split plus Q8 K/V for maximum context capacity and prompt ingestion.
- Tensor split plus F16 K/V for a single coding/reasoning agent with long
  generation.
- A continuous-batching runtime for independent concurrent agents.

No production service, routing, or model placement changes are part of this
plan.

## Output-Length Validation

The 128-token results do not represent long-running coding agents. At 196k,
the measured layer-to-tensor prefill penalty is about 58 seconds, while tensor
saves about 0.025 seconds per generated token. The predicted break-even is
approximately 2,300 generated tokens. The native 262k point has a comparable
2,260-token break-even.

Run fresh 196k prompts with 4,096 forced generated tokens for both physical
profiles. Use the NCCL-enabled llama.cpp v0.4.0-dev build, Flash Attention,
Tensor is the one-agent coding candidate only if the observed long-output wall
time is lower without sustained clock collapse or an execution error.

## Tensor Scheduler Check

Run exactly one moderate-pool tensor concurrency test:

```text
--ctx-size 262144 --parallel 2
slot A: 128,000 prompt tokens
slot B: 64,000 prompt tokens
1,024 forced output tokens each
```

This is not a 524k capacity test. It isolates whether tensor execution changes
the long-prefill starvation seen with layer split. Report both per-request
decode rates, aggregate decode, completion time, peak VRAM, and thermal/clock
samples. If one active stream is starved, treat scheduling rather than split
mode as the llama.cpp concurrency limitation and stop further llama.cpp
concurrency permutations.

## Native-INT4 SGLang Gate

Use the public direct-base checkpoint
`cyankiwi/Qwen3.8-27B-AWQ-INT4` rather than converting the Q4_K_M GGUF. The
checkpoint is Apache-2.0 and is a conventional AWQ INT4 candidate for SGLang
TP=2.

1. Download to the local model store after a dry-run size check.
2. Start SGLang 0.5.16 with TP=2 at 8k and one request. Record architecture,
   allocation, and startup result.
3. If the gate passes, repeat single-agent 196k and 196k+64k concurrent tests
   with the same output lengths and report TTFT, per-stream decode, aggregate
   decode, capacity, and thermals.
4. Before any serving comparison can influence a production decision, perform
   a controlled coding-quality comparison against the existing Q4_K_M GGUF.
   A throughput gain does not compensate for a material capability regression.

SGLang supports native quantized model serving and tensor parallelism, but the
previous exact-GGUF failure remains a loader limitation for `qwen35`, not a
negative scheduler result.

## Execution Outcome

The long-output and tensor scheduler tests completed. The selected AWQ-labelled
checkpoint loaded but failed its trivial quality smoke because SGLang 0.5.16
did not correctly map packed hybrid linear-attention parameters. Its long
context benchmark was correctly stopped; see
`qwen3.8-27b-q4-runtime-follow-up-results-2026-09-09.md`.

## Completion Criteria

- Publish all raw artifacts and a result addendum.
- Keep tensor and layer as separate workload profiles unless long-output data
  disproves the break-even prediction.
- Do not enable long-context `--parallel 2` in llama.cpp if the tensor
  scheduler test also starves a stream.
- Do not promote an AWQ profile without its quality gate and selected-topology
  metrics/accounting work.
