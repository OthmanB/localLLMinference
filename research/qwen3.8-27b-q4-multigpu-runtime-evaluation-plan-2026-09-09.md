# Qwen3.8-27B Q4 Multi-GPU Runtime Evaluation Plan

Date: 2026-09-09

## Decision Scope

Choose the best non-production Qwen3.8-27B Q4_K_M serving profile before any
Q4 promotion, Muse move, Flash decommission, or gateway/OpenCode routing
change. The comparison has three candidates:

1. Existing llama.cpp GPU1+GPU2 layer split, Q8 K/V, Flash Attention, and one
   262,144-token slot.
2. llama.cpp GPU1+GPU2 tensor split under its supported requirements.
3. SGLang tensor-parallel serving using the exact existing Q4_K_M GGUF, if the
   exact model and two-GPU path pass a compatibility gate.

The current layer result remains the control: 18.65 tok/s decode, 939.67 tok/s
prefill, and 216.72 seconds for a 196k prompt with a 262,144-token allocation;
15.58 tok/s decode and 770.22 tok/s prefill at native 262k.

## 524k Shared Context Is the Right Strict Concurrency Test

Use a 524,288-token shared KV pool with two llama.cpp server slots. It is not a
single 524k request: llama.cpp shares `--ctx-size` across `--parallel` slots,
so the required confirmation is two `n_ctx_slot = 262144` slots. This gives two
independent native-262k contexts, exceeding the previous two-times-196k
guarantee rather than relaxing it.

There is no current measurement-based reason to rule it out. The one-GPU Q8
static allocation increased from 21,820 MiB at 196,608 context to 23,996 MiB at
262,144 context. That is 2,176 MiB for 65,536 tokens, or approximately 34 KiB
per token. Applying another 262,144 tokens to the successful two-GPU layer
startup allocation projects:

| Quantity | Observed or projected allocation |
|---|---:|
| GPU1+GPU2 layer startup at 262,144 | 24,426 MiB |
| Additional Q8 KV for another 262,144 tokens | 8,704 MiB |
| Projected 524,288 static allocation | 33,130 MiB |
| Total GPU1+GPU2 VRAM | 49,152 MiB |
| Projected static remaining VRAM | 16,022 MiB |
| Current 262k one-slot runtime overhead above startup | 4,252 MiB |
| Projected remaining VRAM if that overhead did not grow | 11,770 MiB |

This is a feasibility estimate, not a fit guarantee. Two slots add active
sequence, CUDA graph, scheduler, and compute-buffer pressure; allocation need
not remain linear. The 524k test must validate fresh-server startup, two
262,144-token slots in the log, maximum VRAM, and two populated concurrent
requests. Do not infer its result from aggregate free VRAM alone.

## Shared Test Rules

- Use GPU1+GPU2 only. The pair is `PHB` and CUDA P2P is unsupported; do not set
  `GGML_CUDA_P2P`.
- Keep the existing Q4_K_M GGUF and deterministic requests. Do not change model
  quantization while comparing runtimes.
- Record cold-start allocation, per-GPU VRAM/power/clock/core temperature, and
  memory-junction temperature and fan speed if the installed driver exposes
  them. GPU2 reached 87 C at 262k in a 28-30 C room, so thermal stability is a
  first-class metric.
- Record prompt tokens, completion tokens, prompt cache reuse, TTFT, prefill
  rate, per-request decode rate, aggregate decode rate, end-to-end latency,
  GPU energy, CPU energy, and errors/OOMs.
- Use fixed 128-token generation for direct profile comparisons. Use a separate
  realistic agent workload for scheduling and prefix-cache measurements.
- Repeat long points at least twice and reverse candidate order on the second
  pass to avoid attributing room temperature or warmed hardware effects to a
  runtime.
- Q4, Muse, and Flash are currently inactive. A direct user-level benchmark is
  viable while GPU1+GPU2 remain unused. `systemd-run` is optional at home, but
  remains preferable for detached logs, state restoration, and interruption
  safety. It needs sudo only when using the root wrapper.

## Phase 1: llama.cpp Tensor Split

Tensor split must be tested before promotion because it is the only plausible
llama.cpp change that could improve single-stream decode beyond the successful
layer profile. It is also high risk on this host:

- llama.cpp documents tensor split as experimental.
- It requires F16, BF16, or F32 K/V cache rather than Q8 K/V.
- The installed llama.cpp build requires Flash Attention to be enabled for this
  mode; this supersedes the older documentation snapshot used to draft the
  plan.
- It is more sensitive to inter-GPU interconnect bandwidth than layer split.
- This host has no CUDA P2P path, so tensor collectives may be PCIe/host-staged.

Use equal GPU1/GPU2 tensor distribution, F16 K/V as the initial compatible
choice, Flash Attention enabled, all layers offloaded, one slot, and 262,144 total
context. Hold threads, batch size, microbatch size, model, request shape, and
generation parameters at the established layer values.

Run a paired campaign under the same home ambient conditions:

1. Re-run the layer control at 4k, 196k, and 262k.
2. Run the tensor candidate at 4k, 196k, and 262k.
3. Repeat both profiles once in reverse order.

Treat tensor split as better only when it provides a repeatable material decode
gain at 196k and 262k without a material prefill regression, OOM, fallback,
quality regression, or worse sustained GPU2 thermals. A marginal short-context
decode improvement does not justify losing Q8 K/V for this long-context
workload.

If tensor loses, retain layer split as the llama.cpp baseline. Later small
tuning can then test a slightly GPU1-biased layer ratio to reduce GPU2 heat and
batch/microbatch sweeps for long-prefill throughput. Do not lower KV precision
without separate coding-quality validation.

## Phase 2: SGLang Compatibility Gate

SGLang is strategically useful for continuous batching and shared-prefix agent
traffic, not assumed to be faster for a single request. Current SGLang sources
include a Q4_K_M GGUF integration test and support tensor parallelism, radix
prefix caching, and FP8 KV cache on CUDA 11.8+ Ampere. They do not establish
that this exact Qwen3.8-27B GGUF is production-stable with TP=2 on this host.

The first SGLang experiment must therefore use the exact local Q4_K_M GGUF,
not an AWQ/GPTQ replacement. This isolates server behavior from weight
quantization and coding quality.

Gate sequence:

1. Verify the installed SGLang release, CUDA/PyTorch/NCCL compatibility, and
   exact-Qwen model architecture recognition.
2. Start the local GGUF with two-way tensor parallelism at a small context and
   run a deterministic completion.
3. Allocate 262,144 context and repeat the 4k, 196k, and 262k one-request
   points.
4. Confirm the model does not silently dequantize weights, CPU-fallback, cap
   context below the requested value, or consume unbounded static VRAM.

If exact-GGUF TP=2 fails, record the limitation and stop the runtime comparison.
Do not convert models merely to make SGLang run; an AWQ/GPTQ/FP8 comparison is a
separate model-and-quality workstream. If it passes, compare two SGLang KV
profiles separately:

- Default model KV behavior for the most conservative quality baseline.
- `fp8_e4m3` KV as the capacity/concurrency candidate, with long-context coding
  smoke tests before accepting any quality tradeoff.

Radix cache is enabled by default. Measure it in its own cold-versus-warm test;
never compare an SGLang warm-prefix result with a llama.cpp cold request.

## Phase 3: Strict Two-Slot Concurrency

Run this phase first on whichever one-slot profile wins Phase 1 and Phase 2.
For llama.cpp, use a 524,288-token shared pool and two slots, then confirm two
262,144-token slots from the startup log before workload traffic.

Run each scenario first without shared-prefix reuse, then repeat the dedicated
shared-prefix scenario:

| Scenario | Slot A prompt | Slot B prompt | Purpose |
|---|---:|---:|---|
| Strict maximum | 261,760 | 261,760 | Two native-262k slots with output headroom |
| Production equivalent | 196k | 196k | Two large coding agents |
| Main plus subagent | 196k | 64k | Expected asymmetric load |
| Shared-prefix agent tree | 160k shared prefix plus task | Same shared prefix plus task | Prefix reuse and scheduling |

Submit concurrent requests both simultaneously and with a short stagger. Report
aggregate decode throughput and the slower per-request stream, not aggregate
throughput alone. A result is acceptable only if both active streams make
steady progress, neither is starved, native slot capacity remains available,
and GPU2 stays thermally stable.

If 524k does not fit or has unacceptable latency, test 393,216 as the fallback
strict two-times-196k pool. A 327,680 or 393,216 realistic-pool test is useful
only when explicitly labelled best-effort; it cannot promise two native-262k
slots.

## Phase 4: Runtime Scheduling Comparison

If SGLang passes the exact-GGUF gate, run the same two-slot scenarios through
both the winning llama.cpp profile and SGLang. Separate these measurements:

1. Isolated request latency, no shared prefix.
2. Simultaneous independent requests, testing continuous batching.
3. Cold shared-prefix requests.
4. Warm shared-prefix requests, testing SGLang radix cache.

SGLang is the preferred production candidate only if its concurrency/prefix
benefit outweighs its operational complexity and any loss in one-agent
long-context speed, capacity, thermal headroom, or coding quality. Otherwise
keep llama.cpp layer split as the Q4 server and revisit SGLang when agent
concurrency is the primary bottleneck.

## Promotion Gate

No production service, routing, Muse placement, or Flash disposition changes
are part of this plan. Promotion requires a separate explicit approval after:

- Tensor split result is recorded.
- SGLang exact-GGUF gate result is recorded.
- The selected profile passes native-262k and strict two-slot capacity testing.
- GPU2 thermal monitoring is acceptable under home ambient conditions.
- Multi-GPU metrics and energy-accounting updates are ready for the selected
  production topology.

## Sources

- llama.cpp multi-GPU guidance: `docs/multi-gpu.md` in the current llama.cpp
  repository. Tensor split is experimental, requires non-quantized K/V, and is
  interconnect-sensitive. The installed build's startup validation is the
  source of truth for its Flash Attention requirement.
- llama.cpp server context implementation and argument definitions: current
  `tools/server/server-context.cpp` and `common/arg.cpp`. `--ctx-size` is the
  shared pool; `--parallel` creates slots and context is assigned per slot.
- SGLang current project sources: Q4_K_M GGUF integration test,
  `server_args.py`, and project README. These support the compatibility-gate
  rationale but do not replace testing the exact Qwen3.8 GGUF with TP=2.
