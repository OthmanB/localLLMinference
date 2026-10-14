# Qwen3.8-27B Tensor 262k Optimization Plan

Date: 2026-09-09

## Objective

Optimize the balance of prompt ingestion and long-generation decode around the
proven llama.cpp tensor profile:

```text
Qwen3.8-27B Q4_K_M
GPU1 + GPU2
--split-mode tensor --tensor-split 1,1
--ctx-size 262144 --parallel 1
F16 K/V, Flash Attention on
NCCL-enabled llama.cpp v0.4.0-dev 4d91760
```

The current physical result is approximately 710-816 prompt tok/s and 33-36.5
decode tok/s at long context. The campaign must optimize end-to-end coding-agent
latency, not decode or prefill in isolation.

No production service, routing, model placement, or deletion is part of this
plan.

## Full-Precision Feasibility

A two-GPU BF16/F16 test is rejected before execution. A 27B model requires
about 54 GB decimal (50.3 GiB) for two-byte weights alone. GPU1 and GPU2 provide
48 GiB aggregate physical VRAM. The weights therefore exceed aggregate VRAM
before F16 K/V, CUDA/NCCL workspaces, graph buffers, and model state are added.

Tensor parallelism distributes memory; it does not reduce aggregate model
memory. CPU offload would change the experiment into a PCIe/host-memory profile
and is not expected to preserve the Q4 tensor decode result.

Three 24 GiB GPUs may be barely large enough in aggregate for BF16 plus native
context, but that is a separate, low-confidence experiment. It consumes GPU0,
has little estimated headroom, adds a third participant to collectives over the
no-P2P topology, and may encounter tensor-partition constraints in Qwen3.8's
hybrid head dimensions. Do not download or run BF16 unless a separate
three-GPU capacity calculation and explicit approval are completed first.

## Phase 1: Precision Ladder

Use the existing local Q5_K_M GGUF first. Preserve the exact Q4 tensor settings
and binary so model precision is the only changed variable.

Execution order: start the Q5 tensor campaign only after the permanent GPU2 85%
fan unit is applied and verified. Download Q6_K and Q8_0 in parallel with Q5
GPU work because they do not consume GPU VRAM. The selected publisher artifacts
are `Qwen3.8-27B-UD-Q6_K.gguf` (22.0 GB) and
`Qwen3.8-27B-Q8_0.gguf` (29.0 GB). Q8_0 is not assumed to fit; its download
completed first, so it was screened before Q6 at the user's direction.

1. Q4_K_M control.
2. Q5_K_M capacity smoke and benchmark.
3. Stage Q6_K only if Q5 fits with at least 2 GiB free on each GPU during a
   populated near-native request and provides a measurable quality benefit.
4. Consider Q8_0 only after a static allocation estimate. It is expected to be
   close to the two-card VRAM boundary and is lower priority than Q5/Q6.

### Q5 Execution Status

GPU2's permanent 85% fan unit was applied and verified before Q5 execution.
Q5_K_M tensor passed the native-262k capacity smoke at 18,052 MiB peak on each
GPU, leaving about 6.4 GiB per card. Its 196k plus 4,096-output point measured
827.04 prompt tok/s, 35.08 decode tok/s, and 355.15 seconds total; the 257k
plus 4,096-output point measured 729.84 prompt tok/s, 31.86 decode tok/s, and
481.50 seconds. GPU2 peaked at 72 C and 77 C respectively, with a 1515 MHz
minimum graphics clock at the near-native point.

Against Q4's 196k plus 4,096-output tensor control, Q5 gains 1.3% prefill but
loses 4.0% decode and is 0.4% slower end-to-end. Retain Q5 for coding-quality
evaluation; do not advance to a Q6 runtime test merely on performance. Existing
corrected HumanEval+ evidence has Q4 1.9 percentage points above Q5, while the
comparable SWE subset resolved 22 of 40 tasks for both. Q5 therefore does not
meet the planned quality-benefit gate for a Q6 runtime campaign. Raw artifacts
are under:

- `research/qwen3.8-27b-q5-tensor-262k-smoke-20260909T115827Z/results.json`
- `research/qwen3.8-27b-q5-tensor-262k-long-20260909T120139Z/results.json`

### Q8 Execution Status

Q8_0 was downloaded and run before Q6 because its download completed first. It
passes native 262k allocation but peaks at 22,122 MiB per GPU, leaving only
about 2.4 GiB per card. At 196k plus 4,096 output it measured 848.45 prompt
tok/s, 30.56 decode tok/s, and 366.39 seconds; at 257k plus 4,096 output it
measured 743.15 prompt tok/s, 28.12 decode tok/s, and 492.28 seconds. It is
slower than both Q4 and Q5 at the coding-agent workload and does not justify
its reduced headroom. GPU2 remained at 72 C and 77 C with the permanent 85%
fan setting.

Q6_K is downloaded at `models/qwen3.8-27b-q6-k-gguf`, but is not run: Q5 did
not satisfy the explicit quality-benefit gate, and Q8 already demonstrates the
adverse direction of the higher-precision latency/headroom trade-off. Revisit
Q6 only after new coding-quality evidence identifies a precision-sensitive
failure in Q4.

Q8 artifacts are under:

- `research/qwen3.8-27b-q8-0-tensor-262k-smoke-20260909T132957Z/results.json`
- `research/qwen3.8-27b-q8-0-tensor-262k-long-20260909T133337Z/results.json`

For each admitted precision, run:

| Prompt | Output | Purpose |
|---:|---:|---|
| 4k | 1,024 | Short-context decode and startup control |
| 196k | 4,096 | Established coding-agent crossover workload |
| 257k | 4,096 | Native-context workload with output headroom |

Record prompt rate, decode rate, wall time, peak/idle VRAM per GPU, energy,
temperature, and graphics-clock range. Stop a candidate on OOM, UVM/fallback,
less than 1 GiB transient headroom, numerical errors, or unstable clocks.

Run the existing coding-quality suite on Q4 and Q5 before preferring the higher
precision. Higher precision is useful only if its quality gain justifies its
latency and memory cost.

## Phase 2: Batch and Microbatch Sweep

Tune Q4 first, then confirm only the winning setting on the selected higher
precision. Keep the 262,144-token allocation and all other controls fixed.

Screen these configurations at 64k input plus 1,024 output:

| Batch | Microbatch |
|---:|---:|
| 2048 | 256 |
| 2048 | 512 (control) |
| 2048 | 1024 |
| 4096 | 512 |
| 4096 | 1024 |

Promote only the Pareto candidates to 196k plus 4,096 output. Larger batches
primarily target prefill and may consume more workspace; decode must not regress
materially. Stop configurations that add memory pressure without a repeatable
prefill gain.

### Q4 Batch Sweep Result

The 64k plus 1,024-output Q4 tensor screen completed with the same 262,144
context allocation, F16 K/V, Flash Attention, and 300 W limits. `2048/1024`
is the selected control: it has the highest prefill and decode rates, the
lowest request energy, and only 386 MiB more peak GPU memory than the prior
`2048/512` control.

| Batch / microbatch | Prefill tok/s | Decode tok/s | Request s | Peak MiB/GPU | Total kJ |
|---|---:|---:|---:|---:|---:|
| 2048 / 256 | 1081.91 | 45.94 | 83.09 | 16282 | 55.77 |
| 2048 / 512 | 1116.90 | 45.82 | 81.24 | 16476 | 54.55 |
| 2048 / 1024 | 1147.57 | 46.15 | 79.52 | 16862 | 53.72 |
| 4096 / 512 | 1102.12 | 45.70 | 82.11 | 16476 | 55.53 |
| 4096 / 1024 | 1139.42 | 46.10 | 79.97 | 16862 | 53.76 |

The promoted control passed both long points. At 196k plus 4,096 output it
measured 856.45 prompt tok/s, 36.95 decode tok/s, 341.07 seconds, 16,862 MiB
peak per GPU, and 229.99 kJ. This improves on the prior `2048/512` result's
816.06 prompt tok/s, 36.52 decode tok/s, 353.72 seconds, and 243.02 kJ. At
257k plus 4,096 output it measured 763.97 prompt tok/s, 33.49 decode tok/s,
459.53 seconds, 16,862 MiB peak per GPU, and 308.56 kJ. GPU2 remained at 74 C
and 76 C respectively with its permanent 85% fan setting; no clock instability
occurred.

Raw artifacts are under:

- `research/qwen3.8-27b-q4-tensor-batch-sweep-20260909T140914Z/`
- `research/qwen3.8-27b-q4-tensor-batch2048-ubatch1024-long-20260909T142018Z/results.json`
- `research/qwen3.8-27b-q4-tensor-batch2048-ubatch1024-native-20260909T143100Z/results.json`

The earlier `...batch-sweep-20260909T135807Z` directory is invalid and excluded:
systemd expanded shell variables in its runner, overwriting candidate artifacts.

## Phase 3: F16 Versus BF16 K/V

Tensor mode requires a non-quantized K/V cache in the tested build. Compare
F16/F16 (control) with BF16/BF16 using the winning batch configuration. They
have the same nominal cache size, so this is a kernel speed and numerical
behavior test, not a capacity strategy. Do not test F32 at native context.

The installed binary's observed behavior is authoritative: Flash Attention on
is the known-good tensor configuration. Do not disable it merely because
upstream experimental documentation differs from this build's behavior.

### K/V Cache Result

BF16/BF16 was supported and passed the 196k plus 4,096-output request, but is
rejected. Against the `2048/1024` F16/F16 control, it produced essentially the
same prefill rate (852.54 versus 856.45 tok/s) but reduced decode from 36.95 to
20.49 tok/s. It increased peak memory from 16,862 to 17,778 MiB per GPU, total
time from 341.07 to 431.15 seconds, and request energy from 229.99 to 293.07
kJ. Keep F16/F16 for tensor mode and do not run BF16 at the near-native point.

Raw artifact:

- `research/qwen3.8-27b-q4-tensor-bf16kv-196k-20260909T144329Z/results.json`

## Phase 4: Communication and Hardware Controls

The two RTX 3090s communicate across PHB without CUDA P2P. First capture the
actual NCCL topology, selected transport, algorithm, protocol, and channel
count from a verbose control. Then test only one variable at a time against
NCCL defaults. Do not enable `GGML_CUDA_P2P`.

Candidate A/Bs, only when supported by the installed NCCL version:

- Default versus explicit two-GPU Ring.
- Default protocol versus Simple and LL128.
- Default channel count versus one conservative alternate derived from the
  control log.

Reject any setting that helps 4k decode but hurts 196k prefill/decode balance,
causes NCCL fallback, or adds run-to-run variance.

The current tensor run reaches the 300 W limit while maintaining useful clocks.
GPU2 is permanently pinned to an 85% fan speed at boot for sustained dense-model
thermal headroom; do not alter that fan target during this campaign. An optional
final power sweep may compare 300 W with one higher approved limit, restoring
only the original power limit after every run. Judge this by latency and joules
per completed request. Temperature alone is not a failure unless accompanied by
clock loss, instability, or the configured hardware limit.

### NCCL Control And Protocol Result

The fresh NCCL 2.30.4 control confirms a two-GPU PHB topology with CUDA P2P
disabled. NCCL uses `SHM/direct/direct` for local collective transport, two
rings, and two collective channels. The tuning trace selects `RING` plus `LL`
for 40,960-byte AllReduce operations across channels 0 through 1. Do not enable
`GGML_CUDA_P2P` or force Ring: both would contradict the detected topology or
repeat the default selection.

At 64k plus 1,024 output, forcing protocol alternatives is worse than the
default adaptive selection:

| NCCL protocol | Prefill tok/s | Decode tok/s | Request s | Total kJ |
|---|---:|---:|---:|---:|
| Default | 1163.37 | 46.39 | 78.63 | 52.59 |
| Simple | 1156.40 | 42.29 | 81.11 | 54.59 |
| LL128 | 948.85 | 42.29 | 93.51 | 62.81 |

Keep the default protocol. Raw artifacts are under:

- `research/qwen3.8-27b-q4-tensor-nccl-control-20260909T145432Z/`
- `research/qwen3.8-27b-q4-tensor-nccl-tuning-20260909T150430Z/`
- `research/qwen3.8-27b-q4-tensor-nccl-protocol-sweep-20260909T150811Z/`

The `NCCL_MAX_CTAS=1` control reduced the communicator from two collective
channels to one, as verified by NCCL INIT logging. It is also rejected: 64k
prefill fell from 1163.37 to 1122.44 tok/s, request time rose from 78.63 to
80.78 seconds, and energy rose from 52.59 to 53.96 kJ. Keep NCCL defaults and
two collective channels.

Raw artifact:

- `research/qwen3.8-27b-q4-tensor-nccl-maxctas1-20260909T151718Z/`

## Phase 5: Agent-Relevant Cache Behavior

Cold synthetic prompts overstate repeated multi-turn agent prefill cost. Test a
realistic warm conversation with a 160-196k retained prefix and successive
small tool-result additions. Measure:

- Cold first-turn TTFT and total time.
- Warm retained-prefix TTFT.
- Tokens actually re-evaluated.
- Decode rate over a 2k-4k response.
- Slot-cache correctness after tool-call round trips.

This phase may improve effective prefill/decode balance more than low-level
kernel tuning and should be included in the final workload recommendation.

### Warm Retained-Prefix Result

The selected Q4 tensor control was measured with a synthetic 160k retained
prefix, a 512-token first tool result, then a second request adding another
512-token tool result. The cold request evaluated 160,522 prompt tokens in
173.478 seconds and took 226.82 seconds total. The second 161,040-token request
reused 159,494 tokens, evaluated only 1,546 tokens in 2.884 seconds, and took
55.42 seconds total. Decode remained stable at 39.38 tok/s versus the cold
request's 38.76 tok/s; energy fell from 152.32 to 37.05 kJ. GPU2 peaked at 70 C
and peak VRAM remained 16,864 MiB per GPU.

This validates llama.cpp prompt-cache reuse mechanics for a retained prefix and
appended tool-result payload. It is not a semantic coding-agent or tool-call
correctness evaluation, so do not treat it as evidence of agent-task quality.

Raw artifact:

- `research/qwen3.8-27b-q4-tensor-warm-cache-20260909T153433Z/results.json`

## Deferred Experiments

- Native MTP/speculative decoding is low priority. Earlier populated-192k MTP
  tests collapsed decode despite acceptable draft rate. Admit one tensor smoke
  only after the precision and batch sweeps, and stop immediately if long-context
  decode falls below the non-MTP control.
- N-gram speculation may help repetitive code generation but must be evaluated
  on real coding outputs, not deterministic repeated-context probes.
- BF16/F16 model weights on three GPUs require a separate capacity/topology
  plan and do not inherit Q4 performance expectations.

## Decision Method

For each candidate, compute measured or interpolated end-to-end latency for
2k, 4k, and 8k generated-token workloads:

```text
total_time = measured_prefill_time + output_tokens / measured_decode_rate
```

Maintain a Pareto table for total latency, quality, peak VRAM, and energy. The
default coding profile should favor the lowest validated 4k-8k output latency;
retain layer/Q8-KV as the separate maximum-capacity and ingestion profile.

Publish raw artifacts and an addendum before any production change. Restore all
temporary power/fan settings and verify all pre-existing services and routing
remain unchanged after every campaign stage.
