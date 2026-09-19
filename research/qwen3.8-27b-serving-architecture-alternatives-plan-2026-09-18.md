# Qwen3.8-27B Serving Architecture Alternatives Plan

Date: 2026-09-18

## Objective

Evaluate three architecture-changing alternatives to the validated Qwen3.8-27B
llama.cpp deployment. The goal is not incremental llama.cpp tuning. The goal is
to maximize:

```text
quality * useful concurrency * long-context capability / GPU resources
```

The candidates are evaluated in this order:

1. llamAmpere v0.3 with Qwen3.8-27B ATX IQ4_XS-M
2. PrismML Bonsai 2 27B
3. vLLM and/or SGLang Qwen3.8-27B with DFlash2

After each phase, write a short report and stop that branch if it clearly fails
the quality, correctness, context, or deployment-value gate.

No production routing, systemd service, model placement, or reference runtime
change is part of this plan. Promotion requires separate explicit approval.

## Immutable Reference

The current production profile is the control for every comparison and must
remain available and unchanged:

| Property | Reference |
|---|---|
| Model | Qwen3.8-27B UD Q4_K_M |
| Runtime | llama.cpp |
| GPUs | Physical GPU1 + GPU2 |
| Distribution | Tensor split `1,1` |
| Context | `262144` |
| KV cache | F16/F16 |
| Attention | Flash Attention enabled |
| Batch | `2048` |
| Micro-batch | `1024` |
| NCCL | Defaults, no overrides |
| Observed decode | Approximately 40-50 tok/s, depending on context depth |
| Observed prefill | Approximately 750-1100 tok/s |

Reference files and services:

- Service: `llama-qwen3.8-q4-tensor-262k.service`
- Reference unit: `operations/systemd/llama-qwen3.8-q4-tensor-262k.service`
- Reference runtime history: `research/qwen3.8-27b-q4-runtime-follow-up-results-2026-09-09.md`
- Reference deployment record: `operations/deployment-record.md`

Do not replace, edit, disable, or repoint the reference service during a
candidate installation. Candidate tests must stop the reference only inside a
controlled benchmark wrapper, record its prior state, and restore it in an
`EXIT` trap. Candidate runtimes must use isolated directories, ports, model
paths, environments, and output directories.

## Comparability Rules

Headline throughput is not a deployment result. A candidate must be compared
at the same or greater useful context depth before it can disqualify the
reference.

- A result at 64k does not disqualify a 262k reference result.
- A result at 128k or 196k is reported separately from a 262k result.
- A candidate that is faster only because it uses less context, a smaller
  target model, a different quality level, or a different workload is not a
  direct replacement.
- Vendor, Reddit, and social-media benchmark claims are hypotheses only.
- Synthetic code throughput is useful for reproducibility but cannot establish
  coding-agent quality or tool correctness.
- Do not compare warm prompt-cache throughput against a cold reference.
- Record both per-request latency and aggregate throughput for concurrent tests.
- Record failed requests, OOMs, context truncation, CPU fallback, silent
  quantization changes, and scheduler starvation as first-class results.

Every run must record:

- Exact runtime repository and git commit.
- Exact model repository, filename, quantization, file size, and SHA-256 hash.
- Full command line and environment variables relevant to CUDA, NCCL, and the
  runtime.
- GPU assignment and physical GPU identifiers.
- Context allocation and actual prompt/completion token counts.
- Raw server logs, benchmark output, request payload metadata, and errors.
- Per-GPU VRAM, power, temperature, clocks, and GPU energy per request.
- CPU utilization, CPU power, host memory, and swap state where available.
- Cold and warm prompt-cache behavior.

## Common Workloads

Use two workload families for every candidate that reaches quality testing.

### Synthetic Coding Workload

Use a deterministic code-shaped prompt with fixed seed, sampling parameters,
prompt lengths, and forced output lengths. Measure at the common context points
where the candidate fits. Repeat long-context points at least twice and reverse
candidate order on the second pass.

Report prefill tok/s, decode tok/s, wall time, TTFT, prompt-cache state, VRAM,
power, temperature, and energy/request.

### Real OpenCode Workload

Use a small fixed corpus of difficult tasks representative of the actual
agent. The initial corpus should contain approximately 8-10 tasks for the
llamAmpere quality gate and approximately 10-15 tasks for Bonsai and any
runtime that reaches the broader quality gate.

Cover:

- Multi-file implementation.
- Code review and bug diagnosis.
- Long-context repository reasoning.
- Planning followed by implementation.
- Structured tool calls and JSON arguments.
- MCP routing through Semble, Context7, and Repomix.
- Long-horizon task continuation and recovery from tool errors.

Use the same task text, repository snapshot, tool permissions, and evaluation
criteria for the reference and candidate. Record task success, regressions,
reasoning coherence, tool-call validity, MCP selection, and final patch
quality. No large SWE campaign is required for the first quality gates.

## Phase 0: Harness and Reference Capture

Before downloading or building a candidate:

1. Capture the reference service unit, active command line, runtime commit,
   model hash, GPU assignment, and current health endpoints.
2. Capture a fresh reference result for the synthetic workload at the relevant
   context depths if an exact same-shape result is not already available.
3. Verify that candidate directories, ports, and logs cannot overlap production
   paths or listeners.
4. Create a per-phase artifact directory such as:

   ```text
   research/qwen3.8-27b-serving-alternatives-<phase>-<timestamp>/
   ```

5. Keep the production gateway and systemd routing unchanged. Candidate API
   calls must use an isolated loopback port or an explicit candidate endpoint.

The reference capture is a control, not a promotion step.

## Phase 1: llamAmpere v0.3

### Candidate

- Runtime repository: `JakeATX/llamAmpere`
- Runtime: llamAmpere v0.3
- Model repository: `jakeatx/Qwen3.8-27B-ATX-IQ4_XS-M-GGUF`
- Target hardware: one RTX 3090 first

Resolve the documented release commit from the official llamAmpere release
documentation and record it before building. Build in an isolated directory;
do not replace `/home/obenomar/.local/share/llama.cpp-recent-20260905` or the
production binary.

First reproduce the author's recommended single-3090 command exactly, except
for isolated host/port/output paths. Record any required CUDA, compiler, or
environment settings.

### Single-GPU Matrix

Run one physical RTX 3090 at:

- 64k context.
- 128k context.
- 196k context.
- Near maximum supported context, approximately 240k-245k.

At each depth measure:

- Prefill tok/s.
- Decode tok/s.
- TTFT and end-to-end wall time.
- VRAM allocation and peak VRAM.
- GPU power, temperature, clocks, and energy/request.
- Host memory and CPU utilization.
- Prompt-cache cold and warm reuse.
- MTP acceptance statistics if exposed.
- Context actually accepted by the server.

Use both the deterministic synthetic coding workload and real OpenCode tasks.

### ATX Quality Gate

Compare ATX IQ4_XS-M directly against the current UD Q4_K_M reference using
the same approximately 8-10 difficult tasks. Examine:

- Task success and regressions.
- Code correctness and reasoning coherence.
- Tool-call formatting and argument validity.
- MCP behavior through Semble, Context7, and Repomix.
- Long-context retrieval and continuation.

Do not assume equivalent quality from similar model size or speed. A candidate
fails this gate if it has a material quality, tool-use, or long-context
regression even when its decode rate is higher.

### Two-Instance Architecture Test

Run this only if the single-GPU candidate passes the quality gate and shows a
useful long-context speed result. The target is two independent processes:

- GPU1: independent Qwen3.8-27B llamAmpere server.
- GPU2: independent Qwen3.8-27B llamAmpere server.

Do not use llama.cpp `--parallel 2`. Use separate processes, ports, and server
instances. Submit independent simultaneous requests and measure:

- Per-instance decode and prefill tok/s.
- Aggregate tok/s.
- Prefill interference.
- TTFT and wall-time changes.
- PCIe and CPU contention.
- Per-GPU VRAM, power, temperature, and energy/request.
- Prompt-cache behavior per instance.

### llamAmpere Decision Gate

The original gate is superseded by the explicit supervised production decision.
The following remain the relevant readiness checks:

- The bounded objective quality suite shows no demonstrated regression against
  the UD Q4_K_M reference; this is not a subjective or full OpenCode quality
  certification.
- Tool and MCP behavior is correct in the normal-EOS readiness probes.
- Useful long-context operation is stable at the measured approximately 61-65
  tok/s decode range. The former 70-80 tok/s figure was aspirational, not a
  hard acceptance threshold.
- The candidate retains a useful context depth near 240k for the claimed
  comparison.
- Two independent instances remain close to single-instance performance when
  run concurrently.

If the candidate fails quality, correctness, context, or operational readiness,
stop the cutover and retain llama.cpp as fallback. Do not reject llamAmpere
solely because it does not reach the former aspirational speed figure.

## Phase 2: PrismML Bonsai 2 27B

### Candidate

Use the new Bonsai 2 27B released September 17, 2026. Do not substitute the
older Qwen3.6-derived Bonsai 27B.

- Model: `prism-ml/Ternary-Bonsai-2-27B-gguf`
- Runtime: PrismML's current llama.cpp fork and Bonsai-demo tooling
- Packings: `PQ2_0` and `PTQ1_0` when both are available

Do not run Bonsai through llamAmpere unless explicit compatibility is documented
by PrismML. Keep the PrismML runtime, fork, build, model files, and environment
isolated from the reference and llamAmpere trees.

### Capacity and Speed Matrix

For each packing that can load, determine single-3090 behavior at:

- 8k context.
- 64k context.
- 128k context.
- 196k context.
- 262k context if physically supported.

Record the common metrics: prefill, decode, TTFT, wall time, VRAM, power,
temperature, clocks, energy/request, CPU/host memory, prompt-cache reuse,
errors, and actual maximum context.

Do not infer 262k support from metadata alone. Require a real populated
long-context request with no fallback or truncation.

### Bonsai Quality and Role Gate

Run approximately 10-15 real tasks against the current Qwen reference. Cover:

- Coding implementation.
- Code review.
- Planning.
- Tool calling.
- Long-horizon agent behavior.
- MCP routing through Semble, Context7, and Repomix.
- Reasoning under long context.

The primary question is substitution quality, not headline tok/s. Record which
tasks Bonsai solves, where it regresses, and whether the smaller weight
footprint changes context capacity or tool concurrency.

If Bonsai is close enough in quality, evaluate these isolated topologies:

- GPU1: Qwen full-strength main agent; GPU2: Bonsai reviewer/subagent.
- GPU1: Bonsai session A; GPU2: Bonsai session B.
- Multiple independent Bonsai servers or contexts on one 3090 if KV memory
  permits.

For multi-instance tests, report per-instance progress and aggregate throughput
and confirm that no stream is starved by prefill or KV allocation.

### Bonsai Decision Gate

Bonsai is valuable even if slightly weaker than Qwen when it supports two or
more useful independent agents per GPU without sacrificing long-context
correctness. It is not a Qwen replacement if coding, reasoning, tool calling,
or MCP behavior has material regressions at the intended role.

Stop the Bonsai branch if neither packing passes the basic load/context gate or
if quality is clearly unsuitable for any reviewer/subagent role.

## Phase 3: vLLM/SGLang DFlash2

Do not spend more time fixing llama.cpp DFlash2 multi-GPU placement for this
branch. The tested llama.cpp builds demonstrated a target/draft graph and
device-placement limitation.

### Runtime Order

1. Reproduce a known two-RTX-3090 recipe with the fewest modifications.
2. Preserve 262k context if possible.
3. Only after correctness and context are established, tune performance.

Known high-throughput recipes may use AutoRound INT4 or another target
quantization rather than the reference UD Q4_K_M GGUF. This is a different
model-and-runtime comparison, not a pure runtime comparison. Keep the current
llama.cpp Q4 profile as the quality control.

### SGLang DFlash2 Gate

Use the BF16/unquantized `z-lab` Qwen3.8-27B DFlash2 drafter initially. Do not
use a quantized DFlash2 drafter until the known silent-acceptance-collapse issue
is confirmed fixed.

Start with thinking disabled and compare, under the same target and workload:

- Target-only baseline without speculation.
- Built-in MTP.
- DFlash2.

Record:

- Prefill tok/s.
- Decode tok/s.
- End-to-end wall tok/s.
- Speculative acceptance rate and accepted length where available.
- VRAM and maximum usable context.
- GPU power, temperature, clocks, and energy/request.
- Code-shaped versus narrative/reasoning-shaped output behavior.

Then test thinking mode and tool calling. At temperature 0, compare target-only
and DFlash2 deterministic prompts for output identity or an explicitly
documented equivalence criterion. Test reasoning traces, structured tool calls,
and MCP behavior. If thinking-mode DFlash2 diverges incorrectly, silently
collapses acceptance, or corrupts tool calls, it fails the hard correctness
gate regardless of throughput.

### DFlash2 Context Matrix

Only after the correctness gate passes, test real OpenCode workloads near:

- 64k.
- 128k.
- 196k.
- 262k.

Compare cold and warm prompt-cache behavior separately. Do not credit a speed
gain that comes from a smaller context than the reference.

### DFlash2 Decision Gate

Do not consider replacing the production Qwen profile unless all of these hold:

- Target quality remains comparable.
- Thinking and tool calling are correct.
- Long context is retained at the claimed comparison point.
- Real coding/agent workloads show a large speed increase.
- Practical real-workload throughput is approximately 80-100 tok/s or better
  at 196k or greater context.

A 200+ tok/s synthetic code result is interesting but is not a deployment
criterion by itself.

## Cross-Branch Decision Matrix

Populate this table only with validated measurements. Use `not tested`, `does
not fit`, or `failed correctness gate` rather than filling gaps with vendor
claims or extrapolation.

| Candidate | Model/quant | Runtime | GPUs | Maximum validated context | Prefill tok/s at useful long context | Decode tok/s at useful long context | Aggregate concurrent throughput | VRAM/GPU | Quality result | Tool/MCP behavior | Prompt-cache behavior | Known correctness issues | Energy/request | Recommended role |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|
| Current reference | UD Q4_K_M | llama.cpp | 2 | 262k | 857.60 at 196k; 750.42 at 262k | 35.28 at 196k; 32.99 at 262k | Not tested in this campaign | 16,864 MiB observed per GPU | Validated control | Validated control | Validated cold/warm behavior | None known for reference profile | See Phase 0 artifact | Fallback/reference |
| llamAmpere single | ATX IQ4_XS-M | llamAmpere v0.3 | 1 | 240k populated | 502.92 at 240k; 572.94 at 196k | 61.03 at 240k; 65.50 at 196k | N/A | 22,830 MiB | No demonstrated objective regression; subjective quality not scored | Objective tool/MCP tasks passed | Cold and warm behavior recorded | 70-80 tok/s was aspirational, not an acceptance gate | 38.023 Wh at 240k artifact point | Production single-agent baseline |
| llamAmpere optimized | ATX IQ4_XS-M | llamAmpere v0.3, MTP3 b6144/ub1024 | 1 | 240k populated | 503.75 at 240k; 571.58 at 196k | 59.03 at 240k; 64.54 at 196k | N/A | 22,798 MiB | Same bounded objective evidence; sweep probes are not quality evidence | Same as single | Cold and warm behavior recorded | Only 0.2-0.5% better than b4096 in controlled cells | See Axis 1 artifacts | Canary optimization, not required for promotion |
| llamAmpere dual | ATX IQ4_XS-M | llamAmpere v0.3 | 2 | 196k concurrent | Approximately 578/568 per-instance cold at 196k; 984/972 at 64k | Approximately 65.2 per-instance cold at 196k | Approximately 130.4 aggregate decode; independent contexts | Approximately 22,814 MiB per instance | Same objective evidence as single | Same objective evidence as single | Cold/warm behavior recorded per instance | Not a 262k dual replacement result | See dual artifact | Production concurrency topology |
| llamAmpere TP=2 | ATX IQ4_XS-M | llamAmpere v0.3 tensor/NCCL | 2 | 239,963 populated | 738.83 at 240k; 826.77 at 196k | 39.56 at 240k; 42.26 at 196k | One request only | 11,986 MiB per GPU | Non-MTP probes passed; no TP2-specific quality suite | MTP3 aborts after NCCL initialization | Cold/warm behavior recorded | Lower decode than single MTP3; MTP3 tensor draft graph failure | 50.231 Wh combined at 240k | Not production; later latency experiment only |
| Bonsai 2 single | PQ2_0 or PTQ1_0 | PrismML fork/demo | 1 | 262k populated | 479.05 PQ2_0; 364.25 PTQ1_0 | 28.89 PQ2_0; 26.38 PTQ1_0 | N/A | 23,810 PQ2_0; 22,666 PTQ1_0 MiB | Both packings fail T8 code-output contract | T1-T7 and T9-T10 objective tasks pass; T8 fails | Cold/warm behavior recorded | Truncation and malformed JSON at 2,048 tokens | See Phase 2 artifact | No substitution; conditional reviewer/subagent only |
| Bonsai 2 multiple | PQ2_0 | PrismML fork/demo | 1-2 | 196k independent dual; 8k shared | 1,147.74 aggregate independent-dual prefill at 196k | 29.17 aggregate independent-dual decode at 196k | Independent dual and shared `parallel=2` validated; overlap efficiency 1.986 at cold 196k | See single-packing values | T8 regression remains | Topology works; role quality gate fails | Cold/warm and shared-context evidence recorded | Coding-output contract regression | See topology artifact | Conditional reviewer/subagent topology only |
| vLLM/SGLang DFlash2 | AWQ target and BF16 DFlash2 drafter recorded per run | vLLM or SGLang | 2 | 8k only | Not tested at useful long context | SGLang DFlash code probe 219.3; vLLM target-only 67.39 at 8k | Not tested; vLLM DFlash not started | SGLang 23,313/23,008; vLLM 20,632/20,632 MiB | Failed correctness/feasibility gate | Thinking and structured tool behavior fail | Bounded probes only | Malformed thinking; invalid or unconfigured tool calls; unstable acceptance | Not comparable; sampled values only | Stop branch; no promotion |

Final normalized results and recommendations: `research/qwen3.8-27b-serving-alternatives-final-report.md`.

## Supervised Axis Outcome

- llamAmpere v0.3 with Qwen3.8-27B ATX IQ4_XS-M is accepted for the production
  architecture after readiness and canary validation. The 70-80 tok/s figure
  was aspirational and is not a rejection gate.
- Initial production baseline: one GPU, MTP3, batch/ubatch 4096/1024,
  context 245760, GPU1 only. Batch 6144 is a measured canary optimization with
  only a marginal 0.2-0.5% improvement.
- Two independent llamAmpere instances on GPUs 1 and 2 are the preferred
  concurrent-agent topology. This is independent replication, not TP=2.
- llamAmpere TP=2 passed non-MTP long-context probes but failed MTP3 during
  tensor draft-graph setup and is not a production profile.
- Keep the existing two-GPU stock llama.cpp service startable as the fallback
  and reference until the llamAmpere cutover is validated. No cutover was
  performed by this measurement phase.
- No llamAmpere systemd unit, production gateway route, exporter configuration,
  or automatic rollback exists yet. The validated port `18101` is isolated
  research-only. Creating the production unit and choosing the client-facing
  alias/port requires explicit operational approval.
- Axis artifacts:
  - `research/qwen3.8-27b-llamampere-axis1-sweep-20260918T224901Z`
  - `research/qwen3.8-27b-llamampere-axis1-batch6144-20260919T004351Z`
  - `research/qwen3.8-27b-llamampere-tp2-20260918T235826Z`
  - `research/qwen3.8-27b-serving-alternatives-phase1-dual-20260918T144111Z`

The final recommendation must state the role, context, quality evidence, and
concurrency evidence for each surviving candidate. Do not select a candidate
from decode tok/s alone.

## Stop and Promotion Rules

Stop a branch immediately when it fails any hard gate:

- Cannot load the documented model/runtime combination.
- Cannot sustain the claimed context without truncation, fallback, or OOM.
- Has material quality or tool/MCP regressions against the reference.
- Has deterministic thinking-mode divergence or malformed tool calls.
- Requires changing the production service to run the test.
- Produces a speed result that is not comparable in context depth or workload.

After each phase, write a report containing the command, commit, hashes, raw
artifacts, metrics, quality observations, and a go/no-go decision. Do not
automatically promote a runtime, modify production routing, change systemd
units, move GPUs, or decommission the current Qwen service.

The explicit supervised decision is to use llamAmpere after its relevant
quality, correctness, context, operational, and deployment-value readiness
gates pass. The stock llama.cpp deployment remains available as fallback until
the canary is validated and must not be decommissioned as part of this phase.
