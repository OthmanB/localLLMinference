# Qwen3.8-27B RTX 5090 Three-Session Decode Optimization Plan

Date: 2026-09-20
Status: historical Phase F plan. Phase G has now established CUDA-graph B0 at
C=1 and C=3, then rejected graph MTP1 at C=1. No candidate is authorized for
routing or production use.

## Objective

Optimize the calibrated ModelOpt NVFP4/FP8-KV vLLM TP2 candidate for three
independent long-context users. The primary workload is three concurrent
sessions with 196k to near-native prompts, while preserving the best possible
per-user decode rate and bounded p95 inter-token latency (ITL). Aggregate
throughput, prefill, and four-session capacity remain secondary measurements.

The stable llama.cpp Q4 service remains the reference and the only live model.
This plan does not install a vLLM unit, modify gateway routing, change a driver,
change a GPU power limit, or stop an unrelated GPU workload.

## Phase G Supersession

The isolated nightly graph profile supersedes the pinned eager stack as the
current performance baseline, while leaving the pinned results as historical
controls. With the calibrated export, TP2, FP8 KV, Triton attention/GDN prefill,
native sampler, and disabled prefix caching, C=1 B0 graph reached 59.52 tok/s
at 196k and 52.94 tok/s at 250k. The meaningful historical comparison is the
pinned cache-neutral C=1 B0 rate of 16.53 tok/s, a 3.60x improvement; the
nightly eager 0.188 tok/s control is a path-failure diagnostic, not a graph
speedup claim.

G1.5 submitted three independent cache-disabled 196k prompts concurrently with
explicit B=1 and B=3 FULL_DECODE_ONLY graphs. The all-cold client score was
23.57 tok/s median per stream and 90.54 tok/s summed per-stream decode, with
serial long-prefill TTFTs. Once all three requests were decoding, seven
ten-second server intervals reported 153.4-154.7 tok/s aggregate with FULL B=3
replay. This proves the graph serving shape but does not make cold simultaneous
arrival equivalent to three already-populated conversations.

G2 graph MTP1 passed deterministic text, forced tool calling, and FULL replay
at its required B=1/two-token geometry, but reached only 14.36 tok/s at 196k
versus graph B0's 59.52 tok/s despite 93.85% draft-token acceptance. This is a
catastrophic regression, so MTP is closed for this vLLM/runtime/model path.
MTP2/MTP3, C=3 MTP, and 250k MTP are not authorized. The complete current
evidence is in
`qwen3.8-27b-rtx5090-phase-g1p5-g2-graph-concurrency-mtp1-2026-09-21.md`.

For a materially different future runtime, use the revised MTP rule: advance to
MTP2/MTP3 after an MTP1 win; permit only a narrow MTP2/MTP3 test when MTP1 is
within 10% below B0 and draft-token acceptance exceeds 90%; stop when MTP1 is
20% or more below B0. This exception is not authorization to reopen MTP on the
current runtime/model path.

## Phase H Supersession

Phase H moved the remaining work from decode optimization to prefill admission.
With the successful graph B0 C=3 profile, a `4096` token budget gave the best
all-user admission (279.58 s) and completion wall time (359.81 s), but retained
serial 94/187/280 s cold TTFTs. Adding a `1024` long-prefill cap made the same
three simultaneous 196k arrivals fair at 274.09/279.53/281.48 s while preserving
154.5 tok/s stable FULL-B=3 aggregate decode and adding only 1.2 s completion
wall time. However, the selected fair profile raised an established 196k user's
p95 ITL from 35 ms to 1.48 s while a new 196k request prefills. This path is not
approved for interactive serving; retain llama.cpp production and use admission
control rather than treating graph B0 as a routing candidate. Complete evidence
is in `qwen3.8-27b-rtx5090-phase-h-prefill-admission-2026-09-21.md`.

## Phase I Supersession

The final H3 ladder confirmed that scheduler caps alone cannot make an
independent cold 196k arrival interactive: 1024/512/256/128 thresholds yielded
established-user p95 ITL of 1.479/0.752/0.426/0.348 s, with the 128 cap raising
incoming TTFT to 257.59 s. Retain 4096/256 as the practical experimental knee,
not as a cold-arrival QoS solution.

Under the same graph profile with explicit prefix caching, a warmed distinct
196k conversation plus a real 2k multi-turn continuation reused 196,000 local
tokens and computed only 2,075. The continuation TTFT was 2.468 s while the
already-decoding user's p95 ITL was 0.265 s, below the 300 ms target. This
validates prefix-cache locality as the next operational lever. It does not
authorize routing: maintain llama.cpp production, apply future cold-context
admission control at the gateway, and qualify cache retention/load behavior
separately. Complete evidence is in
`qwen3.8-27b-rtx5090-phase-i-cached-continuation-qos-2026-09-22.md`.

## Starting Point

The calibrated export and the pinned vLLM 0.29.0 fallback are the fixed
baseline:

```text
model: /home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-e-kv-fp8-calibration-2026-09-20T052906Z/export
TP: 2 on physical GPUs 0,1
context: 262144
KV cache: fp8_e4m3
attention: {"backend":"TRITON_ATTN","use_trtllm_attention":false}
sampler: VLLM_USE_FLASHINFER_SAMPLER=0
execution: --enforce-eager
communication: NCCL/PYNCCL fallback; no usable GPU P2P
```

This configuration passed restart, cache, tool, sustained-load, retrieval, and
four near-native-context operational tests, but it is not quality-approved.
Its reconstructed 128-example GSM8K smoke was 122/128 (95.3125%), with three
truncations. See `qwen3.8-27b-rtx5090-phase-e-calibrated-validation-2026-09-20.md`.

The live hardware snapshot at plan creation has both cards capped at 500 W.
Every maintenance window must record power caps, topology, P2P state, driver,
temperature, free memory, and compute-process ownership again; older 500/575 W
measurements are historical, not a current tuning input.

## Compatibility Findings

- Installed vLLM 0.29.0 exposes `--spec-method`, `--spec-tokens`,
  `--compilation-config`, CUDA-graph capture controls, and
  `--disable-custom-all-reduce`.
- It recognizes the export's `qwen3_5`/`mtp_num_hidden_layers=1` configuration.
  The supported native invocation is `--spec-method mtp --spec-tokens N`; no
  separate draft model is needed because vLLM loads MTP weights from the target
  checkpoint.
- This is a one-layer MTP head. vLLM warns that `N > 1` repeats forward passes
  through that same layer and can lower acceptance. Therefore MTP1 is the first
  real candidate; MTP2 and MTP3 are experiments, not assumed improvements.
- The currently pinned FlashInfer JIT rejects SM120. Retain Triton attention
  and the non-FlashInfer sampler for every initial MTP and graph comparison.
- NVIDIA reports P2P read/write `GNS` between these GPUs. The public 2x5090
  campaign's `pcie_ipc` and custom-driver path is not a safe local flag change.
  No `pcie_ipc` communication backend was found in this vLLM installation's
  CLI or Python source.

## Stage 0 Result

The primary B0 matrix is complete. With three warmed, distinct long prefixes
and three simultaneous 4,096-token streams, eager Triton sustained a median
16.33 tok/s per user at 196k and 16.37 tok/s at 250k (approximately 49 tok/s
aggregate). Median p95 ITL was 0.121 s and 0.120 s respectively. Cold
simultaneous long-prefill arrivals are materially different and queue TTFT;
they remain a separate prefill/admission metric.

The detailed evidence and artifact list are in
`qwen3.8-27b-rtx5090-phase-f-stage0-b0-2026-09-20.md`. Stage 1 must beat B0 by
at least 10% per-user decode at both contexts while keeping median p95 ITL at
or below 0.133 s (196k) and 0.132 s (250k).

## Non-Negotiable Guardrails

1. Run only through `operations/run-rtx5090-phase-d-benchmark.sh` or a direct
   successor with the same rollback, port, process-ownership, and exit-trap
   protections. Pass a freshly verified unrelated GPU-0 PID to the wrapper.
2. Bind every candidate to loopback, use physical GPUs 0 and 1, and preserve
   the reference service plus monitor's prior state on every exit path.
3. Use a new timestamped run directory and save server logs, command line,
   environment allowlist, model config hash, runtime package versions, request
   JSON, `/metrics` snapshot, telemetry samples, and result summary.
4. Do not alter the driver, kernel module parameters, CUDA installation, power
   cap, fan policy, gateway, systemd unit, or exporter during a benchmark.
5. Stop a run on an error, unexpected GPU process, OOM, failed correctness
   request, or either GPU reaching 85 C. A completed run requires at least
   1 GiB free VRAM per GPU throughout; retain the measured minimum as a result
   rather than treating this floor as capacity proof.

## Measurement Contract

Use separately prefixed prompts and 512 forced output tokens for cold-capacity
and prefill measurements. Use 4,096 forced output tokens after prefix-cache
warmup for the three-session steady-decode score. Run each warm configuration
three times after a recorded warmup; report cold-start/JIT results separately
and do not blend them into steady-state decode figures.

| Workload | Purpose |
| --- | --- |
| 8k, one session | Startup, MTP correctness, graph capture, short-context decode |
| 196,017 tokens, one and three sessions | Primary interactive long-context point |
| 224k, three sessions | Capacity/decode transition point |
| 250,017 tokens, one and three sessions | Near-native primary point used by prior evidence |
| 259k, one session | Native-window boundary only; not a three-user target unless it fits |
| 64k prefill plus active decode | Prefill/decode interference and p95 ITL |
| 300-second three-session sustained load | Thermal stability, scheduler behavior, and goodput |

For each request and aggregate, record prompt/completion tokens, TTFT,
steady-state decode tokens/s, p50/p95 ITL, request wall time, aggregate goodput,
prefill rate, queue time, KV-cache use, free VRAM, power, temperature, and
errors. For speculative runs also record drafted, accepted, and accepted-token
rate from vLLM metrics/logs. Do not infer acceptance from wall time alone.

The comparison score is the median three-user per-user decode rate at 196k and
250k. Aggregate decode is a tie-breaker only when p95 ITL, errors, and quality
are no worse than the competing configuration.

## Stage 0: Establish a Fair Three-Session Baseline (Completed)

1. Extend `tools/rtx5090_phase_d.py` before the first run to record streaming
   TTFT/ITL and scrape the server metrics needed for speculative acceptance.
   Add explicit, allowlisted options for speculative method/token count,
   graph configuration, and `max-num-seqs`; do not add a free-form shell-args
   escape hatch.
2. Re-run MTP-off eager Triton at `max-num-seqs=3` using the measurement
   contract. This is baseline B0. Keep the existing `max-num-seqs=4` result as
   a capacity reference, not the per-user-decode baseline.
3. Confirm normal EOS, a forced tool call, deterministic temperature-zero text,
   and the 249k buried-fact retrieval before timing each configuration.
4. Reject an implementation change if B0 cannot reproduce the calibrated
   operational contracts or differs materially from the documented Phase E
   behavior without an explained environmental cause.

## Stage 1: Native MTP (MTP1 Rejected)

Use the existing calibrated export and pinned fallback stack. Do not combine
MTP with CUDA graphs, a runtime upgrade, a communication change, a new sampler,
or a new checkpoint in this stage.

| Candidate | vLLM addition | Advance only if |
| --- | --- | --- |
| M1 | `--spec-method mtp --spec-tokens 1` | Starts cleanly; model log confirms MTP load; deterministic output, tool call, and retrieval match B0 |
| M2 | M1 with `--spec-tokens 2` | M1 passes; acceptance and per-user decode do not regress at 8k or 196k |
| M3 | M1 with `--spec-tokens 3` | M2 passes; explicitly evaluate the one-layer repeated-forward warning |

For each candidate, run the 8k/one-session preflight, then 196k and 250k at
one and three sessions, interference, and sustained load. Do not run 224k,
259k, or four-session capacity for a candidate that loses at the primary
196k/250k three-session points.

Keep a speculative candidate only when all of the following hold:

- Zero errors, OOMs, unexpected finish reasons, or thermal stops.
- Exact temperature-zero output equality with B0 for the frozen deterministic
  probes, including the tool and long-context retrieval contracts.
- Median three-user per-user decode improves by at least 10% over B0 at both
  196k and 250k, with no more than a 10% p95 ITL regression.
- Acceptance and performance are stable across three warmed repeats; a single
  favorable first run is not evidence.

If M1 fails, stop native MTP work. If M2 or M3 loses to M1, retain M1 and do
not expand the worse configuration's matrix.

### MTP1 Result

MTP1 passed the 8k normal-text, tool, interference, sustained-load, and 249k
retrieval contracts, but its first 196k C=3 run recomputed all three supposedly
warmed prefixes. It recorded zero prefix-cache-hit tokens, 99-299 second TTFT,
and an observed median 11.04 tok/s per user, far below B0 and the MTP gate.
MTP2, MTP3, and the remaining MTP matrix are therefore not run. See
`qwen3.8-27b-rtx5090-phase-f-stage1-mtp1-2026-09-20.md` for artifacts and the
bounded conclusion.

### Stage 1.5 Cache Diagnostic

The narrow 128k identical-prompt reproducer found positive prefix-cache reuse
for both B0 and MTP1 on the pinned vLLM 0.29.0 runtime and an isolated
vLLM 0.29.1rc1 nightly. The original C=3 zero-hit result is therefore
workload/concurrency-specific rather than evidence that MTP universally
disables prefix caching. However, cache-disabled 196k C=1 MTP1 still delivered
14.08 tok/s versus B0's 16.53 tok/s, with 0.147 s versus 0.120 s p95 ITL.
MTP1 remains ineligible for MTP2/MTP3 or Stage 2. See
`qwen3.8-27b-rtx5090-phase-f-stage1p5-cache-diagnostic-2026-09-21.md`.

## Stage 2: Historical CUDA-Graph Proposal

This proposal is superseded by the successful Phase G B0 graph work above. Only
the Stage 1 winner advances. First remove `--enforce-eager` while retaining
Triton attention, the non-FlashInfer sampler, the export, TP2, and
`max-num-seqs=3`. Begin with an 8k startup/correctness run, then 196k and 250k
at one and three sessions.

If default graph selection passes, make a separate, attributed experiment with
explicit captures for the actual three-user decode batches, initially sizes
`1 2 3`. Capture overhead, startup time, graph-memory use, graph hit/miss
metrics, and a fresh-request path; graph success during warmup alone is not a
serving result.

Stop this stage and restore eager mode on a capture OOM, SM120/FlashInfer error,
incorrect output, unstable shutdown, or lost Stage 1 gain. Keep a graph profile
only if it passes the Stage 1 criteria and improves the same primary score by
at least 5% over the eager MTP winner without reducing safe three-session
capacity.

`performance-mode=interactivity` and `performance-mode=throughput` are optional
follow-up comparisons only after a graph profile passes. Do not use throughput
mode as a winner if its aggregate gain is bought by worse per-user decode or
p95 ITL.

## Stage 3: Isolated Runtime and Communication Research

This is not a production upgrade and must start from a fresh, isolated venv or
container. Leave `.phase2-vllm-venv` untouched.

1. Reproduce B0, then the Stage 1/2 winner in the isolated runtime before
   comparing any patch or newer build. Record source revision, wheels, CUDA,
   Torch, FlashInfer, ModelOpt, and model hash.
2. Evaluate public-campaign patches only one category at a time: SM120
   FlashInfer/attention support, CUDA-graph behavior, then communication.
   A benchmark against different weights, an unpinned patch bundle, or a
   changed driver has no attributable result.
3. Do not enable `pcie_ipc`, peer-copy, or a custom all-reduce backend while
   `nvidia-smi topo -p2p r/w` remains non-P2P. A driver experiment requires a
   separately approved maintenance and recovery procedure, including reference
   service recovery after reboot. It is not part of this optimization window.
4. Compare an alternate communication path only against the same model,
   sampler, MTP count, graph profile, power caps, and three-session matrix.
   Retain NCCL/PYNCCL when the alternative does not improve primary-score
   decode by at least 10% or affects stability.

## Stage 4: DFlash2 and Other Challengers

DFlash2 is a fallback, not a parallel first experiment. Start it only when
native MTP/graphs do not meet the objective or the native MTP quality gate
fails. The local DFlash2 asset is GGUF and cannot be used by vLLM's `dflash`
method.

Before downloading an HF/safetensors drafter, record publisher, revision,
license, hash, disk requirement, architecture compatibility, quantization
compatibility with the calibrated target, and a deletion plan. Validate it in
an isolated environment at 8k and 196k before any native-context matrix. Treat
its draft/target memory and three-session capacity independently from MTP.

Do not prioritize llamAmpere on this host: its known profile targets SM86 and
single-GPU replicas, so it cannot answer the TP2 three-session objective before
its SM120 compatibility and capacity are separately established.

## Quality and Promotion Gates

Performance selection is distinct from production promotion.

1. Every candidate must pass deterministic text/tool/retrieval equality against
   B0 before its long matrix.
2. The performance finalist must run the 128-example reconstructed GSM8K smoke
   with the fixed seed, prompt policy, concurrency, and output cap. It must not
   be worse than B0 in score, request errors, finish reasons, or truncations.
3. Freeze and run representative coding-agent and multi-turn tool trajectories
   with the same tokenizer, parser, thinking policy, and API contract. Define
   expected outputs before viewing finalist results.
4. Only a finalist that passes the preceding quality work receives a dedicated
   maintenance window for the full 1,319-example GSM8K evaluation. It must meet
   the recorded promotion floor: at least 96.5% accuracy, 100% stop rate, zero
   request errors, and zero truncations.
5. Only after those gates pass may a separate canary proposal cover an additive
   loopback vLLM unit, authenticated gateway pool, cancellation/streaming
   checks, vLLM-aware metrics parsing, latency SLOs, and rollback monitoring.

The current calibrated baseline fails the historical smoke gate. An MTP or graph
gain does not waive that failure, and no optimization result may be routed to
users before quality is resolved.

## Decision Outcomes

| Outcome | Action |
| --- | --- |
| MTP/graphs improve the primary score and pass quality gates | Retain the exact pinned profile as the performance finalist; prepare the separate full-quality and canary proposal |
| MTP improves decode but a quality gate regresses | Preserve the evidence, reject it for routing, and compare a separately qualified checkpoint only |
| MTP1 does not improve three-session decode | Stop MTP2/MTP3; keep B0 and move to isolated graph/runtime research only if justified |
| Graphs fail or regress | Keep eager MTP or B0; do not alter FlashInfer/driver settings in place |
| Communication experiment requires P2P/driver changes | Defer until an explicit, reversible driver-change approval exists |
| No candidate qualifies | Keep the llama.cpp Q4 reference live and retain all run evidence for a later runtime/model revision |

## Evidence Locations

- Calibrated validation: `research/qwen3.8-27b-rtx5090-phase-e-calibrated-validation-2026-09-20.md`
- TP2 capacity and topology: `research/qwen3.8-27b-rtx5090-phase3-tp2-benchmark-window-2026-09-20.md`
- Consolidated measurements and untested levers: `research/qwen3.8-27b-rtx5090-serving-digest-2026-09-20.md`
- Existing temporary harness: `tools/rtx5090_phase_d.py`
- Guarded maintenance wrapper: `operations/run-rtx5090-phase-d-benchmark.sh`
