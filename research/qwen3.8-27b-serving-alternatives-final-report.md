# Qwen3.8-27B Serving Alternatives Final Report

Date: 2026-09-19 UTC

## Decision

Promote llamAmpere v0.3 with Qwen3.8-27B ATX IQ4_XS-M to the production
architecture after readiness and canary validation. The measured 61 tok/s near
240k and 65 tok/s around 196k, successful objective quality checks, and
single-GPU operation are sufficient for this decision. The earlier 70-80 tok/s
figure was aspirational, not an acceptance criterion.

Keep the existing two-GPU stock llama.cpp Q4 deployment available as the
fallback and reference until the llamAmpere cutover is validated. This report
does not change production routing or perform the cutover.

## Controls

- Physical GPU 0 was forbidden for candidate work. Candidate processes used physical GPUs 1 and 2 only. An unrelated Python training process occupied GPU 0 during part of the main Axis 1 sweep; it was not started or stopped by this work.
- Candidate servers used isolated environments, loopback ports, model paths, and artifact directories.
- No production routing, systemd service, model placement, or reference runtime was changed.
- Candidate processes were stopped and cleanup artifacts verified after each branch.
- Throughput results are not treated as comparable when context depth, model, or workload differs.

## Normalized Comparison

`Cold long context` reports the cold populated request closest to the reference
comparison point. Warm-cache prefill is not used as a substitute for cold
prefill. Energy values are included only where the artifact records a usable
measurement; they are not normalized across runtimes.

| Candidate | Model/runtime | GPUs | Maximum validated context | Cold long-context throughput | Concurrency | VRAM | Quality/correctness | Recommended role |
|---|---|---:|---:|---|---|---|---|---|
| Current reference | UD Q4_K_M / llama.cpp | 1+2 tensor split | 262k | 750.42 prefill / 32.99 decode tok/s at 261,765 prompt tokens | Control; no comparable concurrent result in this campaign | 16,864 MiB/GPU observed at 262k | Validated control; no known issue in the reference profile | Fallback/reference |
| llamAmpere single | ATX IQ4_XS-M / llamAmpere v0.3, commit `36a6bca817755c6ddab2001b78eb0ae3e3eef7c8` | 1 | 240k populated request | 502.92 prefill / 61.03 decode tok/s at 239,960 prompt tokens; 572.94 / 65.50 at 196k | N/A | 22,830 MiB on physical GPU 1 | Six objective tasks passed initially; T4/T8 passed normal-EOS follow-up; no subjective quality score | Production single-agent baseline |
| llamAmpere dual | ATX IQ4_XS-M / llamAmpere v0.3 | 1+2 independent processes | 196k validated concurrently | Cold prefill: 578.41/567.90 tok/s at 196k and 983.99/971.83 tok/s at 64k, GPU1/GPU2; decode was approximately 65.2 tok/s per instance at cold 196k | Two independent instances overlapped successfully; no `--parallel 2` | Approximately 22,814 MiB per instance | Same objective quality evidence as single instance | Production concurrency topology |
| Bonsai PQ2_0 | Ternary Bonsai 2 27B PQ2_0 / PrismML fork commit `5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6` | 1 | 262k populated request | 479.05 prefill / 28.89 decode tok/s at the 262k matrix point | Independent GPU1/GPU2 and shared single-GPU `parallel=2` topology validated | 23,810 MiB on single physical GPU | T1-T7 and T9-T10 passed; T8 failed with truncation and malformed JSON at 2,048 tokens | No substitution; future reviewer/subagent experiment only after human review |
| Bonsai PTQ1_0 | Ternary Bonsai 2 27B PTQ1_0 / same PrismML fork | 1 | 262k populated request | 364.25 prefill / 26.38 decode tok/s at the 262k matrix point | Not separately topology-tested after the shared PQ2_0 topology validation | 22,666 MiB on single physical GPU | Same T8 material regression as PQ2_0 | No substitution; future reviewer/subagent experiment only after human review |
| Bonsai multiple | PQ2_0 / PrismML fork | 1+2 or shared GPU1 | 196k independent dual; 8k shared `parallel=2` | Independent dual cold 196k: 1,147.74 aggregate prefill and 29.17 aggregate decode tok/s under the artifact definition; overlap efficiency 1.986 | Independent dual and shared-weight multi-context concurrency validated | Per-instance values above | Topology works, but T8 contract regression remains | Conditional reviewer/subagent topology only |
| SGLang DFlash2 | Local Qwen AWQ INT4 target + BF16 DFlash2 drafter / SGLang commit `9784d5f979c9eaa5a6d9b31e596a7a73cb76dab8` | 1+2 TP | 8k load and bounded requests only | DFlash code probe 576.0 prefill / 219.3 decode tok/s; acceptance ranged from 14.29% to 79.22% by workload | Not evaluated beyond the failed gate | 23,313/23,008 MiB in the active sample | Thinking format malformed; structured tool call invalid; acceptance unstable | Failed correctness gate; stop branch |
| vLLM target-only | Local Qwen AWQ INT4 target / vLLM `0.29.0`, commit `98dff2a81d747d1dba01a47f939f48c3526d4206` | 1+2 TP | 8k load and bounded requests only | 67.39 decode tok/s on the narrative probe; not a long-context result | DFlash was not started | 20,632 MiB/GPU | Thinking format malformed; forced tool call rejected because no parser was configured | Failed feasibility/correctness gate; stop branch |

## Production Architecture Comparison

| Profile | GPUs/request | Context | Cold prefill | Decode | Concurrent agents | Aggregate decode | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Old stock llama.cpp tensor | 2 | 262k | 750.42 tok/s at 261,765 tokens | 32.99 tok/s | 1 | 32.99 tok/s | Fallback/reference |
| llamAmpere current | 1 | Approximately 240k | 502.92 at 239,960; 572.94 at 196k | 61.03 at 240k; 65.50 at 196k | 1 | Same as decode | Validated production baseline |
| llamAmpere optimized | 1 | Approximately 240k | 503.75 at 240k; 571.58 at 196k | 59.03 at 240k; 64.54 at 196k | 1 | Same as decode | MTP3, batch 6144; only marginally better than batch 4096 in the controlled sweep |
| 2 independent llamAmpere | 1+1 | 196k validated per instance | 578.41/567.90 per instance | Approximately 65.2 per instance | 2 | Approximately 130.4 | No tensor synchronization; default concurrency topology |
| llamAmpere TP=2 | 2 | 239,963 measured | 738.83 | 39.56 | 1 | 39.56 | Non-MTP only; MTP3 fails in the tensor draft graph |

The Axis 1 candidate placement and cell sequencing were controlled within the
new sweep, but host-wide GPU-idle isolation was not because an unrelated GPU0
process was active during part of the main run. The earlier Phase 1
single-instance values remain the validated current baseline; small differences
between runs are expected from runtime variance. Warm-cache prefill is reported
separately and is not substituted for cold prefill.

## Branch Results

### Phase 0: Reference

The immutable llama.cpp control accepted populated prompts at 196k and 262k
context. The 262k request with 261,765 prompt tokens measured 750.42 prefill
and 32.99 decode tok/s. The 196k request measured 857.60 prefill and 35.28
decode tok/s. The reference remains the validated fallback and control.

Artifact: `research/qwen3.8-27b-serving-alternatives-phase0-20260918T132414Z`

### Phase 1: llamAmpere

The candidate accepted populated 64k, 128k, 196k, and approximately 240k
requests on one physical GPU. Cold decode declined from 79.19 tok/s at 64k to
61.03 tok/s at 240k. The 196k and 240k results are useful long-context
operating points; the former 70-80 tok/s figure is not a rejection gate.

The original objective quality run passed six tasks and marked T4 and T8 as
reviewer cases because `ignore_eos=true` caused truncation. The normal-EOS
follow-up reran T4 and T8 with `max_tokens=1024`; both passed. This establishes
no demonstrated objective regression, not a complete subjective quality
approval.

The independent dual-instance test at 64k and 196k passed its close-to-single
performance gate. Both instances overlapped without using llama.cpp
`--parallel 2`.

Artifacts:

- `research/qwen3.8-27b-serving-alternatives-phase1-20260918T133922Z`
- `research/qwen3.8-27b-serving-alternatives-phase1-quality-20260918T141239Z`
- `research/qwen3.8-27b-serving-alternatives-phase1-quality-followup-20260918T143447Z`
- `research/qwen3.8-27b-serving-alternatives-phase1-dual-20260918T144111Z`

Model artifact hash: `5cf05ad901dcaa76f41db13a5629146ed882219339377a80e37b12a8528d963b`.

### Axis 1: Prefill/decode trade-off

The controlled sweep covered MTP2/MTP3 with batch 4096, 6144, and 8192 at
approximately 196k and 240k, with separate cold and warm requests. All 12
cells returned HTTP 200, 128 predicted tokens, and `truncated=false`.

| Depth | Profile | Cold prefill | Cold decode | Cold wall | Warm decode / wall |
|---|---|---:|---:|---:|---:|
| 196k | MTP2, b4096 / b6144 / b8192 | 579.91 / 582.86 / 569.15 | 54.22 / 56.44 / 54.79 | 340.574 / 338.762 / 346.928 s | 54.96 / 56.72 / 55.14 tok/s; 4.339 / 4.176 / 4.244 s |
| 196k | MTP3, b4096 / b6144 / b8192 | 569.96 / 571.58 / 569.65 | 64.20 / 64.54 / 64.26 | 346.107 / 345.109 / 346.285 s | 64.48 / 64.31 / 64.10 tok/s; 3.898 / 3.878 / 3.895 s |
| 240k | MTP2, b4096 / b6144 / b8192 | 503.87 / 506.62 / 502.49 | 51.52 / 50.60 / 50.48 | 479.156 / 476.639 / 480.486 s | 51.92 / 50.93 / 50.55 tok/s; 4.772 / 4.798 / 4.853 s |
| 240k | MTP3, b4096 / b6144 / b8192 | 502.58 / 503.75 / 501.80 | 58.93 / 59.03 / 58.36 | 480.077 / 478.955 / 480.826 s | 58.82 / 58.77 / 58.25 tok/s; 4.445 / 4.464 / 4.497 s |

MTP3 is the useful production setting because it retains the decode advantage;
MTP2 gives lower decode with no consistent prefill benefit. Batch 6144 is the
best measured trade-off in several cells, but its improvement over batch 4096
is only approximately 0.2-0.5%. Batch 4096 therefore remains the lower-risk
production baseline, while batch 6144 is a valid canary optimization.

Peak VRAM was approximately 22,648 MiB for MTP2 and 22,798 MiB for MTP3.
Mean power was approximately 282-285 W with a roughly 300 W maximum. MTP2
acceptance was approximately 81-83 accepted out of 88-91 drafted tokens;
MTP3 acceptance was approximately 90-91 out of 106-109.

Artifacts:

- `research/qwen3.8-27b-llamampere-axis1-sweep-20260918T224901Z`
- `research/qwen3.8-27b-llamampere-axis1-batch6144-20260919T004351Z`

TTFT in these artifacts is a derived non-streaming estimate. The throughput
requests are performance probes, not a replacement for the earlier objective
quality suite; the new sweep uses `ignore_eos=true` and does not assign a
subjective quality score.

### Axis 2: Two-GPU architectures

Two independent llamAmpere instances remain the preferred two-GPU topology for
the multi-agent system. At 196k, the prior test measured approximately 65.2
tok/s decode per instance with cold prefill of 578.41 and 567.90 tok/s on
GPU1/GPU2, giving approximately 130.4 tok/s aggregate decode while preserving
two independent contexts.

Tensor parallelism is source-supported and GPU0-safe with
`CUDA_VISIBLE_DEVICES=1,2`. The measured non-MTP TP=2 run passed at 64k, 196k,
and 239,963 tokens, with cold prefill/decode of 1,180.45/48.92, 826.77/42.26,
and 738.83/39.56 tok/s respectively. It reduced single-request wall time and
per-GPU VRAM, but used both GPUs for one request, had lower decode than the
single-GPU MTP3 profile, and used more combined energy.

MTP3 TP=2 is not currently usable: NCCL initialized successfully, then the
MTP draft graph aborted at `GGML_ASSERT(reg)` before serving a request. TP=2 is
therefore not a production profile. It may be revisited as a latency-oriented
single-agent mode after the tensor MTP failure is fixed and quality is
revalidated.

Artifact: `research/qwen3.8-27b-llamampere-tp2-20260918T235826Z`.

### Operational Cutover Status

The existing production definition is
`llama-qwen3.8-q4-tensor-262k.service` on `127.0.0.1:8080`, with the gateway
mapping the current Qwen model ID to that backend. No llamAmpere systemd unit,
production port, gateway route, exporter configuration, or automatic rollback
exists yet. The validated llamAmpere command used isolated port `18101` only.

No service or route changes were made. A safe cutover requires a dedicated
llamAmpere unit with the pinned binary/model/vocabulary hashes, production
alias and metrics settings, plus an explicit gateway mapping decision. The
fallback is cold rather than hot: the stock service needs GPUs 1+2, so rollback
requires stopping llamAmpere, verifying GPU1 is free, starting the stock unit,
and restoring the gateway mapping. The gateway has no weighted canary or
automatic backend failover, so the cutover must use either a temporary candidate
model ID or an approved all-traffic alias switch.

### Phase 2: Bonsai 2

Both PQ2_0 and PTQ1_0 loaded all requested matrix depths, including real
populated 262k prompts. PQ2_0 was faster and PTQ1_0 used less VRAM. The
quality run was an expanded ten-task suite, not only the original T1-T8
manifest. T8 failed for both packings at 1,024 tokens and again at 2,048
tokens: output truncated at the limit and did not satisfy the JSON contract.

The PQ2_0 topology tests validated independent GPU1/GPU2 servers at 64k and
196k, plus two shared-weight contexts on one GPU. These results demonstrate
useful concurrency but do not override the coding-output regression.

Model hashes:

- PQ2_0: `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`
- PTQ1_0: `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33e3`

Artifacts:

- `research/qwen3.8-27b-serving-alternatives-phase2-20260918T160224Z`
- `research/qwen3.8-27b-serving-alternatives-phase2-quality-20260918T170924Z`
- `research/qwen3.8-27b-serving-alternatives-phase2-quality-followup-20260918T173610Z`
- `research/qwen3.8-27b-serving-alternatives-phase2-topology-20260918T174849Z`

### Phase 3: SGLang and vLLM DFlash2

SGLang target-only, built-in MTP, and DFlash2 loaded at 8k with the explicitly
documented local AWQ target. Target-only and DFlash2 both failed the thinking
and structured-tool correctness checks. DFlash2 ordinary bounded outputs
matched target-only, but acceptance was workload-dependent and collapsed to
14.29% on the readiness probe and 29.87% on narrative output. The hard gate
therefore stopped before long-context testing.

vLLM loaded the same local AWQ target with TP=2 at 8k. The bounded code and
JSON probes passed, but thinking output was malformed and forced tool calling
returned HTTP 400 because no tool-call parser was configured. DFlash was not
started. There is no vLLM DFlash acceptance, long-context, or DFlash throughput
result.

The official BF16 target was not silently replaced: it was not downloaded
because it exceeds the two-RTX-3090 VRAM budget. Both Phase 3 branches are
therefore explicitly AWQ feasibility tests, not pure BF16 reference
comparisons.

Artifacts:

- `research/qwen3.8-27b-serving-alternatives-phase3-20260918T180203Z`
- `research/qwen3.8-27b-serving-alternatives-phase3-vllm-20260918T183721Z`

## Known Limitations

- llamAmpere's quality evidence is objective and bounded; no subjective
  reviewer score or full end-to-end OpenCode campaign was assigned.
- The new Axis 1 throughput probes use `ignore_eos=true` and are not additional
  quality evidence. The earlier normal-EOS T4/T8 follow-up remains the quality
  basis for the production decision.
- The ATX 240k measurement is near the reference context but not an exact
  262k replacement result.
- The Axis 1 main sweep overlapped an unrelated GPU0 Python training process;
  candidate placement remained GPU1-only, but host-wide GPU0 idleness cannot be
  claimed for that artifact. The batch-6144 and TP2 runs preserved GPU0.
- TP2 comparisons use a non-MTP mode because TP2 MTP3 aborts during speculative
  graph setup. They are useful architecture evidence but not an apples-to-apples
  MTP3 throughput comparison.
- Bonsai's topology evidence is strongest for PQ2_0; PTQ1_0 was not separately
  repeated through the topology matrix.
- Phase 3 long-context performance was intentionally not measured after the
  correctness gates failed.
- Runtime energy measurements use different sampling and accounting methods;
  the report does not rank candidates by the raw energy values.

## Final Recommendation

1. Use llamAmpere v0.3 ATX IQ4_XS-M as the production runtime after the
   readiness and canary gates below. Use MTP3, batch 4096, ubatch 1024,
   context 245760, and the validated GPU1-only placement as the initial
   baseline. Batch 6144 is a measured canary optimization, not a required
   promotion condition.
2. Use two independent llamAmpere instances on GPUs 1 and 2 for concurrent
   supervisor/researcher/executor/reviewer workloads. This is independent
   replica concurrency, not TP=2.
3. Keep the existing two-GPU stock llama.cpp Q4 deployment startable as the
   fallback/reference until the llamAmpere cutover is validated. Roll back to
   it on output-contract, stability, or operational regression.
4. Do not use llamAmpere TP=2 as production. Revisit only after the tensor MTP3
   draft-graph failure is fixed, then repeat quality and long-context tests.
5. Do not use Bonsai as a general Qwen substitute. Revisit only for a bounded
   reviewer/subagent role after human review of the T8 contract failure.
6. Stop the SGLang/vLLM DFlash2 branch until thinking templates/parsers and
   structured tool calls are corrected and revalidated.

### Cutover Gates

- Pin the llamAmpere binary, runtime commit, ATX model, and MTP vocabulary-map
  hashes from the completed artifacts.
- Run a production-port readiness check, normal-EOS tool/MCP smoke, and T4/T8
  objective probes with no truncation or malformed structured output.
- Verify cold and warm populated requests at approximately 196k and 240k with
  no OOM, process failure, or context truncation.
- Confirm GPU1-only placement, preserve unrelated GPU0 jobs, and verify the
  llama.cpp fallback remains startable.
- Canary the route while monitoring errors, truncation, VRAM, temperature, and
  wall time; rollback immediately to llama.cpp on any output or operational
  regression.
