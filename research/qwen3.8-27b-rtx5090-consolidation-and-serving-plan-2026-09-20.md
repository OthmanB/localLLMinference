# Qwen3.8-27B RTX 5090 Consolidation and Serving Plan

Date: 2026-09-20

Status: Research and proposed execution plan. No repository consolidation,
service changes, installations, or new inference benchmarks were performed
during the assessment. Creating this document does not authorize a live cutover.

## 1. Objective and Recommendation

Find the best trade-off between:

- Fast cold prefill and time to first token (TTFT).
- Fast decode with acceptable per-user inter-token latency (ITL).
- Maximum simultaneous sessions at the native 262,144-token context window.
- Correct reasoning, tool calling, structured output, and long-context behavior.
- Sustainable thermals, energy consumption, and operational reliability.

Keep `localLLMinference` as the canonical knowledge base. Preserve the local
RTX 5090 work from `localLLMinference-old` as an explicit hardware profile,
without replacing the newer RTX 3090, gateway, or monitoring work.

Evaluate two main deployment candidates:

1. Two independent single-GPU servers behind the existing gateway, using the
   current GGUF deployment as the reference. This is the lowest-risk route to
   two native-context sessions, subject to strict-residency and thermal
   revalidation.
2. One vLLM or SGLang server spanning both GPUs, using a suitable NVFP4
   checkpoint and FP8 attention KV cache. Avoiding duplicated weights could
   improve full-context capacity; native kernels and scheduling could improve
   prefill. Communication and runtime memory overhead must be measured.

vLLM TP2 is the best-supported first challenger in the current published
RTX 5090 evidence, not a proven winner. SGLang merits an equally controlled
comparison. Published single-card NVFP4 recipes do not establish the required
combination of native context and useful concurrency.

Working expectation, not a benchmark conclusion: independent GGUF replicas
will be the simplest reliable two-session solution; a carefully selected
NVFP4 TP2 deployment is the strongest candidate for better prefill and more
than two full-context sessions.

## 2. Evidence Boundaries

The assessment combined read-only local inspection, existing benchmark
artifacts, upstream documentation, model configurations, and GitHub issue and
release searches checked on September 20, 2026.

- Local hardware and service observations are snapshots, not permanent facts.
- Existing local performance results were measured previously, not in this
  assessment.
- Upstream recipe results belong to their documented builds, checkpoints,
  hardware, and workloads. They are not measurements on this workstation.
- Startup cache capacity is not proof of populated-context correctness or
  latency-qualified concurrency.
- Open issue reports identify validation risks; they do not prove every build
  or configuration suffers the reported defect.
- Recheck upstream support and issue status before executing the plan. Pin
  versions and artifact revisions rather than using floating `latest` images.

## 3. Workstation Findings

| Component | Observed state |
|---|---|
| CPU | AMD Threadripper PRO 7995WX, 96 physical cores, 192 logical CPUs |
| RAM | 672 GiB installed; approximately 660 GiB OS-visible |
| Available RAM | Approximately 392 GiB during inspection; other substantial workloads present |
| Memory population | Seven of eight DIMM slots populated; not a fully populated eight-channel configuration |
| NUMA | One OS-visible node |
| GPUs | Two RTX 5090s, approximately 32 GiB each, SM120 |
| GPU 0 PCIe | Gen 5 x16 capability |
| GPU 1 PCIe | Gen 5 x8 capability at its upstream port |
| GPU topology | `NODE`, no NVLink |
| Peer access | NVIDIA reports GPU peer reads/writes unsupported |
| Driver/toolkit | NVIDIA 610.43.03; installed CUDA toolkit 13.3 |
| Active model placement | One llama.cpp process on GPU 0; GPU 1 effectively unused |
| Power limits | GPU 0 at 500 W; GPU 1 at 575 W |

PCIe links were observed at an idle power-saving speed. This does not establish
a loaded-link speed problem. The x8 width and missing peer access are relevant
to distributed serving. No CUDA peer-copy or NCCL performance benchmark was
run. Do not treat the cards as a transparent 64 GiB memory pool.

### 3.1 Current Serving Configuration

- Service: `llama-qwen3.8-q4-native.service`.
- Runtime: local llama.cpp commit
  `b96806d96061049a5b574269b049bf6241d63d46`, built for `sm_120a`.
- Model: `Qwen3.8-27B-UD-Q4_K_M.gguf`, Unsloth export.
- API alias: `qwen3.8-27b-q4-gpukv-native`.
- One 262,144-token slot, Q8_0 K/V, Flash Attention, all layers requested on GPU.
- Batch/microbatch: 2048/512; 32 target and batch CPU threads.
- No active speculative decoding; embedded MTP tensors are not used.
- A small embedding service also uses GPU 0.

Native allocated context does not mean a request has populated that context.
The service had approximately 40k tokens retained in its slot during inspection.

### 3.2 Baseline Issues to Correct

**Managed memory is enabled unintentionally.** The service and benchmark
harness set `GGML_CUDA_ENABLE_UNIFIED_MEMORY=0`. This runtime tests whether the
variable exists, not its value, and selects `cudaMallocManaged` when present:

`/home/michel/LLMs-tests/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:140-165`.

For this revision, the variable must be absent to select the ordinary CUDA
allocator. Existing tests do not prove material spilling occurred, but claims
of strict non-migratable VRAM residency need correction and revalidation.
Also distinguish CUDA managed memory from host prompt snapshots and the
runtime's unrelated slot-cache organization settings.

Additional findings:

- The live server binds `0.0.0.0:8080`, unlike the old repository's
  loopback-only unit. Metadata endpoints responded without authentication.
  External firewall reachability was not assessed.
- The successful native-context test peaked at 79 C against an 80 C cutoff,
  with a 500 W cap and 100% GPU fan. Current policy requests 95% fan.
- The motherboard maximum-fan service is failed. Do not assume its intended
  cooling policy was applied.
- Other workloads consume substantial RAM; swap use was approximately 9.2 GiB.
- Equalize or explicitly account for GPU power, cooling, and co-tenant
  differences before comparing engines.

### 3.3 Existing Local Performance

These September 2 results are historical, single-GPU Q4 measurements:

| Actual prompt | Prefill tok/s | Decode tok/s | Request wall | Peak VRAM | Peak temperature |
|---:|---:|---:|---:|---:|---:|
| 196,227 | 1,428.64 | 38.70 | 141.02 s | 23,049 MiB | 68 C |
| 261,765 | 1,118.05 | 32.26 | 238.56 s | 25,615 MiB | 79 C |

Artifact directory:
`/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/q4-gpu0-500w-native-context-2026-09-02/`.

At native context, approximately 234 seconds were prompt evaluation. The test
generated only 128 forced output tokens; it is not a sustained serving or
normal-EOS quality test. The allocator caveat above applies to its residency
claims.

A two-slot, 192k-per-slot test completed at approximately 30,513 MiB peak VRAM,
but requests took approximately 285-287 seconds. This establishes capacity
under that configuration, not good interactive concurrency. Do not sum its
per-slot timing-derived decode rates as aggregate service throughput.

Artifact directory:
`/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/q4-gpu0-p2-192k-500w-retry-2026-09-02/`.

## 4. Repository Consolidation

### 4.1 Provenance and Scope

At inspection, the canonical checkout was clean at `c7d1860`. The old checkout
was at initial release `0c14557`, twelve commits behind the canonical checkout,
with two modified tracked documentation files and six untracked RTX
5090-specific files. There are no local customization commits to cherry-pick.

Do not overlay the old tree onto the canonical repository. Preserve these
files selectively:

```text
operations/install-native-q4.sh
operations/systemd/llama-qwen3.8-q4-native.service
operations/systemd/nvidia-qwen3.8-gpu0-policy.service
operations/systemd/ai-qwen3.8-q4-native-monitor.service
operations/logrotate/ai-server-qwen3.8-q4-native
tools/qwen_gpu_monitor.py
```

Also preserve useful additions to `operations/README.md` and
`operations/runbooks.md`. Do not port generated bytecode. Generic inherited
files in the old checkout are not RTX 5090 customizations.

### 4.2 Minimal Organization

Prefer additive hardware separation over a broad restructuring:

```text
operations/
  hardware/
    rtx5090/
      README.md
      ... opt-in profile-specific units and installation material ...
  config/profiles/
  monitoring.md
research/
  ... dated, hardware-specific evidence and plans ...
```

Existing RTX 3090 files can remain in place initially. Root and operations
navigation should distinguish the hardware profiles. Share the gateway,
monitoring, model manifest, and benchmark methodology where appropriate.

Each profile must record:

- GPU UUIDs, ports, service user, paths, and resource ownership.
- Runtime commit or image digest and model revision/hash.
- Weight, attention KV, and recurrent-state precision separately.
- Context per request, total cache capacity, concurrency/admission policy.
- Host-specific power, fan, and energy-accounting configuration.
- Validation state: proposed, tested, staged, or live.
- Explicit rollback procedure and mutually exclusive GPU ownership.

Keep UD Q4_K_M, UD Q5_K_M, ATX IQ4_XS-M, and NVFP4 artifacts distinct in the
model manifest. They are not interchangeable instances of a generic "Q4".

### 4.3 Conflicts and Safety Gates

- The old native installer disables shared power/fan services. Adapt it to
  refuse resource conflicts rather than disable unrelated services.
- The canonical installer has RTX 3090-host-specific paths, username, and
  topology assumptions. Do not run it unchanged on this workstation.
- Canonical operations documentation assigns Muse to GPU 0, but
  `operations/systemd/llama-muse-glimmer-30b-131k.service:12` selects GPU 2.
  Reconcile this before applying the committed units; an external live
  override was not established.
- Old and current logrotate rules overlap. Use one rotation owner and honor
  configurable log paths.
- Resolve the old README's 100% fan versus policy unit's 95% discrepancy.
- Replace presence-based managed-memory misconfiguration in affected profiles
  after checking each pinned runtime's semantics.
- Validate GPU identity and permitted power range before applying a policy.
- Readiness must verify expected model identity and context, not merely an
  HTTP health response from a potentially unrelated listener.
- Reconcile stale research cutover descriptions with current staged operations
  documentation without rewriting historical results.
- Preserve secrets outside the repository and avoid new backend LAN exposure.

The canonical repository describes llamAmpere as staged, with isolated native
context gates passed and live acceptance outstanding. This does not establish
the current live deployment at the home machine.

## 5. Current Engine and Checkpoint Evidence

Latest GitHub releases returned during research:

- vLLM 0.29.0, published 2026-09-09.
- SGLang 0.5.20, published 2026-09-18.

Published recipe validation may use older builds. Preserve those distinctions.

### 5.1 vLLM

The official Qwen3.8-27B recipe, updated September 14, explicitly documents
two RTX 5090s using native `FlashInferCutlassNvFp4LinearKernel` execution on
SM120, FP8 KV, and working in-checkpoint MTP. Its tested build is
`0.26.1rc1.dev608+g99a10304d`.

| TP2 checkpoint | Reported startup KV tokens | Capacity / 262,144 |
|---|---:|---:|
| Official FP8 | 377,456 | 1.44 |
| Inferact NVFP4 | 445,875 | 1.70 |
| Unsloth NVFP4 | 920,517 | 3.51 |

These are recipe-reported allocation figures, not proof of three simultaneous
full-context requests meeting a latency target. They make Unsloth NVFP4 plus
FP8 KV and TP2 a high-priority capacity candidate.

The same recipe's single-card Inferact configuration requires eager execution
to avoid graph-capture OOM on the tested build. Its text-only, reduced-sequence
configuration reports approximately 153k KV tokens, below native context.
This is not a universal TP1 engine limit, but it prevents promising native
context from that published configuration.

Use the model-appropriate reasoning and tool parsers. The current recipe
specifies `qwen3` reasoning and `qwen3_xml` tool parsing. Verify their behavior
against the pinned tokenizer/template and API contract.

### 5.2 SGLang

The official cookbook covers SM120 and RTX 5090, including RadixArk and NVIDIA
NVFP4 exports with FP8 attention KV. Its cited validation uses SGLang 0.5.19;
the broad validation sweep uses 8,192 input tokens, 1,024 output tokens, and
concurrency one, not native-context concurrent serving.

Key constraints:

- Published no-speculation 5090 configurations report approximately 69k-97k KV
  tokens, depending on recurrent-state precision. These are configuration
  results, not a universal limit.
- GDN recurrent-state pools and checkpoints can constrain concurrency before
  attention KV does.
- Embedded MTP uses EAGLE/NEXTN and ReplaySSM-related settings; include its
  state and workspace overhead in capacity accounting.
- The cited SM120 GDN prefill path uses Triton. Some Blackwell fast paths are
  SM100-only and cannot be assumed to work on SM120.
- BF16 recurrent-state configurations need workload-specific quality testing;
  reduced state precision is not just a free memory optimization.
- Small prefill chunks protect decode responsiveness; tight speculative
  configurations may require 512-1,024 rather than 2,048 tokens.

SGLang 0.5.20 includes relevant GDN, ReplaySSM, NVFP4, and cache changes. Test
the release, but do not turn kernel speedups into assumed end-to-end gains.

### 5.3 Earlier Local Comparisons Are Not Dispositive

The canonical September 19 alternatives report stopped its SGLang/vLLM
branches at short-context correctness gates. Those branches had thinking and
template issues; the vLLM forced-tool failure included a missing tool parser.
They do not establish native-context performance or general engine
unsuitability. Correct the serving contract before benchmarking.

Do not use experimental vLLM GGUF support as the representative vLLM baseline.
Use engine-native, compatible safetensors quantizations. Retain AWQ/GPTQ
Marlin W4A16 as a secondary candidate where model support and memory accounting
justify it; NVFP4 W4A4 and GGUF Q4 are different numerical/kernel paths.

Keep llamAmpere as the RTX 3090 profile and an optional RTX 5090 experiment.
Its SM86-oriented optimizations and pinned binaries are not an automatic
upgrade for SM120.

## 6. Architecture Candidates

| Architecture | Benefit | Limitation | Priority |
|---|---|---|---|
| Two TP1 replicas | Isolation, no inter-GPU synchronization, straightforward affinity | Duplicated weights reduce cache capacity | Reference and lowest-risk expansion |
| One TP2 server with continuous batching | Shared weight footprint, potentially more cache and faster prefill | Frequent communication over an unfavorable peer topology | Main challenger |
| One PP2 server | Shared weight footprint, fewer communication boundaries | Pipeline bubbles, single-stream latency, model/feature compatibility | Secondary |
| Separate prefill/decode GPUs | Can isolate prefill interference | Duplicated weights, state/cache transfer, only two GPUs | Not initial target |
| CPU/RAM offload | Larger nominal capacity | Transfer/recompute costs and latency variability | Cold-session experiment only |

Tensor parallelism and concurrent serving are not alternatives: a TP2 server
can continuously batch multiple requests, with both cards participating.
Independent replicas can also batch multiple requests if their memory permits.
Two independent TP2 replicas would oversubscribe this two-card workstation.

TP2 can win capacity and aggregate throughput even if it loses single-stream
decode speed. PP2 is worth a secondary check on a no-NVLink system, conditional
on support for this hybrid model, quantization, and selected speculative/cache
features.

### 6.1 Gateway and Admission

- Preserve stable session affinity to improve prefix reuse.
- Keep backend ports private; expose the authenticated gateway deliberately.
- Verify routing, cancellation, streaming, readiness, and failure behavior for
  each new engine rather than assuming OpenAI API compatibility is sufficient.
- Count explicit replica requests as well as pooled requests toward admission;
  the current singleton route can bypass pool accounting.
- For continuously batched engines, replace a fixed one-request limit only
  after measuring safe token/cache budgets and latency-qualified concurrency.
- Keep differently quantized or differently configured candidates under
  explicit model IDs during evaluation; do not silently pool unlike backends.
- Retain the no-replay-after-dispatch behavior for inference requests.

## 7. Memory Model and Tuning Order

Qwen3.8-27B has 64 layers: 16 full-attention and 48 Gated DeltaNet layers. The
attention layers use four KV heads with head dimension 256.

```text
attention KV bytes = tokens * 16 layers * 2(K,V) * 4 heads * 256 * bytes/value
```

At 262,144 tokens, per independent context:

| Attention KV format | Approximate payload |
|---|---:|
| BF16 | 16 GiB |
| FP8 | 8 GiB plus scale/layout overhead |
| llama.cpp Q8_0 | 8.5 GiB |
| Four-bit | 4 GiB before scale/layout overhead |

Exclude neither recurrent state/checkpoints nor graphs, workspaces, MTP,
weights, and allocator overhead from the final deployment budget. Startup
token figures for hybrid engines must be verified against their actual pool
semantics and populated requests.

Capacity targets:

- Two native contexts total: credible initial replica target, not yet a
  sustained two-card validation.
- Three native contexts total: worthwhile TP2 NVFP4 research target.
- Four native contexts total: stretch target, not a supported promise.

Two Q8 native contexts on one GPU require approximately 17 GiB of attention KV
alone. Larger host RAM does not remove this active-serving constraint.

Tune in this order:

1. Select and pin the checkpoint. Compare excluded layers, head/embedding
   precision, FP8 components, MTP tensors, and actual runtime footprint.
2. Establish target-only correctness with FP8 KV and verified scale handling.
3. Use text-only loading when vision is not required.
4. Test prefill chunk sizes around 2k and 4k, then 8k for prefill-oriented
   workloads. Smaller chunks generally protect ITL; larger chunks can improve
   prefill efficiency.
5. Set conservative request limits and graph capture sizes. Measure state and
   graph memory instead of inheriting large default concurrency limits.
6. Compare MTP off with 1-3 speculative tokens. Measure acceptance, per-user
   latency, and aggregate goodput, not decode tok/s alone.
7. Test lower-bit KV or recurrent-state precision only as separately
   quality-gated experiments.

### 7.1 Experimental Features and RAM Caching

Do not make NVFP4 KV the initial production assumption. NVFP4 weight kernels
and NVFP4 attention KV are distinct features. At the research date, vLLM
SM120 NVFP4 KV integration and correctness work remained open. Reported risks
include V-scale layout handling and a separate XQA/FULL CUDA-graph issue
affecting FP8 as well as NVFP4 KV on that path. Eager/piecewise tests are useful
controls, not a blanket cure for unrelated defects.

Host RAM can store inactive sessions or prefixes, but restoring a hybrid model
requires compatible recurrent state as well as attention KV. A recent SGLang
host-cache report concerns Qwen3.8-Flash-Next, a different hybrid model, and
must not be presented as a demonstrated Qwen3.8-27B failure. It does justify
explicit cache eviction/restoration correctness tests before relying on RAM
as a large session tier.

Prefix sharing can improve practical capacity for related conversations. It
must not be used to claim worst-case capacity for unrelated full contexts.

## 8. Benchmark Design

### 8.1 Selection Criterion

Maximize simultaneous near-native-context sessions satisfying agreed TTFT,
ITL, reliability, and quality requirements. Then maximize throughput within
that constraint. Report a Pareto frontier if no configuration dominates.

Set latency and quality thresholds before ranking results. Separate cold
native-context ingestion from warm interactive continuation; their acceptable
TTFT targets may differ. Queued sessions are not simultaneously active sessions.

### 8.2 Candidate Matrix

| Priority | Candidate |
|---|---|
| Reference | Pinned existing llama.cpp Q4/Q8, managed-memory variable absent, one GPU |
| Reference expansion | Same configuration as two independent GPU replicas |
| Main challenger | vLLM, Unsloth NVFP4, FP8 KV, TP2 |
| Matched engine comparison | SGLang, same compatible checkpoint, KV precision, topology, and output policy |
| Best deployment comparison | Each engine's best independently quality-approved checkpoint/settings |
| Secondary | TP1 AWQ/Marlin if memory accounting supports the required context |
| Secondary | PP2 after hybrid-model and feature compatibility checks |

If a common checkpoint is unsupported by one engine, record that compatibility
finding. Do not silently substitute a different quantization in the controlled
comparison. Use a separately labeled best-deployment comparison instead.

### 8.3 Workloads

- Populated depths around 8k, 64k, 128k, 196k, and near 262k.
- Reserve output and template overhead within the 262,144-token total budget.
- Concurrency 1, 2, 3, and 4 where memory permits; test short-context throughput
  at larger concurrency separately if relevant to procurement.
- Cold unique prompts, warm multi-turn continuation, and shared-prefix cases.
- Distinct prefixes for worst-case independent-session capacity tests.
- One long prefill arriving while another request is decoding.
- Meaningful output lengths, such as 512 and 2,048 tokens, not only 128 forced
  tokens.
- Normal-EOS quality workloads separate from fixed-output performance probes.
- Representative code, documents, tools, reasoning, and multi-turn agent traces.
- Cache churn followed by device-cache and host-cache restoration tests.

Use both controlled concurrency and an arrival-rate sweep for finalists to
identify queueing and service-level collapse under load.

### 8.4 Measurements

- Client-side streaming TTFT, p50/p95 ITL, and end-to-end latency.
- Per-user decode rate and completed requests per second.
- Aggregate generated tokens divided by a common measured wall interval.
- Cold prefill rate, actual cached tokens, and cache-reuse behavior separately.
- Queue time, rejection, timeout, cancellation, preemption, and recomputation.
- VRAM, attention KV capacity, recurrent-state allocation, graphs, workspaces.
- GPU temperature, power, throttling, PCIe behavior, and integrated energy.
- Speculative drafted/accepted tokens and acceptance by workload.
- Quality, tool contract validity, truncation, and incorrect cache restoration.

Do not sum timing-derived per-request decode rates when requests wait for
different intervals. Do not substitute warm-cache prefill for cold prefill.
Use identical measurement definitions across engines.

### 8.5 Quality and Reproducibility

- Pin model revisions, tokenizer, chat template, runtime/image, PyTorch,
  CUDA runtime, FlashInfer, and relevant kernel packages.
- Record actual selected kernels and precision, not just requested flags.
- Use equivalent sampling and thinking policies; distinguish thinking and
  non-thinking results. Match output budgets and reasoning preservation.
- Verify reasoning separation, automatic and forced tools, JSON contracts,
  normal EOS, and no hidden prompt truncation.
- Include long-context retrieval and representative end-to-end coding/agent
  tasks. Short-context GSM8K alone does not validate native-context quality.
- Test prefix reuse and eviction against cold controls, including recurrent
  state correctness.
- Use repeated runs and randomized engine-to-GPU placement where possible.
- Keep raw telemetry and result manifests; put stable summaries and artifact
  identifiers in the repository without secrets or proprietary prompts.

### 8.6 Sustained and Operational Validation

Run finalists under sustained two-card load, not just brief isolated requests.
Agree on the thermal stop policy first; do not raise the existing cutoff to
make a candidate pass. Verify chassis cooling and account for power supply
and shared-workload constraints.

Test startup, restart, cancellation, backend failure, gateway behavior,
monitoring, and rollback. A benchmark winner is not production-approved until
these checks pass.

## 9. Corporate SGLang versus vLLM Evaluation

Maintain two separate comparisons:

1. Controlled engine comparison: identical checkpoint, precision, tokenizer,
   template, workload, topology, and generation policy.
2. Best achievable deployment: each engine's best quality-approved settings
   under identical hardware and service-level constraints.

The first explains differences; the second informs purchasing and deployment.
Report service-level-qualified goodput per dollar and per watt, failure
recovery, observability, upgrade/reproducibility costs, and operational
complexity alongside throughput.

Repeat finalists on the proposed procurement hardware. A dual-5090 result
cannot establish relative performance on large-VRAM GPUs or NVLink-connected
SM100 systems. SM120 workstation Blackwell and SM100 datacenter Blackwell
must remain separate hardware categories in the report.

## 10. Execution Phases and Gates

### Phase A: Preserve and Consolidate Knowledge

- [ ] Preserve the old checkout's local modifications and untracked files.
- [ ] Add an opt-in RTX 5090 profile to the canonical repository.
- [ ] Keep existing RTX 3090, gateway, and monitoring behavior intact.
- [ ] Resolve profile conflicts, GPU ownership, path assumptions, and log rules.
- [ ] Correct residency claims and document historical/staged/live status.

Gate: reviewed repository-only changes; no service activation or resource
policy changes as a side effect of consolidation.

### Phase B: Revalidate the Baseline

- [ ] Agree on latency, quality, thermal, and rollback criteria.
- [ ] Inspect current workload ownership and preserve unrelated services.
- [ ] Correct the managed-memory environment in a controlled configuration.
- [ ] Confirm runtime/model provenance, backend exposure, and cooling policy.
- [ ] Reproduce populated native-context behavior and strict-residency evidence.

Gate: trustworthy baseline with known sampling, memory, and thermal behavior.

### Phase C: GPU 1 Feasibility

- [ ] Use isolated environments, private ports, and explicit GPU selection.
- [ ] Leave the GPU 0 service available; monitor host-wide thermal interference.
- [ ] Test vLLM/SGLang TP1 loading, selected kernels, parsers, and bounded quality.
- [ ] Measure memory allocation before attempting long-context requests.
- [ ] Reject incompatible or incorrect candidates before expensive sweeps.

Gate: pinned, correct candidate stacks and a justified two-GPU test shortlist.

### Phase D: Two-GPU Benchmark Window

- [ ] Schedule a maintenance window before consuming both production GPUs.
- [ ] Establish and verify rollback before stopping the existing model service.
- [ ] Compare two replicas against matched vLLM/SGLang TP2 candidates.
- [ ] Test capacity, prefill/decode interference, and sustained operation.
- [ ] Add PP2 or other secondary candidates only when the main results justify it.

Gate: reproducible performance/quality results and measured capacity at the
agreed latency thresholds, not startup estimates.

### Phase E: Canary and Promotion

- [ ] Select the measured Pareto winner or separate latency/capacity profiles.
- [ ] Integrate authenticated routing, admission, and per-engine metrics.
- [ ] Canary under an explicit model ID before replacing a stable alias.
- [ ] Verify native context, tools, cache correctness, restart, and rollback.
- [ ] Record exact deployed artifacts and acceptance evidence.

Gate: explicit production acceptance. Retain the known-good reference for
rollback; no automatic substitution between unlike checkpoints or runtimes.

## 11. Sources

### Local Evidence

- [Canonical operations and staged status](../operations/README.md).
- [Prior serving alternatives report](qwen3.8-27b-serving-alternatives-final-report.md).
- [llamAmpere upgrade plan](qwen3.8-27b-llamampere-production-upgrade-plan.md).
- [Canonical Muse unit](../operations/systemd/llama-muse-glimmer-30b-131k.service).
- Old checkout: `/home/michel/LLMs-tests/localLLMinference-old/`.
- Workstation benchmarks: `/home/michel/LLMs-tests/rtx5090-qwen38-bench/`.
- Allocator semantics: `/home/michel/LLMs-tests/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:140-165`.
- Live service: `/etc/systemd/system/llama-qwen3.8-q4-native.service`.

### Upstream Sources Checked 2026-09-20

- [Official Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B).
- [vLLM Qwen3.8-27B recipe, including RTX 5090](https://recipes.vllm.ai/Qwen/Qwen3.8-27B).
- [SGLang Qwen3.8-27B cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B).
- [Unsloth NVFP4 configuration](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4/blob/main/config.json).
- [Inferact NVFP4 configuration](https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4/blob/main/config.json).
- [vLLM 0.29.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.29.0).
- [SGLang 0.5.20 release](https://github.com/sgl-project/sglang/releases/tag/v0.5.20).
- [vLLM parallelism and scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/).
- [vLLM optimization and chunked prefill](https://docs.vllm.ai/en/latest/configuration/optimization/).
- [vLLM quantized KV cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/).
- [vLLM GGUF support caveats](https://docs.vllm.ai/en/latest/features/quantization/gguf/).
- [SGLang server arguments](https://docs.sglang.ai/advanced_features/server_arguments.html).
- [llamAmpere upstream](https://github.com/JakeATX/llamAmpere).
- [vLLM SM120 NVFP4 KV integration issue #49011](https://github.com/vllm-project/vllm/issues/49011).
- [vLLM NVFP4 KV integration PR #46329](https://github.com/vllm-project/vllm/pull/46329).
- [vLLM V-scale layout issue #50084](https://github.com/vllm-project/vllm/issues/50084).
- [vLLM XQA FULL-graph issue #49010](https://github.com/vllm-project/vllm/issues/49010).
- [SGLang hybrid host-cache issue #39830, different Qwen model](https://github.com/sgl-project/sglang/issues/39830).
