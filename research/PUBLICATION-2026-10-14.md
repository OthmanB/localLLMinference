# Historical benchmark publication — 2026-10-14

This archive adds previously unpublished benchmark sources and filtered
measurements from September 4 through October 14. It also retains the existing
August context/runtime and SWE-bench reports. Publication did not rerun inference.

## Coverage

| Period | Published material |
|---|---|
| September 4–8 | Large-MoE, Mistral, DeepSeek and FreeToken probe/sweep sources; available scalar results; Flash NVFP4 2048 profile report |
| September 9 | Qwen Q4 layer/tensor, NCCL, batch/KV, long-output, parallel-request and 524k-window experiments; Q5/Q8 controls; plans, results and per-run aggregates |
| September 11–12 | DFlash blocked-run status aggregates alongside the existing blocker report |
| September 18–19 | Serving alternatives (llamAmpere, Bonsai, vLLM), synthetic quality rubric/follow-ups, topology, batch6144 and native 262k validation aggregates |
| September 30 | Flash GSQ-RCO Coder report, HTTP probes, authored quality suite and orchestration benchmark source |
| October 2 | HyperQwen/KVarN batch, MTP, DFlash k7/k15/int4; matched ATX MTP/ngram/hybrid native 262k sources and aggregate measurements |
| October 14 | Strata Q4/IQ3_S native screens; IQ3_S native/YaRN controls, 384k, 512k and independent 512k repetition; fixture generation, grading and summary sources |

The exact file inventory and hashes are in
[`publication-manifest-2026-10-14.json`](publication-manifest-2026-10-14.json).
Each experiment's `aggregate-measurements.json` identifies the original result
file and retains numeric/bool measurements and a small set of categorical labels.
It excludes response text, raw prompts, token arrays, process metadata, command
output and resource time-series. Fields retain their original units; absent fields
are omitted, not zero. Some failed trials retain only status; consult their report
for the failure explanation. Duplicate summary/per-task records are not independent
repeats and must not be pooled as such.

## Reproducing the source harnesses

Python harnesses are archived under their original `tools/` and `research/`
locations. Model/runtime assets are external prerequisites. Historical absolute
locations are replaced with explicit environment variables, preserving their
relative suffixes:

```bash
export EXPERIMENT_ROOT="$PWD"       # run from the public checkout root
export RUNTIME_ROOT="/path/to/runtimes"
export MODEL_STORE="/path/to/model-data"
```

`RUNTIME_ROOT` is the parent of the named runtime directories, such as
`strata-flash-next-20261014` and `llama.cpp-recent-20260905`. `MODEL_STORE` is
the historical model-data parent; individual older harnesses also use it as a
direct model directory override, so inspect each model-path definition before use.
Use each harness's CLI to override paths where supported. Bonsai sources expect
their pinned builds/models at the relative paths listed in the script; provide
those externally or configure those constants. None of these assets is included.

GPU indices, loopback ports, runtime flags and read-only service-health checks
retain the historical profiles. Select an isolated test machine and configure
the profiles for it. HyperQwen/Strata sources use GPU0, ports 18120/18144 and
read-only checks for the historical Muse/Qwen unit names. They do not operate
those services. Their runners can launch their own benchmark child processes;
Strata `execute.py` also prepares the compact comparison model if explicitly run.
Its historical preparation-unit check requires local setup. These are source
harnesses, not turnkey installers. No benchmark was executed during publication.

Typical entry points:

- Qwen multi-GPU: `python3 tools/qwen_q4_dual_gpu_layer_benchmark.py --help`.
- Synthetic serving quality:
  `python3 research/qwen3.8-27b-serving-alternatives-phase1-quality-20260918T141239Z/quality_gate.py --init-manifest`
  generates its authored test manifest locally without inference. Follow-up
  scripts require that manifest; raw historical manifests are omitted.
- GSQ Coder: `probes.py` accepts probe kind, endpoint and model as positional
  arguments. The quality-suite Markdown contains the authored test instructions;
  its grader commentary should be excluded from model-facing review prompts.
- HyperQwen: `native_probe.py` builds fresh near-full native prompts with the
  supplied tokenizer and checks actual server token usage. `benchmark_native.py`
  and `benchmark_atx.py` implement the matched longer-output/copy workloads.
  `run-gpu0.sh` records the runtime launch profile and requires `HYPERQWEN_ROOT`.
- Strata: `probe.py` and `context_probe.py` issue requests to a separately
  configured endpoint. The latter uses Strata's tokenizer and upstream authored
  quant-quality corpus. `extend_iq3.py` constructs unique synthetic documents and
  grades exact facts/citations; `grade_extensions.py` checks generated LRU code.
  The validated profile is recorded in `validated-iq3-512k-profile.json` as portable
  options, not as a loader config containing local asset paths.

Dependencies vary: the HTTP-only probes generally use the standard library;
Strata runner/resource accounting needs `psutil`, HyperQwen prompt construction
needs `transformers` and the pinned tokenizer, and inference requires each pinned
runtime's own environment. Summary/grading scripts expect local raw results from
a new run. Historical raw inputs/outputs are deliberately absent.

## Interpretation

- HyperQwen did not meet the joint native-capacity/prefill/sustained-decode gate.
  ATX MTP achieved about 495/45 tok/s prefill/noncopy decode on the matched workload;
  DFlash k7/k15 achieved about 588/15.8 and 585/13.5. Short 22-token answers are
  screening data, not sustained decode evidence. Quantizations differ.
- GSQ Coder retrieved the marker at native window size but populated-native
  decode was only 4.4 tok/s. The report distinguishes repaired-weight results
  from the earlier corrupt-download failure.
- Strata IQ3_S's initial native screen achieved 1915.7/78.6 tok/s but hit its
  output cap; that is not a natural-completion quality pass. The later 512k YaRN
  trials used 515650 uncached input tokens, 1629.9–1636.4 prefill tok/s and
  58.7–59.8 document decode tok/s. Coding continuations intentionally used cached
  history (73.4–73.7 tok/s). Both document and independently graded code answers
  completed naturally. Extension is distinct from native context and from broad
  agent evaluation.

## Omitted artifacts

Weights, packed experts, model checkouts/builds, virtual environments and caches
are excluded (some individual models are tens to hundreds of GB). Raw prompts,
responses, SSE/agent trajectories, datasets/gold patches, server/machine logs and
resource JSONL are excluded. Authored synthetic fixture generators and test
instructions are included as reproducible source. Deployment/preparation and
service start-stop wrappers are excluded; the HyperQwen wrapper is an explicitly
configured benchmark launcher with read-only service checks. The upcoming
qwen27-vs-flash comparison is outside this archival export.
