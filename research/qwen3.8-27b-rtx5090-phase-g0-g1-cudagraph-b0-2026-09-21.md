# Qwen3.8-27B RTX 5090 Phase G0/G1 CUDA-Graph B0

Date: 2026-09-21

Status: completed for isolated C=1 B0 graph qualification. The graph profile is
operationally viable at 8k, 196k, and 250k with an exact 249k retrieval result.
Subsequent C=3 B0 and C=1 MTP1 graph evidence is recorded in
`qwen3.8-27b-rtx5090-phase-g1p5-g2-graph-concurrency-mtp1-2026-09-21.md`.
Neither result is a production authorization.

## Scope And Safety

This phase evaluated only no-speculation B0 on an isolated nightly vLLM runtime.
It did not install a vLLM unit, alter routing, change Docker, alter drivers or
power limits, or run MTP1/2/3. Every run used the guarded maintenance wrapper,
TP2 on physical GPUs 0 and 1, loopback-only serving, an 85 C stop, and the
calibrated export below. The reference llama.cpp service and monitor were
restored at each exit. The unrelated GPU-0 RAG process PID 2598220 was
allowlisted and remained running.

```text
model: /home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-e-kv-fp8-calibration-2026-09-20T052906Z/export
config SHA-256: 731d9ea745aa5c171822e4e64b4a863ae3269620a43ca2bec8a1e42c13a89b68
runtime: vLLM 0.29.1rc1.dev438+g01f1f58f1
torch/CUDA: 2.13.0+cu132 / 13.2
FlashInfer package: 0.6.18.post1, but no FlashInfer attention or sampler path
TP: 2; max-num-seqs: 1; KV cache: fp8_e4m3; prefix caching: disabled
attention: {"backend":"TRITON_ATTN","use_trtllm_attention":false}
GDN prefill: triton
sampler: VLLM_USE_FLASHINFER_SAMPLER=0
communication: PYNCCL fallback; FlashInfer all-reduce disabled at world size 2
```

The cards have no usable P2P (`GNS` for peer read and write), so no PCIe IPC,
custom peer all-reduce, or driver experiment was introduced.

## G0 Research And Candidate

The upstream vLLM tree was inspected at
`17e50b9b761023d5f2499f062507df1ff49092a4`, together with the current
optimization documentation at
`https://docs.vllm.ai/en/latest/configuration/optimization/` and upstream issues
`#49010` and `#49011`.

The calibrated model identifies as `model_type=qwen3_5` and
`Qwen3_5ForConditionalGeneration`, with hybrid GDN layers. The isolated nightly
source supports uniform-batch decode CUDA graphs for this architecture. The
smallest attributable profile is therefore FULL decode-only graphs for batch
size one, while retaining Triton for GDN prefill:

```text
--cudagraph-metrics
--cudagraph-capture-sizes 1
--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}
--gdn-prefill-backend triton
VLLM_USE_BREAKABLE_CUDAGRAPH=0
```

The prior pinned eager constraint was specific to the automatic FlashInfer/XQA
path on SM120: automatic attention selected FlashInfer, its JIT did not support
the local Blackwell path, and automatic top-k/top-p sampling also needed to be
disabled. The graph candidate does not re-enable either path. In particular,
the open FULL-graph XQA correctness issue `#49010` is not treated as fixed or
irrelevant; it is avoided by explicit Triton attention and the disabled
FlashInfer sampler.

The current nightly logs FlashInfer as unavailable for runtime JIT because
`nvcc` and pre-downloaded cubins are absent. This does not affect the selected
candidate because it explicitly uses Triton attention, Triton GDN prefill, the
native sampler, and PYNCCL.

## Harness Improvements

`tools/rtx5090_phase_d.py` now records runtime package versions, accepts
allowlisted graph controls, records capture/fallback evidence, and requires
post-work FULL runtime statistics for a graph result. Its smoke contract now
requires two identical temperature-zero normal responses, an exact forced tool
call, FULL replay evidence, and no error before server shutdown.

The wrapper now runs the candidate in its own session, tracks the candidate
process group, and forwards termination to it before restoring production. The
Python runner turns SIGINT/SIGTERM into controlled cleanup so its separately
sessioned vLLM process group is stopped. Validation after these changes passed:

```text
python3 -m py_compile tools/rtx5090_phase_d.py
python3 -m pytest tools/tests/test_operations_scripts.py  # 7 passed
bash -n operations/run-rtx5090-phase-d-benchmark.sh
```

The nightly still writes an `EngineDeadError` only after the harness has sent
SIGTERM for intentional vLLM teardown. It occurs after each successful request
and after the server's `[shutdown]` marker, including eager controls. It is not
counted as a workload error. The new evidence record separately captures all
pre-shutdown `ERROR` and traceback lines; they were empty for the qualifying
smoke, 250k, and retrieval runs.

## Correctness Evidence

The repeated calibrated 8k smoke passed with FULL B=1 capture on both TP ranks
and runtime graph statistics. The two normal requests returned the same SHA-256
and exact text `phase-d-ok`; the forced tool request produced exactly one
`lookup_symbol({"symbol":"Record"})` call.

| Contract | Result |
| --- | --- |
| Repeated normal EOS | Passed, both hashes `82ab982deddf07640f2269e8f5ffbd4896927a7d8c8cf96cd81ac0ed2b922f98` |
| Forced tool | Passed, `lookup_symbol` with `{"symbol":"Record"}` |
| FULL graph capture/replay | Passed; both ranks captured B=1 and runtime stats reported FULL execution |
| Graph fallback | None |
| Workload errors before shutdown | None |
| 249k buried-fact retrieval | Passed; exact `phase-d-retrieval-raven-73` in 135.67 s |

## C=1 B0 Results

All decode measurements are cache-neutral: prefix caching was disabled, all
prompt tokens were local compute, and local/external cache-hit deltas were zero.
The prefill figure is API prompt tokens divided by client TTFT; it is provided
only as a comparable end-to-first-token approximation.

| Prompt workload | API prompt tokens | Forced output | Decode tok/s | TTFT | Approx. prefill tok/s | p50 ITL | p95 ITL | Request wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8k graph B0 | 8,026 | 64 | 107.61 | 2.26 s | 3,551.53 | 0.0094 s | 0.0190 s | 2.85 s |
| 196k graph B0 | 196,026 | 4,096 | 59.52 | 93.84 s | 2,089.04 | 0.0171 s | 0.0344 s | 162.66 s |
| 250k graph B0 | 250,026 | 4,096 | 52.94 | 137.06 s | 1,824.17 | 0.0189 s | 0.0377 s | 214.44 s |

The 196k run manually verified FULL B=1 runtime samples through the entire
decode, with periodic counts near 593-595. The 250k run recorded periodic FULL
B=1 counts of 68, 532, 531, 529, 528, 528, 529, 528, and 322. The changing
counts are vLLM's interval snapshots, not a total counter; every sample remained
FULL with no fallback.

Peak observed telemetry remained below the 85 C stop condition and left more
than 3 GiB free per card:

| Workload | Peak VRAM used | Minimum free VRAM | Peak temperature | Peak reported power | Peak GPU utilization |
| --- | ---: | ---: | ---: | ---: | ---: |
| 196k graph B0 | 29,407 MiB | 3,200 MiB | 63 C | 473.61 W | 100% |
| 250k graph B0 | 29,461 MiB | 3,146 MiB | 68 C | 492.07 W | 100% |
| 249k retrieval | 29,419 MiB | 3,188 MiB | 72 C | 503.27 W | 100% |

The reported 503.27 W retrieval sample is telemetry above the configured 500 W
limit, not a changed power limit. No thermal stop was triggered.

## Matched 8k Eager Control

The graph and eager controls used the same isolated nightly, export hash, TP2,
Triton attention, Triton GDN prefill, FP8 KV, disabled prefix cache, native
sampler, 8,026 API prompt tokens, fixed 64-token output, and `max-num-seqs=1`.
The sole profile difference is graph compilation/FULL decode replay versus
`--enforce-eager`.

| Setting | Decode tok/s | TTFT | p50 ITL | p95 ITL | Request wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| FULL_DECODE_ONLY graph | 107.61 | 2.26 s | 0.0094 s | 0.0190 s | 2.85 s |
| `--enforce-eager` | 0.188 | 23.15 s | 5.4170 s | 10.7985 s | 363.86 s |

The arithmetic 572.87x decode and 569.71x p95-ITL differences to the nightly
eager control are retained only as a diagnostic: that eager path is pathological
on this host and must not be presented as CUDA graphs' attributable multiplier.
The meaningful historical cache-neutral C=1 comparison is the pinned B0 result
of 16.53 tok/s against this graph B0 result of 59.52 tok/s, or approximately
3.60x. The earlier nightly 196k eager attempt likewise reached only 0.1-0.2
tok/s after prefill and did not complete its requested 4,096 tokens in the
maintenance window. It has no `results.json` and is retained only as an aborted
operational observation, not a numeric benchmark.

The forced-length 64-token graph and eager continuations have different content
hashes. They are not correctness probes because `ignore_eos=true` deliberately
continues past the requested phrase. The graph profile's exact deterministic
normal-text, forced-tool, and 249k retrieval contracts above are the correctness
evidence.

## Excluded Attempts

- `phase-g1-graph-b0-smoke-vllm-2026-09-20T232706Z` used the uncalibrated
  default model and automatic FlashInfer attention. It is not comparable.
- `phase-g1-eager-b0-rawdecode196k-nightly-vllm-2026-09-20T234806Z` was
  interrupted after an outer maintenance timeout; its workers were manually
  terminated. It produced no result summary and motivated the teardown repair.
- `phase-g1-graph-b0-8k-control-vllm-2026-09-21T004306Z` was rejected before a
  candidate launch because validation incorrectly checked unused workload
  defaults. Validation is now mode-aware.
- `phase-g1-graph-b0-8k-control-vllm-2026-09-21T004510Z` also predates the
  post-work graph-stat wait. Its short request finished before vLLM's periodic
  logger could emit replay statistics and is excluded in favor of the completed
  `T004929Z` control.

## Decision

1. Accept the isolated FULL_DECODE_ONLY B=1 Triton graph profile as the current
   C=1 B0 graph baseline. It passes deterministic text, tool, retrieval,
   capture/replay, resource, and matched eager-control checks at 8k, 196k, and
   250k.
2. Do not alter production or route this profile. The calibrated export still
   fails its recorded GSM8K promotion gate at 122/128 with three truncations.
3. The subsequent G1.5 C=3 graph run uses explicit B=1 and B=3 captures,
   cache-disabled concurrent 196k inputs, and per-stream evidence. Its result
   is deliberately separate from this C=1 baseline.
4. Native MTP1 was subsequently tested under the graph profile. It passed its
   B=1 two-token FULL replay and correctness preflight but regressed severely at
   C=1 196k; MTP2 and MTP3 are not authorized. See the G1.5/G2 report.

## Qualifying Artifacts

```text
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-g1-graph-b0-smoke-repeat-vllm-2026-09-21T010353Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-g1-graph-b0-8k-control-vllm-2026-09-21T004929Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-g1-eager-b0-8k-control-vllm-2026-09-21T005147Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-g1-graph-b0-rawdecode196k-vllm-2026-09-20T233943Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-g1-graph-b0-rawdecode250k-vllm-2026-09-21T010639Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-g1-graph-b0-retrieval249k-vllm-2026-09-21T011218Z/
```

Each candidate directory contains the launched command, environment record,
runtime versions, model hash, metrics snapshots, GPU/host telemetry, server
log, and maintenance-window restoration log.
