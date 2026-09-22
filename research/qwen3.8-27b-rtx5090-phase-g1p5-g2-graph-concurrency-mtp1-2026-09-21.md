# Qwen3.8-27B RTX 5090 Phase G1.5/G2 Graph Concurrency And MTP1

Date: 2026-09-21

Status: G1.5 establishes the no-speculation B0 graph behavior for three
simultaneous 196k arrivals. G2 rejects graph MTP1 at C=1. Neither profile is
production-approved; llama.cpp Q4 remains the live service.

## Scope

Every run used the existing guarded maintenance wrapper, loopback-only TP2 on
physical GPUs 0 and 1, the calibrated FP8-KV export (config SHA-256
`731d9ea745aa5c171822e4e64b4a863ae3269620a43ca2bec8a1e42c13a89b68`), an
85 C stop, and unchanged 500 W card limits. The unrelated GPU-0 RAG PID
2598220 remained allowlisted. The reference llama.cpp service and its monitor
were restored after every window.

The isolated runtime is vLLM `0.29.1rc1.dev438+g01f1f58f1`, Torch
`2.13.0+cu132`, CUDA `13.2`, TP2, FP8 E4M3 KV, Triton attention, Triton GDN
prefill, native sampler, PYNCCL fallback, disabled prefix caching, and
`VLLM_USE_BREAKABLE_CUDAGRAPH=0`. No FlashInfer attention/sampler, driver,
power, routing, or communication experiment was introduced.

## G1.5: B0 C=3 At 196k

The candidate used `max-num-seqs=3`, `FULL_DECODE_ONLY`, and explicit
`--cudagraph-capture-sizes 1 3`. Three distinct 196,004-content-token prompts
were submitted concurrently, each with a forced 4,096-token response. Prefix
caching was disabled and no request was warmed: all 588,078 prompt tokens were
recorded as local compute, with zero prefix-cache queries, hits, or local-cache
tokens.

| Stream | Decode tok/s from first token to finish | TTFT | Client p95 ITL |
| --- | ---: | ---: | ---: |
| decode-0 | 23.57 | 187.67 s | 0.0403 s |
| decode-1 | 15.36 | 92.91 s | 0.0627 s |
| decode-2 | 51.61 | 283.66 s | 0.0391 s |
| Median / sum | 23.57 / 90.54 | median 187.67 s | median 0.0403 s |

All three streams ended by `length`, generated 12,288 tokens total, and had no
request error. The client-rate spread was 3.36x. It is real interference from
simultaneous cold long prefills: requests begin decoding at different times
while the remaining 196k prompts are still being computed. It must not be
reported as three equal 51 tok/s interactive sessions.

Once all three requests were active, the server recorded seven ten-second
intervals with `Running: 3`, FULL B=3 replay, and aggregate generation
throughput from 153.4 to 154.7 tok/s (median 154.2 tok/s). This establishes the
important mechanism: the serving batch really captures and replays B=3 rather
than silently using B=1 or eager execution. The client score and the stable
FULL-B=3 interval answer different questions and are both retained.

Both TP ranks captured B=1 and B=3. Runtime statistics recorded FULL B=3 counts
of 170, 515, 516, 514, 515, 512, 514, 513, and 134, then correctly used padded
B=3 and B=1 graphs as requests finished. There were no fallback markers or
pre-shutdown error/traceback lines.

| Resource | GPU 0 | GPU 1 |
| --- | ---: | ---: |
| Peak VRAM used | 29,367 MiB | 28,564 MiB |
| Minimum free VRAM | 3,240 MiB | 4,043 MiB |
| Peak temperature | 74 C | 58 C |
| Peak reported power | 480.00 W | 474.78 W |

The first G1.5 startup-only attempt
`phase-g1p5-graph-b0-c3-196k-vllm-2026-09-21T024547Z` is excluded. It made no
model request and was stopped by a harness false positive matching the DEBUG
environment key `VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY`. The evidence parser now
matches actual ERROR-level log records or traceback headers. The succeeding run
uses the corrected guardrail.

## G2: MTP1 Graph At C=1

MTP1 changes the uniform decode query geometry to two tokens for one request.
The preflight therefore used explicit capture size 2, rather than reusing B0's
size 1. It passed repeated deterministic `phase-d-ok` responses, the exact
`lookup_symbol({"symbol":"Record"})` tool contract, no graph fallback, no
pre-shutdown workload error, and FULL runtime replay at one request/two tokens.

The cache-disabled 196k C=1 measurement used the same graph profile with native
`--spec-method mtp --spec-tokens 1`, one 196,004-content-token prompt (196,026
API local-compute tokens), and a forced 4,096-token response.

| Profile | Decode tok/s | TTFT | p95 ITL | Drafted | Accepted | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Graph B0 | 59.52 | 93.84 s | 0.0344 s | n/a | n/a | n/a |
| Graph MTP1 | 14.36 | 98.09 s | 0.1523 s | 2,113 | 1,983 | 93.85% |

MTP1 is 75.87% slower than graph B0 and its p95 ITL is 4.43x higher. The run is
not an eager or fallback artifact: both TP ranks captured the one-request,
two-token graph and runtime statistics repeatedly recorded FULL `2 -> 2`
replay. Prefix-cache queries/hits and local-cache tokens were zero, all 4,096
completion tokens were returned by length, and there were no request errors.

MTP1's high acceptance does not offset its graph-path overhead. The result is
well beyond the allowed within-10% MTP1 exception, so MTP is closed for this
vLLM/runtime/model path. Do not test MTP2, MTP3, C=3 MTP, or 250k MTP. The next
useful work is cold-prefill admission, not deeper speculation or topology
experiments.

| Resource | GPU 0 | GPU 1 |
| --- | ---: | ---: |
| Peak VRAM used | 29,413 MiB | 28,554 MiB |
| Minimum free VRAM | 3,194 MiB | 4,053 MiB |
| Peak temperature | 63 C | 54 C |
| Peak reported power | 473.35 W | 476.45 W |

## Artifacts

```text
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-g1p5-graph-b0-c3-196k-vllm-2026-09-21T025031Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-g2-graph-mtp1-smoke-vllm-2026-09-21T030528Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-g2-graph-mtp1-rawdecode196k-vllm-2026-09-21T031011Z/
```
