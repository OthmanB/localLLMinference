# Qwen3.8-27B RTX 5090 Phase I Cached Continuation QoS

Date: 2026-09-22

Status: A realistic cached 196k conversation continuation avoids the cold-arrival
QoS failure. The result is an isolated graph B0 mechanism qualification, not a
production approval: GSM8K remains below gate and no routing or service change
is authorized.

## Workload

The run retained the Phase H graph B0 runtime: vLLM
`0.29.1rc1.dev438+g01f1f58f1`, Torch `2.13.0+cu132`, CUDA `13.2`, TP2 on GPUs
0 and 1, calibrated FP8-KV export (config SHA-256
`731d9ea745aa5c171822e4e64b4a863ae3269620a43ca2bec8a1e42c13a89b68`), Triton
attention/GDN prefill, native sampler, PYNCCL fallback, and FULL_DECODE_ONLY
graphs with capture sizes 1 and 3. It used explicit prefix caching,
`max_num_batched_tokens=4096`, `long_prefill_token_threshold=256`, chunked
prefill, 500 W card limits, and an 85 C stop.

The benchmark does not compare two identical prompts. It models two distinct
users:

1. B first creates a 196,005-content-token conversation cache entry with a
   deterministic normal-EOS response.
2. A starts a distinct, cache-cold 195,992-content-token request with a forced
   8,192-token response.
3. After A's first token plus 20 seconds of established decode, B submits a
   real multi-message next turn: the cached user message, B's known prior
   assistant response, and a new 2,040-content-token user message.

All requests had distinct leading markers where appropriate. This prevents A
from accidentally sharing B's prefix while preserving B's actual conversation
history in the continuation request.

## Cache Contract

The harness rejects the QoS result unless all five statements hold. They did:

| Guard | Evidence |
| --- | --- |
| Prefix caching enabled | `vllm:cache_config_info` recorded `enable_prefix_caching="True"` |
| B's base created locally | 196,016 local-compute prompt tokens; zero cache hits |
| A was cold relative to B | 196,003 local-compute tokens; zero local cache hits |
| B's continuation reused B locally | 196,000 prefix/local-cache/cached tokens; 2,075 local-compute tokens; zero external hits |
| Both contracts completed | B returned exact normal EOS text; A returned all 8,192 forced tokens by length |

The result therefore measures locality rather than an accidental cache miss or
shared prefix between A and B.

## QoS Result

| Measure | Cached continuation result |
| --- | ---: |
| A p95 ITL before B continuation | 0.0336 s |
| A p95 ITL during B continuation | 0.2646 s |
| A full-request decode rate | 58.26 tok/s |
| B continuation TTFT | 2.468 s |
| B continuation wall | 2.556 s |
| B local prefill work | 2,075 tokens |
| B local cache reuse | 196,000 tokens |

The 265 ms established-user p95 is below the 300 ms target and is qualitatively
different from the independent cold-196k case at the same 4096/256 scheduler
profile: that workload gave the arriving user 155.69 s TTFT and raised the
established user's p95 ITL to 426 ms. B's cached continuation reached first
token in 2.47 s because its actual local prefill was only about 2k tokens.

Both TP ranks captured FULL B=1/B=3 graphs. Runtime statistics recorded FULL
B=1 and padded two-request-to-B=3 execution, with no graph fallback or
pre-shutdown error. Peak GPU0/GPU1 VRAM was 29,135/28,326 MiB, peak temperatures
were 71/57 C, and peak power was 491.82/496.83 W. The known `EngineDeadError`
appeared only after harness-initiated vLLM shutdown.

## Decision

- Prefix-cache locality materially changes the operational result and is a
  higher-value lever than further decode or MTP work for realistic agent turns.
- Retain `4096/256` as the next scheduler/cache experimental profile: it does
  not meet the cold-independent-arrival 300 ms target, but does meet it for the
  cache-local 2k continuation.
- Keep a cold 196k arrival as an admission-control case. Scheduler thresholding
  alone did not reduce its established-user p95 below 300 ms; the 128-token
  cap still measured 348 ms while increasing new-user TTFT to 257.59 s.
- For a future gateway qualification, combine scheduler protection, cache
  locality, and explicit cold-context admission/queueing. `max_num_queued_tokens`
  is a candidate TTFT backpressure control, but it was not configured in this
  isolated run and must not be introduced into production without a separate
  gateway/load test.
- MTP remains closed for this vLLM/runtime/model path. TP1 and DFlash remain
  out of scope.

## Artifact

```text
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-i-graph-b0-cached-continuation-append2048-threshold256-vllm-2026-09-21T225815Z/
```
