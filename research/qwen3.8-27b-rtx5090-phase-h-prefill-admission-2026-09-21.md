# Qwen3.8-27B RTX 5090 Phase H Prefill Admission And Decode QoS

Date: 2026-09-21

Status: `max_num_batched_tokens=4096` with
`long_prefill_token_threshold=1024` is the fair three-cold-arrival graph B0
profile. It is not production-approved: a new 196k prefill still raises an
established user's p95 ITL from 35 ms to 1.48 s. MTP is closed for this vLLM
runtime/model path; TP1 and DFlash are out of scope.

## Scope

Every run used the guarded maintenance wrapper, loopback-only TP2 on physical
GPUs 0 and 1, the calibrated FP8-KV export (config SHA-256
`731d9ea745aa5c171822e4e64b4a863ae3269620a43ca2bec8a1e42c13a89b68`), an
85 C stop, unchanged 500 W card limits, Triton attention, Triton GDN prefill,
native sampling, PYNCCL fallback, disabled prefix caching, and
`FULL_DECODE_ONLY` CUDA graphs with explicit capture sizes 1 and 3. The
isolated runtime was vLLM `0.29.1rc1.dev438+g01f1f58f1`, Torch
`2.13.0+cu132`, and CUDA `13.2`. The unrelated GPU-0 RAG PID 2598220 remained
allowlisted, and the reference llama.cpp service plus monitor were restored
after every window.

No MTP, TP, attention, sampler, communication, power, driver, routing, or
cache setting changed. `--enable-chunked-prefill` was explicit for every Phase H
run. The only scheduler settings under test were
`--max-num-batched-tokens` and, after the budget sweep,
`--long-prefill-token-threshold`.

## H1: Token Budget Sweep

Each point submitted three distinct simultaneous 196,000-content-token cold
prompts with 4,096 forced output tokens, `max-num-seqs=3`, and no warmup. Every
run completed by length, had zero prefix-cache queries/hits/local-cache tokens,
captured FULL B=1 and B=3 on both TP ranks, replayed FULL B=3 at runtime, and
remained below the 85 C stop.

The client p95 during another user prefill is deliberately separate from each
stream's whole-response p95. Whole-response p95 hides rare but operationally
visible multi-second pauses after long stable decode periods.

| Budget | Median TTFT | Worst TTFT / all-enter-decode | p95 ITL while another cold prefill runs | Stable B=3 aggregate | Completion wall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 194.67 s | 294.65 / 294.66 s | 1.50-1.90 s | 154.1 tok/s | 374.12 s |
| 4,096 | 186.82 s | 279.57 / 279.58 s | 3.02-3.41 s | 152.9 tok/s | 359.81 s |
| 8,192 | 187.03 s | 280.25 / 280.26 s | 7.14-9.57 s | 153.4 tok/s | 360.04 s |
| 16,384 | 189.48 s | 281.52 / 281.53 s | 20.25-20.41 s | 155.0 tok/s | 360.78 s |

`4096` is the throughput/TTFT knee: it is the fastest all-user admission and
completion point. `2048` reduces decode stalls but makes the last user 15.1 s
later, while 8192 and 16384 sharply worsen in-progress decode stalls without a
TTFT or wall-time benefit. None of these budgets alone makes the three cold
prefills fair: the TTFTs remain approximately 94 s, 187 s, and 280 s because
the first eligible long prefill consumes each scheduler iteration.

## H2: Long-Prefill Admission Cap

The current scheduler caps a long request's new tokens before it deducts the
per-iteration budget. A `1024` cap under the 4096 budget therefore lets all
three 196k requests progress in the same iteration instead of one request
consuming the complete budget.

| 4,096 budget profile | TTFTs | TTFT spread | All users in decode | Completion wall | Stable B=3 aggregate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Threshold disabled | 93.80 / 186.82 / 279.57 s | 185.77 s | 279.58 s | 359.81 s | 152.9 tok/s |
| Threshold 1,024 | 274.09 / 279.53 / 281.48 s | 7.39 s | 281.49 s | 361.03 s | 154.5 tok/s |

The threshold changes arrival fairness without sacrificing the eventual decode
engine. In the fair run, client decode was 47.18, 50.30, and 51.51 tok/s
(1.09x max/min), while seven stable server intervals were 153.6-154.9 tok/s
aggregate. Peak GPU0/GPU1 VRAM was 29,617/28,756 MiB; peak temperatures were
77/60 C; peak power was 495.08/487.62 W.

For three simultaneous giant arrivals, select `4096/1024` when fairness is the
goal. It delays the first user's response rather than letting that user leap
ahead, but changes all-user admission by only 1.9 s and total completion by
1.2 s versus uncapped 4096.

## H3: Established Decode Under A New 196k Prefill

The selected `4096/1024` profile then started a 196,003-content-token user,
waited for its first output plus 20 seconds of established decode, and injected
a distinct 196,006-content-token request. The established request forced 8,192
output tokens; the new request forced 1,024. There were zero prefix-cache
queries/hits/local-cache tokens, both requests completed by length, and FULL
runtime replay recorded B=1 plus padded two-request-to-B=3 execution. There
were no pre-shutdown errors; the known `EngineDeadError` appeared only after the
harness initiated vLLM shutdown.

| Measure | Result |
| --- | ---: |
| Established user's TTFT | 102.48 s |
| New prefill injection | 122.50 s after established request submission |
| New user's TTFT after injection | 109.74 s |
| Established user p95 ITL before injection | 0.0352 s |
| Established user p95 ITL during new prefill | 1.4793 s |
| Established user decode rate over its full response | 33.13 tok/s |
| New user's decode rate | 52.35 tok/s |

This is the operational result that the cold C=3 test cannot show. The
long-prefill threshold makes equal arrivals fair, but a later giant arrival
still produces 42x p95 ITL degradation for an active user. Do not claim that
`4096/1024` protects interactive decode.

## H4: Final Cold-Arrival Threshold Ladder

H3 was repeated with the same 4096 budget and the requested lower caps. Every
point was cache-disabled, completed both requests by length, had zero cache
hits, passed FULL replay, and restored the reference service. The wall column
is the established user's request wall from its initial submission to its forced
8,192-token completion.

| Threshold | Existing-user p95 ITL during cold prefill | Existing-user decode | New-user TTFT | Established-user wall |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 1.4793 s | 33.13 tok/s | 109.74 s | 349.78 s |
| 512 | 0.7516 s | 32.40 tok/s | 118.96 s | 349.23 s |
| 256 | 0.4265 s | 28.86 tok/s | 155.69 s | 388.94 s |
| 128 | 0.3478 s | 22.01 tok/s | 257.59 s | 537.93 s |

The cap produces the expected monotonic protection/TTFT tradeoff, but no tested
cap reaches the 300 ms target for an independent cold 196k arrival. `256` is
the practical knee: halving it again saves only 79 ms p95 while adding 102 s to
new-user TTFT and 149 s to the established request wall. Do not lower the cap
further in this path.

The realistic cache-local continuation result is materially better and is
recorded in `qwen3.8-27b-rtx5090-phase-i-cached-continuation-qos-2026-09-22.md`.

## Decision

- Retain graph B0 TP2 as the isolated decode baseline.
- Use `max_num_batched_tokens=4096` and `long_prefill_token_threshold=1024`
  only as the fair simultaneous-cold-arrival experimental profile.
- Do not promote, route, or install vLLM: the existing GSM8K gate remains
  failed, and H3 does not meet active-user decode QoS.
- Close MTP for this vLLM/runtime/model path. Do not run MTP2/MTP3, C=3 MTP,
  or 250k MTP.
- Do not investigate TP1 or DFlash in this phase.

## Artifacts

```text
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-h-graph-b0-c3-budget2048-vllm-2026-09-21T043738Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-h-graph-b0-c3-budget4096-vllm-2026-09-21T044543Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-h-graph-b0-c3-budget8192-vllm-2026-09-21T045544Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-h-graph-b0-c3-budget16384-vllm-2026-09-21T050542Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-h-graph-b0-c3-budget4096-threshold1024-vllm-2026-09-21T052032Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-h-graph-b0-decode-under-prefill-budget4096-threshold1024-vllm-2026-09-21T053026Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-h-graph-b0-decode-under-prefill-budget4096-threshold512-vllm-2026-09-21T222619Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-h-graph-b0-decode-under-prefill-budget4096-threshold256-vllm-2026-09-21T223430Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-h-graph-b0-decode-under-prefill-budget4096-threshold128-vllm-2026-09-21T224303Z/
```
