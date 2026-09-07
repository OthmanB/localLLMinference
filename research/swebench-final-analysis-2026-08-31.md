# SWE-bench Final Analysis

## Run

- Run directory: `/home/obenomar/localLLMinference/swebench/runs/qwen38-q4-q5-verified40-r2-sampled-89b534d4`
- Dataset: frozen 40-task SWE-bench Verified manifest
- Manifest SHA-256: `89b534d44bfcfb63f0d5810261ed7ced70611f4968a493f9f499911a17ccb6a1`
- Runtime: stock mini-SWE configuration with sampled non-thinking policy
- Power change cutoff: first confirmed 300 W resource sample at `2026-08-31T05:07:06Z`
- Paired run wall interval: `2026-08-31T04:21:53Z` to `2026-08-31T09:24:27Z` (5 h 2 m 34 s)

## Official Results

| Model | Tasks | Submitted | Completed | Empty patches | Step-limit failures | Resolved | Score | Ambiguous |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q4 | 40 | 40 | 37 | 3 | 3 | 22 | 55.0% | 0 |
| Q5 | 40 | 40 | 37 | 3 | 3 | 22 | 55.0% | 1 |

The 3 empty patches in each evaluation are exactly the 3 step-limit failures.
There were no infrastructure failures or evaluator errors. The Q5 ambiguous
case is `pytest-dev__pytest-5787`, classified as `no_tests_collected`; it was
resolved by Q4.

## Solve Overlap

- Both models resolved 21 identical tasks.
- Q4-only resolved task: `pytest-dev__pytest-5787`.
- Q5-only resolved task: `django__django-16560`.
- Neither model resolved 17 tasks.

The final benchmark score is therefore a tie. The two quantizations differ on
one resolved task in each direction, despite Q5 having the same aggregate score.

## Step-Limit Failures

All six failures were agent-level step exhaustion, not hardware or Docker
failures. Each consumed exactly 250 tool calls and produced an empty patch.

| Model | Task | Wall seconds | Max context tokens |
|---|---|---:|---:|
| Q4 | `django__django-15375` | 2460.4 | 108902 |
| Q5 | `django__django-15375` | 2760.5 | 92228 |
| Q4 | `pydata__xarray-4687` | 1380.1 | 42373 |
| Q5 | `matplotlib__matplotlib-24177` | 1800.3 | 60071 |
| Q4 | `sphinx-doc__sphinx-10614` | 1800.2 | 94841 |
| Q5 | `sphinx-doc__sphinx-10614` | 2940.4 | 102332 |

The shared failures are `django__django-15375` and
`sphinx-doc__sphinx-10614`; the third failed task differs by quantization.

## Token Throughput

Rates below are weighted generation rates from the trajectory-level llama.cpp
timings. Only response records at or after the first confirmed 300 W sample
are included in the post-change figures.

| Model | Phase | Responses | Generated tokens | Decode tok/s | P10 / median / P90 |
|---|---|---:|---:|---:|---:|
| Q4 | Before 300 W | 362 | 58183 | 27.37 | 25.55 / 28.37 / 30.34 |
| Q4 | After 300 W | 1670 | 244344 | 32.16 | 26.79 / 35.18 / 38.20 |
| Q5 | Before 300 W | 262 | 49287 | 22.33 | 21.23 / 23.16 / 24.85 |
| Q5 | After 300 W | 1906 | 330028 | 27.03 | 23.81 / 29.34 / 32.26 |

The weighted decode rate increased by approximately 17.5% for Q4 and 21.0%
for Q5 after the power-limit change. This is a useful operational result, but
not a perfectly controlled causal comparison because prompt lengths, context
states, task mix, and idle/test time differ between the two phases.

Weighted post-change prefill rates were approximately 399 tok/s for Q4 and 367
tok/s for Q5. These are prompt-processing rates excluding cached prompt tokens,
not end-to-end agent throughput.

## Power And Thermal Data

For the post-cutoff resource-monitoring window, including inference, tool/test,
and idle intervals:

| GPU | Power limit | Mean draw | Median draw | Maximum draw | Mean temperature | Maximum temperature |
|---:|---:|---:|---:|---:|---:|---:|
| 0 / Q4 | 300 W | 165.93 W | 206.24 W | 302.54 W | 63.14 C | 76 C |
| 1 / Q5 | 300 W | 235.08 W | 298.58 W | 303.48 W | 72.75 C | 82 C |

The lower mean draw for Q4 reflects long CPU-side tool/test intervals and idle
gaps; the median and peak values show that the card reached the new limit during
generation. Both temporary model services were stopped after evaluation, and
the GPUs returned to approximately 1 MiB used.

## Task Runtime

For the 37 successfully submitted tasks, mean task wall time was 257.9 seconds
for Q4 and 287.9 seconds for Q5. Including the three step-limit failures, the
means were 379.6 seconds and 453.8 seconds respectively. These are sums of
individual task durations; the paired workers ran concurrently.

## Artifacts

- Q4 predictions: `q4/preds.jsonl`
- Q5 predictions: `q5/preds.jsonl`
- Q4 evaluator report: `eval-q4/q4.qwen38-q4-verified40-r2-sampled-89b534d4.json`
- Q5 evaluator report: `eval-q5/q5.qwen38-q5-verified40-r2-sampled-89b534d4.json`
- Durable ledger: `ledger.sqlite3`
- GPU and host telemetry: `resources.csv`
