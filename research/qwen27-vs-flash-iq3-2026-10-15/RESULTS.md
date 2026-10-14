# Qwen3.8-27B versus Flash-Next IQ3_S — paired campaign

<!-- generated-by: publish_results.py -->

Updated: 2026-10-14T23:50:41Z. Campaign phase: **blocked-or-failed**.

## Configurations

| | Qwen3.8-27B | Flash-Next |
|---|---|---|
| Quant | ATX IQ4_XS-M | full-expert IQ3_S |
| Runtime | deployed llamAmpere, MTP3 | Strata 0.1.44, MTP4 |
| Primary allocation | one RTX 3090, GPU1 | one RTX 3090, GPU0 + host RAM |
| Matched context | 262144 native | 262144 native, no scaling |

Identical task text and generation policy per benchmark. This is a comparison
of serving configurations, not an isolated quantization experiment.

## SWE-bench Verified (frozen 40-task subset)

Phase: **smoke_complete**. Patch submission is not resolution.

| Metric | Qwen | Flash |
|---|---:|---:|
| Attempted tasks | 2.0 | 2.0 |
| Submitted patches | 2.0 | 1.0 |
| Officially evaluated submissions | 2.0 | 1.0 |
| Officially resolved / 40 | 2.0 | 1.0 |
| Total model calls | 40.0 | 159.0 |
| Total output tokens | 8822.0 | 45270.0 |
| Mean resolved-task wall seconds | 97.2 | 32.1 |
| All attempt seconds per resolved task | 97.2 | 847.8 |

Denominator stays 40 for each model. Missing official reports and unattempted
tasks remain explicit; incomplete coverage is not a completed accuracy score.

## HumanEval+ and MBPP+

| Dataset | Qwen base / plus pass@1 | Flash base / plus pass@1 | Coverage/status |
|---|---:|---:|---|
| humaneval | 96.3% / 92.1% | 97.6% / 95.7% | qwen: 164/164 complete; flash: 164/164 complete |
| mbpp | pending / pending | pending / pending | qwen: 0/? pending; flash: 0/? pending |

## Matched natural coding workload: median engine rates and TTFT

| Body target | Qwen prefill / decode tok/s | Flash prefill / decode tok/s | Qwen / Flash TTFT s |
|---:|---:|---:|---:|
| 4096 | 1161.6 / 98.8 | 1251.4 / 87.5 | 3.9 / 3.3 |
| 32768 | 1143.8 / 100.5 | 2142.4 / 91.3 | 29.8 / 15.4 |
| 131072 | 725.9 / 81.2 | 2069.4 / 88.2 | 183.6 / 63.5 |
| 245760 | 507.0 / 66.7 | 1917.9 / 87.0 | 489.9 / 128.3 |

Three fresh repeats are planned per point. These medians include valid responses;
the aggregate JSON separately exposes caps, cache reuse/unknowns, failures, actual
token counts, ranges and zero-reported-cache distributions. Do not interpret
different natural output lengths as equal-token work.

## Deployment capacity

Measured separately: two Qwen replicas on GPU1+GPU2 versus two queued requests
to one Flash replica on GPU0. This is a hardware-allocation comparison.

## Interpretation and reproduction

Use PLAN.md, README.md and SUPPLEMENTARY.md for fixed budgets, sampler settings,
held-out evaluation, failure accounting and reproduction. Flash separately passed
512k YaRN document/code trials; that capability is outside this native matched test.
Raw agent/model outputs, held-out fixtures, credentials and local telemetry are
not published. results-summary.json contains filtered aggregate measurements.
