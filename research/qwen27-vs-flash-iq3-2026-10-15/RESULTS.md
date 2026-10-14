# Qwen3.8-27B versus Flash-Next IQ3_S — paired campaign

<!-- generated-by: publish_results.py -->

Updated: 2026-10-14T22:06:16Z. Campaign phase: **swe-smoke**.

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

Phase: **smoke_generation**. Patch submission is not resolution.

| Metric | Qwen | Flash |
|---|---:|---:|
| Attempted tasks | 2.0 | 2.0 |
| Submitted patches | 2.0 | 1.0 |
| Officially evaluated submissions | 0.0 | 0.0 |
| Officially resolved / 40 | 0.0 | 0.0 |
| Total model calls | 40.0 | 9.0 |
| Total output tokens | 8822.0 | 938.0 |
| Mean resolved-task wall seconds | pending | pending |
| All attempt seconds per resolved task | pending | pending |

Denominator stays 40 for each model. Missing official reports and unattempted
tasks remain explicit; incomplete coverage is not a completed accuracy score.

## HumanEval+ and MBPP+

| Dataset | Qwen base / plus pass@1 | Flash base / plus pass@1 | Coverage/status |
|---|---:|---:|---|
| humaneval | pending / pending | pending / pending | qwen: 0/? pending; flash: 0/? pending |
| mbpp | pending / pending | pending / pending | qwen: 0/? pending; flash: 0/? pending |

## Matched natural coding workload: median engine rates and TTFT

| Body target | Qwen prefill / decode tok/s | Flash prefill / decode tok/s | Qwen / Flash TTFT s |
|---:|---:|---:|---:|
| 4096 | pending / pending | pending / pending | pending / pending |
| 32768 | pending / pending | pending / pending | pending / pending |
| 131072 | pending / pending | pending / pending | pending / pending |
| 245760 | pending / pending | pending / pending | pending / pending |

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
