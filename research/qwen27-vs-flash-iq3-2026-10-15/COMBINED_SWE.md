# Paired SWE cohort comparison — expanded evaluation

**Incomplete snapshot: pending outcomes are not a final accuracy score.**

| Measure | Original 40 | Additional 40 | Combined 80 |
|---|---:|---:|---:|
| Tasks per model | 40 | 40 | 80 |
| qwen: Attempted | 40 | 0 | 40 |
| qwen: Submitted | 34 | 0 | 34 |
| qwen: Failed generation, no patch | 6 | 0 | 6 |
| qwen: Failed state | 6 | 0 | 6 |
| qwen: Infra exhausted | 0 | 0 | 0 |
| qwen: Infra retry attempts | 0 | 0 | 0 |
| qwen: Official scored | 2 | 0 | 2 |
| qwen: Official scored coverage | 0.0500 | 0.0000 | 0.0250 |
| qwen: Submitted without official bool | 32 | 0 | 32 |
| qwen: Determinate | 8 | 0 | 8 |
| qwen: Unknown/pending | 32 | 40 | 72 |
| qwen: Observed successes | 2 | 0 | 2 |
| qwen: Deployment submission fraction | 0.8500 | 0.0000 | 0.4250 |
| qwen: Success fraction lower bound (progress) | 0.0500 | 0.0000 | 0.0250 |
| qwen: Success fraction upper bound (progress) | 0.8500 | 1.0000 | 0.9250 |
| qwen: Complete-cohort score | withheld / unavailable | withheld / unavailable | withheld / unavailable |
| qwen: Wilson 95% CI | withheld / unavailable | withheld / unavailable | withheld / unavailable |
| qwen: wall_seconds observed total | 8577.6858 | 0 | 8577.6858 |
| qwen: wall_seconds tasks fully measured | 40 | 0 | 40 |
| qwen: wall_seconds mean task total (fully measured) | 214.4421 | withheld / unavailable | 214.4421 |
| qwen: output_tokens observed total | 305168 | 0 | 305168 |
| qwen: output_tokens tasks fully measured | 40 | 0 | 40 |
| qwen: output_tokens mean task total (fully measured) | 7629.2000 | withheld / unavailable | 7629.2000 |
| qwen: backend_calls observed total | 2219 | 0 | 2219 |
| qwen: backend_calls tasks fully measured | 40 | 0 | 40 |
| qwen: backend_calls mean task total (fully measured) | 55.4750 | withheld / unavailable | 55.4750 |
| qwen: agent_calls observed total | 2219 | 0 | 2219 |
| qwen: agent_calls tasks fully measured | 40 | 0 | 40 |
| qwen: agent_calls mean task total (fully measured) | 55.4750 | withheld / unavailable | 55.4750 |
| qwen: official patch failures | 0 | 0 | 0 |
| flash: Attempted | 40 | 0 | 40 |
| flash: Submitted | 35 | 0 | 35 |
| flash: Failed generation, no patch | 5 | 0 | 5 |
| flash: Failed state | 5 | 0 | 5 |
| flash: Infra exhausted | 0 | 0 | 0 |
| flash: Infra retry attempts | 0 | 0 | 0 |
| flash: Official scored | 1 | 0 | 1 |
| flash: Official scored coverage | 0.0250 | 0.0000 | 0.0125 |
| flash: Submitted without official bool | 34 | 0 | 34 |
| flash: Determinate | 6 | 0 | 6 |
| flash: Unknown/pending | 34 | 40 | 74 |
| flash: Observed successes | 1 | 0 | 1 |
| flash: Deployment submission fraction | 0.8750 | 0.0000 | 0.4375 |
| flash: Success fraction lower bound (progress) | 0.0250 | 0.0000 | 0.0125 |
| flash: Success fraction upper bound (progress) | 0.8750 | 1.0000 | 0.9375 |
| flash: Complete-cohort score | withheld / unavailable | withheld / unavailable | withheld / unavailable |
| flash: Wilson 95% CI | withheld / unavailable | withheld / unavailable | withheld / unavailable |
| flash: wall_seconds observed total | 8950.1971 | 0 | 8950.1971 |
| flash: wall_seconds tasks fully measured | 40 | 0 | 40 |
| flash: wall_seconds mean task total (fully measured) | 223.7549 | withheld / unavailable | 223.7549 |
| flash: output_tokens observed total | 363348 | 0 | 363348 |
| flash: output_tokens tasks fully measured | 40 | 0 | 40 |
| flash: output_tokens mean task total (fully measured) | 9083.7000 | withheld / unavailable | 9083.7000 |
| flash: backend_calls observed total | 1968 | 0 | 1968 |
| flash: backend_calls tasks fully measured | 40 | 0 | 40 |
| flash: backend_calls mean task total (fully measured) | 49.2000 | withheld / unavailable | 49.2000 |
| flash: agent_calls observed total | 1968 | 0 | 1968 |
| flash: agent_calls tasks fully measured | 40 | 0 | 40 |
| flash: agent_calls mean task total (fully measured) | 49.2000 | withheld / unavailable | 49.2000 |
| flash: official patch failures | 0 | 0 | 0 |
| Pairs: both_resolved | 1 | 0 | 1 |
| Pairs: qwen_only | 1 | 0 | 1 |
| Pairs: flash_only | 0 | 0 | 0 |
| Pairs: both_unresolved | 0 | 0 | 0 |
| Pairs: unknown_pairs | 38 | 40 | 78 |
| Paired difference (Qwen − Flash) | withheld / unavailable | withheld / unavailable | withheld / unavailable |
| Paired bootstrap 95% CI | withheld / unavailable | withheld / unavailable | withheld / unavailable |
| Exact McNemar p | withheld / unavailable | withheld / unavailable | withheld / unavailable |
| Paired wall_seconds: mean Qwen − Flash | -9.3128 | withheld / unavailable | -9.3128 |
| Paired wall_seconds: terminal pairs measured | 40 | 0 | 40 |
| Paired output_tokens: mean Qwen − Flash | -1454.5000 | withheld / unavailable | -1454.5000 |
| Paired output_tokens: terminal pairs measured | 40 | 0 | 40 |
| Paired backend_calls: mean Qwen − Flash | 6.2750 | withheld / unavailable | 6.2750 |
| Paired backend_calls: terminal pairs measured | 40 | 0 | 40 |
| Paired agent_calls: mean Qwen − Flash | 6.2750 | withheld / unavailable | 6.2750 |
| Paired agent_calls: terminal pairs measured | 40 | 0 | 40 |

## Evaluator and integration status

Complete-comparison gate: **BLOCKED / incomplete**.
Thresholds: zero unknown outcomes, zero submitted patches without official bool, zero unfinished evaluations, both controllers complete, and all frozen-protocol comparisons verified equal.
- unknown_task_outcomes
- submitted_patches_missing_official_bool
- controller_or_evaluator_incomplete
- resource_instructions_sampling_or_deadlines_unverified_or_mismatched
- original40: initialized=True; controller_complete=False; evaluation records=3; unfinished=1; failed/timed-out historical records=0 (a later retry may supersede these).
- additional40: initialized=False; controller_complete=False; evaluation records=0; unfinished=0; failed/timed-out historical records=0 (a later retry may supersede these).

## Selection provenance

Zero overlap enforced; exact identical manifest IDs paired across models.
- original40: selection seed `20260830`; manifest body SHA-256 `89b534d44bfcfb63f0d5810261ed7ced70611f4968a493f9f499911a17ccb6a1`; file SHA-256 `fcd5d0eaadf285f61687e54e9bf0cec26182d9e5d19f516b42dd519637628710`.
- additional40: selection seed `20261016`; manifest body SHA-256 `0007c250d4f8b1c2519fc7e55e3b9a7d08f5cefc0d06ad1eee0c87fa529107ab`; file SHA-256 `1abcc2046987cb09e618e045a769e9970afd8104d01fc2b843d75de0dc038650`.

## Protocol and statistical interpretation

Bootstrap: 20000 replicates; seed 20261016; paired percentile 95% interval within repository/cohort strata. Exact McNemar uses the doubled conditional binomial tail.
Frozen protocol verified equal: False; mismatches=0; unverified checks=14.
- Progress snapshot, not a final accuracy or publication claim. Sources: selected manifests and atomic task ledgers, not status summaries.
- Success means official resolved=True. Submitted patches need official bool. Terminal failed generation without a patch is a determinate protocol failure, not an official score.
- Infrastructure exhaustion, deadlines, running, queued, interrupted and unscored submitted patches remain unknown; never imputed False. Unknown pairs are excluded from the four determinate pair cells.
- Full-cohort success scores, Wilson intervals, paired difference intervals and McNemar p values are withheld until all relevant task outcomes are determinate. Observed lower/upper bounds are not accuracy estimates.
- Deployment submission availability includes infrastructure/deadline effects. Official patch failures and no-patch generation failures are separate categories; neither isolates intrinsic code quality from this deployment and agent protocol.
- Wilson intervals assume independent Bernoulli task outcomes with a common probability as a descriptive approximation. Bootstrap assumes exchangeable paired tasks within repository/cohort cells; singleton cells provide no estimated variation. Degenerate intervals can underestimate uncertainty.
- Fixed input-metadata-stratified selection is not a simple random sample of Verified500; single agent seed. No population-generalization probability guarantee. Repository/cohort cells approximate the finer input-length selection strata.
- Increasing 40 to 80 gives rough SE ratio sqrt(40/80)=0.707 (about 29% reduction) under comparable independent variance, not a guaranteed winner or realized precision gain.
- All attempt costs include retries. Missing measurements remain missing; observed totals can be incomplete, especially running attempts. Backend calls fall back to agent_calls when backend_requests is absent. Paired resource means use terminal pairs with every attempt measured.
- Frozen stock instructions, model sampling overlays, controller/evaluator implementations, config limits and inherited deadlines are compared. Missing additional-run files leave protocol unverified. Historical selection-manifest runtime settings are not active settings.
