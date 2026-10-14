# Qwen3.8-27B ATX IQ4_XS-M versus Flash-Next IQ3_S

User authorized parallel, substantial benchmarking over approximately 30 hours,
followed by comprehensible side-by-side reporting and public commit/push.
Campaign preparation begins 2026-10-14 UTC; directory names use the next-day
benchmark label. All limits are anchored to the recorded start epoch, not the
calendar label.

## Models and fairness

| | Qwen3.8-27B | Qwen3.8-Flash-Next |
|---|---|---|
| Quant | deployed ATX IQ4_XS-M (not generic Q4_K_M) | full-expert GSQ IQ3_S |
| Runtime | deployed llamAmpere 36a6bca817 | Strata 0.1.44 |
| Primary allocation | one RTX 3090, GPU1 | one RTX 3090, GPU0 + host experts |
| Context for matched tests | 262144 native | 262144 native, no YaRN |
| Speculation | deployed MTP3 | validated MTP4 |
| KV | deployed Q8 K / turbo3 V | int8; 32768 cells resident, remainder host |
| Replicas | GPU2 is a separately measured second replica | one server, parallel=1 |

This compares complete serving configurations, not quantization in isolation.
Identical task text, agent instructions, generation policy and task budgets.
Actual model token counts, cache state and output lengths are measured. Existing
Qwen service settings/PIDs remain unchanged; user now authorizes inference on
GPU1/2 for the comparison. No GPU1/2 experimental runtime is launched.

## Ordered phases (parallel within each, no competing suites on same endpoints)

1. Load Flash native262k with 1536 MiB reserve, skipping off; snapshot active
   model IDs/services. Run two paired SWE smoke tasks plus official hidden tests.
   Pause after the smoke gate; diagnose protocol/infrastructure errors first.
2. Matched speed/document tests: 4k/32k/128k/245760 body tokens, three fresh
   repeats, both backends simultaneously, JSON source/fact/arithmetic answers
   and independent natural coding completions. Record medians/ranges, TTFT,
   prefill/decode, cache counts, correctness and output caps. Allow 3 hours.
3. Deployment capacity: two Qwen replicas on GPU1+2 versus two requests queued
   to one Flash on GPU0. Three rounds, distinctly labelled hardware allocation;
   do not mistake aggregate throughput for a single-GPU model advantage.
4. HumanEval+164 and MBPP+378: one greedy sample per task per model, paired
   generation, identical sanitizer, official base+extra tests, fixed denominator.
   Allow up to 5 hours, keep partial coverage/error status explicit. Local
   EvalPlus evaluator uses its execution guards, process/resource limits, private
   work directory, parallel=2; it is not Docker isolation.
5. SWE-bench Verified frozen stratified40: one active agent per model concurrently,
   150 steps/1800s per task, 8192 max output, sampled nonthinking policy identical
   across models. Docker 2 CPUs/6 GiB, network none, cached task images only.
   Primary generation cutoff = global start +24h; all hidden evaluation cutoff
   = global start +30h. Resume incomplete infrastructure work only, preserve
   failed/limited tasks in the 40-task denominator. Official resolved, not patch
   submission, is the task-success authority.

Report: resolved/40 with evaluation coverage, HumanEval+/MBPP+ pass@1, errors,
caps/loops/timeouts, output tokens, model calls, time per resolved task, common
context TTFT and useful decode, resources. Include Flash's separately verified
512k capability as context, not as a shared-context comparison.

## Durability and publication

run_campaign.py owns the sequential phases and only its own Flash process.
Record deadline/phase/resource files and per-task durable ledgers. Do not restart
the campaign into duplicate requests; use its lock/resume paths. No production
gateway or system service deployment changes. Stop Flash after campaign/failure;
Muse remains stopped unless user restores it.

Public checkout is a separate Git tree. Publish portable scripts, instructions,
filtered task/aggregate metrics and authored tests; raw responses/trajectories,
held-out datasets/gold patches, credentials, machine-local loader configurations,
large model/runtime assets and telemetry remain private. Earlier unpublished
test work has been collected in a separate archival manifest for review/commit.
Public origin may advance independently; fast-forward before committing and
stage only intended publication paths. Publish source/protocol now and scored
results when the campaign finishes; never invent final scores during execution.
