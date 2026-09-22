# Phase J: C=3 Client-Side Cold Admission Qualification

Status: passed on 2026-09-22.

Successful artifact:
`/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-j-graph-b0-cached-c3-client-queued-cold-vllm-2026-09-22T005955Z/candidate/results.json`.

## Scope

This RTX 5090 host is modeled as a local 5-6-user system, with at most three
simultaneously decoding expensive long-context agent sessions. It is not a
50-user local-server target. A 50-user estimate belongs to a separate larger
6-8 RTX 4000 cluster and requires independent measurement there.

Phase J qualifies an application or gateway admission policy, not a vLLM
internal C4/C6 queue. The runner fixes `--max-num-seqs 3` and does not set
`max_num_queued_tokens`.

## Fixed Profile

- TP2 on GPUs 0 and 1; 262,144-token context; FP8 E4M3 KV cache; GPU memory
  utilization 0.90.
- Prefix caching, chunked prefill, `max_num_batched_tokens=4096`, and
  `long_prefill_token_threshold=256`.
- Triton attention/GDN, native sampler, PYNCCL, `FULL_DECODE_ONLY` graphs,
  captures 1 and 3, and breakable CUDA graphs disabled.
- 85 C thermal stop, 600-second maximum per request, and a 900-second limit
  starting when the candidate becomes ready.

## Qualification Workload

1. Create three distinct approximately 196k-token cached conversations with
   deterministic normal-EOS replies.
2. Prove cache reuse separately for every conversation with a small unique
   normal-EOS next-turn probe.
3. Start the three next turns simultaneously with 1,024, 2,048, and 4,096
   input-token appends and force each to generate 2,048 tokens.
4. When all three streams produce content, construct but do not POST a fourth,
   independent approximately 196k-token cold request.
5. Sample the scheduler at 1 Hz. POST that cold request only after all three
   streams finish and two consecutive samples report `Running: 0` and
   `Waiting: 0`.
6. Force 256 cold output tokens, then issue a unique normal-EOS retention
   probe for each original conversation.

The C=0 release rule is intentional. Phase H showed that a 196k cold arrival
at the 4096/256 profile still increased an active user's p95 ITL to 426 ms;
lowering the threshold to 128 did not achieve 300 ms. Releasing a cold prefill
when only one `max_num_seqs` slot is free would not demonstrate protected
interactive service.

## Required Evidence

- Every warm request must locally compute at least 90% of its base prompt,
  have no local/external cache hit, and return its deterministic normal-EOS
  reply.
- Every preflight and retention probe must reuse at least 90% of its own base
  in prefix-hit, local-cache-hit, and cached-token counters; small local
  compute only; no external hit; correct normal-EOS reply.
- The aggregate C=3 delta must reuse at least 90% of all three bases. Per-user
  attribution is established by the isolated probes before and after C=3.
- Record per-user TTFT, p50/p95/p99 ITL, output rate, token timestamps, 1 Hz
  scheduler/GPU/host samples, and 10-second server intervals with `Running: 3`,
  `Waiting: 0`, and zero prompt throughput.
- Record gateway-held delay, post-admission TTFT, user-visible TTFT, cold wall
  time, and backend queue histogram delta. Backend queue time should remain
  near zero because the request was intentionally not posted while C=3 ran.
- Fail on C>3 evidence, fewer than two all-three-active scheduler samples,
  graph fallback, missing actual B=3 FULL replay, preemption, thermal stop,
  errors, cache-contract failure, or post-cold retention miss.

## Result

An initial `2026-09-22T004914Z` attempt stopped before cold admission because
the harness did not initialize its server-log wall-clock origin. It is not used
for qualification. The instrumentation fix was validated against its retained
B=3 graph log, then the single replacement qualification run passed all
contracts.

- Each of the three distinct 196,006-token base warm requests was fully cold:
  196,006 local-compute tokens, zero local/external cache hits, and
  114.964-115.313 s request time.
- Each 196,281-token preflight reused 196,000 local cached tokens, computed
  281 tokens, had no external hit, and completed in 1.054-1.147 s.
- The concurrent C=3 turns reused 588,000 cached/local-hit tokens in total,
  computed 8,059 tokens, had no external hit, and generated 148.92 tok/s
  aggregate. The 1k/2k/4k append streams had TTFTs of 4.312/7.248/7.465 s,
  p95 ITL of 39.88/39.26/39.41 ms, p99 ITL of 45.45/40.85/41.57 ms, and
  47.37/50.65/50.90 tok/s respectively.
- All 40 one-second C=3 scheduler samples reported three running, zero
  waiting, and zero preemptions. Their cache gauge range was 57.44-58.17%;
  three server ten-second intervals reported 152.3/152.6/152.5 tok/s with
  `Running: 3`, `Waiting: 0`, and zero prompt throughput.
- The cold request was held in the harness for 41.408 s, then posted only
  after two one-second-separated C=0 samples. Its backend queue delay was
  0.000450 s, post-admission TTFT was 115.317 s, user-visible TTFT was
  156.725 s, total wall time was 160.988 s, and decode rate was 60.05 tok/s.
  It locally computed 196,005 tokens and had zero cached/local/external hits.
- All three post-cold probes retained a 196,000-token local cache hit with
  only 281 local-compute tokens and completed in 1.059/1.076/1.115 s. This is
  the retention proof; the C=0 cache gauge alone is not interpreted as an
  eviction signal.
- FULL CUDA graph B=3 was captured and replayed before cold admission;
  runtime request and batch-size evidence contains both 1 and 3. There were
  zero preemptions, no graph fallback, no pre-shutdown workload errors, no
  thermal stop, and 500 GPU/host samples. Peak GPU temperature was 80 C.

The candidate log records an `EngineDeadError` only after the runner's
explicit zero-timeout shutdown began and force-stopped the EngineCore. It is
outside the pre-shutdown workload-error window and does not invalidate the
completed request contracts.

## Decision

The client-side C=0 gate is a demonstrated protection reference: it preserves
three expensive cache-resident agent sessions while an independent cold request
waits outside vLLM. It is not an operational admission decision, evidence for
six simultaneous long contexts, or evidence for a 50-user local target. Phase K
must compare this reference with vLLM-native admission under the exact same
four-user workload before selecting a user-facing policy. Phase K completed that
comparison: native vLLM kept the fourth request waiting at C=3 but measured
about 393 ms active-user p95/p99 ITL during cold-prefill activation. Explicit
C<=2 admission measured about 36 ms p95/p99 ITL and was selected for the
qualification gateway. The C=0 result remains a protection reference.

## Follow-On Policy

The selected C<=2 external gate conservatively classifies unverified long work
as cold, queues it outside vLLM while three slots are active, and releases it
when the target reports C<=2. It does not infer cache reuse from session IDs or
prompt text. Queue bounds, deadlines, cancellation, user-visible progress, and
fairness were implemented in Phase K's disabled qualification gateway. The
guarded loopback acceptance passed the selected C<=2 contracts; this does not
authorize production routing or promotion.

Model-quality remediation is a separate branch. It must resolve the GSM8K
122/128 result and three truncations before any production promotion, without
revisiting MTP, TP1, DFlash, or scheduler sweeps.
