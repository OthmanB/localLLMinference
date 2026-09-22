# Phase K: Gateway Admission Implementation Plan

Status: selected C<=2 controller implemented and loopback acceptance passed. The
feature remains disabled; Phase K must not promote, route, or install the RTX
5090 vLLM candidate.

## Objective And Boundary

Select the smallest admission mechanism that provides fluid interaction for the
measured local 5-6-user system. Phase J's C=0 gate is a protection reference,
not the default operational policy. Start with vLLM-native waiting/admission;
only consider a disabled-by-default gateway policy if the exact four-user
comparison fails the selected interactive QoS target. Neither option claims
support for six simultaneous long contexts or a 50-user local server.

The existing LAN gateway is currently deployed for the separate three-RTX-3090
profiles. The RTX 5090 profile does not participate in that profile switcher.
Phase K therefore targets only a dedicated, unpooled loopback vLLM backend in a
non-production qualification environment. No existing stock or `atx-dual`
backend/pool may enable this feature.

## Measured Contract

Phase J passed with `max_num_seqs=3`, three 196k cached conversations, and a
fourth independent 196k request held client-side. During C=3, the gateway saw
40 one-second `Running: 3`, `Waiting: 0` samples and three ten-second decode
intervals at 152.3-152.6 tok/s. The cold request was released only after two
one-second-separated C=0 samples; gateway-held delay was 41.408 s while vLLM
backend queue delay was 0.000450 s. All three 196k cache entries were retained
after the cold request.

Phase K preserves that distinction: gateway admission wait is not vLLM queue
time. The Phase J reference deliberately used no `max_num_queued_tokens`; the
native-scheduler comparison may evaluate bounded vLLM admission controls, but
must report their behavior separately from gateway-held wait.

## Native Scheduler Decision Gate

vLLM remains responsible for token-level continuous batching, KV allocation,
prefix-cache matching, decode scheduling, and all concurrency after a request
is admitted. The gateway is not a replacement scheduler. It only decides
whether an independent cold request may enter vLLM at all.

Before implementing an external cold queue, run one narrow four-user comparison
with the same three verified 196k cache-resident streams and one independent
196k cold request. The policies are:

| Policy | Fourth request handling | Question answered |
| --- | --- | --- |
| Phase J reference | Gateway holds it until C=0 | Best existing-user protection, and its cold-user delay. |
| Native C=3 | POST it to vLLM immediately with `max_num_seqs=3` | Whether vLLM keeps it `WAITING` while C=3 and how it behaves once a slot opens. |
| Explicit C<=2 | POST it only after one of the original three completes | Whether a minimal gate changes the native result or merely duplicates it. |

Here C<=2 is the condition immediately before cold admission. After admission,
the engine again has three active sequences: two decoders plus one cold prefill.
It is not the same as holding the cold request until C=0.

For all three policies, record D's submit-to-first-token time, time in gateway
or vLLM waiting, vLLM queue histogram, B/C p50/p95/p99 ITL during D prefill,
B/C decode rate, D prefill TTFT, and time until stable three-user decode resumes.
Set the acceptable B/C interactive SLO before treating any policy as a winner;
Phase H supplies a trade-off frontier, not a chosen operational threshold.

For this qualification, use the existing Phase H active-user p95 ITL target of
300 ms as the primary SLO. Report p99 as a diagnostic, not as a reason to hide
tail behavior. A policy with no active B/C samples during D's prefill must
report that fact rather than treating the missing value as zero.

Use vLLM native admission if it keeps D waiting without GPU work while A/B/C
are full, then gives acceptable B/C interaction once D begins prefill. The
expected native behavior with `max_num_seqs=3` is that D remains `WAITING` at
C=3 and becomes runnable when a slot opens, but this must be measured rather
than assumed. `max_num_queued_reqs` and `max_num_queued_tokens` are optional
backpressure limits: they reject excess work with HTTP 503; they are not a
fairness scheduler or a replacement for the normal vLLM waiting queue.

If native admission fails the selected SLO, evaluate a minimal gateway gate.
It must prevent only the harmful activation condition, perform no token-level
scheduling, and leave all admitted concurrency to vLLM. Do not build the full
cache ledger or long gateway-held queue unless that comparison demonstrates a
benefit that native vLLM cannot provide. Measure any canonical-prefix
classification and streaming overhead during hardware acceptance.

### Comparison Result

The guarded four-user runs selected explicit C<=2 admission:

| Policy | D waiting | D TTFT | B/C p95/p99 ITL during D prefill | Result |
| --- | ---: | ---: | ---: | --- |
| Phase J C=0 reference | 41.408 s gateway wait | 156.725 s user-visible | no active B/C sample | protection reference only |
| Native vLLM C=3 | 40.431 s vLLM wait | 156.335 s | 393/393 ms | fails 300 ms p95 SLO |
| Explicit C<=2 | 0 s gateway wait | 115.992 s | 36/36 ms | selected |

Native vLLM did keep D in `WAITING` while C=3. The native run passed the
cache, graph, retention, and no-preemption contracts, but its activation still
missed the interactive SLO. The C<=2 run posted D after exactly one original
stream completed, passed the same contracts, and had a 0.438 ms vLLM queue
histogram delay. Stable C=3 decode after D was not applicable because the fixed
2,048-token A/B/C streams completed while D prefetched; record this as `null`,
not as a passing recovery measurement.

Artifacts:

- `/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-k-native-c3-admission-vllm-2026-09-22T031655Z/candidate/results.json`
- `/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-k-c2-admission-vllm-2026-09-22T032734Z/candidate/results.json`

## Loopback Gateway Acceptance

The selected controller was exercised through a loopback gateway on `18080`,
with the candidate vLLM backend on `18081` and direct vLLM metrics sampled from
`18081`. The production service and monitor were restored after each guarded
run, and the gateway was not installed or enabled.

- Selected C<=2 workload passed all cold/queue contracts. Gateway wait was
  `0.007 ms`; direct vLLM queue delay was `0.000396 s`. Artifact:
  `/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-k-gateway-c2-acceptance-vllm-2026-09-22T035134Z/candidate/results.json`
- Native-submission workload passed all cache, capacity, graph, cold, and queue
  contracts while the gateway applied C<=2. The cold request waited
  `40.020870 s` at the gateway, while direct vLLM queue delay was only
  `0.000346 s`; user-visible TTFT was `155.371 s` and post-admission TTFT was
  `115.350 s`. Artifact:
  `/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-k-gateway-native-acceptance-vllm-2026-09-22T041535Z/candidate/results.json`

The native-submission run validates the gateway hold path only; it does not
change the direct native-vLLM policy decision. Gateway-held wait and backend
queue time are reported separately.

## Selected C<=2 Gateway Policy

The first gateway implementation is a minimal, disabled-by-default C<=2 gate.
It treats every request above the configured long-input estimate as unclassified
cold work. It never infers cache reuse from session IDs, prompt text, or
`kv_cache_usage_perc`; the Phase J cache ledger is deferred until an independent
canonical-prefix proof exists.

For one dedicated, unpooled backend, the controller keeps at most three active
leases and a FIFO queue of at most three long requests. A long request dispatches
immediately when the lease count is below three, a fresh loopback vLLM metrics
sample reports `num_requests_running <= 2`, and
`num_requests_waiting == 0`. Otherwise it waits outside vLLM. Short requests
bypass the controller. This preserves the initial C=3 workload but prevents a
fourth long prefill from entering while all three slots are active.

Metric failures, stale observations, malformed samples, and model mismatches
fail closed. Queue wait is distinct from vLLM queue time. Queue cancellation,
deadlines, no-retry behavior, streaming lease release, bounded labels, and
pool incompatibility remain required. No local Prometheus/Grafana deployment
is required.

The strict optional `cold_admission` configuration must contain only the
loopback vLLM metrics URL, target model label, C3 lease limit, C<=2 dispatch
threshold, long-input estimate, poll interval, queue depth, queue deadline, and
upstream timeout. Unknown fields are rejected, and the feature is rejected on
any backend referenced by `PoolConfig`. Do not edit production profile env
files or profile-switch behavior; use only a disabled RTX 5090 qualification
example.

Bounded metrics use only configured backend/model and fixed enum labels:

- `ai_gateway_cold_admission_queue_depth`
- `ai_gateway_cold_admission_inflight{request_class="long"}`
- `ai_gateway_cold_admission_events_total{event=...}`
- `ai_gateway_cold_admission_wait_seconds{outcome="dispatched|expired|cancelled"}`
- `ai_gateway_cold_admission_idle_observation_info{status="idle|busy|unknown"}`
- `ai_gateway_cold_admission_idle_probe_failures_total`

Return `X-Gateway-Request-ID` on every response and, when dispatched,
`X-Inference-Admission: c2` plus
`X-Inference-Gateway-Admission-Wait-Ms`. Never log prompts, sessions, token
IDs, authorization, or user-provided labels.

## Deferred C=0 Gateway Model

This section is retained only as the Phase J protection-reference design. It is
not the selected policy or the first implementation target. Do not build it
unless later measurements show that C<=2 cannot meet the operational SLO.

The controller owns one admission domain per eligible backend. It has two
views:

- Gateway leases: every successfully upstream-dispatched request remains leased
  until its non-streaming response finishes or its SSE stream closes.
- Fresh target-vLLM metrics: aggregate `num_requests_running`,
  `num_requests_waiting`, and `num_preemptions_total` from the configured
  loopback `/metrics` endpoint.

A C=0 cold release requires all of the following:

1. No gateway lease or dispatch reservation exists for the backend.
2. Two valid metrics samples, at least one second apart, report running zero
   and waiting zero for the target model.
3. Metrics are fresh, parse correctly, profile/engine epoch is unchanged, and
   no preemption increase has occurred.

Metrics failures, stale metrics, counter resets, profile changes, upstream
ambiguity, or preemption invalidate hot cache state and reset idle evidence.
They fail closed: queued cold work waits until its deadline and new work is
rejected rather than posted upstream.

## Deferred Classification And Cache Ledger

The gateway must never infer a prefix-cache hit from an arbitrary session ID or
prompt text. For the initial implementation, only a request whose canonical
token prefix has been gateway-tracked and independently verified may be
`cache_hot`.

The canonicalizer is specific to the pinned Phase J vLLM profile: exact model,
tokenizer revision, chat-template hash and arguments, prefix caching, context
length, and KV/profile fingerprint. Unsupported multimodal inputs, unpinned
template options, tools, or uncanonicalizable request shapes are never hot.
They fall back safely to unverified handling.

Cache ledger entries are in-memory, process-ephemeral, session-scoped, and use
an HMAC of the aligned token prefix. Prometheus labels and logs never contain
session IDs, prompts, token IDs, or cache hashes. Each entry records the backend
engine epoch, profile fingerprint, aligned length, state, timestamps, and its
reserved hot-session budget.

- `observed`: a C=0 long request locally computed its base prefix.
- `verified_hot`: a later matching C=0 probe proved at least 196k local cached
  tokens, no external cache hit, and at most 4,096 locally computed prompt
  tokens.
- `suspect`: verification failed, expiry occurred, restart/preemption/profile
  change happened, or upstream dispatch was ambiguous.

Only `verified_hot` continuations with an exact shared prefix of at least 196k
tokens and a suffix no larger than 4,096 tokens may bypass the cold queue. Keep
at most three such entries and at most 588k verified prefix tokens. A fourth
session must wait for an explicit demotion of an inactive entry. Do not use
`kv_cache_usage_perc` as cache-residency evidence.

Untracked long work at or above 100k tokens is `cold_long`. An unverified
candidate is also C=0 work. Unverified short work must not join a vLLM queue:
when busy it receives retryable overload; at C=0 it competes behind the existing
FIFO queue head. This is conservative because Phase J only qualified hot C=3
traffic and a 196k cold prefill, not concurrent short prefills.

## Deferred C=0 Queue, Deadline, And Cancellation Contract

The controller state is `REJECTED`, `QUEUED`, `RESERVED`, `DISPATCHED`,
`STREAMING`, or `TERMINAL`. A single async lock linearizes admission,
cancellation, deadline expiry, drain, and dispatch; at an exact deadline,
expiry wins.

- FIFO C=0 queue, maximum three requests and maximum one queued/dispatched
  request per session or trusted principal.
- Maximum queued body: 16 MiB. The request-body limit must run before FastAPI
  model parsing so three retained payloads are bounded.
- Gateway queue deadline: 900 seconds by default; an optional request header
  may lower it, never raise it above the configured maximum.
- Upstream lifetime for this backend: at most 600 seconds, separate from queue
  time and from legacy gateway backends' current global timeout.
- Full queue or per-session limit: OpenAI-shaped `503` with `Retry-After: 1`;
  no upstream POST.
- Deadline before POST: `504 admission_queue_deadline_exceeded`; no upstream
  POST.
- Queued-client disconnect: atomically remove the ticket; no upstream POST.
- After dispatch, close the upstream response best-effort on cancellation,
  release the lease only after closure, mark cache state suspect if completion
  is ambiguous, and never retry.
- A cold request is exclusive through its first content token. A subsequent
  cold request again requires new C=0 proof.

To prevent cache-continuation traffic starving a queued cold user, make the
fairness rule explicit and configurable: after the oldest cold request waits
30 seconds, stop admitting new hot requests, drain existing leases, and release
the FIFO head at confirmed C=0. The bound and drain interval are policy values,
not Phase J measurements.

## Deferred C=0 Configuration And Deployment

Add a strict optional `cold_admission` object to `BackendConfig`; reject unknown
backend fields before enabling it. Reject this configuration for any backend
referenced by `PoolConfig`. It remains disabled by default.

The target configuration needs: loopback metrics URL; target model label;
session header; metric interval and two-sample C=0 requirement; C=3 limit;
100k cold and 196k/4k hot thresholds; three-entry/588k hot budget; cache-entry
TTL; queue depth, per-session limit, body cap and deadlines; drain delay;
upstream timeout; pinned tokenizer/template/profile fingerprint; and an HMAC
key environment variable.

Do not edit `stock-q4-tensor.gateway.env`, `atx-dual.gateway.env`, or the
canonical profile-switch behavior. Add a separate disabled/non-production
example for the RTX 5090 qualification backend only. The hardware acceptance
run must use a guarded candidate service and loopback gateway, never a client
facing production unit.

## Deferred C=0 Gateway Observability

The controller's own `/metrics` output is optional direct-debug and future
integration data. Phase K does not require a local Prometheus, Grafana, or
exporter deployment, and a missing external scrape must never affect request
admission. This is distinct from the required loopback vLLM metrics reads used
to prove C=0 before releasing cold work.

Add bounded gateway metrics with only configured backend/model and fixed enum
labels:

- `ai_gateway_cold_admission_queue_depth`
- `ai_gateway_cold_admission_inflight{request_class="cached|cold|short"}`
- `ai_gateway_cold_admission_events_total{event=...}`
- `ai_gateway_cold_admission_wait_seconds{outcome="dispatched|expired|cancelled"}`
- `ai_gateway_cold_admission_idle_observation_info{status="idle|busy|unknown"}`
- `ai_gateway_cold_admission_idle_probe_failures_total`

Events are `enqueued`, `dispatched_immediate`, `dispatched_queued`,
`bypassed_cached`, `rejected_queue_full`, `expired`, `cancelled`,
`drained_shutdown`, and `idle_probe_failed`. Gateway wait is strictly dispatch
monotonic time minus queue-accept monotonic time; it must not be labelled as
TTFT or backend queue delay.

Return `X-Gateway-Request-ID` on every response and, when dispatched,
`X-Inference-Admission` plus `X-Inference-Gateway-Admission-Wait-Ms`. Emit one
structured transition log without prompts, session identifiers, token IDs,
authorization, cache HMACs, or user-provided labels. Extend the metrics exporter
allowlist and its tests so these fixed schemas survive re-exporting.

## Implementation Sequence

1. Completed the native/C<=2 comparison and selected the 300 ms active-user
   p95 ITL SLO; C<=2 passed and native submission failed it.
2. Completed the strict, disabled-by-default, pool-incompatible `cold_admission`
   configuration and standalone C<=2 controller.
3. Completed lifecycle acquisition/release around non-streaming and SSE
   forwarding, with queue cancellation, deadlines, bounded depth, and no retry.
4. Completed deterministic ASGI coverage for C<=2 dispatch, configuration
   rejection, streaming release, and bounded direct-debug metrics.
5. Completed guarded non-production loopback-gateway hardware acceptance and
   compared gateway-held wait with direct vLLM queue time.
6. Keep the feature disabled after acceptance. Do not promote or route the RTX
   5090 candidate; the quality gates remain blocked.

## Required Tests

P0 tests cover long-request classification, FIFO, max depth, exact-deadline
precedence, C<=2 metric gating, malformed/stale metrics fail-closed, queue full,
no upstream POST before dispatch, cancellation, stream lease release, and
dispatch/cancel/expiry races. P1 tests cover unsupported request fallback,
pool incompatibility, per-backend timeout, metrics schemas, exporter re-export,
and absence of sensitive labels. Hardware testing is separate and must not
replace deterministic tests.

## Promotion Block

The RTX 5090 candidate remains blocked from any production service, gateway
routing, or canary. Its reconstructed GSM8K smoke scored 122/128 (95.3125%) vs
the 96.5% floor, stopped normally only 125/128 times, and truncated three
responses. Before reconsidering promotion it needs a corrected full 1,319-case
quality run, zero truncation/stop failures, and representative coding-agent and
multi-turn tool validation. Phase K must not revisit MTP, TP1, DFlash, or the
completed scheduler sweeps.
