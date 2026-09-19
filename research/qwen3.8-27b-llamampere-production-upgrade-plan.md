# Qwen3.8-27B llamAmpere Production Upgrade Plan

Date: 2026-09-19 UTC
Status: approved architecture, implementation pending

## 1. Decision

Replace the current two-GPU stock llama.cpp Qwen3.8 deployment with two
independent llamAmpere v0.3 instances serving the fixed model:

```text
Qwen3.8-27B ATX IQ4_XS-M
```

The production topology is:

```text
physical GPU1 -> llamAmpere replica 1 -> independent context
physical GPU2 -> llamAmpere replica 2 -> independent context
```

The measured operating point is approximately 65 tok/s decode per replica at
196k context, or approximately 130 tok/s aggregate while preserving two
independent agent contexts.

The 70-80 tok/s number was aspirational. It is not an acceptance criterion.

The upgrade will not use a weighted canary. After the required offline capacity
and readiness tests pass, the operator will switch the active profile and test
the production service directly.

The existing stock llama.cpp service remains installed as a manually selectable
profile. It is not an automatic fallback and must not be started concurrently
with the llamAmpere profile because both profiles use GPUs 1 and 2.

No `Conflicts=` rules are required. Mutual exclusion is enforced by the profile
switch procedure, which stops the active Qwen profile before starting the new
one.

## 2. Context Capacity Decision

The ATX GGUF declares native context length `262144`, and llamAmpere accepts
that value. The exact ATX model has not yet been proven at full 262144 on one
RTX 3090.

Current evidence:

- Model declaration: native `262144`.
- Runtime support: `ctx-size 262144` is accepted by the model/runtime path.
- Validated single-GPU operation: `ctx=245760` with approximately 240k
  populated tokens.
- Validated dual-replica operation: approximately 196k populated tokens per
  replica.
- Unvalidated claim: full 262144 populated operation on one or both GPUs.

Production context policy:

1. Run the full-context validation in Section 5 before the service cutover.
2. If it passes, configure both production replicas at `ctx=262144` and use
   the `262144` model IDs.
3. If it fails, configure both replicas at the already validated `ctx=245760`
   and use the `245760` model IDs. Do not silently advertise 262144.

## 3. Frozen Runtime Artifacts

The following artifacts are pinned for deployment:

| Artifact | Value |
|---|---|
| llamAmpere commit | `36a6bca817755c6ddab2001b78eb0ae3e3eef7c8` |
| llamAmpere binary SHA-256 | `bcee5b8939763da8f87e87d77325142d9b2d4b686036bb4450d3c4754d77ec46` |
| ATX model SHA-256 | `5cf05ad901dcaa76f41db13a5629146ed882219339377a80e37b12a8528d963b` |
| MTP vocabulary map SHA-256 | `8405ff0f8970da24b72d68c9e61fa4309fdcf3719ecc1d7626f0ae8bd2a8e326` |

Source artifacts:

- `research/qwen3.8-27b-llamampere-axis1-sweep-20260918T224901Z`
- `research/qwen3.8-27b-llamampere-axis1-batch6144-20260919T004351Z`
- `research/qwen3.8-27b-serving-alternatives-phase1-dual-20260918T144111Z`

Do not deploy only the `llama-server` executable. Promote the complete
llamAmpere build directory, including colocated shared libraries and runtime
assets, into an immutable production release directory. Suggested layout:

```text
/opt/ai-server/llamampere/36a6bca817/
  build-sm86/bin/llama-server
  build-sm86/lib/...
  docs/mtp-vocab/atx_65536.txt
```

Suggested model location:

```text
/opt/ai-server/models/qwen3.8-27b-atx-iq4_xs_m/
  Qwen3.8-27B-ATX-4-XS.gguf
```

The release directory and model should be root-owned, readable by `obenomar`,
and not modified in place. Every deployment must verify the hashes before
starting a service.

## 4. Production Profiles

Use one root-owned profile variable to select the active Qwen topology:

```text
QWEN_SERVING_PROFILE=atx-dual
```

Allowed values:

```text
atx-dual
stock-q4-tensor
```

The profile is consumed by a root-only switch command, for example:

```text
/usr/local/sbin/ai-qwen-profile-switch atx-dual
/usr/local/sbin/ai-qwen-profile-switch stock-q4-tensor
```

The switch command must:

1. Validate the requested profile.
2. Acquire an exclusive deployment lock.
3. Back up the active gateway and exporter environment files.
4. Stop the gateway before changing backend topology.
5. Stop the active Qwen service or services.
6. Verify ports and physical GPU ownership are clear.
7. Atomically install the selected gateway and exporter profile files.
8. Start the selected Qwen services.
9. Verify direct health, models, and metrics endpoints.
10. Start or restart the exporter, then start the gateway.
11. Verify authenticated gateway readiness and model routing.
12. Leave the previous profile files available for an explicit rollback.

The switch must fail closed. It must not start the new profile if the old
profile still owns GPU1, GPU2, or a required port.

There must be no automatic profile switch on service failure. A failed
llamAmpere process should be visible through systemd, gateway readiness, and
Prometheus. The operator chooses whether to restart llamAmpere or switch to
`stock-q4-tensor`.

## 5. Full Native-Context Validation

Run this before production activation on an isolated loopback port and with no
production gateway route pointing at it.

Use the validated MTP3 baseline. Change only the context from `245760` to
`262144`:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=1
GGML_Q8_TURBO3_MMA_FUSED=1
llama-server \
  --model /opt/ai-server/models/qwen3.8-27b-atx-iq4_xs_m/Qwen3.8-27B-ATX-4-XS.gguf \
  --ctx-size 262144 \
  --batch-size 4096 \
  --ubatch-size 1024 \
  --threads 8 \
  --threads-batch 8 \
  --n-gpu-layers 99 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v turbo3 \
  --parallel 1 \
  --jinja \
  --fit off \
  --cache-prompt \
  --cache-ram 8192 \
  --ctx-checkpoints 24 \
  --checkpoint-min-step 10240 \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-p-min 0 \
  --spec-draft-type-k q8_0 \
  --spec-draft-type-v q8_0 \
  --spec-draft-vocab-map /opt/ai-server/llamampere/36a6bca817/docs/mtp-vocab/atx_65536.txt \
  --device CUDA0 \
  --host 127.0.0.1 \
  --port 18121 \
  --metrics \
  --perf \
  --no-ui
```

Generate a deterministic prompt with at least `261760` input tokens. Include a
sentinel near the beginning and another near the end. Use normal EOS behavior,
not `ignore_eos=true`.

Run two fresh-server cold requests. Do not use a warm cache as capacity proof.

Required result for each run:

- startup reports context size `262144`;
- input token count is at least `261760`;
- HTTP 200 and valid completion;
- both sentinels are retrieved correctly;
- `truncated=false`;
- no OOM, CUDA error, process restart, or context shift;
- peak VRAM, power, temperature, host RAM, and swap are recorded;
- server and port cleanup succeeds.

Repeat the same full-context request on physical GPU2. After both single-replica
tests pass, run one cold 262k request on each replica concurrently. Require both
to complete without GPU interference or host swap exhaustion.

If full 262k fails, stop the context claim and deploy the validated
`245760` profile instead. Do not change quantization or KV-cache types to make
the capacity test pass.

## 6. Systemd Units

Add two units:

```text
llamampere-qwen3.8-atx-iq4xs-m-gpu1.service
llamampere-qwen3.8-atx-iq4xs-m-gpu2.service
```

Each unit uses:

```text
User=obenomar
Group=obenomar
CUDA_DEVICE_ORDER=PCI_BUS_ID
GGML_Q8_TURBO3_MMA_FUSED=1
```

GPU1 unit:

```text
CUDA_VISIBLE_DEVICES=1
--device CUDA0
--host 127.0.0.1
--port 8080
```

GPU2 unit:

```text
CUDA_VISIBLE_DEVICES=2
--device CUDA0
--host 127.0.0.1
--port 8081
```

Both units use:

```text
--ctx-size 262144
--batch-size 4096
--ubatch-size 1024
--threads 8
--threads-batch 8
--n-gpu-layers 99
--flash-attn on
--cache-type-k q8_0
--cache-type-v turbo3
--parallel 1
--jinja
--fit off
--cache-prompt
--cache-ram 8192
--ctx-checkpoints 24
--checkpoint-min-step 10240
--spec-type draft-mtp
--spec-draft-n-max 3
--spec-draft-p-min 0
--spec-draft-type-k q8_0
--spec-draft-type-v q8_0
--spec-draft-vocab-map /opt/ai-server/llamampere/36a6bca817/docs/mtp-vocab/atx_65536.txt
--metrics
--perf
--no-ui
```

Use `--alias` values that include both the explicit lane ID and the pooled ID:

```text
GPU1:
  qwen3.8-27b-atx-iq4xs-m-262144-gpu1
  qwen3.8-27b-atx-iq4xs-m-262144

GPU2:
  qwen3.8-27b-atx-iq4xs-m-262144-gpu2
  qwen3.8-27b-atx-iq4xs-m-262144
```

The service logs must be separate:

```text
/var/log/ai-server/llamampere-qwen-gpu1.log
/var/log/ai-server/llamampere-qwen-gpu2.log
```

Use `Restart=on-failure`, a short restart delay, `LimitNOFILE=1048576`, and a
stop timeout long enough to release the CUDA context cleanly. Do not add
`Conflicts=` between these units and the stock unit. Profile switching provides
the operational exclusivity.

The stock unit remains installed with its existing two-GPU command and alias:

```text
llama-qwen3.8-q4-tensor-262k.service
qwen3.8-27b-q4-tensor262k
```

It is selected only by `stock-q4-tensor`.

## 7. Gateway Routing

The existing gateway is at:

```text
/home/obenomar/localLLMinference/lan-inference-gateway
```

It currently maps one model ID to one backend. `BackendRegistry` rejects a
model assigned to two backends, so the gateway needs a replica-pool extension.

### Public model IDs

Always expose the two explicit IDs:

```text
qwen3.8-27b-atx-iq4xs-m-262144-gpu1
qwen3.8-27b-atx-iq4xs-m-262144-gpu2
```

Add the convenience pooled ID:

```text
qwen3.8-27b-atx-iq4xs-m-262144
```

The explicit IDs bypass load balancing and allow deterministic lane selection.
The pooled ID selects between the two replicas.

### Session-affinity policy

Use the request header:

```text
X-Inference-Session: <stable-conversation-id>
```

For the pooled model:

1. If the header is present, rendezvous-hash the session ID across healthy
   replicas.
2. Keep the session on that replica while it remains healthy.
3. If the selected replica is unavailable, route to another healthy replica
   only as an explicit cold failover and emit a response header indicating the
   change.
4. If no session header is present, choose the healthy replica with the lowest
   in-flight count; use deterministic tie-breaking.
5. Never use client IP as the session identity.
6. Validate the header length and character set. Do not log the raw value.
7. Add `X-Inference-Replica: gpu1|gpu2` to responses for diagnostics.

The gateway must not retry a request after upstream dispatch, especially for
SSE streams, tool calls, or non-idempotent agent operations.

Do not silently spill a cache-warm session to the other GPU merely because the
preferred replica is busy. Initially return a retryable `503` when a pinned
replica reaches its configured saturation limit. A future explicit cold-spill
mode can be added after production observations.

### Gateway implementation

Modify:

```text
lan-inference-gateway/src/lan_inference_gateway/config.py
lan-inference-gateway/src/lan_inference_gateway/routing.py
lan-inference-gateway/src/lan_inference_gateway/app.py
lan-inference-gateway/tests/test_app.py
```

Add configuration for singleton backends and pools. Keep the existing static
backend format for Muse and explicit replica IDs. Add a pool structure similar
to:

```json
{
  "name": "qwen3.8-27b-atx-iq4xs-m-262144-pool",
  "model": "qwen3.8-27b-atx-iq4xs-m-262144",
  "replicas": ["qwen_atx_gpu1", "qwen_atx_gpu2"],
  "session_header": "X-Inference-Session",
  "max_inflight": 1
}
```

The final environment representation may use a separate
`LAN_INFERENCE_POOLS` JSON variable rather than adding pool fields to backend
objects. Keep configuration validation strict and reject unknown replica names.

Add tests for:

- explicit GPU1 and GPU2 model routing;
- stable pooled routing for repeated session IDs;
- different sessions distributing across both replicas;
- least-inflight routing without a session ID;
- invalid, oversized, and missing session headers;
- unhealthy selected replica behavior;
- saturation `503` behavior;
- response replica headers;
- streaming response cleanup and in-flight decrement;
- tool-call forwarding;
- no unsafe retry after upstream dispatch;
- gateway `/v1/models` listing all public IDs;
- `/readyz` reporting each replica and pool state.

Do not add raw session IDs as Prometheus labels or logs.

## 8. Gateway Service Configuration

Update `lan-inference-gateway.service` so it no longer `Wants=` or orders itself
after the stock Qwen unit. The gateway should depend on the active profile
through the profile switch, not hard-code one Qwen topology.

The gateway remains the only LAN-facing inference endpoint:

```text
0.0.0.0:8088 -> authenticated gateway
127.0.0.1:8080 -> GPU1 llamAmpere
127.0.0.1:8081 -> GPU2 llamAmpere
```

Keep ports 8080 and 8081 loopback-only. Do not expose either model server to
the LAN.

## 9. Prometheus and Metrics Exporter

Prometheus requires no new scrape target. Continue scraping:

```text
ai_metrics_exporter:9108
CPU profiler:9109
```

Change the exporter profile from one Qwen backend with `gpus: ["1", "2"]`
to two independent backends:

```json
{
  "qwen_atx_gpu1": {
    "url": "http://127.0.0.1:8080/metrics",
    "model": "qwen3.8-27b-atx-iq4xs-m-262144-gpu1",
    "gpus": ["1"]
  },
  "qwen_atx_gpu2": {
    "url": "http://127.0.0.1:8081/metrics",
    "model": "qwen3.8-27b-atx-iq4xs-m-262144-gpu2",
    "gpus": ["2"]
  }
}
```

Retain the Muse backend unchanged.

The exporter must reject duplicate physical GPU ownership in one active
profile. Never configure stock Qwen and llamAmpere backends simultaneously,
because both claim GPUs 1 and 2 and metrics attribution becomes ambiguous.

llamAmpere exposes the main llama metrics, but it does not expose the global
cumulative cached-prompt-token metric expected by current cost accounting.
Before enabling cache-inclusive API-cost panels for llamAmpere, choose one:

1. Add an aggregate cached-token metric to the pinned llamAmpere runtime; or
2. Mark cached-token accounting as `unobserved` for llamAmpere and suppress
   cache-inclusive cost estimates rather than reporting false zero-cache data.

The second option is the initial production choice.

Add exporter metrics for:

- active profile;
- backend/replica health;
- pool route decisions by replica and reason;
- session-affinity hits and misses;
- saturation rejections;
- explicit cold-failover events;
- cache-accounting observation status.

Do not include session IDs in metric labels.

## 10. Grafana

The existing Prometheus datasource and scrape job remain unchanged.

Update the dashboard to:

- group model health by `model`, `gpu`, and `gpu_uuid`;
- show one health tile per llamAmpere replica;
- show prefill/decode throughput per replica;
- show active requests and route decisions per replica;
- show GPU utilization, VRAM, temperature, power, and energy per replica;
- show affinity hit/miss and saturation rejection counters;
- show whether cached-token accounting is observed or unavailable;
- avoid legends containing only `{{model}}` when two replicas are active.

The model selector must use active `ai_model_up` values rather than historical
Prometheus label values.

Add an optional MTP acceptance panel if the runtime metrics expose the draft
and accepted token counters. Keep that panel separate from decode throughput.

## 11. Installer and Profile Files

Update:

```text
operations/install.sh
operations/config/ai-metrics-exporter.env.example
operations/systemd/lan-inference-gateway.service
operations/systemd/ai-metrics-exporter.service
operations/promote-muse-glimmer-30b-131k.sh
```

The installer currently hard-codes the stock Qwen backend and unconditionally
enables/restarts stock Qwen. Replace that behavior with:

- immutable profile files under `operations/config/profiles/`;
- a root-only profile switch;
- backups of root-owned environment files before every switch;
- no unconditional stock-Qwen restart;
- no deletion of the stock service or model;
- no automatic fallback.

Suggested profile files:

```text
operations/config/profiles/atx-dual.gateway.env
operations/config/profiles/atx-dual.metrics.env
operations/config/profiles/stock-q4-tensor.gateway.env
operations/config/profiles/stock-q4-tensor.metrics.env
```

The profile files must preserve the existing gateway authentication token and
request timeout. They must not contain credentials in the repository.

Update any promotion script that currently reinstalls or starts stock Qwen so
it invokes the profile switch instead.

## 12. Direct Cutover Procedure

This is a direct switch, not a weighted canary. Perform it only after Sections
5 and 13 pass.

### Preflight

1. Verify the immutable runtime, model, and vocabulary hashes.
2. Verify both systemd units with `systemd-analyze verify`.
3. Verify the profile files contain the correct two aliases and endpoints.
4. Verify GPU0 is not selected by either replica.
5. Verify ports 8080, 8081, 8088, 9108, and 9109 ownership.
6. Confirm host RAM and swap headroom for two prompt-cache/checkpoint users.
7. Back up installed units and root-owned environment files.
8. Confirm the gateway client token is unchanged.

### Switch

1. Stop the gateway and wait for its process to exit.
2. Stop the active Qwen service profile.
3. Verify physical GPUs 1 and 2 have no Qwen compute processes.
4. Install/start both llamAmpere units under `atx-dual`.
5. Wait for direct `/health`, `/v1/models`, and `/metrics` on ports 8080 and
   8081.
6. Verify GPU1/GPU2 placement and GPU0 remains untouched.
7. Install the ATX exporter profile and restart the exporter.
8. Install the ATX gateway profile and start the gateway.
9. Verify authenticated `/readyz`, `/v1/models`, non-streaming completions,
   streaming completions, and both explicit model IDs.
10. Test the pooled model with two distinct session IDs and confirm response
    replica headers.
11. Run the operator's direct production tests.

Do not call `operations/install.sh` in its current form during the cutover; it
would restore/start the stock Qwen topology.

## 13. Production Acceptance Tests

### Service tests

- Both units are active and stable under systemd.
- GPU1 has only the GPU1 replica; GPU2 has only the GPU2 replica.
- GPU0 is not used by either replica.
- Direct `/health` returns success on both ports.
- Direct `/v1/models` exposes the correct aliases.
- Direct `/metrics` returns compatible metrics.
- No listener exposes ports 8080 or 8081 beyond loopback.

### Gateway tests

- Authentication remains required.
- `/v1/models` lists both explicit IDs and the pooled ID.
- Explicit GPU1 requests always reach GPU1.
- Explicit GPU2 requests always reach GPU2.
- Repeated pooled requests with the same session header stay on one replica.
- Different session headers can use both replicas concurrently.
- Session routing does not log raw session IDs.
- Unkeyed requests use least-inflight selection.
- A saturated pinned replica returns retryable `503` rather than silently
  destroying cache locality.
- SSE streams complete and decrement in-flight state.
- Tool calls are forwarded without replay.

### Model tests

- Normal-EOS thinking output.
- T4/T8 objective probes.
- Tool/MCP smoke.
- Cold and warm 196k requests.
- Cold and warm 240k requests.
- Full 262k requests if Section 5 passed.
- Two simultaneous long-context requests, one per replica.
- No truncation, malformed structured output, OOM, restart, or context shift.

### Observability tests

- Exporter reports both replica model IDs as up.
- Prometheus sees both GPU labels and UUID labels.
- Grafana distinguishes GPU1 and GPU2.
- Cached-input cost is visibly marked unavailable/unobserved for llamAmpere.
- Pool route and affinity metrics increment with bounded labels.
- Gateway readiness identifies an unavailable replica.

### Reboot tests

- `atx-dual` is the selected profile after reboot.
- Both llamAmpere units start and become healthy.
- Stock Qwen remains stopped unless explicitly selected.
- Gateway and exporter start after model services are ready.
- Muse remains unchanged on GPU0.

## 14. Explicit Profile Rollback

Rollback is manual profile selection, not automatic fallback:

```text
sudo /usr/local/sbin/ai-qwen-profile-switch stock-q4-tensor
```

The switch must:

1. Stop gateway.
2. Stop both llamAmpere units.
3. Verify ports 8080 and 8081 are free and GPUs 1 and 2 are clear.
4. Install the stock gateway and exporter profiles.
5. Start `llama-qwen3.8-q4-tensor-262k.service`.
6. Verify stock `/health`, `/v1/models`, and `/metrics`.
7. Start exporter and gateway.
8. Verify model ID `qwen3.8-27b-q4-tensor262k` and authenticated requests.

The stock service remains installed, hash-verifiable, and selectable. It is not
started automatically if llamAmpere fails.

## 15. Client and Documentation Updates

Update:

```text
operations/clients.md
operations/README.md
operations/runbooks.md
operations/network.md
operations/deployment-record.local.md
```

Document:

- both explicit GPU model IDs;
- the pooled model ID and `X-Inference-Session` header;
- the active context contract, either 262144 or 245760 based on Section 5;
- direct profile switching;
- manual stock rollback;
- no automatic fallback;
- GPU ownership and port mapping;
- cache-accounting limitation.

Update benchmark wrappers and SWE-bench launchers so they do not blindly claim
ports 8080/8081 or GPUs 1/2 while `atx-dual` is active.

## 16. Deferred llm-router Evaluation

Do not deploy `llm-router` in the first production cutover.

Evaluate it later only if one of these becomes true:

- gateway routing grows beyond a small, testable replica-pool change;
- prefix-aware routing materially outperforms session affinity;
- more than two inference workers are added;
- the gateway must route across multiple hosts or runtimes.

The evaluation must run behind the existing authenticated boundary or in an
isolated test port. Compare:

- cold and warm TTFT at 196k and 240k;
- cache hit/reuse behavior;
- p50/p95/p99 latency;
- least-loaded versus prefix-aware behavior;
- session branching behavior;
- stream/tool failure semantics;
- metrics and Grafana integration;
- process and security complexity.

The current two-replica deployment should not depend on an unvalidated second
proxy.

## 17. Implementation Order

1. Add the full-context validation harness and run Section 5.
2. Stage and hash-verify the immutable llamAmpere release.
3. Add the two systemd units and profile files.
4. Refactor installer/profile switching without changing live services.
5. Implement gateway replica pools and session-affinity tests.
6. Implement exporter multi-backend profiles and cache-accounting status.
7. Update Grafana dashboard and monitoring documentation.
8. Run unit tests, configuration validation, and offline integration tests.
9. Run the two-replica direct readiness tests on loopback ports.
10. Execute the direct production cutover.
11. Run the production acceptance suite in Section 13.
12. Record the selected context contract and deployment hashes.
13. Keep the stock profile available for explicit manual rollback.

## 18. Completion Record

The upgrade is complete only when this file is updated with:

- full-context result and selected context contract;
- deployed runtime/model/vocabulary hashes;
- active systemd unit names and ports;
- gateway pool and session-header behavior;
- exporter and dashboard validation results;
- direct production test results;
- reboot result;
- explicit rollback test result;
- operator sign-off.

Until then, this document is the implementation plan and no production service
or routing change is implied by its existence.
