# Qwen3.8 RTX 5090 vLLM TP2 Production Cutover

Status: deployed on 2026-09-22 with an operator-approved quality exception.

## Quality Exception

The qualified 256-example recovery-aware GSM8K run completed with no evaluator,
transport, or scorer errors:

| Metric | Result |
| --- | ---: |
| Strict first pass | 246/256 (96.09375%) |
| Recovered result | 247/256 (96.484375%) |
| Recovery attempts / successes | 3 / 1 |
| Length truncations | 3 |

The historical promotion floor is 96.5% with zero truncations. The recovered
result is 0.015625 percentage points below that accuracy floor. The operator
explicitly waived that threshold for this deployment, acknowledging that the
256-case result is accepted operational evidence rather than a passed 1,319-case
promotion gate.

Evidence:

- `/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-e-gsm8k-recovery-256-vllm-2026-09-22T115654Z/candidate/results.json`
- `/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-e-gsm8k-recovery-256-vllm-2026-09-22T115654Z/candidate/gsm8k-recovery/summary.json`

## Deployment Contract

The replacement preserves the existing public address and model ID:

- Public listener: `0.0.0.0:8080`
- Public model ID: `qwen3.8-27b-q4-gpukv-native`
- Public API: `GET /v1/models` and `POST /v1/chat/completions`, including SSE
  streaming.
- Client authentication: disabled, preserving the existing unauthenticated
  endpoint. Authentication is a separately scoped hardening change.

This is intentionally not a transparent llama.cpp replacement. The former
llama.cpp-specific endpoints such as `/health`, `/completion`, `/tokenize`,
`/props`, and `/slots` are not exposed by the gateway. Gateway liveness is
`/healthz`; backend readiness is `/readyz`.

The existing `llama-qwen3.8-q4-native.service` remains installed, disabled
during the vLLM deployment, and is the immediate rollback target.

## Selected Runtime

The internal backend is loopback-only on `127.0.0.1:18081`; the dedicated
gateway owns public port 8080. This is required to retain the selected C<=2
long-request admission policy. Exposing vLLM directly would bypass that policy.

Pinned backend:

- Model: `/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-e-kv-fp8-calibration-2026-09-20T052906Z/export`
- Model `config.json` SHA-256:
  `731d9ea745aa5c171822e4e64b4a863ae3269620a43ca2bec8a1e42c13a89b68`
- vLLM: `0.29.1rc1.dev438+g01f1f58f1`
- Torch/CUDA: `2.13.0+cu132` / `13.2`
- TP: 2 across physical GPUs 0 and 1
- Context: 262,144
- KV cache: FP8 E4M3
- Maximum sequences: 3
- Attention: Triton attention and Triton/FLA GDN prefill
- Graphs: FULL_DECODE_ONLY, capture shapes 1 and 3
- Prefix caching and chunked prefill enabled

The deployed vLLM served-model name changes only the public alias from the
qualification artifact to `qwen3.8-27b-q4-gpukv-native`; no model or runtime
setting is otherwise changed.

The unpooled gateway admits long work using:

```text
max_active_leases=3
dispatch_running_limit=2
cold_min_input_tokens=100000
poll_interval_seconds=1.0
queue_depth=3
queue_timeout_seconds=900
upstream_timeout_seconds=600
```

The GSM8K runner used a threshold of one only to force each short evaluation
request through the admission path. The 100,000-token production threshold is

## Cutover Procedure

1. Capture the legacy unit files, enablement, endpoint responses, GPU ownership,
   and the exact candidate runtime/model manifests in a root-owned deployment
   record.
2. Install but do not start a new private vLLM unit, public gateway unit, and
   two GPU monitor instances. Validate their environment files and systemd
   syntax before stopping production.
3. Stop and disable only the legacy Q4 server and its monitor. Keep the GPU
   policy and the approved unrelated GPU 0 process running. Abort if another
   unexpected GPU compute process exists.
4. Start vLLM privately. Require `/health`, the preserved model ID in
   `/v1/models`, correct scheduler metrics, and the pinned runtime/model hash
   before exposing any public listener.
5. Start both monitors and the public gateway. Require `/healthz`, `/readyz`,
   `/v1/models`, non-streaming and streaming chat completions, C<=2 admission
   headers for a controlled long request, graph evidence, and expected two-GPU
   ownership.
6. Keep the Q4 service definition and model untouched throughout the deployment
   window. Roll back immediately on failed readiness, model mismatch, missing
   admission evidence, thermal issue, monitor failure, or client-visible chat
   regression.

## Rollback

1. Stop and disable the public gateway, both vLLM monitors, and the private
   vLLM service.
2. Confirm public port 8080 and private port 18081 are released and vLLM has
   released both GPUs.
3. Re-enable and start `llama-qwen3.8-q4-native.service` and
   `ai-qwen3.8-q4-native-monitor.service`.
4. Require legacy `/health`, `/v1/models`, and model ID validation before
   declaring the rollback complete.

## Deployment Outcome

The successful guarded activation started at `2026-09-22T20:01:07Z` and is
recorded in the root-only rollback capture:

`/var/lib/ai-server/qwen3.8-vllm-tp2-cutover-20260922T200107Z-651997`

The private backend, public gateway, and GPU 0/1 monitor instances are active
and enabled. The legacy Q4 service and monitor are installed but inactive and

Acceptance completed through public port 8080:

- `/healthz` and `/readyz` returned healthy/ready.
- The preserved model ID was present in both gateway and direct vLLM
  `/v1/models` responses.
- Non-streaming, streaming, and three concurrent chat completions succeeded.
- The server recorded FULL CUDA graph capture and C=3 graph statistics, with
  the Triton/FLA GDN prefill kernel enabled.
- A controlled long request returned C<=2 admission headers; gateway metrics
  recorded one enqueue and one immediate dispatch with no queue or probe
  failures.
- Both GPUs were below the 85 C operational limit after validation: GPU 0 at
  43 C and GPU 1 at 33 C. The approved unrelated GPU 0 RAG process remained
  present alongside the TP0 worker.
