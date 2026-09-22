# Qwen3.8 vLLM TP2 Cookbook

This is the operational cookbook for the deployed Qwen3.8-27B vLLM TP2 service
on the two RTX 5090 GPUs. It replaces the former single-GPU llama.cpp Q4
service. The original Q4 units remain installed as the rollback target.

## Service Contract

| Item | Value |
| --- | --- |
| Public listener | `0.0.0.0:8080` |
| Private vLLM listener | `127.0.0.1:18081` |
| Public model ID | `qwen3.8-27b-q4-gpukv-native` |
| Context limit | 262,144 tokens |
| Backend concurrency | 3 sequences |
| Long-input admission | Gateway C<=2 policy |
| Client authentication | Disabled, preserving the previous endpoint behavior |

The public gateway provides `GET /healthz`, `GET /readyz`, `GET /metrics`,
`GET /v1/models`, and `POST /v1/chat/completions`, including SSE streaming.
It is not a transparent llama.cpp proxy: former llama.cpp-specific endpoints
such as `/health`, `/completion`, `/tokenize`, `/props`, and `/slots` are not
available.

The selected long-input policy uses a conservative message-character estimate.
Requests estimated at 100,000 tokens or more are admitted only when the vLLM
scheduler has at most two running requests and no waiting work. The gateway
returns `X-Inference-Admission: c2` and
`X-Inference-Gateway-Admission-Wait-Ms` when that policy applies. Short
requests bypass this long-input gate.

## Expected Performance And Admission

These are measured reference values, not service-level guarantees. Prompt
length, cache residency, output length, reasoning effort, tool use, and the
first use of a new shape materially affect latency.

| Concurrent users and workload | Expected prefill / first token | Expected decode | Admission behavior |
| --- | --- | --- | --- |
| 1-3 cached 196k conversations, each adding 1k-4k tokens | The qualified C=3 appends reached first token in 4.31-7.47 s. | About 149 tok/s client aggregate; vLLM reported 152.3-152.6 tok/s aggregate, or about 47-51 tok/s per stream. C=3 p95 ITL was about 39-40 ms. | All three run directly; vLLM reports `Running: 3`, `Waiting: 0`. |
| 1-3 short GSM8K-like requests | Short prompt prefill is workload-dependent. | The 256-case validation logged about 239-259 tok/s aggregate at C=3. This is not a long-context agent-rate estimate. | Short work bypasses the long-input gateway gate. |
| A new 196k cold context when the backend has at most two active long requests | In the selected C<=2 test, post-admission TTFT was 115.992 s. | The two established decoders retained 36 ms p95/p99 ITL during this prefill. | The request is posted immediately when the scheduler reports no waiters and at most two running requests. |
| Fourth or later long context while three long requests are active | It waits at the gateway until one active request finishes and the backend reaches C<=2. After dispatch, use the approximately 116 s cold TTFT reference above. | The fourth cold prefill is not allowed to enter a full C=3 backend, avoiding the measured native-C=3 393 ms active-user p95/p99 regression. | The gateway holds up to three long requests FIFO outside vLLM. A fourth queued long request beyond that capacity receives `503` with `Retry-After: 1`; a queued request expires after 900 s. |

The fourth-user wait is intentionally workload-dependent: it is the remaining
duration of the active requests, not a fixed latency. The qualifying C<=2
comparison submitted the cold user immediately after one original request
completed, so its measured gateway wait was effectively zero. A direct native
C=3 submission instead spent 40.431 s waiting inside vLLM, had 156.335 s
user-visible TTFT, and violated the 300 ms active-user p95 target.

No performance contract has been qualified for arbitrary mixtures of four or
more short requests, tool-heavy payloads, or unclassified long requests. The
gateway classifies only `messages` content with a conservative character
estimate; callers should retry the documented `503` response rather than
expecting unbounded queueing.

Measurement sources:

- `research/qwen3.8-27b-rtx5090-phase-j-client-side-cold-admission-plan-2026-09-22.md`
- `research/qwen3.8-27b-rtx5090-phase-k-gateway-admission-implementation-plan-2026-09-22.md`
- `/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-e-gsm8k-recovery-256-vllm-2026-09-22T115654Z/candidate/vllm-0.server.log`

## Units

| Unit | Purpose |
| --- | --- |
| `ai-qwen3.8-vllm-tp2.service` | Private vLLM TP2 backend on GPUs 0 and 1 |
| `ai-qwen3.8-vllm-tp2-gateway.service` | Public port 8080 OpenAI chat gateway |
| `ai-qwen3.8-vllm-tp2-monitor@0.service` | GPU 0 telemetry |
| `ai-qwen3.8-vllm-tp2-monitor@1.service` | GPU 1 telemetry |
| `ai-qwen3.8-vllm-tp2-metrics.service` | Tailscale-only Prometheus exporter on port 9108 |
| `llama-qwen3.8-q4-native.service` | Disabled legacy Q4 rollback backend |
| `ai-qwen3.8-q4-native-monitor.service` | Disabled legacy Q4 monitor |

All deployed units are enabled at boot. The backend requires the existing
Qwen GPU policy, which applies the 500 W power cap and 95% fan target to both
GPUs.

## Status And Health

```bash
systemctl is-active \
  ai-qwen3.8-vllm-tp2.service \
  ai-qwen3.8-vllm-tp2-gateway.service \
  ai-qwen3.8-vllm-tp2-monitor@0.service \
  ai-qwen3.8-vllm-tp2-monitor@1.service \
  ai-qwen3.8-vllm-tp2-metrics.service

systemctl is-enabled \
  ai-qwen3.8-vllm-tp2.service \
  ai-qwen3.8-vllm-tp2-gateway.service \
  ai-qwen3.8-vllm-tp2-monitor@0.service \
  ai-qwen3.8-vllm-tp2-monitor@1.service \
  ai-qwen3.8-vllm-tp2-metrics.service

curl -fsS http://127.0.0.1:8080/healthz
curl -fsS http://127.0.0.1:8080/readyz
curl -fsS http://127.0.0.1:8080/v1/models
ss -ltnp '( sport = :8080 or sport = :18081 )'
```

Expected listeners are public `0.0.0.0:8080` for the gateway and private
`127.0.0.1:18081` for vLLM. `/readyz` must report the
`qwen3.8-vllm-tp2` backend as ready and `/v1/models` must include
`qwen3.8-27b-q4-gpukv-native`.

Check the vLLM scheduler, admission state, and GPU ownership:

```bash
curl -fsS http://127.0.0.1:18081/metrics | rg \
  'vllm:(num_requests_(running|waiting)|generation_tokens_total).*qwen3.8-27b-q4-gpukv-native'
curl -fsS http://127.0.0.1:8080/metrics | rg 'ai_gateway_cold_admission_'
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,power.limit,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

Treat a sustained temperature at or above 85 C, unexpected GPU ownership, an
unhealthy `/readyz`, or a private vLLM listener exposed beyond loopback as an
operational failure. Roll back rather than changing the qualified runtime in
place.

## Prometheus Export

Install or refresh the exporter after deploying the vLLM service:

```bash
sudo /home/michel/LLMs-tests/localLLMinference/operations/install-qwen3.8-vllm-tp2-metrics.sh
systemctl status ai-qwen3.8-vllm-tp2-metrics.service
```

The installer discovers this machine's Tailscale IPv4 address and binds port
9108 only to that address. It does not open a LAN listener and does not enable
the unqualified CPU-power profiler. On `rtx-4000-test`, merge
`operations/config/prometheus-rtx5090-vllm-tp2.yml` and replace
`<RTX5090_TAILSCALE_IP>` with the address printed by the installer.

The exporter normalizes vLLM token counters, active requests, mean TTFT, and
counter-derived prefill/decode rates into `ai_model_*` metrics. It also relays
the gateway's bounded `ai_gateway_cold_admission_*` metrics with `host_id`, so
Grafana can show cold-work queue depth, waits, and admission outcomes without
scraping the public gateway directly.

## Client Check

```bash
curl -fsS --max-time 600 \
  -H 'Content-Type: application/json' \
  --data '{
    "model":"qwen3.8-27b-q4-gpukv-native",
    "messages":[{"role":"user","content":"Give a short greeting."}],
    "max_tokens":512,
    "temperature":0
  }' \
  http://127.0.0.1:8080/v1/chat/completions
```

The model's reasoning mode can consume a small output budget before producing
visible content. Use an output budget appropriate for the requested reasoning
work; a tiny `max_tokens` value is only an API-liveness check.

## Restart

Restarting the backend interrupts active requests. The gateway is bound to the
backend and must be healthy again before client traffic resumes.

```bash
sudo systemctl restart ai-qwen3.8-vllm-tp2.service
sudo systemctl restart ai-qwen3.8-vllm-tp2-gateway.service

until curl -fsS http://127.0.0.1:8080/readyz; do sleep 1; done
```

Startup normally takes about one minute. Validate `/v1/models` and the private
loopback listener after every backend restart. Inspect startup and graph logs
with:

```bash
journalctl -u ai-qwen3.8-vllm-tp2.service -u ai-qwen3.8-vllm-tp2-gateway.service --since '-30 min'
journalctl -u ai-qwen3.8-vllm-tp2-monitor@0.service -u ai-qwen3.8-vllm-tp2-monitor@1.service --since '-30 min'
```

## Initial Cutover Or Unit Update

The root-only cutover script is:

```bash
/home/michel/LLMs-tests/localLLMinference/operations/deploy-qwen3.8-vllm-tp2-cutover.sh
```

Without arguments it stages and validates the unit templates and environment
files without changing live services:

```bash
sudo /home/michel/LLMs-tests/localLLMinference/operations/deploy-qwen3.8-vllm-tp2-cutover.sh
```

`--activate` is for the initial migration from an enabled, active legacy Q4
service. It backs up the legacy units under `/var/lib/ai-server/`, starts the
private backend first, validates the public gateway, C=3 CUDA graphs, and C<=2
admission, and automatically restores Q4 if activation fails.

If the approved unrelated GPU 0 RAG process is still present, supply its live
PID. Verify the PID immediately before running the command:

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
sudo env AI_SERVER_QWEN38_ALLOWED_GPU0_PID=<rag-pid> \
  /home/michel/LLMs-tests/localLLMinference/operations/deploy-qwen3.8-vllm-tp2-cutover.sh --activate
```

Do not use `--activate` for a normal restart or to reapply a modified deployed
unit. It correctly refuses when the legacy Q4 service is no longer the active
deployment. Stage changes, inspect them with `systemd-analyze verify`, reload
systemd, then restart the affected deployed units in a maintenance window.

The exact model/runtime settings, quality exception, and initial acceptance
record are in:

`research/qwen3.8-27b-rtx5090-vllm-tp2-production-cutover-2026-09-22.md`

## Rollback To Q4

Rollback is cold and interrupts active vLLM requests. It disables the gateway,
both monitors, and private vLLM, waits for ports to release, then restores the
legacy Q4 enablement and health checks from the most recent activation capture.

```bash
sudo /home/michel/LLMs-tests/localLLMinference/operations/deploy-qwen3.8-vllm-tp2-cutover.sh --rollback
```

Confirm the legacy endpoint after rollback:

```bash
systemctl is-active llama-qwen3.8-q4-native.service ai-qwen3.8-q4-native-monitor.service
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/v1/models
```
