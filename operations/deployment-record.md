# Deployment Record Template

This public template records the supported deployment shape without exposing
hostnames, addresses, usernames, filesystem paths, or credentials. Keep the
actual inventory and validation history in the ignored
`deployment-record.local.md` file.

## Inventory

Record these values only in the local deployment record:

- AI server hostname and LAN address
- AI server Tailscale address
- NAS hostname and monitoring addresses
- Service account and repository path
- Model-server binary and model-file paths
- Model and GPU assignments

## Supported Runtime

- Qwen3.8-Flash-Next IQ3_S via the Strata runtime on GPUs 0 and 1
- 524,288-token context, int8 KV cache, YaRN scale 2, one active session
- Qwen3.8-27B ATX IQ4_XS single llamAmpere replica on GPU 2
- 262,144-token context, one slot, Q8_0 K / turbo3 V cache, MTP speculative drafting
- Model services bound to loopback on ports 8081 and 8082
- Authenticated OpenAI-compatible gateway on port 8088
- Prometheus exporter on port 9108
- Fixed fan speeds at boot when supported by NVML: GPU 0 at 70%, GPU 1 at 80%, GPU 2 at 85%
- Reasoning default: xhigh for Flash (persisted as `high`), `auto` with medium effort for Qwen
- Sampling parameters supplied by clients

## Active Profile

The active profile is `flash512-qwen-single`: the Strata Flash engine on
physical GPUs 0 and 1 at loopback port 8082, plus one llamAmpere replica on
GPU 2 at port 8081, and the pooled gateway model ID
`qwen3.8-27b-atx-iq4xs-m-262144` with that single member and
`X-Inference-Session` affinity (OpenCode's `X-Session-Id` is also accepted).

Muse Glimmer on GPU 0, the second llamAmpere replica on GPU 1, and the stock
`stock-q4-tensor` profile are retired: their units are disabled at boot and
conflict-guarded against the Flash unit. Restoring them is a deliberate manual
procedure; the two legacy topologies must never run concurrently with Flash.

## Historical Qwen Tensor Validation

The validated 2026-09-09 Qwen tensor profile uses the existing PCIe Gen3 x16
links and DDR4-3066 memory configuration. It runs on GPUs 1 and 2 with the
native 262,144-token context, F16 K/V, Flash Attention, and default NCCL.

The production selection measured 856.45 prefill tok/s and 36.95 decode tok/s
with a 196k input plus 4,096 output workload, and 763.97 prefill tok/s and 33.49
decode tok/s with a 257k input plus 4,096 output workload. Each Qwen GPU peaked
at approximately 16,862 MiB with no OOM or swap.

The profile uses batch/ubatch `2048/1024`; rejected variants included Q5, Q8,
BF16 K/V, forced NCCL protocols, and one-channel NCCL overrides.

## Required Secrets

The gateway token and exporter configuration belong in root-owned files outside
the repository. The public examples in `config/` must contain placeholders only.
Never record the token, a Tailscale authentication key, a Grafana password, or a
client API key in this repository.

## Monitoring

Merge `config/prometheus-ai-server.yml` into the monitoring server's
configuration, validate it with `promtool`, and restart Prometheus only after
validation succeeds. Import `config/grafana-ai-server-dashboard.json` with
overwrite enabled and verify the dashboard UID documented in `monitoring.md`.

## Validation Checklist

After installation, apply the Flash promotion and verify:

```bash
cat /etc/ai-server/qwen-serving-profile                         # flash512-qwen-single
systemctl is-active nvidia-power-limit.service nvidia-fan-control.service
systemctl is-active strata-flash-512k.service llamampere-qwen3.8-atx-iq4xs-m-gpu2.service
systemctl is-active lan-inference-gateway.service ai-metrics-exporter.service
systemctl is-enabled strata-flash-512k.service llamampere-qwen3.8-atx-iq4xs-m-gpu2.service
nvidia-smi --query-gpu=index,power.limit --format=csv
curl -sS http://127.0.0.1:8081/health
curl -sS http://127.0.0.1:8082/health
curl -sS -H "Authorization: Bearer $LAN_GATEWAY_CLIENT_TOKEN" \
  http://127.0.0.1:8088/readyz
cat /var/lib/ai-server/flash-deployment.json
```

The retired Muse, GPU1 llamAmpere, and stock tensor-split services must be
inactive and disabled. Confirm that the gateway rejects unauthenticated `/v1/*` requests, Prometheus
reports the exporter target as healthy, and the host-memory metrics appear in
the monitoring dashboard.
