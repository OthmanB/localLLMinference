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

- Qwen3.8-27B UD Q4_K_M tensor-split across GPUs 1 and 2
- 262,144-token context with F16 GPU KV cache
- Flash Attention, 2,048 batch size, and 1,024 micro-batch size
- Native NCCL defaults with no UVM or NCCL environment overrides
- Muse Glimmer 30B K-Quant on GPU 0 with a 131,072-token context
- llama.cpp services bound to loopback on ports 8080 and 8082
- Authenticated OpenAI-compatible gateway on port 8088
- Prometheus exporter on port 9108
- Fixed fan speeds at boot when supported by NVML: GPU 0 at 70%, GPU 1 at 80%, GPU 2 at 85%
- Reasoning default: `auto` with medium effort for Qwen and high for Muse
- Sampling parameters supplied by clients

## Qwen Tensor Validation

The validated Qwen tensor profile uses the existing PCIe Gen3 x16 links and
DDR4-3066 memory configuration. It runs on GPUs 1 and 2 with the native
262,144-token context, F16 K/V, Flash Attention, and default NCCL.

The production profile passed long-context validation without OOM or swap. The
selected runtime uses batch/ubatch `2048/1024`; rejected variants included Q5,
Q8, BF16 K/V, forced NCCL protocols, and one-channel NCCL overrides.

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

After installation, verify:

```bash
  systemctl is-active nvidia-power-limit.service nvidia-fan-control.service
  systemctl is-active llama-qwen3.8-q4-tensor-262k.service llama-muse-glimmer-30b-131k.service
  systemctl is-active lan-inference-gateway.service ai-metrics-exporter.service
  nvidia-smi --query-gpu=index,power.limit --format=csv
  curl -sS http://127.0.0.1:8080/health
  curl -sS http://127.0.0.1:8082/health
  curl -sS http://127.0.0.1:8088/readyz
```

Confirm that the gateway rejects unauthenticated `/v1/*` requests, Prometheus
reports the exporter target as healthy, and the host-memory metrics appear in
the monitoring dashboard.
