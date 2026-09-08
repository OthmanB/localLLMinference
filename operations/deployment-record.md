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

- Qwen3.8 Flash Next NVFP4 on GPU 0
- 262,144-token context
- QSA sparse attention with a 262,144-token GPU KV pool
- 2,048-slot GPU MoE cache with hybrid CPU/GPU expert decode
- Disk-backed PLE and 2,048-token prefill chunks
- FreeToken bound to loopback on port 1901
- Authenticated OpenAI-compatible gateway on port 8088
- Prometheus exporter on port 9108
- Fixed fan speeds at boot when supported by NVML: GPU 0 and GPU 2 at 70%, GPU 1 at 80%
- Reasoning default: `auto` with medium effort
- Sampling parameters supplied by clients

## Current Flash Validation

The validated 2026-09-08 Flash profile uses the existing PCIe Gen3 x16 link and
DDR4-3066 memory configuration, with 2,048 MoE cache slots and 2,048-token
prefill chunks. A fresh OpenCode window completed successfully; its first call
was slow before the first response because it paid the cold prompt prefill cost.

The native-context probe accepted 261,949 prompt tokens and 127 output tokens
in 656.85 seconds. It measured approximately 401 tok/s prefill and 36.7 tok/s
settled scheduler decode, with the marker returned exactly first. GPU0 peaked at
23,450 MiB allocated, with approximately 677 MiB free; no OOM or swap occurred.
Host-level monitoring showed approximately 85 GB RAM used in `htop`.

For comparison, the preceding 2,344-slot/512-token profile measured 39.61 tok/s
settled decode at native context. The current 2,048/2,048 profile is therefore
approximately 7.4% slower on decode while providing much faster prefill.

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
systemctl is-active nvidia-power-limit.service freetoken-qwen3.8-flash-next-262k.service
systemctl is-active lan-inference-gateway.service ai-metrics-exporter.service
nvidia-smi --query-gpu=index,power.limit --format=csv
curl -sS http://127.0.0.1:1901/health
curl -sS http://127.0.0.1:8088/readyz
```

Confirm that the gateway rejects unauthenticated `/v1/*` requests, Prometheus
reports the exporter target as healthy, and the host-memory metrics appear in
the monitoring dashboard.
