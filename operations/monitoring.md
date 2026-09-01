# Monitoring

Prometheus and Grafana run on the monitoring host. The AI server exporter listens
on port 9108 and combines llama.cpp metrics with `nvidia-smi` metrics.

Expected model labels:

- `model="qwen3.8-27b-q4-gpukv192"`
- `gpu="0"`

The dashboard should show request processing, prompt tokens/s, generated
tokens/s, cumulative prompt/generated tokens, GPU utilization, power draw and
limit, temperature, VRAM, host memory, and exporter/service health.

Host memory metrics are exposed as:

- `ai_host_memory_total_bytes`
- `ai_host_memory_available_bytes`
- `ai_host_memory_used_bytes`

The Prometheus scrape job is in `config/prometheus-ai-server.yml`. Grafana
should use the existing Prometheus data source and import
`config/grafana-ai-server-dashboard.json` into dashboard UID
`ai-server-qwen-q4`. The authenticated API procedure is documented in
`deployment-record.md`; verify the resulting dashboard at
`/d/ai-server-qwen-q4`.

After changing the exporter or dashboard source files, restart the exporter and
re-import the dashboard with overwrite enabled:

```bash
sudo systemctl restart ai-metrics-exporter.service
```
