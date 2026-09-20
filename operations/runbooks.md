# Operations Runbook

## Service Status

```bash
systemctl status nvidia-power-limit.service
systemctl status nvidia-fan-control.service
systemctl status llamampere-qwen3.8-atx-iq4xs-m-gpu1.service
systemctl status llamampere-qwen3.8-atx-iq4xs-m-gpu2.service
systemctl status llama-muse-glimmer-30b-131k.service
systemctl status lan-inference-gateway.service
systemctl status ai-metrics-exporter.service
systemctl status ai-cpu-power-profiler.service
cat /etc/ai-server/qwen-serving-profile
nvidia-smi
```

The active Qwen profile is selected by the root-only profile switch. The
`atx-dual` profile owns GPU1/port 8080 and GPU2/port 8081; Muse continues to own
GPU0/port 8082. The stock Qwen unit is stopped unless `stock-q4-tensor` is
selected explicitly.

The standalone RTX 5090 profile is not selected by this switcher. Its status
uses distinct units:

```bash
systemctl status ai-rtx5090-gpu0-policy.service
systemctl status ai-rtx5090-qwen3.8-q4-native.service
systemctl status ai-rtx5090-monitor.service
```

Use `operations/hardware/rtx5090/README.md` for its ownership checks and
installer. Never resolve a conflict by disabling the canonical global GPU
policy or an unrelated model service from that profile installer.

The llamAmpere replicas append to `/var/log/ai-server/llamampere-qwen-gpu1.log`
and `llamampere-qwen-gpu2.log`; the installer creates that directory and installs
`/etc/logrotate.d/ai-server` to rotate the files daily. Both `LogsDirectory=` in
the units and the `append:` targets require the directory to exist before the
unit starts, so never remove it without recreating it (`systemd-tmpfiles
--create`).

## Reinstall or Apply Unit Changes

The installer installs the canonical units, preserves an existing gateway token,
and applies the GPU power and fan settings. It does not choose a Qwen topology.
After installation, select the profile explicitly:

```bash
sudo env \
  AI_SERVER_LAN_SUBNET=192.0.2.0/24 \
  AI_SERVER_PROMETHEUS_IP=192.0.2.10 \
  /path/to/localLLMinference/operations/install.sh
sudo /usr/local/sbin/ai-qwen-profile-switch atx-dual
```

Run this command again after changing a canonical unit file. It performs the
systemd daemon reload; the profile switch starts the services for the selected
topology.

Gateway restart shutdown is bounded to five seconds so an in-flight request
cannot hold the profile switch or a deliberate restart indefinitely; that
request may be interrupted during a cutover or restart.

## API Check

```bash
curl -sS http://127.0.0.1:8080/health
curl -sS http://127.0.0.1:8081/health
curl -sS http://127.0.0.1:8082/health
curl -sS http://127.0.0.1:8088/healthz
curl -sS http://127.0.0.1:8088/metrics
curl -sS -H "Authorization: Bearer $LAN_GATEWAY_CLIENT_TOKEN" \
  http://127.0.0.1:8088/readyz
curl -sS -H "Authorization: Bearer $LAN_GATEWAY_CLIENT_TOKEN" \
  http://127.0.0.1:8088/v1/models
```

`/v1/*` and `/readyz` are authenticated. Gateway `/metrics` is the bounded,
unauthenticated diagnostics endpoint; it must not contain raw session IDs.

## Restart

Restart only the services belonging to the selected profile. For `atx-dual`:

```bash
sudo systemctl restart llamampere-qwen3.8-atx-iq4xs-m-gpu1.service
sudo systemctl restart llamampere-qwen3.8-atx-iq4xs-m-gpu2.service
sudo systemctl restart llama-muse-glimmer-30b-131k.service
sudo systemctl restart lan-inference-gateway.service
sudo systemctl restart ai-metrics-exporter.service
```

Do not start stock Qwen alongside either llamAmpere replica. Use the profile
switch for a topology change so ports, GPU ownership, environment backups, and
readiness checks are performed in order.

## Rollback

Rollback is manual, cold, and mutually exclusive. It is not an automatic
fallback:

```bash
sudo /usr/local/sbin/ai-qwen-profile-switch stock-q4-tensor
```

The switch stops the gateway and both llamAmpere units, verifies ports 8080 and
8081 and GPUs 1 and 2 are clear, installs the stock gateway/exporter profiles,
starts `llama-qwen3.8-q4-tensor-262k.service`, and verifies authenticated
gateway routing. If llamAmpere fails, leave it visible through systemd,
`/readyz`, and Prometheus until the operator chooses restart or rollback.

## Reboot Acceptance

After a reboot verify that the selected profile marker is `atx-dual`, both
llamAmpere units own only their intended GPU and loopback port, Muse reports
model ID `muse-glimmer-30b-kquant17` on GPU0, the gateway requires a token for
`/v1` and `/readyz`, `/metrics` is reachable without a token, the exporter is
reachable by Prometheus, and the stock Qwen service did not start. If the
operator selected `stock-q4-tensor`, verify that profile and its single Qwen
model ID instead.

The local deployment state, active profile, and acceptance gates are in
`deployment-record.local.md`.

## CPU Power Profiler & Monitoring

The always-on `ai-cpu-power-profiler.service` (port 9109) measures CPU package
power from the RAPL `package-0` counter plus `/proc/stat` utilization; the main
exporter (port 9108) resolves it per scrape as `rapl -> interpolated -> linear`.
Metric semantics are in `monitoring.md`.

Quick check (AI server):

```bash
systemctl is-active ai-cpu-power-profiler.service ai-metrics-exporter.service
curl -s http://127.0.0.1:9109/metrics | grep -E 'rapl_available|ai_cpu_power_watts|samples_total'
curl -s http://127.0.0.1:9108/metrics | grep ai_host_cpu_power_source_info
```

`ai_cpu_profiler_rapl_available` must be `1`. If `0`, the counter is unreadable;
`ai_cpu_profiler_rapl_enabled` reports the powercap `enabled` flag for diagnosis
(some kernels report `enabled=0` while the counter still counts, so the flag is
not the cause).

### Reload the 9109 scrape on the monitoring host (privileged)

Add the `ai-server-cpu-power` job (port 9109) from `config/prometheus-ai-server.yml`
to the NAS `prometheus.yml` (back up first; Synology has no `scp`, so transfer via
`ssh <nas> 'cat > file'`). Then validate and reload:

```bash
sudo docker exec prometheus promtool check config /etc/prometheus/prometheus.yml \
  && sudo docker restart prometheus
curl -s 'http://<NAS_IP>:9090/api/v1/query' --data-urlencode 'query=up{job="ai-server-cpu-power"}'
```

The AI server firewall must allow 9109 from the monitoring host only:
`sudo ufw allow from <MONITOR_IP> to any port 9109 proto tcp`.

### Cost config: CPU power and baseline mode (privileged)

`/etc/ai-server/ai-cost-accounting.json` carries a `cpu_power` block
(`profiler_url` + `linear` bootstrap) and a `baseline` that selects the host
model. Deployed mode is `wall_idle_plus_cpu_w` with `base_idle_total_w` 127 W:
whole-host = `base + cpu (full RAPL) + sum(max(gpu - 20, 0))`, so idle GPUs stay
in the base and CPU/GPU are added on top. The split is exposed as
`ai_host_power_attribution_watts{source=base|cpu|gpu}`.

The service runs as `User=obenomar`, so the config must stay group-readable
(`root:obenomar 0640`). Editing it with `tempfile.mkstemp` + `os.replace` under
sudo rewrites it as `root:root 0600` and the exporter crash-loops with a
`PermissionError` at start; after any such edit run
`sudo chown root:obenomar /etc/ai-server/ai-cost-accounting.json && sudo chmod 640
/etc/ai-server/ai-cost-accounting.json`, then
`sudo systemctl restart ai-metrics-exporter.service`.

## Context And Reasoning Effort

The llamAmpere profile files request context `262144`, and the native 262144
acceptance gate passed in isolated runs on both GPUs and concurrently. The
matching `...-262144` IDs are the validated staged configuration; live endpoint
and cutover acceptance are still required. Muse remains at 131072.

Use the model-specific OpenCode variants: Qwen defaults to `medium` and
supports `none`, `low`, `medium`, `high`, and `xhigh`; Muse Glimmer defaults to
`high` and supports `low`, `medium`, `high`, and `xhigh`. Muse does not expose a
reliable non-thinking mode. For an exact Q4 non-thinking benchmark profile,
use a separate one-off llama.cpp invocation with `--reasoning off`; do not
change the general-purpose service for that test.

Sampling parameters are caller-controlled. Use the Qwen-recommended values for
the selected thinking or non-thinking mode when reproducibility is required.

An OpenCode request also contains its system and tool context, so a fresh
session can still be slow before the first output token. For pooled llamAmpere
requests, send a stable `X-Inference-Session` value to preserve replica cache
locality. The response identifies the selected lane with
`X-Inference-Replica`.

Thinking example:

```json
{
  "reasoning_effort": "medium",
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 20,
  "presence_penalty": 0,
  "repeat_penalty": 1.0
}
```

Non-thinking example:

```json
{
  "reasoning_effort": "none",
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "min_p": 0,
  "presence_penalty": 1.5,
  "repeat_penalty": 1.0
}
```
