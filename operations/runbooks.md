# Operations Runbook

## Service Status

```bash
systemctl status nvidia-power-limit.service
systemctl status nvidia-fan-control.service
systemctl status strata-flash-512k.service
systemctl status llamampere-qwen3.8-atx-iq4xs-m-gpu2.service
systemctl status lan-inference-gateway.service
systemctl status ai-metrics-exporter.service
systemctl status ai-cpu-power-profiler.service
cat /etc/ai-server/qwen-serving-profile
nvidia-smi
```

The active profile marker must read `flash512-qwen-single`. Flash owns
GPU0+GPU1/port 8082 and the remaining llamAmpere replica owns GPU2/port 8081.
The Muse, GPU1 llamAmpere, and stock Qwen units are disabled at boot and stay
stopped.

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

The deployed RTX 5090 vLLM TP2 replacement has its own public gateway on port
8080 and is also outside the profile switcher. Use
`operations/qwen3.8-vllm-tp2-cookbook.md` for its status, restart, initial
cutover, and rollback procedures. Do not start it alongside the legacy Q4 unit.

The active services append to `/var/log/ai-server/llamampere-qwen-gpu2.log`,
`flash-512k-server.log`, and `flash-512k-engine.log`; the installer creates
that directory and installs `/etc/logrotate.d/ai-server` to rotate the files
daily. Both `LogsDirectory=` in the units and the `append:` targets require the
directory to exist before the unit starts, so never remove it without
recreating it (`systemd-tmpfiles --create`).

## Reinstall or Apply Unit Changes

The installer installs the canonical units, preserves an existing gateway token,
and applies the GPU power and fan settings. It does not choose a topology.
After installation, apply the active profile explicitly:

```bash
sudo env \
  AI_SERVER_LAN_SUBNET=192.0.2.0/24 \
  AI_SERVER_PROMETHEUS_IP=192.0.2.10 \
  /path/to/localLLMinference/operations/install.sh
sudo python3 /path/to/localLLMinference/operations/promote_flash_512k.py --check
sudo python3 /path/to/localLLMinference/operations/promote_flash_512k.py
```

The promotion preserves the private gateway credential, replaces the
gateway/exporter backend lists, disables the retired units, and verifies both
services through the gateway before recording completion under
`/var/lib/ai-server/flash-deployment.json`. To re-apply unit-file changes for
the active profile without a full promotion:

```bash
sudo systemctl daemon-reload
sudo systemctl restart strata-flash-512k.service llamampere-qwen3.8-atx-iq4xs-m-gpu2.service
```

Gateway restart shutdown is bounded to five seconds so an in-flight request
cannot hold the profile switch or a deliberate restart indefinitely; that
request may be interrupted during a cutover or restart.

## API Check

```bash
curl -sS http://127.0.0.1:8081/health
curl -sS http://127.0.0.1:8082/health
curl -sS http://127.0.0.1:8088/healthz
curl -sS http://127.0.0.1:8088/metrics
curl -sS -H "Authorization: Bearer $LAN_GATEWAY_CLIENT_TOKEN" \
  http://127.0.0.1:8088/readyz
curl -sS -H "Authorization: Bearer $LAN_GATEWAY_CLIENT_TOKEN" \
  http://127.0.0.1:8088/v1/models
```

Gateway `/healthz` must list exactly `flash_iq3s_512k` and `qwen_atx_gpu2`.
Port 8080 is unused while the retired profiles stay stopped. `/v1/*` and
`/readyz` are authenticated. Gateway `/metrics` is the bounded,
unauthenticated diagnostics endpoint; it must not contain raw session IDs.

## Restart

Restart only the services belonging to the active profile:

```bash
sudo systemctl restart strata-flash-512k.service
sudo systemctl restart llamampere-qwen3.8-atx-iq4xs-m-gpu2.service
sudo systemctl restart lan-inference-gateway.service
sudo systemctl restart ai-metrics-exporter.service
```

Flash model loading can take minutes; `systemctl status` may stay healthy while
`/health` on 8082 still reports loading. Do not start retired services while
Flash runs; the units declare `Conflicts=`, but starting a retired unit first
would block Flash instead.

## Restoring A Retired Profile

Rollback is manual, cold, and mutually exclusive. It is not an automatic
fallback. First stop and disable Flash, then use the legacy profile switcher,
then restore the gateway/exporter settings that the promotion replaced:

```bash
sudo systemctl stop disable strata-flash-512k.service
sudo /usr/local/sbin/ai-qwen-profile-switch atx-dual   # or stock-q4-tensor
```

The switch verifies ports and GPU ownership and installs its profile's
gateway/exporter environment files. Alternatively, restore the pre-promotion
private files from the root-only backup the promotion created under
`/var/backups/ai-server/flash512-<UTC timestamp>/`. If a model service fails,
leave it visible through systemd, `/readyz`, and Prometheus until the operator
chooses restart or rollback.

## Reboot Acceptance

After a reboot verify that the selected profile marker is
`flash512-qwen-single`, `strata-flash-512k.service` and
`llamampere-qwen3.8-atx-iq4xs-m-gpu2.service` are enabled and own only their
intended GPUs and loopback ports, the gateway requires a token for `/v1` and
`/readyz`, `/metrics` is reachable without a token, the exporter is reachable
by Prometheus, and the Muse, GPU1 llamAmpere, and stock Qwen services did not
start.

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

The active services publish context `524288` for Flash (one session) and
`262144` for the remaining Qwen replica (one slot); both were verified at
promotion time. The retired Muse profile used 131072 and is not served.

Use the model-specific OpenCode variants. Flash supports `none`, `low`,
`medium`, and `xhigh`; Strata also accepts `high` and maps it to `xhigh`, so
they are not two different levels. The persisted server configuration uses
`reasoning_effort: high` (Strata 0.1.56 rejects `xhigh` in that field), which
means thinking at xhigh by default; `set_flash_thinking.py` re-applies that
default to an installed service. Qwen defaults to `medium` and supports
`none`, `low`, `medium`, `high`, and `xhigh`.

Sampling parameters are caller-controlled. Use the model-recommended values for
the selected thinking or non-thinking mode when reproducibility is required.

An OpenCode request also contains its system and tool context, so a fresh
session can still be slow before the first output token. For pooled llamAmpere
requests, send a stable `X-Inference-Session` value to preserve replica cache
locality; OpenCode's own `X-Session-Id` is accepted as well, so its sessions and
subagent sessions are pinned automatically. Requests without a session header
use the least-inflight replica with round-robin tie-breaking. The response
identifies the selected lane with `X-Inference-Replica`.

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
