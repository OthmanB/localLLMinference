# Operations Runbook

## Service Status

```bash
systemctl status nvidia-power-limit.service
systemctl status nvidia-fan-control.service
systemctl status freetoken-qwen3.8-flash-next-262k.service
systemctl status llama-qwen3.8-q4-192k.service
systemctl status lan-inference-gateway.service
systemctl status ai-metrics-exporter.service
nvidia-smi
```

## Reinstall or Apply Unit Changes

The installer is idempotent for the generated secrets. It installs the
canonical units, preserves an existing gateway token, applies the GPU power and
fan settings, disables the old user llama services, starts the Flash Next model,
and starts the gateway and exporter:

```bash
sudo /path/to/localLLMinference/operations/install.sh
```

Run this command again after changing a canonical unit file. It performs the
systemd daemon reload and restarts the affected services.

Gateway restart shutdown is bounded to five seconds so an in-flight request
cannot hold the installer indefinitely; that request may be interrupted during
an installation or deliberate restart.

## API Check

```bash
curl -sS http://127.0.0.1:1901/health
curl -sS http://127.0.0.1:8088/healthz
curl -sS -H "Authorization: Bearer $LAN_GATEWAY_CLIENT_TOKEN" \
  http://127.0.0.1:8088/v1/models
```

## Restart

Restart the model before the gateway so the dependency remains clear:

```bash
sudo systemctl restart freetoken-qwen3.8-flash-next-262k.service
sudo systemctl restart lan-inference-gateway.service
sudo systemctl restart ai-metrics-exporter.service
```

## Rollback

Stop only the service being changed. Flash Next owns GPU 0 and Q4 owns GPU 1,
so they are intended to run together. Do not enable the old Q5 service without
checking GPU ownership first.

## Reboot Acceptance

After a reboot verify that the inference GPUs report their power limits
(GPUs 0-2 = 300 W), GPU 0's fixed fan service is active, Flash
Next reports model ID `qwen3.8-flash-next-nvfp4-262k`, Q4 reports model ID
`qwen3.8-27b-q4-gpukv192` on GPU 1, the gateway requires a token, the exporter
is reachable by Prometheus, and the old Q5 service did not start.

The full deployment history and NAS configuration are in
`deployment-record.md`.

## Reasoning Effort

Send one of `none`, `low`, `medium`, `high`, or `xhigh` as the OpenAI
`reasoning_effort` request field. The production default is medium via
`--reasoning auto --reasoning-effort medium`. For an exact non-thinking
benchmark profile, use a separate one-off server invocation with
`--reasoning off`; do not change the general-purpose service for that test.

Sampling parameters are caller-controlled. Use the Qwen-recommended values for
the selected thinking or non-thinking mode when reproducibility is required.

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
