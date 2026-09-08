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
(GPUs 0-2 = 300 W), the fixed fan service reports GPU 0 and GPU 2 at 70% and
GPU 1 at 80%, Flash
Next reports model ID `qwen3.8-flash-next-nvfp4-262k`, Q4 reports model ID
`qwen3.8-27b-q4-gpukv192` on GPU 1, the gateway requires a token, the exporter
is reachable by Prometheus, and the old Q5 service did not start.

The full deployment history and NAS configuration are in
`deployment-record.md`.

## Reasoning Effort

Use the model-specific OpenCode variants: Q4 defaults to `medium` and supports
`none`, `low`, `medium`, `high`, and `xhigh`; Flash Next defaults to `xhigh` and
supports `none`, `low`, `medium`, and `xhigh`; Muse Glimmer defaults to `high`
and supports `low`, `medium`, `high`, and `xhigh`. Muse does not expose a
reliable non-thinking mode. For an exact Q4 non-thinking benchmark profile,
use a separate one-off llama.cpp invocation with `--reasoning off`; do not
change the general-purpose service for that test.

Sampling parameters are caller-controlled. Use the Qwen-recommended values for
the selected thinking or non-thinking mode when reproducibility is required.

Flash Next is single-request and the validated profile uses a 2,048-slot GPU
MoE cache with 2,048-token prefill chunks. The native-context validation measured
approximately 401 tok/s prefill and 36.7 tok/s settled scheduler decode. The
preceding 2,344-slot/512-token profile measured 39.61 tok/s settled decode, so
the current profile trades approximately 7.4% decode throughput for much faster
prefill. An OpenCode request also contains its system and tool context, so a
fresh session can still be slow before the first output token. GPU activity
during this interval is prefill, not decode; subsequent calls benefit from the
runtime's populated context and expert state.

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
