# AI Server Operations

Operational reference for a self-hosted AI server. Research results stay under
`../research/`; this directory describes the supported production setup.

## Supported Profile

- Qwen3.8-27B Q4_K_M on GPU 1.
- Qwen3.8 Flash Next NVFP4 on GPU 0 with a 262144-token context.
- Flash Next uses 2,048-token prefill chunks and a 2,048-slot GPU MoE cache.
- Muse Glimmer 30B K-Quant on GPU 2 with a 131072-token context.
- 196608-token context with Q8 GPU KV and Flash Attention.
- Internal llama.cpp endpoint: `http://127.0.0.1:8080`.
- Authenticated OpenAI gateway on port `8088` on the LAN and tailnet.
- GPU power limit: 300 W on all three RTX 3090 GPUs, applied at system boot.
- Fixed fan speeds at system boot: GPU 0 and GPU 2 at 70%, GPU 1 at 80%.
- OpenCode reasoning defaults are Q4 `medium`, Flash Next `xhigh`, and Muse
  Glimmer `high`. Q4 supports `none`, `low`, `medium`, `high`, and `xhigh`;
  Flash Next supports `none`, `low`, `medium`, and `xhigh`; Muse Glimmer
  supports `low`, `medium`, `high`, and `xhigh`.
- Sampling parameters are not baked into the service. Clients should provide
  mode-specific sampling parameters when reproducibility matters.

## Files

- `systemd/`: canonical unit files installed under `/etc/systemd/system/`.
- `config/`: secret-free configuration examples and monitoring snippets.
- `install.sh`: root installer for the system services and firewall rules.
- `runbooks.md`: start, stop, recovery, and rollback procedures.
- `network.md`: LAN, gateway, firewall, and Tailscale access.
- `monitoring.md`: Prometheus and Grafana setup.
- `clients.md`: OpenCode, Oh My Pi, and generic OpenAI-compatible client setup.
- `deployment-record.md`: host inventory, installation history, final topology,
  validation results, and remaining acceptance checks.

## API

Use the gateway, not the private llama.cpp port. Clients should set:

```text
OPENAI_BASE_URL=http://<AI_SERVER_HOST>:8088/v1
OPENAI_API_KEY=<gateway token>
```

For remote access, replace the host with the Tailscale address or MagicDNS name.

## Safety

- Port 8080 remains loopback-only.
- Port 8088 requires a bearer token for `/v1/*` routes.
- The metrics exporter is restricted to the Prometheus host by firewall rules.
- If fixed fan control is unsupported by the driver or board, the model falls
  back to the driver's automatic fan policy.
- Do not put API keys, Tailscale auth keys, or GitHub credentials in this tree.

## Installation

After Tailscale authentication, run:

```bash
sudo /path/to/localLLMinference/operations/install.sh
```

The installer creates the gateway token only if it does not already exist.
