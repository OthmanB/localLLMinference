# Local LLM Inference

Deployment templates, an OpenAI-compatible gateway, metrics tooling, benchmark
harnesses, and research notes for self-hosted language-model inference.

This repository intentionally excludes model weights, credentials, local
sessions, virtual environments, generated benchmark runs, and machine-specific
telemetry. Download model files separately and provide credentials through the
environment or an operating-system secret manager.

## Components

- `lan-inference-gateway/`: authenticated, runtime-independent OpenAI-compatible
  gateway with routing and readiness checks.
- `operations/`: systemd templates, installer, monitoring configuration, and
  operational runbooks.
- `tools/`: metrics exporter and benchmark preparation/control utilities.
- `swebench/`: reproducible SWE-bench launcher and configuration files.
- `research/`: benchmark summaries and design notes.

## Gateway Development

```bash
cd lan-inference-gateway
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/pytest
```

## Deployment

Review the files in `operations/config/` and the systemd templates before
installing. The current installer is tailored to a Linux host and requires the
repository path, service user, model path, firewall ranges, and backend profile
to match the target machine.

```bash
sudo env \
  AI_SERVER_LAN_SUBNET=192.0.2.0/24 \
  AI_SERVER_PROMETHEUS_IP=192.0.2.10 \
  /path/to/localLLMinference/operations/install.sh
```

The reserved documentation network values above are examples only. The full
variable list is in `operations/config/install.env.example`.

The installer generates the gateway token outside the repository. Never copy
that token, a Tailscale authentication key, a Grafana password, or a client API
key into this tree.

## Security

Keep the private llama.cpp listener loopback-only and expose only the
authenticated gateway. Do not commit files ignored by the root `.gitignore`,
and review generated logs and benchmark outputs before sharing them elsewhere.
