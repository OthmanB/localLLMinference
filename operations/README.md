# AI Server Operations

Operational reference for the self-hosted AI server. Research results stay under
`../research/`; this directory describes profile boundaries and installation
contracts.

## Active Profile

The live three-GPU RTX 3090 profile is `flash512-qwen-single`, promoted by
`promote_flash_512k.py` and verified through the gateway:

- Strata `qwen3.8-flash-next-iq3_s` owns physical GPUs 0 and 1 on loopback
  port 8082 with a 524,288-token context and one active session.
- llamAmpere `qwen3.8-27b-atx-iq4xs-m-262144-gpu2` owns physical GPU 2 on
  loopback port 8081 with a 262,144-token context and one slot.
- Pooled model ID `qwen3.8-27b-atx-iq4xs-m-262144` remains published as a
  single-member pool over the GPU2 replica, with session affinity
  (`X-Inference-Session`, or the `X-Session-Id` header OpenCode already sends).

Muse Glimmer and the GPU1 llamAmpere replica are retired. Their units, and the
stock two-GPU `stock-q4-tensor` profile, are disabled at boot; the Flash unit
carries `Conflicts=` guards so they cannot start alongside it. Restoring any
retired profile is a deliberate manual procedure described in `runbooks.md`.

Gateway `/healthz` reports exactly `flash_iq3s_512k` and `qwen_atx_gpu2`. The
promotion completion record lives on the server at
`/var/lib/ai-server/flash-deployment.json`. Allocation, promotion, and
thinking-mode details are in `flash-512k-promotion.md`.

Profiles are mutually exclusive; there is no automatic fallback.

## Hardware Profiles

The profile above is the canonical three-GPU RTX 3090 host layout. The
standalone RTX 5090 profile is separate:

- `hardware/rtx5090/`: opt-in single-card llama.cpp reference profile.
- It targets physical GPU 0 and loopback port 8080 on the two-RTX 5090 host.
- It uses distinct unit names and does not participate in the canonical Qwen
  profile switcher or gateway installation.
- Its installer requires `AI_SERVER_RTX5090_CONFIRM=install` and refuses active
  global GPU policy units, known model conflicts, an occupied port, or an
  already-used selected GPU.
- It must not be treated as a live deployment until its strict-residency,
  thermal, quality, and concurrency gates are rerun.

Do not copy the RTX 5090 installer over `operations/install.sh`. The canonical
installer assumes the three-GPU host paths, account, and topology. Do not run
both profiles on a host until GPU, port, policy, and log ownership are
explicitly resolved.

## Gateway And Metrics

Clients use the authenticated OpenAI-compatible gateway on port 8088, never the
private model ports:

```text
OPENAI_BASE_URL=http://<AI_SERVER_HOST>:8088/v1
OPENAI_API_KEY=<gateway token>
```

`/v1/*` and `/readyz` require the bearer token. The gateway `/metrics` endpoint
is intentionally unauthenticated but exposes only bounded service and pool
counters; it must remain within the configured network boundary. The exporter
on port 9108 scrapes the Flash engine (labeled across GPUs 0 and 1) and the
GPU2 llamAmpere replica, adding `model`, `gpu`, and `gpu_uuid` labels. Strata
cached-input accounting is `observed`; llamAmpere cached-input accounting is
`unobserved`, not zero, so cache-inclusive cost estimates are not reported for
that replica.

For the pooled model, send a stable conversation identifier:

```text
X-Inference-Session: <stable-conversation-id>
```

OpenCode's built-in `X-Session-Id` is also accepted, so its sessions get
affinity without extra configuration. Requests with no session header use the
least-inflight replica with round-robin tie-breaking.

Responses include `X-Inference-Replica: gpu2`; the pool currently has one
member. The gateway does not retry a request after upstream dispatch. A failed
or saturated backend is surfaced to the operator; rollback to a retired
profile is manual and requires first stopping the Flash service, because the
profiles share GPUs. See `runbooks.md`.

## Files

- `systemd/`: canonical unit files installed under `/etc/systemd/system/`,
  including the active `strata-flash-512k.service` and the retired Muse,
  GPU1 llamAmpere, and stock tensor-split units kept for manual restore.
- `hardware/rtx5090/`: isolated RTX 5090 profile templates and installer.
- `config/`: secret-free profile, tmpfiles, logrotate, and monitoring configuration.
- `flash-512k-promotion.md`: active Flash + single-Qwen allocation and procedure.
- `promote_flash_512k.py`: privileged promotion installer with backup and
  rollback of prior state; `flash_preflight.py` pins GPU UUIDs and runtime hashes.
- `set_flash_thinking.py`, `verify_flash_thinking_wire.py`: Flash thinking-mode
  default and wire-level verification.
- `runbooks.md`: start, stop, recovery, profile switch, and rollback procedures.
- `network.md`: LAN, gateway, firewall, and Tailscale access.
- `monitoring.md`: Prometheus, exporter, and Grafana setup.
- `clients.md`: OpenCode, Oh My Pi, and generic OpenAI-compatible client setup.
- `deployment-record.local.md`: local topology, gate status, and acceptance record.

## Safety

- Ports 8080, 8081, and 8082 remain loopback-only.
- Port 8088 is the only inference endpoint exposed to LAN or Tailscale clients.
- Port 9108 is restricted to the Prometheus host by firewall rules.
- Never start the retired Muse, GPU1 llamAmpere, or stock tensor-split services
  alongside `strata-flash-512k.service`; they share GPUs 0 and/or 1. Stop
  Flash first when restoring any of them.
- If fixed fan control is unsupported by the driver or board, the driver keeps
  its automatic fan policy.
- Do not put API keys, Tailscale auth keys, or GitHub credentials in this tree.

## Installation

Install the canonical units, then apply the active profile explicitly with the
promotion installer (root):

```bash
sudo env \
  AI_SERVER_LAN_SUBNET=192.0.2.0/24 \
  AI_SERVER_PROMETHEUS_IP=192.0.2.10 \
  /path/to/localLLMinference/operations/install.sh
sudo python3 /path/to/localLLMinference/operations/promote_flash_512k.py --check
sudo python3 /path/to/localLLMinference/operations/promote_flash_512k.py
```

The promotion backs up private configuration and unit states, preserves the
gateway token, disables the retired profiles, and verifies both services
through the gateway before recording completion. The legacy
`ai-qwen-profile-switch` remains only for restoring `atx-dual` or
`stock-q4-tensor` after Flash has been stopped.
