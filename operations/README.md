# AI Server Operations

Operational reference for the self-hosted AI server. Research results stay under
`../research/`; this directory describes profile boundaries and installation
contracts.

## Staged Profile

The staged Qwen profile is `atx-dual`. It is not a live-service claim until the
privileged installer and profile switch complete:

- llamAmpere replica `qwen3.8-27b-atx-iq4xs-m-262144-gpu1` uses physical GPU 1
  on loopback port 8080.
- llamAmpere replica `qwen3.8-27b-atx-iq4xs-m-262144-gpu2` uses physical GPU 2
  on loopback port 8081.
- Pooled model ID `qwen3.8-27b-atx-iq4xs-m-262144` selects between those two
  replicas using session affinity (`X-Inference-Session`, or the `X-Session-Id`
  header OpenCode already sends).
- Muse Glimmer remains unchanged on physical GPU 0 and loopback port 8082 as
  `muse-glimmer-30b-kquant17`.

The native-context gate passed in six isolated runs on GPUs 1 and 2, including
concurrent execution. The detailed local results stay in the ignored deployment
record; the live-service acceptance checks remain outstanding.

The stock two-GPU `stock-q4-tensor` profile remains installed for explicit
manual rollback. Profiles are mutually exclusive; there is no automatic
fallback.

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
on port 9108 scrapes both llamAmpere replicas and Muse, adding `model`, `gpu`,
and `gpu_uuid` labels. llamAmpere cached-input accounting is `unobserved`, not
zero, so cache-inclusive cost estimates are not reported for those replicas.

For the pooled model, send a stable conversation identifier:

```text
X-Inference-Session: <stable-conversation-id>
```

OpenCode's built-in `X-Session-Id` is also accepted, so its sessions get
affinity without extra configuration. Requests with no session header use the
least-inflight replica with round-robin tie-breaking.

Responses include `X-Inference-Replica: gpu1|gpu2`. The gateway does not retry a
request after upstream dispatch. A failed or saturated replica is surfaced to
the operator; select `stock-q4-tensor` manually when rollback is required.

## Files

- `systemd/`: canonical unit files installed under `/etc/systemd/system/`.
- `hardware/rtx5090/`: isolated RTX 5090 profile templates and installer.
- `config/`: secret-free profile, tmpfiles, logrotate, and monitoring configuration.
- `runbooks.md`: start, stop, recovery, profile switch, and rollback procedures.
- `network.md`: LAN, gateway, firewall, and Tailscale access.
- `monitoring.md`: Prometheus, exporter, and Grafana setup.
- `clients.md`: OpenCode, Oh My Pi, and generic OpenAI-compatible client setup.
- `deployment-record.local.md`: local topology, gate status, and acceptance record.

## Safety

- Ports 8080, 8081, and 8082 remain loopback-only.
- Port 8088 is the only inference endpoint exposed to LAN or Tailscale clients.
- Port 9108 is restricted to the Prometheus host by firewall rules.
- Stop the active Qwen profile before selecting the other profile; never run
  stock Qwen and llamAmpere together on GPUs 1 and 2.
- If fixed fan control is unsupported by the driver or board, the driver keeps
  its automatic fan policy.
- Do not put API keys, Tailscale auth keys, or GitHub credentials in this tree.

## Installation

Install the canonical units and then select the desired profile explicitly:

```bash
sudo env \
  AI_SERVER_LAN_SUBNET=192.0.2.0/24 \
  AI_SERVER_PROMETHEUS_IP=192.0.2.10 \
  /path/to/localLLMinference/operations/install.sh
sudo /usr/local/sbin/ai-qwen-profile-switch atx-dual
```

The profile switch preserves the gateway token and backs up the root-owned
environment files. It is the only supported way to switch between Qwen
topologies.
