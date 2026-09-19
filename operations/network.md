# Network Access

## Endpoints

- Gateway LAN endpoint: `http://<AI_SERVER_LAN_IP>:8088/v1`.
- Gateway remote endpoint: `http://<AI_SERVER_TAILSCALE_IP>:8088/v1`.
- llamAmpere replica GPU1: `http://127.0.0.1:8080`.
- llamAmpere replica GPU2: `http://127.0.0.1:8081`.
- Muse Glimmer: `http://127.0.0.1:8082`.
- Metrics exporter: `http://<AI_SERVER_LAN_IP>:9108/metrics`; permit only the
  monitoring server.

Ports 8080, 8081, and 8082 are loopback-only and must never be used as client
endpoints. Port 8088 is the only LAN or Tailscale inference endpoint.

## Authentication

The gateway requires `Authorization: Bearer <token>` on every `/v1/*` route and
on `/readyz`. `/healthz` remains unauthenticated for process checks. Gateway
`/metrics` is intentionally unauthenticated and bounded to configured pool,
replica, and reason labels; it must not contain raw `X-Inference-Session` values.

The token lives in `/etc/ai-server/lan-inference-gateway.env`, owned by root and
mode 0600. The profile switch preserves it while changing backend topology.

For the pooled Qwen model, clients should send a stable
`X-Inference-Session: <conversation-id>` header. The gateway returns
`X-Inference-Replica: gpu1|gpu2`. It does not retry after upstream dispatch and
does not automatically switch to the stock profile.

## Profiles And Context

The staged profile is `atx-dual`: independent llamAmpere processes on physical
GPU1 and GPU2, plus unchanged Muse on GPU0. The stock `stock-q4-tensor` profile
is a manual rollback option and must not run concurrently with llamAmpere.

The llamAmpere native `262144` context gate passed in isolated single-GPU and
concurrent runs. The matching `...-262144` IDs are valid for the staged profile;
live endpoint acceptance remains pending.

Authenticate the Tailscale device interactively through the browser. Do not
reuse a GitHub CLI token as an application credential. Use `tailscale ip -4`
and `tailscale status` because Tailscale addresses can change.

## Firewall

Allow TCP 8088 from the configured LAN subnet and the Tailscale interface. Allow
TCP 9108 only from the configured monitoring host. Do not allow TCP 8080, 8081,
or 8082 from any external interface. Do not add router port forwarding.
