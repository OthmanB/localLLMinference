# Network Access

## Endpoints

- Gateway LAN endpoint: `http://<AI_SERVER_LAN_IP>:8088/v1`.
- Gateway remote endpoint: `http://<AI_SERVER_TAILSCALE_IP>:8088/v1`.
- Private llama.cpp endpoint: `http://127.0.0.1:8080/v1`; never expose it directly.
- Metrics exporter: `http://<AI_SERVER_LAN_IP>:9108/metrics`; permit only the
  monitoring server.

## Authentication

The gateway requires `Authorization: Bearer <token>` on every `/v1/*` route.
Health checks remain unauthenticated. The token lives in
`/etc/ai-server/lan-inference-gateway.env`, owned by root and mode 0600.

Qwen tensor uses `--reasoning auto --reasoning-effort medium` as its
general-purpose default. Muse Glimmer is configured with `high` as the OpenCode
default and supports `low`, `medium`, `high`, and `xhigh`. The gateway forwards
`reasoning_effort` to the appropriate backend. Sampling parameters remain
request-specific and are not fixed by the service.

Authenticate the Tailscale device interactively through the browser. Do not
reuse a GitHub CLI token as an application credential. Use `tailscale ip -4` and
`tailscale status` because Tailscale addresses can change.

## Firewall

Allow TCP 8088 from the configured LAN subnet and the Tailscale interface. Allow
TCP 9108 and 9109 only from the configured monitoring host. Do not allow TCP 8080
from any external interface.
Do not add router port forwarding.
