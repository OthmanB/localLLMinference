# Network Access

## Endpoints

- Gateway LAN endpoint: `http://<AI_SERVER_LAN_IP>:8088/v1`.
- Gateway remote endpoint: `http://<AI_SERVER_TAILSCALE_IP>:8088/v1`.
- Flash 512K (Strata, GPUs 0+1): `http://127.0.0.1:8082`.
- llamAmpere replica GPU2: `http://127.0.0.1:8081`.
- Port 8080 is unused by the active profile; retired services use it or 8082.
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
`X-Inference-Session: <conversation-id>` header; the pool also accepts the
`X-Session-Id` header OpenCode sends automatically. The gateway returns
`X-Inference-Replica: gpu2` (the pool currently has one member). It does not
retry after upstream dispatch and does not automatically switch profiles.

## Profiles And Context

The active profile is `flash512-qwen-single`: the Strata Flash engine on
physical GPUs 0 and 1 with a 524,288-token context and one session, plus one
llamAmpere replica on physical GPU 2 with a 262,144-token context. Muse
Glimmer, the GPU1 llamAmpere replica, and the stock `stock-q4-tensor` profile
are retired, disabled at boot, and must not run concurrently with Flash.
See `flash-512k-promotion.md` for the promotion procedure and
`runbooks.md` for restoring a retired profile.

Both context sizes were verified live through health checks and gateway
completion checks at promotion time.

Authenticate the Tailscale device interactively through the browser. Do not
reuse a GitHub CLI token as an application credential. Use `tailscale ip -4`
and `tailscale status` because Tailscale addresses can change.

## Firewall

Allow TCP 8088 from the configured LAN subnet and the Tailscale interface. Allow
TCP 9108 only from the configured monitoring host. Do not allow TCP 8080, 8081,
or 8082 from any external interface. Do not add router port forwarding.
