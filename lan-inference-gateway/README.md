# LAN Inference Gateway

`lan-inference-gateway` gives LAN clients one OpenAI-compatible API while routing requests to independently configured `llama.cpp`, FreeToken, or vLLM backends. It does not load models, manage GPUs, or encode model-specific launch flags.

## Client contract

- `GET /healthz`: gateway process status.
- `GET /readyz`: configured backend health probes; uses the gateway bearer token when configured.
- `GET /metrics`: unauthenticated Prometheus text for bounded pool-routing counters.
- `GET /v1/models`: configured model IDs and their owning backend.
- `POST /v1/chat/completions`: OpenAI-compatible chat-completions request and response, including SSE streaming.

The first adapter is `openai`, which expects an upstream OpenAI-compatible `/v1/chat/completions` endpoint. Adding a non-compatible runtime means adding an adapter; clients and model routing remain unchanged.

## Install and run

```bash
cd /path/to/localLLMinference/lan-inference-gateway
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
export LAN_INFERENCE_BACKENDS='[
  {
    "name": "llama-cpp",
    "base_url": "http://127.0.0.1:8080",
    "models": ["qwen3.8-27b-q5-gpukv64"],
    "health_path": "/health"
  },
  {
    "name": "freetoken",
    "base_url": "http://127.0.0.1:1919",
    "models": ["qwen3.8-27b-nvfp4"],
    "health_path": "/health"
  }
]'
.venv/bin/lan-inference-gateway
```

For independent replicas of one model, keep the explicit model IDs on their
backends and configure the pooled public ID separately. Pool entries must name
existing backends; the limit applies independently to each replica. A pool
accepts one or more session headers (`session_headers`), checked in order; the
legacy single-value `session_header` is still supported. With several headers,
the first one present on the request wins, which lets clients that already send
a standard header (for example OpenCode's `X-Session-Id`) get affinity without a
custom header:

```bash
export LAN_INFERENCE_POOLS='[
  {
    "name": "qwen-atx-pool",
    "model": "qwen3.8-27b-atx-iq4xs-m-262144",
    "replicas": ["qwen_atx_gpu1", "qwen_atx_gpu2"],
    "session_headers": ["X-Inference-Session", "X-Session-Id"],
    "max_inflight": 1
  }
]'
```

The console command binds to `0.0.0.0:8088` for LAN use. Restrict it with a host firewall. To require a client bearer token, store the token in a separate environment variable and configure its name:

```bash
export LAN_GATEWAY_CLIENT_TOKEN='replace-with-a-random-secret'
export LAN_INFERENCE_CLIENT_API_KEY_ENV=LAN_GATEWAY_CLIENT_TOKEN
```

Backends can use their own bearer token without exposing it to clients:

```bash
export VLLM_UPSTREAM_TOKEN='replace-with-a-backend-secret'
```

Then add `"api_key_env": "VLLM_UPSTREAM_TOKEN"` to that backend object. Do not place secrets in `LAN_INFERENCE_BACKENDS`.

Two mutually exclusive Qwen profiles can be prepared on GPU 0: FreeToken for
short-context work, and llama.cpp for a native long-context window. See the
local runtime documentation for the selected backend. Configure only the
currently active profile as ready, and expose only this authenticated gateway
to the LAN.

## Routing and health

Singleton model IDs map exactly to one configured backend. Pool model IDs are
selected across healthy replicas: a request carrying a configured session header
is routed by deterministic rendezvous affinity, while requests without one use
the least-inflight replica with round-robin tie-breaking, so sequential traffic
spreads across the pool. A pinned saturated replica returns a retryable `503`
rather than spilling the session to another replica.

An unlisted model returns OpenAI-style `model_not_found` unless
`LAN_INFERENCE_DEFAULT_BACKEND` is explicitly configured. Configure explicit
IDs for predictable model selection. Inference responses include
`X-Inference-Replica`; cold health failover also includes
`X-Inference-Failover: true`.

Each backend's `health_path` defaults to `/health`. Set it to `null` when the upstream runtime has no suitable probe; `/readyz` will report that backend as `not_configured` and return `503` until a probe is configured.

```bash
curl http://gateway-host:8088/v1/models
curl http://gateway-host:8088/readyz -H 'authorization: Bearer replace-with-a-random-secret'
curl http://gateway-host:8088/metrics
curl http://gateway-host:8088/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3.8-27b-q5-gpukv64","messages":[{"role":"user","content":"Explain this error."}]}'
```

## Verification

```bash
.venv/bin/pytest
```

The tests use an in-memory upstream transport. They validate singleton and
replica-pool routing, session affinity, saturation, failover, non-streaming and
streaming OpenAI response forwarding, gateway authentication, and readiness
probes without requiring a local model runtime.

`/metrics` exposes pool counters with only `pool`, `replica`, and bounded
`reason` labels. It exports route decisions, affinity hits and misses,
saturation rejections, and cold failovers. Session IDs are never emitted as
metric labels or logs.
