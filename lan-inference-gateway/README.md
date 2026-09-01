# LAN Inference Gateway

`lan-inference-gateway` gives LAN clients one OpenAI-compatible API while routing requests to independently configured `llama.cpp`, FreeToken, or vLLM backends. It does not load models, manage GPUs, or encode model-specific launch flags.

## Client contract

- `GET /healthz`: gateway process status.
- `GET /readyz`: configured backend health probes.
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

Model IDs map exactly to one configured backend. An unlisted model returns OpenAI-style `model_not_found` unless `LAN_INFERENCE_DEFAULT_BACKEND` is explicitly configured. Configure explicit IDs for predictable model selection.

Each backend's `health_path` defaults to `/health`. Set it to `null` when the upstream runtime has no suitable probe; `/readyz` will report that backend as `not_configured` and return `503` until a probe is configured.

```bash
curl http://gateway-host:8088/v1/models
curl http://gateway-host:8088/readyz
curl http://gateway-host:8088/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3.8-27b-q5-gpukv64","messages":[{"role":"user","content":"Explain this error."}]}'
```

## Verification

```bash
.venv/bin/pytest
```

The tests use an in-memory upstream transport. They validate model routing, non-streaming and streaming OpenAI response forwarding, gateway authentication, and readiness probes without requiring a local model runtime.
