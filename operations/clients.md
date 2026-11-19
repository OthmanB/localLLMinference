# Client Integrations

The inference endpoint is OpenAI-compatible and is exposed through the
authenticated gateway. Use the gateway on port 8088; do not configure clients
to call the loopback model servers on ports 8080, 8081, or 8082.

## Common Environment

Set the gateway token in the shell or operating-system secret manager used to
launch the client. Do not put the token in a repository, `opencode.json`,
`models.yml`, shell history, or a shared screenshot.

For a generic OpenAI-compatible client on the LAN:

```bash
export AI_SERVER_BASE_URL=http://<AI_SERVER_LAN_IP>:8088/v1
```

For a generic OpenAI-compatible client over Tailscale:

```bash
export AI_SERVER_BASE_URL=http://<AI_SERVER_TAILSCALE_IP>:8088/v1
```

The OpenCode and Oh My Pi examples below put the selected URL directly in
their configuration files. Replace the LAN URL with the Tailscale URL when
using those harnesses remotely; `AI_SERVER_BASE_URL` is used by the generic
connectivity check and other clients that support it.

Set `AI_SERVER_API_KEY` from the gateway token using the client's secret
management mechanism. For an interactive shell only:

```bash
read -r -s -p 'AI server gateway token: ' AI_SERVER_API_KEY
printf '\n'
export AI_SERVER_API_KEY
```

The active `flash512-qwen-single` profile exposes these gateway identifiers:

```text
qwen3.8-flash-next-iq3_s
qwen3.8-27b-atx-iq4xs-m-262144-gpu2
qwen3.8-27b-atx-iq4xs-m-262144
```

The `...-gpu2` ID selects the deterministic llamAmpere lane. The pooled ID is
published for compatibility and currently routes to that single replica. For
pooled conversations, send a stable header:

```text
X-Inference-Session: <stable-conversation-id>
```

OpenCode already sends its session ID as `X-Session-Id`, which the pool also
accepts, so pooled OpenCode sessions get replica affinity without a custom
header. Requests without any session header use the least-inflight replica with
round-robin tie-breaking.

The gateway returns `X-Inference-Replica: gpu2`. It does not retry a
request after upstream dispatch, and it does not automatically switch profiles.

Flash serves a 524,288-token context with one active session on GPUs 0 and 1
and supports `none`, `low`, `medium`, and `xhigh` reasoning; the server default
is xhigh. Strata also accepts `high` and maps it to `xhigh`. Qwen keeps the
validated 262,144-token single slot on GPU 2. Muse Glimmer and the GPU1
llamAmpere lane are retired and no longer served.

## OpenCode

OpenCode supports project configuration in `opencode.json` or `opencode.jsonc`
and global configuration under `$XDG_CONFIG_HOME/opencode/` (normally
`~/.config/opencode/`). Project configuration takes precedence over global
configuration.

Add this provider to the chosen configuration file. Keep the API key as an
environment reference:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ai-server": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "AI Server (Qwen3.8 Models)",
      "options": {
        "baseURL": "http://<AI_SERVER_LAN_IP>:8088/v1",
        "apiKey": "{env:AI_SERVER_API_KEY}"
      },
      "models": {
        "qwen3.8-flash-next-iq3_s": {
          "name": "Qwen3.8-Flash-Next IQ3_S Strata, 512k",
          "limit": {
            "context": 524288,
            "output": 8192
          },
          "options": {
            "reasoningEffort": "xhigh"
          },
          "variants": {
            "none": {"reasoningEffort": "none"},
            "low": {"reasoningEffort": "low"},
            "medium": {"reasoningEffort": "medium"},
            "xhigh": {"reasoningEffort": "xhigh"}
          }
        },
        "qwen3.8-27b-atx-iq4xs-m-262144-gpu2": {
          "name": "Qwen3.8-27B ATX llamAmpere GPU2, validated 262k",
          "limit": {
            "context": 262144,
            "output": 8192
          },
          "options": {
            "reasoningEffort": "medium"
          },
          "variants": {
            "none": {"reasoningEffort": "none"},
            "low": {"reasoningEffort": "low"},
            "medium": {"reasoningEffort": "medium"},
            "high": {"reasoningEffort": "high"},
            "xhigh": {"reasoningEffort": "xhigh"}
          }
        },
        "qwen3.8-27b-atx-iq4xs-m-262144": {
          "name": "Qwen3.8-27B ATX llamAmpere pool, validated 262k",
          "limit": {
            "context": 262144,
            "output": 8192
          },
          "options": {
            "reasoningEffort": "medium"
          },
          "variants": {
            "none": {"reasoningEffort": "none"},
            "low": {"reasoningEffort": "low"},
            "medium": {"reasoningEffort": "medium"},
            "high": {"reasoningEffort": "high"},
            "xhigh": {"reasoningEffort": "xhigh"}
          }
        }
      }
    }
  }
}
```

Use `http://<AI_SERVER_TAILSCALE_IP>:8088/v1` instead of the LAN URL for a client that
reaches the server through Tailscale. Start OpenCode from a shell where
`AI_SERVER_API_KEY` is set. Use `/models` inside OpenCode to select Flash, the
Qwen lane, or the pool. If a service context changes, update the model IDs and
context limits together; do not leave a 262144 alias pointing at a smaller
service.

OpenCode sends its session ID to non-OpenCode providers as `X-Session-Id`
(and `x-session-affinity`), and subagents send `x-parent-session-id`. The
Qwen pool accepts `X-Session-Id`, so each OpenCode session and each
subagent session is pinned to a replica for prefix-cache locality. With the
pool's single current member all pooled requests share that replica. No custom
header or plugin is required.

The model-level `reasoningEffort` settings provide the model-specific defaults.
When a task needs a different level, use the OpenCode model/request controls;
the gateway accepts the selected value and forwards it to llama.cpp.

## Oh My Pi

Oh My Pi reads custom providers from `~/.omp/agent/models.yml`. Add this
provider without putting the token in the file. `apiKey` is resolved as an
environment variable name first, so the token stays out of the file:

```yaml
providers:
  ai-server:
    baseUrl: http://<AI_SERVER_LAN_IP>:8088/v1
    api: openai-completions
    apiKey: AI_SERVER_API_KEY
    models:
      - id: qwen3.8-flash-next-iq3_s
        name: Qwen3.8-Flash-Next IQ3_S Strata, 512k
        reasoning: true
        input: [text]
        contextWindow: 524288
        maxTokens: 8192
        compat:
          supportsReasoningEffort: true
      - id: qwen3.8-27b-atx-iq4xs-m-262144-gpu2
        name: Qwen3.8-27B ATX llamAmpere GPU2, validated 262k
        reasoning: true
        input: [text]
        contextWindow: 262144
        maxTokens: 8192
        compat:
          supportsReasoningEffort: true
      - id: qwen3.8-27b-atx-iq4xs-m-262144
        name: Qwen3.8-27B ATX llamAmpere pool, validated 262k
        reasoning: true
        input: [text]
        contextWindow: 262144
        maxTokens: 8192
        compat:
          supportsReasoningEffort: true
```

Use the Tailscale URL in `baseUrl` when appropriate:

```yaml
baseUrl: http://<AI_SERVER_TAILSCALE_IP>:8088/v1
```

Export `AI_SERVER_API_KEY` before launching `omp`. In Oh My Pi, inspect the
provider and model with:

```bash
omp models find qwen3.8-flash-next-iq3_s
```

Oh My Pi does not send a session header for a custom provider, so its pooled
requests would bypass cache affinity. Pin its roles to explicit models instead.
Put this in `~/.omp/agent/config.yml`. Flash serves one active session, so
keep a single heavy role on it and put the lighter roles on the Qwen lane:

```yaml
modelRoles:
  default: ai-server/qwen3.8-flash-next-iq3_s
  plan: ai-server/qwen3.8-flash-next-iq3_s
  slow: ai-server/qwen3.8-flash-next-iq3_s
  commit: ai-server/qwen3.8-27b-atx-iq4xs-m-262144-gpu2
  task: ai-server/qwen3.8-27b-atx-iq4xs-m-262144-gpu2
  smol: ai-server/qwen3.8-27b-atx-iq4xs-m-262144-gpu2
  tiny: ai-server/qwen3.8-27b-atx-iq4xs-m-262144-gpu2
  advisor: ai-server/qwen3.8-27b-atx-iq4xs-m-262144-gpu2
```

The explicit `...-gpu2` ID is a singleton lane that bypasses pool routing, so
each role keeps its own warm prompt cache. The pooled ID stays available for
interactive `/model` selection; it currently routes to the same GPU2 replica.
No model currently serves inline-image input; the retired Muse profile was the
vision lane.

## Reasoning And Sampling

The server does not force reasoning off and does not bake in a single sampling
policy. Clients should send request-level settings when mode-specific behavior
matters.

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

For an exact SWE-bench-style non-thinking run, use a separate llama.cpp
invocation with `--reasoning off`; do not change the production service.

## Connectivity Check

After setting the endpoint and token, verify the client host can reach the
gateway:

```bash
curl -sS -H "Authorization: Bearer $AI_SERVER_API_KEY" \
  "$AI_SERVER_BASE_URL/models"
```

The response should list the Flash ID, the explicit `...-gpu2` llamAmpere ID,
and the pooled llamAmpere ID. A missing or incorrect token should return HTTP
401. The `qwen3.8-27b-atx-iq4xs-m-262144-gpu1`, `muse-glimmer-30b-kquant17`,
and `qwen3.8-27b-q4-tensor262k` IDs appear only after an explicit manual
restore of a retired profile.
