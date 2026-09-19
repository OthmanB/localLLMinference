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

The active `atx-dual` profile exposes these staged llamAmpere identifiers:

```text
qwen3.8-27b-atx-iq4xs-m-262144-gpu1
qwen3.8-27b-atx-iq4xs-m-262144-gpu2
qwen3.8-27b-atx-iq4xs-m-262144
muse-glimmer-30b-kquant17
```

The `gpu1` and `gpu2` IDs select deterministic lanes. The pooled ID distributes
requests across both replicas. For pooled conversations, send a stable header:

```text
X-Inference-Session: <stable-conversation-id>
```

The gateway returns `X-Inference-Replica: gpu1|gpu2`. It does not retry a
request after upstream dispatch, and it does not automatically fall back to the
stock profile.

The native llamAmpere 262144-token gate passed on both GPUs and during
concurrent isolated execution. The three Qwen IDs above are therefore the
validated staged contract; live service acceptance remains pending. Muse Glimmer runs
on GPU 0 with a 131072-token limit, accepts text and
inline-image input, and supports `low`, `medium`, `high`, or `xhigh` reasoning;
`high` is the recommended local agent default.

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
        "qwen3.8-27b-atx-iq4xs-m-262144-gpu1": {
          "name": "Qwen3.8-27B ATX llamAmpere GPU1, validated 262k",
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
        },
        "muse-glimmer-30b-kquant17": {
          "name": "Muse Glimmer 30B K-Quant, 131k",
          "limit": {
            "context": 131072,
            "output": 8192
          },
          "options": {
            "reasoningEffort": "high"
          },
          "variants": {
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
`AI_SERVER_API_KEY` is set. Use `/models` inside OpenCode to select a lane, the
pool, or Muse. If the context gate fails, update the three Qwen IDs and context
limits together; do not leave a 262144 alias pointing at a 245760 service.

The model-level `reasoningEffort` settings provide the model-specific defaults.
When a task needs a different level, use the OpenCode model/request controls;
the gateway accepts the selected value and forwards it to llama.cpp.

## Oh My Pi

Oh My Pi reads custom providers from `~/.omp/agent/models.yml`. Add this
provider without putting the token in the file:

```yaml
providers:
  ai-server:
    baseUrl: http://<AI_SERVER_LAN_IP>:8088/v1
    api: openai-completions
    apiKey: AI_SERVER_API_KEY
    models:
      - id: qwen3.8-27b-atx-iq4xs-m-262144-gpu1
         name: Qwen3.8-27B ATX llamAmpere GPU1, validated 262k
        reasoning: true
        input: [text]
        contextWindow: 262144
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
      - id: muse-glimmer-30b-kquant17
        name: Muse Glimmer 30B K-Quant, 131k
        reasoning: true
        input: [text, image]
        contextWindow: 131072
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
omp models find qwen3.8-27b-atx-iq4xs-m-262144
```

Use `/model` to select the provider-prefixed model interactively. To make it
the default model, add this to `~/.omp/agent/config.yml`:

```yaml
modelRoles:
  default: ai-server/qwen3.8-27b-atx-iq4xs-m-262144
```

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

The response should list the two explicit llamAmpere IDs, the pooled llamAmpere
ID, and Muse. A missing or incorrect token should return HTTP 401. The stock
`qwen3.8-27b-q4-tensor262k` ID is available only after an explicit manual
rollback to `stock-q4-tensor`.
