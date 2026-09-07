# Client Integrations

The inference endpoint is OpenAI-compatible and is exposed through the
authenticated gateway. Use the gateway on port 8088; do not configure clients
to call llama.cpp directly on port 8080.

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

The available model identifiers are:

```text
qwen3.8-27b-q4-gpukv192
qwen3.8-flash-next-nvfp4-262k
```

Q4 runs on GPU 1 with a 196608-token limit and defaults to medium reasoning.
Flash Next runs on GPU 0 with a 262144-token limit and defaults to xhigh
reasoning; clients can set `reasoning_effort` per request.

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
        "qwen3.8-27b-q4-gpukv192": {
          "name": "Qwen3.8-27B Q4, 192k",
          "limit": {
            "context": 196608,
            "output": 8192
          },
          "options": {
            "reasoningEffort": "medium"
          }
        },
        "qwen3.8-flash-next-nvfp4-262k": {
          "name": "Qwen3.8 Flash Next NVFP4, 262k",
          "limit": {
            "context": 262144,
            "output": 128
          },
          "options": {
            "reasoningEffort": "xhigh"
          }
        }
      }
    }
  }
}
```

Use `http://<AI_SERVER_TAILSCALE_IP>:8088/v1` instead of the LAN URL for a client that
reaches the server through Tailscale. Start OpenCode from a shell where
`AI_SERVER_API_KEY` is set. Use `/models` inside OpenCode to select
either model.

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
      - id: qwen3.8-27b-q4-gpukv192
        name: Qwen3.8-27B Q4, 192k
        reasoning: true
        input: [text]
        contextWindow: 196608
        maxTokens: 8192
        compat:
          supportsReasoningEffort: true
      - id: qwen3.8-flash-next-nvfp4-262k
        name: Qwen3.8 Flash Next NVFP4, 262k
        reasoning: true
        input: [text]
        contextWindow: 262144
        maxTokens: 128
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
omp models find qwen3.8-27b-q4-gpukv192
```

Use `/model` to select the provider-prefixed model interactively. To make it
the default model, add this to `~/.omp/agent/config.yml`:

```yaml
modelRoles:
  default: ai-server/qwen3.8-27b-q4-gpukv192
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

The response should list both model identifiers. A missing or incorrect token
should return HTTP 401.
