# Qwen 3.8 27B — local vLLM endpoint for the team

Qwen 3.8 27B (Q4 weights, FP8 KV cache) served with vLLM, tensor-parallel 2 across
two RTX 5090 GPUs. OpenAI-compatible API, 262,144-token context, reasoning and
tool calling enabled.

| | |
|---|---|
| Tailscale URL | `http://100.114.27.79:8080/v1` |
| LAN URL | `http://192.168.68.72:8080/v1` |
| Model ID | `qwen3.8-27b-q4-gpukv-native` |
| Auth | `Authorization: Bearer <token>` — token is shared on Slack |
| Max concurrency | 3 simultaneous requests (4th+ is queued, see table below) |

## Quick test

```bash
export AI_SERVER_API_KEY='...paste the token from Slack...'
curl -sS -H "Authorization: Bearer $AI_SERVER_API_KEY" \
  http://100.114.27.79:8080/v1/models

curl -sS -H "Authorization: Bearer $AI_SERVER_API_KEY" \
  -H 'Content-Type: application/json' \
  --data '{
    "model": "qwen3.8-27b-q4-gpukv-native",
    "messages": [{"role": "user", "content": "Give a short greeting."}],
    "max_tokens": 512
  }' http://100.114.27.79:8080/v1/chat/completions
```

Keep the token out of repos, config files committed to git, and shell history.

## opencode

Add to `opencode.json` (project) or `~/.config/opencode/opencode.json` (global).
The key stays out of the file via an environment reference:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ai-server": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Qwen 3.8 27B (local vLLM)",
      "options": {
        "baseURL": "http://100.114.27.79:8080/v1",
        "apiKey": "{env:AI_SERVER_API_KEY}"
      },
      "models": {
        "qwen3.8-27b-q4-gpukv-native": {
          "name": "Qwen 3.8 27B Q4, 262k context",
          "limit": { "context": 262144, "output": 32768 }
        }
      }
    }
  }
}
```

Start opencode from a shell with `AI_SERVER_API_KEY` exported, then pick the model
with `/models`.

## oh-my-pi (omp)

Add to `~/.omp/agent/models.yml` (the `apiKey` value is an environment variable
name, so the token itself never lands in the file):

```yaml
providers:
  ai-server:
    baseUrl: http://100.114.27.79:8080/v1
    api: openai-completions
    apiKey: AI_SERVER_API_KEY
    models:
      - id: qwen3.8-27b-q4-gpukv-native
        name: Qwen 3.8 27B Q4, 262k context
        reasoning: true
        input: [text]
        contextWindow: 262144
        maxTokens: 32768
```

Then `omp models find qwen3.8-27b` to verify, and select it with `/model`.

## What to expect (measured on this box)

| Concurrent users | Prefill / first token | Decode | What happens |
|---|---|---|---|
| 1, continuing a cached ~196k conversation | ~4–7 s | ~50 tok/s per stream | Runs immediately |
| 3, cached conversations | ~4–7 s | ~47–51 tok/s each, ~150 tok/s aggregate, p95 inter-token ~40 ms | All run immediately |
| 3 short requests | short prefill | ~240–260 tok/s aggregate (~80/user) | Runs immediately |
| 1 brand-new 196k context (cold prefill) | ~116 s one-off | normal afterwards | One user pays the cold prefill; others keep ~36 ms p95 inter-token during it |
| 4th concurrent user | waits in gateway queue | — | FIFO queue of up to 3; starts as soon as a slot frees. A 4th queued request gets `503 Retry-After: 1` — just retry. Queued requests expire after 900 s. |

Notes:

- Prefill bursts at ~20–25k tok/s internally; the long first-token number above
  is what you actually observe.
- Thinking is on by default and consumes output budget — give `max_tokens` some
  headroom. To disable thinking, send
  `"chat_template_kwargs": {"enable_thinking": false}` in the request body.
- Tool calling is enabled (qwen3_coder parser), so agent loops work.

## Where the model stands

Public-benchmark positioning, September 2026. The vanilla Qwen 3.8 27B sits
just above GPT-5.3-Codex and just below GPT-5.6 Luna on the independent
Artificial Analysis composite, leads this comparison set on SWE-bench Pro and
computer use, and trails the GPT-5.6 tiers and Sonnet 5 on terminal and
long-horizon agentic work.

```text
Artificial Analysis Intelligence Index v4.3.2 (independent, max reasoning)
GPT-5.6 Terra        █████████████████████████   42
Claude Sonnet 5      ████████████████████        38
GPT-5.6 Luna         ███████████████████▌        37
Qwen 3.8 27B (BF16)  ██████████████████          34
GPT-5.3-Codex        ██████████████████          33 (AA estimate)
Qwen 3.8 27B Q4      █████████████████           ~31 (estimate)
```

Per-benchmark scores (public, vendor or independent):

| Benchmark | Qwen3.8-27B Q4 (this endpoint) ¹ | Qwen3.8-27B (BF16) | GPT-5.6 Luna | GPT-5.6 Terra | GPT-5.3-Codex | Claude Sonnet 5 |
|---|---|---|---|---|---|---|
| AA Intelligence Index v4.3.2 (max) | ~31–32 | 34 | 37 | 42 | 33 (est.) | 38 |
| SWE-bench Pro | ~59 | 61.7 | not reported | not reported | 56.8 | 63.2 |
| Terminal-Bench 2.1 | ~70 | 73.0 | ~80–83 | ~84–85 | 77.3 ² | 80.4 |
| DeepSWE 1.1 | ~40 | 42.2 | 67.2 | 69.6 | not reported | not reported |
| OSWorld-Verified | ~80 | 84.3 | not reported ³ | not reported ³ | 64.7 | 81.2 |
| GPQA Diamond | ~85–87 | 89.2 | not reported | not reported | not reported | not reported |
| LiveCodeBench v6 | ~88 | 90.3 | not reported | not reported | not reported | not reported |
| HLE (no tools) | ~28 | 30.8 | not reported | not reported | not reported | 57.4 ⁴ |

¹ **Estimated, not measured.** This deployment runs Q4_K_M weights + FP8 KV
cache (calibrated, needed to fit four full 262k contexts in 2x 32 GiB) and
sits a few percent below the BF16 vanilla on internal quality gates; the
column applies that haircut to each vanilla score. Run your own workload to
confirm.
² Reported on Terminal-Bench 2.0, not 2.1.
³ OpenAI reported GPT-5.6 on OSWorld 2.0 instead (Sol 62.6), a different
benchmark revision; Terra/Luna figures were not published.
⁴ Different evaluation setup (tools enabled, Anthropic harness), not
directly comparable to the no-tools column.

Reading the table:

- **Agentic SWE (SWE-bench Pro):** the vanilla model (61.7) beats
  GPT-5.3-Codex (56.8) by ~5 points and lands ~1.5 below Claude Sonnet 5
  (63.2). OpenAI did not publish SWE-bench Pro for the Luna/Terra tiers
  (their Sol tier scored 64.6).
- **Computer use (OSWorld-Verified):** leads this set at 84.3 vs Sonnet 5's
  81.2 and GPT-5.3-Codex's 64.7.
- **Terminal work (Terminal-Bench 2.1)** and **long-horizon agentic
  engineering (DeepSWE):** the GPT-5.6 tiers and Sonnet 5 are clearly ahead.
- **Knowledge/reasoning (GPQA Diamond, HLE):** strong for a 27B, but the
  bigger frontier models keep their usual edge on HLE.
- The general **GPT-5.3** model (non-Codex) has no separately published
  benchmark table; the documented GPT-5.3 variant is GPT-5.3-Codex, which is
  why it anchors the "GPT 5.3" column.

Caveats: SWE-bench Pro numbers come from different harnesses per vendor
(Qwen used the Claude Code harness; OpenAI/Anthropic their own), so treat
cross-vendor deltas smaller than ~3 points as noise. The GPT-5.3-Codex AA
index value is Artificial Analysis's estimate, and the whole Q4 column is an
estimate.

Sources: Qwen3.8-27B model card (huggingface.co/Qwen/Qwen3.8-27B),
Artificial Analysis model pages (qwen3-8-27b, gpt-5-6-luna, gpt-5-6-terra,
gpt-5-3-codex, claude-sonnet-5), OpenAI "Introducing GPT-5.3-Codex"
(2026-02-05), OpenAI GPT-5.6 launch material (2026-07-09), Anthropic Claude
Sonnet 5 system card (2026-06-30).

All of that for electricity: the two GPUs run under 500/575 W caps, so the
whole inference box draws around a kilowatt under load.

## House rules

- Be a good neighbor: three concurrent streams is the ceiling. Long-running
  agent work is fine; parallel fan-out of dozens of short requests is not.
- If you get a `503`, retry after the `Retry-After` seconds — the request
  simply waits for a free slot.
- The service restarts occasionally (upgrades, reboots). Startup takes about a
  minute; a hung client request will fail or take long on restart, just retry.
- Questions / incidents: ping Michel.
