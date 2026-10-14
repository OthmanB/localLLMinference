import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE = sys.argv[1].rstrip("/")
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)


def get(path):
    started = time.perf_counter()
    try:
        with urlopen(BASE + path, timeout=30) as response:
            body = response.read().decode()
            return {"status": response.status, "body": body, "wall_seconds": time.perf_counter() - started}
    except (HTTPError, URLError) as exc:
        return {"error": repr(exc), "wall_seconds": time.perf_counter() - started}


def post(name, payload, stream=False):
    body = json.dumps(payload).encode()
    request = Request(
        BASE + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token = None
    chunks = []
    raw_lines = []
    final = None
    try:
        with urlopen(request, timeout=300) as response:
            if not stream:
                raw = response.read().decode()
                record = {"status": response.status, "payload": payload, "response": json.loads(raw), "wall_seconds": time.perf_counter() - started}
                (OUT / f"{name}-response.json").write_text(json.dumps(record, indent=2) + "\n")
                return record
            for line in response:
                text = line.decode().rstrip("\n")
                raw_lines.append(text)
                if not text.startswith("data: "):
                    continue
                data = text[6:]
                if data == "[DONE]":
                    continue
                item = json.loads(data)
                chunks.append(item)
                choice = (item.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                if first_token is None and (delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls")):
                    first_token = time.perf_counter()
                if item.get("usage"):
                    final = item
        finished = time.perf_counter()
        usage = (final or {}).get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        total_seconds = finished - started
        decode_seconds = finished - first_token if first_token else None
        record = {
            "status": 200,
            "payload": payload,
            "chunks": chunks,
            "raw_sse_lines": raw_lines,
            "usage": usage,
            "wall_seconds": total_seconds,
            "time_to_first_token_seconds": first_token - started if first_token else None,
            "decode_seconds_after_first_token": decode_seconds,
            "end_to_end_completion_tok_s": completion_tokens / total_seconds if completion_tokens and total_seconds else None,
            "decode_tok_s_after_first_token": completion_tokens / decode_seconds if completion_tokens and decode_seconds else None,
        }
        (OUT / f"{name}-response.json").write_text(json.dumps(record, indent=2) + "\n")
        return record
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        record = {"error": repr(exc), "payload": payload, "wall_seconds": time.perf_counter() - started, "raw_sse_lines": raw_lines}
        (OUT / f"{name}-response.json").write_text(json.dumps(record, indent=2) + "\n")
        return record


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


common = {
    "model": "qwen3.8-27b-awq-int4",
    "temperature": 0,
    "top_p": 1,
    "seed": 7,
    "stream": True,
    "stream_options": {"include_usage": True},
    "max_tokens": 128,
    "chat_template_kwargs": {"enable_thinking": False},
}

health = get("/health")
models = get("/v1/models")
save("health.json", health)
save("models.json", models)
metrics_before = get("/metrics")
(OUT / "metrics-before.txt").write_text(metrics_before.get("body", json.dumps(metrics_before)))

code_payload = dict(common)
code_payload["messages"] = [{"role": "user", "content": "Write a Python function named clamp_int(value, low, high) that returns value bounded inclusively by low and high. Return only a fenced Python code block."}]
code = post("code", code_payload, stream=True)
code_repeat = post("code-repeat", code_payload, stream=True)

narrative_payload = dict(common)
narrative_payload["messages"] = [{"role": "user", "content": "In exactly three short sentences, explain why a binary search runs in logarithmic time."}]
narrative = post("narrative", narrative_payload, stream=True)

thinking_payload = {
    "model": common["model"],
    "messages": [{"role": "user", "content": "Solve 17 * 19. Show your reasoning, then give the final answer."}],
    "temperature": 0,
    "top_p": 1,
    "seed": 7,
    "max_tokens": 128,
    "chat_template_kwargs": {"enable_thinking": True},
}
thinking = post("thinking", thinking_payload, stream=False)

tool_payload = {
    "model": common["model"],
    "messages": [{"role": "user", "content": "What is the weather in Paris? You must call the get_weather tool."}],
    "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"], "additionalProperties": False}}}],
    "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    "temperature": 0,
    "top_p": 1,
    "seed": 7,
    "max_tokens": 64,
    "chat_template_kwargs": {"enable_thinking": False},
}
tool = post("tool", tool_payload, stream=False)

json_payload = {
    "model": common["model"],
    "messages": [{"role": "user", "content": "Return a JSON object with keys answer and language. Set answer to 4 and language to English."}],
    "response_format": {"type": "json_object"},
    "temperature": 0,
    "top_p": 1,
    "seed": 7,
    "max_tokens": 64,
    "chat_template_kwargs": {"enable_thinking": False},
}
json_result = post("json", json_payload, stream=False)

metrics_after = get("/metrics")
(OUT / "metrics-after.txt").write_text(metrics_after.get("body", json.dumps(metrics_after)))
summary = {
    "environment": {"CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")},
    "health": health,
    "models": models,
    "code": code,
    "code_repeat": code_repeat,
    "narrative": narrative,
    "thinking": thinking,
    "tool": tool,
    "json": json_result,
}
save("probe-summary.json", summary)
