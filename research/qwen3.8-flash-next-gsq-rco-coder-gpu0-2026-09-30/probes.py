#!/usr/bin/env python3
"""OpenAI-compatible probes for the GPU0 Flash-Next evaluation (stdlib only)."""
import json
import sys
import time
import urllib.error
import urllib.request
import uuid as uuidlib


def _post(url, payload, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from {url}: {e.read().decode()[:600]}") from None


def http_json(url, payload, timeout=3600):
    with _post(url, payload, timeout) as r:
        return json.loads(r.read().decode())


def tokenize(base, model, text, timeout=300):
    out = http_json(f"{base}/tokenize", {"content": text}, timeout)
    toks = out.get("tokens") or out.get("ids") or []
    return len(toks), toks


def chat(base, model, messages, timeout=3600, **kw):
    """Non-streaming chat. Returns dict(content, reasoning, finish_reason, usage)."""
    payload = {"model": model, "messages": messages, "stream": False}
    payload.update(kw)
    with _post(f"{base}/v1/chat/completions", payload, timeout) as r:
        d = json.loads(r.read().decode())
    msg = d["choices"][0]["message"]
    return {"content": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or "",
            "finish_reason": d["choices"][0].get("finish_reason"),
            "usage": d.get("usage")}


def chat_stream(base, model, messages, timeout=3600, **kw):
    """Streaming chat. Returns dict(text, ttft, n_chunks, t_end, tool_calls)."""
    payload = {"model": model, "messages": messages, "stream": True}
    payload.update(kw)
    r = _post(f"{base}/v1/chat/completions", payload, timeout)
    t0 = time.time()
    first = None
    n = 0
    parts = []
    tool_calls = []
    with r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            ch = json.loads(body)["choices"][0]
            delta = ch.get("delta", {})
            if delta.get("content"):
                if first is None:
                    first = time.time() - t0
                parts.append(delta["content"])
            if delta.get("tool_calls"):
                tool_calls.extend(delta["tool_calls"])
            if ch.get("finish_reason") in ("stop", "length", "tool_calls", "content_filter"):
                pass
            n += 1
    t_end = time.time() - t0
    return {"text": "".join(parts), "ttft": first, "n_chunks": n, "t_end": t_end,
            "tool_calls": tool_calls}


def build_marker_prompt(base, model, target_tokens, marker, timeout=600):
    filler = ("The quick brown fox jumps over the lazy dog near the old stone bridge. "
              "A quiet river winds past the meadow where the lighthouse keeper walks at dawn. ")
    n, _ = tokenize(base, model, filler)
    reps = max(1, (target_tokens - 64) // max(1, n))
    body = (f"Hidden in the middle of a long stream of sentences is a unique code. "
            f"Your first output must be exactly that code, then a newline, then one short sentence. "
            f"The code is {marker}. Read the whole stream carefully before answering. "
            f"Here is the stream:\n\n")
    prompt = body + filler * reps + f"\n\nAt the end of the stream, the code repeats: {marker}.\n"
    got, _ = tokenize(base, model, prompt)
    while got > target_tokens + 4:
        reps -= 2
        prompt = body + filler * reps + f"\n\nAt the end of the stream, the code repeats: {marker}.\n"
        got, _ = tokenize(base, model, prompt)
    return prompt, got


def main():
    args = sys.argv[1:]
    kind = args[0]
    base = args[1]
    model = args[2]
    if kind == "determinism":
        m = [{"role": "user", "content": "Say exactly: The answer is 42. Then stop."}]
        out1 = chat(base, model, m, temperature=0, seed=1234, max_tokens=512)
        out2 = chat(base, model, m, temperature=0, seed=1234, max_tokens=512)
        same = (out1["content"], out1["reasoning"]) == (out2["content"], out2["reasoning"])
        print(json.dumps({"identical": same,
                          "a": {"content": out1["content"], "reasoning_len": len(out1["reasoning"])},
                          "b": {"content": out2["content"], "reasoning_len": len(out2["reasoning"])}}))
    elif kind == "eos":
        m = [{"role": "user", "content": "What is 17*19? Reply with just the number."}]
        d = chat(base, model, m, temperature=0, max_tokens=1024)
        print(json.dumps({"reply": d["content"], "reasoning_len": len(d["reasoning"]),
                          "finish": d["finish_reason"], "usage": d["usage"]}))
    elif kind == "coding":
        m = [{"role": "user", "content": (
            "Write a Python function `parse_ranges(s: str) -> list[tuple[int,int]]` that parses a "
            "string like '1-4,7,9-11' into a list of inclusive (lo,hi) tuples. Raise ValueError on "
            "invalid input (empty parts, lo>hi, non-integers). Include 4 doctest examples. "
            "Output code only.")}]
        d = chat(base, model, m, temperature=0.2, max_tokens=4000)
        print(json.dumps({"content": d["content"], "reasoning_len": len(d["reasoning"]),
                          "chars": len(d["content"])}))
    elif kind == "json":
        m = [{"role": "user", "content": (
            "Return JSON only (no prose) describing the planet Mars with exactly these keys: "
            '{"name": str, "moons": int, "gravity_m_s2": number, "day_length_hours": number, '
            '"facts": array of 3 strings}.')}]
        d = chat(base, model, m, temperature=0, max_tokens=2000,
                 response_format={"type": "json_object"})
        txt = d["content"]
        try:
            parsed = json.loads(txt)
            ok = set(parsed) == {"name", "moons", "gravity_m_s2", "day_length_hours", "facts"}
        except Exception as e:
            parsed, ok = str(e), False
        print(json.dumps({"valid_json": ok, "parsed": parsed,
                          "reasoning_len": len(d["reasoning"]), "raw_head": txt[:200]}))
    elif kind == "tool":
        m = [{"role": "user", "content": "What is the weather like in Zurich right now?"}]
        tools = [{"type": "function", "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]}}}]
        r = chat_stream(base, model, m, temperature=0, max_tokens=256, tools=tools)
        print(json.dumps({"tool_calls": r["tool_calls"], "text": r["text"][:200]}))
    elif kind == "marker":
        target = int(args[3])
        marker = "FT-" + uuidlib.uuid4().hex.upper()
        prompt, got = build_marker_prompt(base, model, target, marker)
        m = [{"role": "user", "content": (
            f"{prompt}\n\nNow: output the code first, exactly as it appears, then a newline, "
            f"then one sentence confirming you read the end of the stream.")}]
        r = chat_stream(base, model, m, temperature=0, max_tokens=192)
        first_line = r["text"].strip().splitlines()[0].strip() if r["text"].strip() else ""
        print(json.dumps({
            "marker": marker,
            "prompt_tokens": got,
            "first_line": first_line,
            "marker_exact_first": first_line == marker,
            "marker_anywhere_first10": marker in r["text"][:10 + len(marker) * 2],
            "ttft_s": round(r["ttft"], 2) if r["ttft"] else None,
            "n_chunks": r["n_chunks"],
            "t_end_s": round(r["t_end"], 2),
            "decode_tps": round((r["n_chunks"] - 1) / max(0.001, r["t_end"] - (r["ttft"] or 0)), 2)
            if r["n_chunks"] > 1 and r["ttft"] else None,
            "prefill_tps": round(got / r["ttft"], 1) if r["ttft"] else None,
            "text_head": r["text"][:300],
        }))
    elif kind == "prompt":
        text = open(args[3]).read()
        maxtok = int(args[4]) if len(args) > 4 else 8000
        t0 = time.time()
        d = chat(base, model, [{"role": "user", "content": text}],
                 temperature=0.2, max_tokens=maxtok)
        print(json.dumps({"wall_s": round(time.time() - t0, 1),
                          "content": d["content"], "reasoning_len": len(d["reasoning"]),
                          "finish": d["finish_reason"], "usage": d["usage"]}))
    else:
        print(f"unknown kind {kind}")
        sys.exit(2)


if __name__ == "__main__":
    main()
