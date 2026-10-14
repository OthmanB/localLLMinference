"""Standard-library OpenAI SSE client. No retries, token estimates, or server control."""
import http.client
import json
import socket
import time
from urllib.parse import urlsplit
from typing import Any


def payload(model, messages, max_tokens, cache_prompt=None) -> dict[str, Any]:
    result: dict[str, Any] = dict(model=model, messages=messages, max_tokens=max_tokens,
                  temperature=0, top_p=1, n=1, stream=True,
                  stream_options={"include_usage": True}, reasoning_effort="none",
                  chat_template_kwargs={"enable_thinking": False})
    if cache_prompt is not None:
        result["cache_prompt"] = cache_prompt
    return result


def sse_events(lines):
    """Handle comments, CRLF, multi-line data, and an EOF without a blank line."""
    data = []
    for line in lines:
        line = line.rstrip(b"\r\n")
        if not line:
            if data:
                yield b"\n".join(data).decode("utf-8")
                data = []
        elif line.startswith(b"data:"):
            data.append(line[5:].lstrip(b" "))
    if data:
        yield b"\n".join(data).decode("utf-8")


class StreamState:
    def __init__(self):
        self.content = ""
        self.reasoning = ""
        self.usage = {}
        self.timings = {}
        self.finish = None
        self.first = None
        self.first_content = None
        self.last = None
        self.done = False

    def feed(self, text, elapsed):
        if text.strip() == "[DONE]":
            self.done = True
            return
        event = json.loads(text)
        if event.get("error"):
            raise RuntimeError(json.dumps(event["error"]))
        self.usage.update(event.get("usage") or {})
        self.timings.update(event.get("timings") or {})
        for choice in event.get("choices", []):
            if choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta") or {}
            content = delta.get("content") or ""
            reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if content or reasoning:
                if self.first is None:
                    self.first = elapsed
                self.last = elapsed
            if content and self.first_content is None:
                self.first_content = elapsed
            self.content += content
            self.reasoning += reasoning
            self.finish = choice.get("finish_reason") or self.finish

    def result(self, wall):
        cached = (self.usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if cached is None:
            cached = self.timings.get("cache_n")
        count = self.usage.get("completion_tokens")
        return dict(content=self.content, reasoning_content=self.reasoning,
                    usage=self.usage, timings=self.timings, finish_reason=self.finish,
                    stream_done=self.done, wall_s=wall, ttft_s=self.first,
                    first_content_s=self.first_content, last_output_s=self.last,
                    cached_prompt_tokens=cached,
                    cache_observation=("unknown" if cached is None else
                                       "zero_reported" if cached == 0 else "reuse_reported"),
                    engine_prefill_tps=self.timings.get("prompt_per_second"),
                    engine_decode_tps=self.timings.get("predicted_per_second"),
                    output_tokens_per_wall_s=(count / wall if count is not None and wall > 0 else None),
                    client_decode_tps=None,
                    timing_note="TTFT is first nonempty output (including reasoning); usage counts, never SSE events, are tokens. Client decode TPS is deliberately null: streamed chunks may batch multiple tokens. Engine rates use engine denominators; output/wall includes queue and prefill.")


def stream_chat(base_url, request_payload, deadline, raw_path, timeout=1800):
    """deadline is absolute time.monotonic(); raw bytes are private caller-owned files."""
    start = time.monotonic()
    end = min(deadline, start + timeout)
    state = StreamState()
    conn = None
    error = None
    category = None
    status = None
    timer = None
    try:
        if end <= start:
            raise TimeoutError("overall deadline exhausted before request")
        url = urlsplit(base_url)
        if url.scheme not in ("http", "https") or not url.hostname:
            raise ValueError("base URL must be http(s)")
        cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
        conn = cls(url.hostname, url.port, timeout=end - start)
        conn.connect()
        import threading
        sock = conn.sock
        def abort():
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        timer = threading.Timer(max(0, end - time.monotonic()), abort)
        timer.daemon = True
        timer.start()
        conn.request("POST", url.path.rstrip("/") + "/chat/completions",
                     body=json.dumps(request_payload).encode(),
                     headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
        response = conn.getresponse()
        status = response.status
        # A watchdog closes even a slow-drip response at the absolute deadline.
        try:
            with raw_path.open("wb") as raw:
                def lines():
                    while True:
                        remaining = end - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("request/overall deadline")
                        sock.settimeout(remaining)
                        line = response.readline()
                        if not line:
                            return
                        raw.write(line)
                        raw.flush()
                        yield line
                if status != 200:
                    for _ in lines():
                        pass
                    raise ConnectionError(f"HTTP {status}; see private raw response")
                if "text/event-stream" not in response.getheader("Content-Type", ""):
                    for _ in lines():
                        pass
                    raise ValueError("expected text/event-stream response")
                for text in sse_events(lines()):
                    state.feed(text, time.monotonic() - start)
                    if state.done:
                        break
                if not state.done or state.finish is None:
                    raise ConnectionError("incomplete stream: missing DONE or finish_reason")
        finally:
            timer.cancel()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        category = "deadline" if time.monotonic() >= end or isinstance(exc, TimeoutError) else "infrastructure"
    finally:
        if timer:
            timer.cancel()
        if conn:
            conn.close()
    result = state.result(time.monotonic() - start)
    result.update(error=error, error_category=category, http_status=status,
                  started_monotonic=start, ended_monotonic=time.monotonic())
    result["outcome"] = category or ("output_cap" if state.finish in ("length", "max_tokens")
                                     else "natural_finish" if state.finish == "stop" else "unexpected_finish")
    return result
