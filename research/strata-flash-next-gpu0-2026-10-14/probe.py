#!/usr/bin/env python3
"""Serial GPU0-only HTTP probes; save raw responses and honest streaming timing."""
import argparse
import json
import pathlib
import time
import urllib.request
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent
OLD = ROOT.parent / 'qwen3.8-flash-next-gsq-rco-coder-gpu0-2026-09-30'


def request(base, model, messages, **extra) -> dict[str, Any]:
    body = dict(model=model, messages=messages, temperature=0, max_tokens=4096,
                stream=True, stream_options={'include_usage': True},
                reasoning_effort='none', chat_template_kwargs={'enable_thinking': False})
    body.update(extra)
    req = urllib.request.Request(base + '/v1/chat/completions',
                                 data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    first = last = None
    events, content, reasoning, tools = [], [], [], {}
    usage, timings, finish = {}, {}, None
    with urllib.request.urlopen(req, timeout=7200) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith('data: '):
                continue
            if line[6:] == '[DONE]':
                break
            event = json.loads(line[6:])
            if event.get('error'):
                raise RuntimeError(str(event['error']))
            now = time.monotonic()
            events.append({'elapsed_s': now - start, 'event': event})
            usage = event.get('usage') or usage
            timings = event.get('timings') or timings
            for choice in event.get('choices', []):
                delta = choice.get('delta') or {}
                if delta.get('content') or delta.get('reasoning_content') or delta.get('tool_calls'):
                    first = now if first is None else first
                    last = now
                content.append(delta.get('content') or '')
                reasoning.append(delta.get('reasoning_content') or '')
                finish = choice.get('finish_reason') or finish
                for call in delta.get('tool_calls', []):
                    dst = tools.setdefault(call.get('index', 0),
                                           {'id': '', 'type': 'function',
                                            'function': {'name': '', 'arguments': ''}})
                    if call.get('id'):
                        dst['id'] = call['id']
                    for key in ('name', 'arguments'):
                        dst['function'][key] += (call.get('function') or {}).get(key) or ''
    n = usage.get('completion_tokens')
    interval = last - first if first is not None and last is not None else 0
    return dict(content=''.join(content), reasoning=''.join(reasoning),
                tool_calls=list(tools.values()), finish_reason=finish, usage=usage,
                timings=timings, wall_s=time.monotonic() - start,
                ttft_s=first - start if first is not None else None,
                decode_interval_s=interval,
                client_decode_tps=(n - 1) / interval if n and interval > 0 else None,
                timing_note='Client tokens from usage, first-to-last output interval; '
                            'not SSE event count. Cross-check with engine timings.',
                events=events)


def save(label, case, result):
    folder = ROOT / 'results' / label
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (case + '.json')).write_text(json.dumps(result, indent=2))
    brief = {k: v for k, v in result.items() if k not in ('events', 'content', 'reasoning')}
    brief['content_head'] = result.get('content', '')[:160]
    print(case, json.dumps(brief), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--label', required=True)
    a = p.parse_args()
    if a.base not in ('http://127.0.0.1:8082', 'http://127.0.0.1:18144'):
        p.error('Only Muse GPU0 and the GPU0 experimental endpoint are allowed')
    cases = {
        'coding': "Write a Python function `parse_ranges(s: str) -> list[tuple[int,int]]` "
                  "that parses a string like '1-4,7,9-11' into a list of inclusive "
                  "(lo,hi) tuples. Raise ValueError on invalid input (empty parts, "
                  "lo>hi, non-integers). Include 4 doctest examples. Output code only.",
        'planning': "Plan, do not implement: add a retry with exponential backoff and "
                    "jitter wrapper to an existing Python HTTP client module that "
                    "currently has a fetch(url) -> bytes function with no error handling. "
                    "It has 12 call sites in a long-running service. Give the interface, "
                    "retryable exceptions, migration, tests, risks and verification.",
        'review': (OLD / 'prompts' / 'R3-review.txt').read_text()
                  if (OLD / 'prompts' / 'R3-review.txt').exists() else '',
        'json': 'Return JSON only (no prose) describing Mars with exactly these keys: '
                'name (string), moons (integer), gravity_m_s2 (number), '
                'day_length_hours (number), facts (array of 3 strings).',
    }
    # Historical prompt filenames differ across harness versions; extract the
    # actual model-facing R3 text, excluding the grader's notes if necessary.
    if not cases['review']:
        text = (OLD / 'quality-suite.md').read_text()
        section = text.split('## R3.')[1].split('> (Planted for the grader')[0]
        cases['review'] = '\n'.join(line[2:] if line.startswith('> ') else line[1:]
                                    for line in section.splitlines() if line.startswith('>'))
    for name, text in cases.items():
        result = request(a.base, a.model, [{'role': 'user', 'content': text}])
        if name == 'json':
            try:
                obj = json.loads(result['content'])
                result['valid_json_keys'] = set(obj) == {
                    'name', 'moons', 'gravity_m_s2', 'day_length_hours', 'facts'}
            except ValueError:
                result['valid_json_keys'] = False
        save(a.label, name, result)
    messages: list[dict[str, Any]] = [{'role': 'user', 'content': 'What is the weather like in Zurich right now?'}]
    tools = [{'type': 'function', 'function': {
        'name': 'get_weather', 'description': 'Get current weather for a city.',
        'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}},
                       'required': ['city']}}}]
    result = request(a.base, a.model, messages, tools=tools, max_tokens=512)
    save(a.label, 'tool-call', result)
    if result['tool_calls']:
        messages.append({'role': 'assistant', 'content': result['content'],
                         'tool_calls': result['tool_calls']})
        for call in result['tool_calls']:
            messages.append({'role': 'tool', 'tool_call_id': call['id'],
                             'content': '{"error":"weather service unavailable"}'})
        messages.append({'role': 'user', 'content': 'Explain what you can conclude; do not invent weather.'})
        save(a.label, 'tool-error', request(a.base, a.model, messages, tools=tools, max_tokens=512))
    messages = []
    for i, text in enumerate([
        'Plan a small document-search CLI. Requirements: cache results, use JSON output, no network. Keep it brief.',
        'Specify the component interfaces and validation approach.',
        'Change of requirements: drop caching entirely, keep JSON and offline operation. Revise the plan.',
        'List tests for the revised requirements; do not reinstate dropped features.',
        'Give the final implementation checklist for the current requirements.']):
        messages.append({'role': 'user', 'content': text})
        result = request(a.base, a.model, messages, max_tokens=768)
        save(a.label, f'state-turn-{i+1}', result)
        messages.append({'role': 'assistant', 'content': result['content']})


if __name__ == '__main__':
    main()
