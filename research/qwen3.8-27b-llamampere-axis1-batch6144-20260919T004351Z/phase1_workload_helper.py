#!/usr/bin/env python3
import json
import sys
import time
import urllib.request
from pathlib import Path


ARTIFACT, LABEL, TARGET, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
ROOT = Path(ARTIFACT)
BASE_URL = "http://127.0.0.1:18101"


def post(path, payload):
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            body = response.read()
            status = response.status
            headers = dict(response.headers.items())
    except Exception as exc:
        ended = time.time()
        return {
            "status": None,
            "headers": {},
            "body": b"",
            "error": repr(exc),
            "started": started,
            "ended": ended,
            "wall_seconds": ended - started,
        }
    ended = time.time()
    return {
        "status": status,
        "headers": headers,
        "body": body,
        "error": None,
        "started": started,
        "ended": ended,
        "wall_seconds": ended - started,
    }


def token_count(text, raw_name):
    result = post("/tokenize", {"content": text})
    (ROOT / raw_name).write_bytes(result["body"])
    if result["error"]:
        raise RuntimeError(result["error"])
    if result["status"] != 200:
        raise RuntimeError("tokenize status %s" % result["status"])
    return len(json.loads(result["body"].decode("utf-8"))["tokens"])


prefix = (
    "Synthetic Phase 1 coding and neutral prefill workload. "
    "Read every record in order and retain the final instruction. "
    "The records are deliberately repetitive but include code-shaped and neutral text. "
    "Depth label: %s.\n\n" % LABEL
)
units = [
    "Record class code: def normalize_value(value: int, offset: int) -> int: return value + offset.\n",
    "Record class neutral: The stable observation is blue cedar seven, with no external action.\n",
    "Record class code: if total < 0: total = 0; return min(total, 1000003).\n",
    "Record class neutral: Preserve ordering, preserve exact integers, and do not invent records.\n",
    "Record class code: result = {\"name\": \"sample\", \"enabled\": True, \"count\": 3}.\n",
    "Record class neutral: This sentence is a fixed synthetic datum for long-context measurement.\n",
    "Record class code: for item in values: checksum = (checksum * 31 + item) % 1000000007.\n",
    "Record class neutral: The expected final response is short and must not quote the records.\n",
]
cycle = "".join(units)

prefix_count = token_count(prefix, "%s-prefix-tokenize.json" % LABEL)
sample_count = token_count(prefix + cycle * 64, "%s-sample-tokenize.json" % LABEL)
cycle_tokens = max(1, (sample_count - prefix_count) // 64)
cycles = max(1, (TARGET - prefix_count) // cycle_tokens)
suffix = (
    "\nFinal instruction: after reading the complete context, output exactly one line "
    "with the label %s and the integer checksum 3817. Do not repeat the records.\n" % LABEL
)

text = prefix + cycle * cycles + suffix
count = token_count(text, "%s-tokenize-0.json" % LABEL)
for attempt in range(8):
    if abs(count - TARGET) <= max(64, cycle_tokens // 2):
        break
    delta = TARGET - count
    step = max(1, abs(delta) // cycle_tokens)
    cycles += step if delta > 0 else -step
    cycles = max(1, cycles)
    text = prefix + cycle * cycles + suffix
    count = token_count(text, "%s-tokenize-%d.json" % (LABEL, attempt + 1))

prompt_path = ROOT / ("%s-prompt.txt" % LABEL)
(ROOT / ("%s-token-count.txt" % LABEL)).write_text(
    "target=%d\nactual_tokenize_count=%d\nprefix_tokens=%d\ncycle_tokens_estimate=%d\ncycles=%d\n"
    % (TARGET, count, prefix_count, cycle_tokens, cycles),
    encoding="utf-8",
)

settings = {
    "n_predict": 128,
    "temperature": 0.0,
    "top_k": 0,
    "top_p": 1.0,
    "min_p": 0.0,
    "seed": SEED,
    "ignore_eos": True,
    "stream": False,
    "cache_prompt": True,
    "target_token_depth": TARGET,
    "actual_tokenize_count": count,
    "prompt_file": str(prompt_path),
}
request_settings = {
    key: settings[key]
    for key in (
        "n_predict",
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "seed",
        "ignore_eos",
        "stream",
        "cache_prompt",
    )
}
(ROOT / ("%s-request-settings.json" % LABEL)).write_text(
    json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)

events_path = ROOT / "events.tsv"
summaries = []
for phase in ("cold", "warm"):
    with events_path.open("a", encoding="utf-8") as events:
        events.write("%s_%s_start\t%.9f\n" % (LABEL, phase, time.time()))
    request_payload = dict(request_settings)
    request_payload["prompt"] = text
    result = post("/completion", request_payload)
    body_path = ROOT / ("%s-%s-response.json" % (LABEL, phase))
    headers_path = ROOT / ("%s-%s-headers.txt" % (LABEL, phase))
    body_path.write_bytes(result["body"])
    headers_path.write_text(
        "status=%s\nerror=%s\nwall_seconds=%.9f\n%s"
        % (
            result["status"],
            result["error"],
            result["wall_seconds"],
            "".join("%s: %s\n" % item for item in sorted(result["headers"].items())),
        ),
        encoding="utf-8",
    )
    with events_path.open("a", encoding="utf-8") as events:
        events.write("%s_%s_end\t%.9f\n" % (LABEL, phase, time.time()))
    summary = {
        "label": LABEL,
        "phase": phase,
        "target_token_depth": TARGET,
        "actual_tokenize_count": count,
        "status": result["status"],
        "error": result["error"],
        "wall_seconds": result["wall_seconds"],
        "response_bytes": len(result["body"]),
    }
    if result["body"]:
        try:
            parsed = json.loads(result["body"].decode("utf-8"))
            summary["timings"] = parsed.get("timings", {})
            summary["tokens_evaluated"] = parsed.get("tokens_evaluated")
            summary["tokens_predicted"] = parsed.get("tokens_predicted")
            summary["stop_type"] = parsed.get("stop_type")
            summary["truncated"] = parsed.get("truncated")
            summary["tokens_cached"] = parsed.get("tokens_cached")
        except Exception as exc:
            summary["parse_error"] = repr(exc)
    summaries.append(summary)
    (ROOT / ("%s-%s-summary.json" % (LABEL, phase))).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

(ROOT / ("%s-summary.json" % LABEL)).write_text(
    json.dumps({"settings": settings, "requests": summaries}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps({"label": LABEL, "actual_tokenize_count": count, "requests": summaries}, sort_keys=True))
