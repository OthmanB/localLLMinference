#!/usr/bin/env python3
"""Near-native 262k prefill/decode benchmark with a longer non-copying reply.

GPU0 capacity must be proved separately by native_probe.py at >=261760 input.
This probe reserves space for up to 640 output tokens while keeping >261k input.
"""

import argparse
import hashlib
from importlib import import_module
import json
from pathlib import Path
import subprocess
import threading
import time
from difflib import SequenceMatcher
from urllib.request import urlopen
from uuid import uuid4

from native_probe import BEGIN, END, CONTEXT, api, samples, template_tokens

MAX_OUTPUT = 640
MIN_PROMPT = 261_200
MAX_PROMPT = CONTEXT - 768
BASE_URL = "http://127.0.0.1:18120"
PROTECTED = (
    "llamampere-qwen3.8-atx-iq4xs-m-gpu1.service",
    "llamampere-qwen3.8-atx-iq4xs-m-gpu2.service",
)

CODE_BLOCK = """from __future__ import annotations
import hashlib
import json
from pathlib import Path

def read_chunks(path: Path, size: int = 65536):
    if size <= 0:
        raise ValueError("size must be positive")
    with path.open("rb") as source:
        while True:
            chunk = source.read(size)
            if not chunk:
                return
            yield chunk

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    for chunk in read_chunks(path):
        digest.update(chunk)
    return digest.hexdigest()

def load_json(path: Path) -> dict:
    raw = b"".join(read_chunks(path))
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value
"""


def build(model_dir: Path, output_dir: Path, task: str) -> None:
    tokenizer = import_module("transformers").AutoTokenizer.from_pretrained(model_dir)
    nonce = uuid4().hex.upper()
    prefix = (
        f"{BEGIN}\nRun {nonce}. You will read a long input and answer an independent "
        "software-review question at the end.\n"
    )
    unit = (
        "Record code: def normalize(value, offset): return value + offset. "
        "Record neutral: blue cedar seven is a fixed observation. "
        "Record rule: preserve ordering and do not invent records.\n"
    )
    if task == "copy":
        suffix = (
            "\nSource code from the task context:\n" + CODE_BLOCK + "\n"
            f"First output {BEGIN}, then {END}, on separate lines. "
            "After the two lines, reproduce the source code exactly as it appears "
            "above, starting with its first line. Do not add fences or commentary.\n"
        )
    else:
        suffix = (
            f"\n{END}\nFirst output {BEGIN}, then {END}, on separate lines. "
            "Then write twelve numbered, *different* checks for reviewing a Python "
            "JSON parsing function in production. Explain each check in one sentence; "
            "include malformed input, Unicode, memory limits, and error handling. "
            "Do not repeat, quote, or rewrite the records above.\n"
        )
    unit_size = max(1, template_tokens(tokenizer, prefix + unit * 16 + suffix)
                    - template_tokens(tokenizer, prefix + unit * 15 + suffix))
    lo, hi = 0, CONTEXT // unit_size + 128
    while lo < hi:
        mid = (lo + hi + 1) // 2
        n = template_tokens(tokenizer, prefix + unit * mid + suffix)
        if n <= MAX_PROMPT:
            lo = mid
        else:
            hi = mid - 1
    prompt = prefix + unit * lo + suffix
    count = template_tokens(tokenizer, prompt)
    if not MIN_PROMPT <= count <= MAX_PROMPT:
        raise RuntimeError(f"Long-reply prompt is outside the near-native range: {count}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt.txt").write_text(prompt)
    metadata = {"context": CONTEXT, "prompt_tokens_offline": count, "task": task,
                "output_reserve": 768, "max_output": MAX_OUTPUT,
                "sha256": hashlib.sha256(prompt.encode()).hexdigest(), "nonce": nonce}
    (output_dir / "prompt-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def protected_pids() -> dict[str, str]:
    return {unit: subprocess.check_output([
        "systemctl", "show", "-p", "MainPID", "--value", unit], text=True).strip()
        for unit in PROTECTED}


def run(prompt_dir: Path, output_dir: Path) -> None:
    if subprocess.run(["systemctl", "is-active", "--quiet",
                       "llama-muse-glimmer-30b-131k.service"]).returncode == 0:
        raise RuntimeError("Muse is active; refusing GPU0 experiment")
    with urlopen(BASE_URL + "/health", timeout=10) as health:
        if health.status != 200:
            raise RuntimeError("GPU0 trial is not ready")
    prompt = (prompt_dir / "prompt.txt").read_text()
    meta = json.loads((prompt_dir / "prompt-metadata.json").read_text())
    if hashlib.sha256(prompt.encode()).hexdigest() != meta["sha256"]:
        raise RuntimeError("Prompt content does not match its metadata")
    before = protected_pids()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    stop = threading.Event()
    sampler = threading.Thread(target=samples, args=(stop, rows), daemon=True)
    request = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": prompt}],
               "chat_template_kwargs": {"enable_thinking": False},
               "temperature": 0, "top_p": 1.0, "max_tokens": MAX_OUTPUT,
               "seed": 3817, "stream": False}
    sampler.start()
    start = time.monotonic()
    try:
        response = api(BASE_URL, "/v1/chat/completions", request, timeout=7200)
    except Exception as exc:
        (output_dir / "error.json").write_text(json.dumps({
            "error": repr(exc), "seconds": time.monotonic() - start,
            "gpu_host_samples": rows}, indent=2) + "\n")
        raise
    finally:
        stop.set()
        sampler.join(timeout=5)
    choice = response.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content") or ""
    usage = response.get("usage", {})
    metrics = response.get("metrics") or {}
    after = protected_pids()
    actual = usage.get("prompt_tokens", 0)
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens")
    generated = usage.get("completion_tokens", 0)
    errors = []
    if not isinstance(actual, int) or not MIN_PROMPT <= actual <= MAX_PROMPT:
        errors.append(f"near-native prompt count invalid: {actual}")
    if cached not in (None, 0):
        errors.append(f"prompt was cached: {cached}")
    if content.strip().splitlines()[:2] != [BEGIN, END]:
        errors.append("incorrect beginning/end sentinels")
    if not isinstance(generated, int) or generated < 192:
        errors.append(f"too few output tokens for sustained decode measurement: {generated}")
    copy_similarity = None
    if meta.get("task") == "copy":
        produced = "\n".join(content.strip().splitlines()[2:]).strip()
        copy_similarity = SequenceMatcher(None, CODE_BLOCK.strip(), produced).ratio()
        if copy_similarity < 0.98:
            errors.append(f"code reproduction similarity too low: {copy_similarity:.3f}")
    if before != after or any(pid in ("", "0") for pid in after.values()):
        errors.append("protected Qwen service PID changed")
    generation_ms = metrics.get("generation_time_ms")
    ttft_ms = metrics.get("time_to_first_token_ms")
    result = {"valid": not errors, "failures": errors,
              "wall_s": time.monotonic() - start, "prompt_tokens": actual,
              "output_tokens": generated, "cached_tokens": cached,
              "task": meta.get("task"), "copy_similarity": copy_similarity,
              "prefill_tps_est": round(1000 * actual / ttft_ms, 2) if ttft_ms else None,
              "decode_tps_est": round(1000 * generated / generation_ms, 2) if generation_ms else None,
              "metrics": metrics, "finish_reason": choice.get("finish_reason"),
              "content": content, "protected_before": before, "protected_after": after,
              "gpu_host_samples": rows}
    (output_dir / "benchmark-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("content", "gpu_host_samples")}, indent=2))
    if errors:
        raise RuntimeError("Near-native throughput gate invalid; see benchmark-result.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    builder = sub.add_parser("build")
    builder.add_argument("--model-dir", type=Path, required=True)
    builder.add_argument("--output-dir", type=Path, required=True)
    builder.add_argument("--task", choices=("analysis", "copy"), default="analysis")
    runner = sub.add_parser("run")
    runner.add_argument("--prompt-dir", type=Path, required=True)
    runner.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.model_dir, args.output_dir, args.task)
    else:
        run(args.prompt_dir, args.output_dir)
