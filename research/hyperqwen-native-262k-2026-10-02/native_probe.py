#!/usr/bin/env python3
"""Offline prompt builder and online near-native vLLM capacity probe.

Generate with CUDA hidden. Run only against a loopback GPU0 trial already started
by run-gpu0.sh. The probe never starts or stops a model or production service.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from urllib.request import Request, urlopen
from uuid import uuid4

CONTEXT = 262_144
MIN_INPUT = 261_760
MAX_OUTPUT = 128
OUTPUT_RESERVE = 256  # allow for server-side template accounting differences
BEGIN = "BEGIN_NATIVE_262K_SENTINEL"
END = "END_NATIVE_262K_SENTINEL"


def template_tokens(tokenizer, prompt: str) -> int:
    result = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False,
    )
    return len(result["input_ids"])


def build(model_dir: Path, output: Path) -> None:
    AutoTokenizer = import_module("transformers").AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    nonce = uuid4().hex.upper()
    prefix = (
        f"{BEGIN}\nUnique run {nonce}. Read the complete prompt before answering. "
        "Return only the two sentinel lines in their original order.\n"
    )
    unit = (
        "Record code: def normalize(value, offset): return value + offset. "
        "Record neutral: blue cedar seven is a fixed observation. "
        "Record rule: preserve ordering and do not invent records.\n"
    )
    suffix = (
        "\nAfter reading the complete context, reply with exactly two lines, "
        f"first {BEGIN}, then {END}. Do not add an explanation.\n{END}\n"
    )
    unit_size = max(1, template_tokens(tokenizer, prefix + unit * 16 + suffix)
                    - template_tokens(tokenizer, prefix + unit * 15 + suffix))
    lo, hi = 0, (CONTEXT // unit_size) + 128
    # Largest complete repetition with room for output and tokenizer differences.
    while lo < hi:
        mid = (lo + hi + 1) // 2
        n = template_tokens(tokenizer, prefix + unit * mid + suffix)
        if n <= CONTEXT - OUTPUT_RESERVE:
            lo = mid
        else:
            hi = mid - 1
    prompt = prefix + unit * lo + suffix
    count = template_tokens(tokenizer, prompt)
    if not (MIN_INPUT <= count <= CONTEXT - OUTPUT_RESERVE):
        raise RuntimeError(f"Native-context prompt did not fit: {count} tokens")
    output.mkdir(parents=True, exist_ok=True)
    (output / "prompt.txt").write_text(prompt, encoding="utf-8")
    info = {"context": CONTEXT, "offline_chat_tokens": count, "output_reserve": OUTPUT_RESERVE,
            "nonce": nonce, "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "begin": BEGIN, "end": END}
    (output / "prompt-metadata.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


def api(url: str, path: str, body: dict | None = None, timeout: int = 30) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = Request(url + path, data=data,
                  headers={"Content-Type": "application/json"} if data else {},
                  method="POST" if data else "GET")
    with urlopen(req, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError(f"Unexpected {path} response")
    return value


def samples(stop: threading.Event, rows: list[dict]) -> None:
    while not stop.is_set():
        stamp = time.time()
        row: dict[str, object] = {"time": stamp}
        try:
            output = subprocess.check_output([
                "nvidia-smi", "--query-gpu=index,memory.used,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits"], text=True, timeout=8)
            row["gpus"] = [list(map(str.strip, record)) for record in csv.reader(output.splitlines())]
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith(("MemAvailable:", "SwapFree:")):
                    name, value, *_ = line.split()
                    row[name[:-1] + "_kib"] = int(value)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            row["sample_error"] = str(exc)
        rows.append(row)
        stop.wait(2)


def run(base_url: str, prompt_dir: Path, output: Path) -> None:
    if base_url != "http://127.0.0.1:18120":
        raise RuntimeError("Capacity probe may only target the GPU0 trial on loopback port 18120")
    if subprocess.run(["systemctl", "is-active", "--quiet",
                       "llama-muse-glimmer-30b-131k.service"]).returncode == 0:
        raise RuntimeError("Muse is still active; refusing trial probe")
    prompt = (prompt_dir / "prompt.txt").read_text(encoding="utf-8")
    meta = json.loads((prompt_dir / "prompt-metadata.json").read_text())
    if hashlib.sha256(prompt.encode()).hexdigest() != meta["sha256"]:
        raise RuntimeError("Prompt hash changed")
    with urlopen(base_url + "/health", timeout=10) as health:
        if health.status != 200:
            raise RuntimeError("GPU0 trial server is not healthy")
    models = api(base_url, "/v1/models").get("data", [])
    if not any(model.get("id") == "qwen3.8-27b" for model in models):
        raise RuntimeError("Expected trial model was not served")

    before = {unit: subprocess.check_output([
        "systemctl", "show", "-p", "MainPID", "--value", unit], text=True).strip()
        for unit in ("llamampere-qwen3.8-atx-iq4xs-m-gpu1.service",
                     "llamampere-qwen3.8-atx-iq4xs-m-gpu2.service")}
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    stop = threading.Event()
    sampler = threading.Thread(target=samples, args=(stop, rows), daemon=True)
    body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0, "top_p": 1.0, "max_tokens": MAX_OUTPUT,
            "seed": 3817, "stream": False}
    sampler.start()
    start = time.monotonic()
    try:
        reply = api(base_url, "/v1/chat/completions", body, timeout=7200)
    except Exception as exc:
        (output / "native-error.json").write_text(json.dumps({
            "error": repr(exc), "seconds": time.monotonic() - start,
            "protected_before": before, "gpu_host_samples": rows,
        }, indent=2) + "\n")
        raise
    finally:
        stop.set()
        sampler.join(timeout=5)
    wall = time.monotonic() - start
    after = {unit: subprocess.check_output([
        "systemctl", "show", "-p", "MainPID", "--value", unit], text=True).strip()
        for unit in before}
    msg = reply.get("choices", [{}])[0]
    content = msg.get("message", {}).get("content") or ""
    usage = reply.get("usage", {})
    prompt_n = usage.get("prompt_tokens")
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens")
    lines = [line.strip() for line in content.strip().splitlines()]
    errors = []
    if not isinstance(prompt_n, int) or not MIN_INPUT <= prompt_n <= CONTEXT - MAX_OUTPUT:
        errors.append(f"actual prompt tokens outside native gate: {prompt_n}")
    if cached not in (0, None):
        errors.append(f"unexpected cached prompt tokens: {cached}")
    if lines[:2] != [BEGIN, END]:
        errors.append("the two sentinels were not the first two lines in order")
    if msg.get("finish_reason") != "stop":
        errors.append(f"completion did not stop normally: {msg.get('finish_reason')}")
    if before != after or any(pid in ("0", "") for pid in after.values()):
        errors.append("protected Qwen PIDs changed or stopped")
    result = {"capacity_pass": not errors, "failures": errors, "seconds": wall,
              "offline_prompt_tokens": meta["offline_chat_tokens"], "usage": usage,
              "finish_reason": msg.get("finish_reason"), "content": content,
              "metrics": reply.get("metrics"), "protected_before": before,
              "protected_after": after, "gpu_host_samples": rows}
    (output / "native-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("capacity_pass", "failures", "seconds", "usage", "finish_reason")}, indent=2))
    if errors:
        raise RuntimeError("Native 262144-token capacity gate failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    builder = sub.add_parser("build")
    builder.add_argument("--model-dir", type=Path, required=True)
    builder.add_argument("--output-dir", type=Path, required=True)
    prober = sub.add_parser("run")
    prober.add_argument("--base-url", default="http://127.0.0.1:18120")
    prober.add_argument("--prompt-dir", type=Path, required=True)
    prober.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.model_dir, args.output_dir)
    else:
        run(args.base_url, args.prompt_dir, args.output_dir)
