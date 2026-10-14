#!/usr/bin/env python3
"""GPU0-only ATX benchmark against the same near-native prompts as vLLM.

The server is external and is never started, stopped or reconfigured here.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time

from benchmark_native import CODE_BLOCK
from native_probe import BEGIN, END, CONTEXT, api, samples

URL = "http://127.0.0.1:18120"
PROTECTED = (
    "llamampere-qwen3.8-atx-iq4xs-m-gpu1.service",
    "llamampere-qwen3.8-atx-iq4xs-m-gpu2.service",
)


def pids() -> dict[str, str]:
    return {unit: subprocess.check_output([
        "systemctl", "show", "-p", "MainPID", "--value", unit], text=True).strip()
        for unit in PROTECTED}


def run(prompt_dir: Path, output_dir: Path) -> None:
    if subprocess.run(["systemctl", "is-active", "--quiet",
                       "llama-muse-glimmer-30b-131k.service"]).returncode == 0:
        raise RuntimeError("Muse is active; refusing GPU0 benchmark")
    props = api(URL, "/props")
    settings = props.get("default_generation_settings") or {}
    if settings.get("n_ctx") != CONTEXT:
        raise RuntimeError(f"ATX context is {settings.get('n_ctx')} rather than {CONTEXT}")
    prompt = (prompt_dir / "prompt.txt").read_text()
    meta = json.loads((prompt_dir / "prompt-metadata.json").read_text())
    if hashlib.sha256(prompt.encode()).hexdigest() != meta["sha256"]:
        raise RuntimeError("Prompt content changed")
    before = pids()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    stop = threading.Event()
    sampler = threading.Thread(target=samples, args=(stop, rows), daemon=True)
    body = {"prompt": prompt, "n_predict": 640, "temperature": 0,
            "top_k": 0, "top_p": 1.0, "seed": 3817, "ignore_eos": False,
            "stream": False, "cache_prompt": False}
    sampler.start()
    start = time.monotonic()
    try:
        reply = api(URL, "/completion", body, timeout=7200)
    except Exception as exc:
        (output_dir / "error.json").write_text(json.dumps({
            "error": repr(exc), "seconds": time.monotonic() - start,
            "gpu_host_samples": rows}, indent=2) + "\n")
        raise
    finally:
        stop.set()
        sampler.join(timeout=5)
    after = pids()
    timings = reply.get("timings", {})
    content = reply.get("content") or ""
    prompt_n = reply.get("tokens_evaluated", 0)
    errors = []
    if not isinstance(prompt_n, int) or not 261_000 <= prompt_n <= CONTEXT - 640:
        errors.append(f"prompt not near-native or exceeds output reserve: {prompt_n}")
    if reply.get("truncated") is not False or timings.get("cache_n") not in (0, None):
        errors.append("prompt was truncated or prefetched from cache")
    if content.strip().splitlines()[:2] != [BEGIN, END]:
        errors.append("missing exact first two sentinel lines")
    generated = reply.get("tokens_predicted", 0)
    if not isinstance(generated, int) or generated < 192:
        errors.append(f"too few generated tokens for sustained decode: {generated}")
    if meta.get("task") == "copy":
        from difflib import SequenceMatcher
        copied = content.rsplit(END, 1)[-1].strip()
        similarity = SequenceMatcher(None, CODE_BLOCK.strip(), copied).ratio()
        if similarity < 0.98:
            errors.append(f"code reproduction similarity too low: {similarity:.3f}")
    else:
        similarity = None
    if before != after or any(pid in ("0", "") for pid in after.values()):
        errors.append("protected Qwen service PID changed")
    result = {"valid": not errors, "failures": errors,
              "wall_s": time.monotonic() - start, "task": meta.get("task"),
              "prompt_tokens": prompt_n, "output_tokens": generated,
              "copy_similarity": similarity, "timings": timings,
              "content": content, "protected_before": before,
              "protected_after": after, "gpu_host_samples": rows}
    (output_dir / "benchmark-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("content", "gpu_host_samples")}, indent=2))
    if errors:
        raise RuntimeError("ATX benchmark failed validation; inspect benchmark-result.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.prompt_dir, args.output_dir)
