#!/usr/bin/env python3
"""Generate EvalPlus samples with explicit Qwen chat-template controls."""

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from evalplus.data import get_human_eval_plus
from evalplus.sanitize import sanitize


def request_completion(base_url, model, message):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant good at coding."},
            {"role": "user", "content": message},
        ],
        "max_tokens": 4096,
        "temperature": 0,
        "top_p": 0.95,
        "n": 1,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    }
    request = Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    try:
        with urlopen(request, timeout=600) as response:
            return json.load(response)
    except HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--raw-samples", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=164)
    args = parser.parse_args()

    problems = get_human_eval_plus()
    selected = list(problems.items())[args.start : args.stop]
    for path in (args.samples, args.raw_samples, args.metadata):
        path.parent.mkdir(parents=True, exist_ok=True)

    with args.samples.open("w", encoding="utf-8") as sanitized_file, args.raw_samples.open(
        "w", encoding="utf-8"
    ) as raw_file, args.metadata.open("w", encoding="utf-8") as metadata_file:
        for task_id, task in selected:
            prompt = task["prompt"].strip() + "\n"
            message = (
                "Please provide a self-contained Python script that solves the following problem "
                "in a markdown code block:\n```python\n"
                f"{prompt.strip()}\n```"
            )
            response = request_completion(args.base_url, args.model, message)
            choice = response["choices"][0]
            assistant = choice.get("message") or {}
            solution = assistant.get("content") or ""
            sanitized_solution = sanitize(solution, entrypoint=task["entry_point"])

            json.dump({"task_id": task_id, "solution": sanitized_solution}, sanitized_file)
            sanitized_file.write("\n")
            json.dump({"task_id": task_id, "solution": solution}, raw_file)
            raw_file.write("\n")
            json.dump(
                {
                    "task_id": task_id,
                    "finish_reason": choice.get("finish_reason"),
                    "usage": response.get("usage"),
                    "raw_chars": len(solution),
                    "sanitized_chars": len(sanitized_solution),
                    "has_reasoning_content": bool(assistant.get("reasoning_content")),
                },
                metadata_file,
            )
            metadata_file.write("\n")
            print(task_id, choice.get("finish_reason"), len(solution), flush=True)


if __name__ == "__main__":
    main()
