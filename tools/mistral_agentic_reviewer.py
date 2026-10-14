#!/usr/bin/env python3
"""Small repository-aware reviewer diagnostic for Mistral IQ3_S.

This deliberately avoids mini-SWE-agent. It gives the model a real SWE-bench
repository in Docker and two explicit tools: bash(command) and
submit_review(text). A review is captured only through submit_review, so no
marker command, output-file convention, or agent-framework history rewriting is
involved.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path((os.environ['EXPERIMENT_ROOT'] + ''))
CASES_PATH = ROOT / "research/mistral-small-4-119b-context-sweep/phase4/cases.json"
IMAGES = {
    "case1": "docker.io/swebench/sweb.eval.x86_64.sphinx-doc_1776_sphinx-7985:latest",
    "case2": "docker.io/swebench/sweb.eval.x86_64.mwaskom_1776_seaborn-3187:latest",
    "case3": "docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-21930:latest",
    "case4": "docker.io/swebench/sweb.eval.x86_64.django_1776_django-16256:latest",
}

sys.path.insert(0, str(ROOT / "tools"))
import mistral_context_sweep as sweep  # noqa: E402
import mistral_review_sweep as review  # noqa: E402

MODEL = "mistral-s"
CONTEXT = 131072
TEMPERATURE = 0.7
REQUEST_TIMEOUT = int(os.environ.get("AGENTIC_REVIEW_REQUEST_TIMEOUT", "600"))
MAX_TOKENS = int(os.environ.get("AGENTIC_REVIEW_MAX_TOKENS", "8192"))
DEFAULT_CASES = ("case2", "case4")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run one shell command in the disposable repository at /testbed.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_review",
            "description": "Submit the final plain-text code review and end the task.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
]


def reviewer_messages(case: dict, max_steps: int) -> list[dict]:
    instructions = f"""You are a meticulous adversarial software engineer reviewing a proposed fix.
The issue is below. The candidate patch is mounted read-only at /candidate.patch.
The real SWE-bench base repository is at /testbed.

Use bash to inspect the patch and repository, search every producer and consumer
of any changed interface, and run one focused test or reproduction when useful.
You may apply the candidate patch only in this disposable container to test it.
Do not implement a replacement patch or modify project tests/configuration.

This is a bounded review with at most {max_steps} tool steps. By step 20, stop
broad exploration and rank your hypotheses. By step 25, test only the highest
value unresolved hypothesis. By step 28, synthesize the review. You MUST call
submit_review by step {max_steps}, even if uncertain; state residual uncertainty.

Each finding in submit_review must include priority, file/location, concrete
none remain, say what you checked and why it is sufficient. Use exactly one tool
per response: bash while investigating, then submit_review to finish.
"""
    return [
        {"role": "system", "content": "You are a concise, evidence-driven code reviewer."},
        {"role": "user", "content": f"<issue>\n{case['problem']}\n</issue>\n\n{instructions}"},
    ]


def chat(port: int, alias: str, messages: list[dict]) -> dict:
    payload = {
        "model": alias,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": 0.95,
        "stream": False,
        "chat_template_kwargs": {"reasoning_effort": "high"},
        "parse_tool_calls": True,
        "parallel_tool_calls": False,
    }
    return sweep.http_json(f"http://127.0.0.1:{port}/v1/chat/completions", payload, timeout=REQUEST_TIMEOUT)


def tool_smoke(port: int, alias: str, output: Path) -> bool:
    messages = [
        {"role": "system", "content": "Use the supplied bash tool when asked."},
        {"role": "user", "content": "Call bash with exactly: printf tool-smoke"},
    ]
    response = chat(port, alias, messages)
    output.write_text(json.dumps({"messages": messages, "response": response}, indent=2), encoding="utf-8")
    message = response.get("choices", [{}])[0].get("message", {})
    calls = message.get("tool_calls") or []
    return len(calls) == 1 and calls[0].get("function", {}).get("name") == "bash"


def start_container(image: str, patch_path: Path, name: str) -> str:
    command = [
        "docker", "run", "-d", "--rm", "--name", name, "--network", "none",
        "-w", "/testbed", "-v", f"{patch_path}:/candidate.patch:ro", image, "sleep", "1h",
    ]
    return subprocess.check_output(command, text=True).strip()


def run_bash(container: str, command: str) -> str:
    try:
        completed = subprocess.run(
            ["docker", "exec", "-w", "/testbed", container, "bash", "-lc", command],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        output = (completed.stdout + completed.stderr).strip()
        if len(output) > 12000:
            output = output[:12000] + "\n<output truncated>"
        return f"<returncode>{completed.returncode}</returncode>\n<output>{output}</output>"
    except subprocess.TimeoutExpired:
        return "<returncode>124</returncode>\n<output>Command timed out after 600 seconds.</output>"


def cleanup_container(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


def assistant_message(message: dict) -> dict:
    result = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("tool_calls"):
        result["tool_calls"] = message["tool_calls"]
    return result


def budget_note(step: int, max_steps: int) -> str:
    if step >= 28:
        return f"<review_budget>Step {step}/{max_steps}: synthesize and call submit_review now.</review_budget>\n"
    if step >= 25:
        return f"<review_budget>Step {step}/{max_steps}: test only the highest-value hypothesis, then synthesize.</review_budget>\n"
    if step >= 20:
        return f"<review_budget>Step {step}/{max_steps}: stop broad exploration and rank hypotheses.</review_budget>\n"
    return f"<review_budget>Step {step}/{max_steps}.</review_budget>\n"


def run_case(case: dict, outdir: Path, alias: str, max_steps: int) -> dict:
    case_dir = (outdir / case["id"]).resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    patch_path = case_dir / "candidate.patch"
    patch_path.write_text(case["candidate_patch"], encoding="utf-8")
    image = IMAGES[case["id"]]
    container = f"mistral-review-{case['id']}-{uuid.uuid4().hex[:8]}"
    messages = reviewer_messages(case, max_steps)
    events: list[dict] = []
    submission = ""
    status = "step_limit"
    started = time.monotonic()
    try:
        start_container(image, patch_path, container)
        for step in range(1, max_steps + 1):
            response = chat(sweep.MODELS[MODEL]["port"], alias, messages)
            message = response.get("choices", [{}])[0].get("message", {})
            messages.append(assistant_message(message))
            calls = message.get("tool_calls") or []
            event = {"step": step, "response": response}
            events.append(event)
            if not calls:
                # This preserves user/assistant alternation rather than injecting a
                # malformed recovery message after an unrecorded response.
                messages.append({
                    "role": "user",
                    "content": f"You must use exactly one tool now. {budget_note(step, max_steps)}",
                })
                continue
            if len(calls) != 1:
                status = "multiple_tool_calls"
                break
            call = calls[0]
            function = call.get("function", {})
            name = function.get("name")
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            if name == "submit_review":
                submission = str(arguments.get("text") or "").strip()
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": "Review submitted."})
                status = "submitted" if submission else "empty_submission"
                break
            if name != "bash" or not isinstance(arguments.get("command"), str):
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": "Invalid tool arguments. Use bash(command) or submit_review(text).",
                })
                continue
            output = budget_note(step, max_steps) + run_bash(container, arguments["command"])
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
    finally:
        cleanup_container(container)
    wall_seconds = round(time.monotonic() - started, 2)
    review_path = case_dir / "review.txt"
    trajectory_path = case_dir / "trajectory.json"
    review_path.write_text(submission, encoding="utf-8")
    trajectory_path.write_text(json.dumps({"messages": messages, "events": events}, indent=2), encoding="utf-8")
    return {
        "case_id": case["id"],
        "instance_id": case["instance_id"],
        "image": image,
        "status": status,
        "steps": len(events),
        "review_chars": len(submission),
        "wall_seconds": wall_seconds,
        "trajectory": str(trajectory_path),
        "review": str(review_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct repository-aware Mistral reviewer diagnostic")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--cases", nargs="*", default=list(DEFAULT_CASES))
    parser.add_argument("--ctx", type=int, default=CONTEXT)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--startup-timeout", type=int, default=360)
    parser.add_argument("--tensor-split", default="1,1")
    parser.add_argument("--verbosity", type=int, default=4)
    args = parser.parse_args()
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be positive")
    args.outdir.mkdir(parents=True, exist_ok=True)

    all_cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    by_id = {case["id"]: case for case in all_cases}
    missing = [case_id for case_id in args.cases if case_id not in by_id or case_id not in IMAGES]
    if missing:
        raise SystemExit(f"unknown or unsupported case IDs: {missing}")
    selected = [by_id[case_id] for case_id in args.cases]

    spec = sweep.MODELS[MODEL]
    server_log = args.outdir / "mistral-s.server.log"
    process = review.start_review_server(spec, args.ctx, args.tensor_split, args.verbosity, server_log)
    results: list[dict] = []
    try:
        sweep.wait_ready(spec["port"], process, args.startup_timeout)
        residency = review.wait_resident(server_log)
        if not residency.get("fully_resident"):
            raise RuntimeError(f"Mistral server is not fully resident: {residency}")
        if not tool_smoke(spec["port"], spec["alias"], args.outdir / "tool-smoke.json"):
            raise RuntimeError("Mistral bash tool-call smoke test failed")
        for case in selected:
            result = run_case(case, args.outdir, spec["alias"], args.max_steps)
            result["fully_resident"] = True
            result["temperature"] = TEMPERATURE
            result["reasoning_effort"] = "high"
            results.append(result)
            (args.outdir / "results.json").write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
            print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        sweep.terminate(process)


if __name__ == "__main__":
    main()
