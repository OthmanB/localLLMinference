#!/usr/bin/env python3
"""Stage 5 — limited orchestrated test.

Roles (all via plain HTTP inference; no service is modified):
  planner/researcher/reviewer: flashnext-coder at :1901 (the candidate)
  executors:                   qwen3.8-27b ATX replicas at :8080 / :8081
Supervisor (this script) issues tasks, relays outputs, and records the trace.
The baseline variant (BASELINE=1) swaps the planner role to Muse at :8082.
"""
import json
import os
import sys
import time

import probes

CANDIDATE = "http://127.0.0.1:1901"
CANDIDATE_MODEL = "flashnext-coder"
EXECUTOR_A = ("http://127.0.0.1:8080", "qwen3.8-27b-atx-iq4xs-m-262144-gpu1")
EXECUTOR_B = ("http://127.0.0.1:8081", "qwen3.8-27b-atx-iq4xs-m-262144-gpu2")
MUSE = ("http://127.0.0.1:8082", "muse-glimmer-30b-kquant17")

BASELINE = os.environ.get("BASELINE", "0") == "1"
PLANNER = MUSE if BASELINE else (CANDIDATE, CANDIDATE_MODEL)
OUT = f"results/orchestrated-{'muse' if BASELINE else 'coder'}/"
os.makedirs(OUT, exist_ok=True)

TASK = """You are a small software team. The planner produces the work; the executor implements it.

PLANNER (you, first): given the task below, produce a concise implementation plan:
the files to create/modify, the exact function signatures, the algorithm, edge
cases, and a short verification checklist. Keep it under 400 words.

TASK: Write a Python module `chunked.py` that provides:
1) `chunked_read(path, size=65536) -> Iterator[bytes]` — memory-bounded file
   reading that also tolerates files that grow while being read (stop at the
   size observed at start, do not raise on short reads);
2) `sha256_of(path) -> str` computed via the same chunked reading;
3) a `__main__` block: `python chunked.py <path> [size]` prints the digest.
No third-party dependencies.

EXECUTOR (second): implement the module exactly per the plan. Output ONLY the
complete file content of chunked.py in a single python code block.

REVIEWER (third): review the implementation against the plan. List any
concrete deviations or bugs (with line references) and give a verdict:
APPROVE or CHANGES_REQUESTED. Be strict but fair; do not invent issues."""


def call(base, model, messages, max_tokens=4000):
    t0 = time.time()
    kw = {}
    if model == CANDIDATE_MODEL:
        kw["chat_template_kwargs"] = {"enable_thinking": False}
    d = probes.chat(base, model, messages, temperature=0.2, max_tokens=max_tokens, **kw)
    return d["content"], round(time.time() - t0, 1)


def main():
    trace = {"planner": PLANNER[1] if not isinstance(PLANNER, str) else PLANNER,
             "baseline": BASELINE, "started": time.time()}
    plan, t_plan = call(PLANNER[0], PLANNER[1],
                        [{"role": "user", "content": TASK + "\n\nYou are the PLANNER. Output the plan only."}])
    trace["plan"] = {"content": plan, "wall_s": t_plan}
    impl, t_impl = call(EXECUTOR_A[0], EXECUTOR_A[1],
                        [{"role": "user", "content": TASK + "\n\nYou are the EXECUTOR.\n\nPlan:\n" + plan +
                                            "\n\nOutput the complete chunked.py file only."}])
    trace["impl"] = {"content": impl, "wall_s": t_impl, "executor": EXECUTOR_A[1]}
    review, t_review = call(PLANNER[0], PLANNER[1],
                           [{"role": "user", "content": TASK + "\n\nYou are the REVIEWER.\n\nPlan:\n" + plan +
                                                 "\n\nImplementation:\n" + impl +
                                                 "\n\nOutput the review and verdict only."}])
    trace["review"] = {"content": review, "wall_s": t_review}
    trace["finished"] = time.time()
    with open(OUT + "trace.json", "w") as f:
        json.dump(trace, f, indent=1)
    print(json.dumps({"planner_wall": t_plan, "impl_wall": t_impl,
                      "review_wall": t_review, "total_s": round(trace["finished"] - trace["started"], 1),
                      "verdict_line": [l for l in review.splitlines() if "APPROVE" in l or "CHANGES" in l][-1:]}))


if __name__ == "__main__":
    main()
