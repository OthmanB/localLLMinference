#!/usr/bin/env python3
"""Freeze a paired, input-visible SWE-bench Verified task manifest."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi


DATASET = "princeton-nlp/SWE-bench_Verified"
SPLIT = "test"
SEED = 20260830
ROOT = Path(os.environ.get("SWEBENCH_ROOT", Path(__file__).resolve().parents[1]))
QUOTAS = {
    "django/django": 10,
    "sympy/sympy": 5,
    "sphinx-doc/sphinx": 4,
    "matplotlib/matplotlib": 4,
    "scikit-learn/scikit-learn": 3,
    "astropy/astropy": 3,
    "pydata/xarray": 3,
    "pytest-dev/pytest": 2,
    "pylint-dev/pylint": 2,
    "psf/requests": 2,
    "mwaskom/seaborn": 1,
    "pallets/flask": 1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tie_key(instance_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{instance_id}".encode()).hexdigest()


def select_rows(rows: list[dict]) -> list[dict]:
    selected = []
    for repo, quota in QUOTAS.items():
        candidates = [row for row in rows if row["repo"] == repo]
        if len(candidates) < quota:
            raise RuntimeError(f"Repository {repo} has only {len(candidates)} rows; need {quota}")
        candidates.sort(key=lambda row: (len(row["problem_statement"]), tie_key(row["instance_id"])))
        for position in range(quota):
            index = min(len(candidates) - 1, int((position + 0.5) * len(candidates) / quota))
            selected.append(candidates[index])
    selected.sort(key=lambda row: (row["repo"], len(row["problem_statement"]), tie_key(row["instance_id"])))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "swebench/manifest-q4-q5-40.json",
    )
    args = parser.parse_args()

    dataset = load_dataset(DATASET, split=SPLIT)
    hub_info = HfApi().dataset_info(DATASET)
    selected = select_rows([dataset[index] for index in range(len(dataset))])

    q4_path = Path(
        os.environ.get(
            "SWEBENCH_Q4_MODEL",
            ROOT / "models/qwen3.8-27b-q4-k-m-gguf/Qwen3.8-27B-UD-Q4_K_M.gguf",
        )
    )
    q5_path = Path(
        os.environ.get(
            "SWEBENCH_Q5_MODEL",
            ROOT / "models/qwen3.8-27b-q5-k-m-gguf/Qwen3.8-27B-UD-Q5_K_M.gguf",
        )
    )
    llama_server = os.environ.get("LLAMA_SERVER_BIN", "llama-server")
    llama_version = subprocess.run(
        [llama_server, "--version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout.strip()

    payload = {
        "schema_version": 1,
        "dataset": {"name": DATASET, "split": SPLIT, "hub_sha": hub_info.sha, "fingerprint": dataset._fingerprint},
        "selection": {
            "seed": SEED,
            "method": "repository quotas and input-visible problem-statement length quantiles",
            "quotas": QUOTAS,
            "task_count_per_model": len(selected),
        },
        "runtime": {
            "mini_swe_agent": "2.4.6",
            "swebench": "5.0.2",
            "llama_cpp": llama_version,
            "context_tokens": 131072,
            "kv_cache": "q8_0",
            "max_tokens": 8192,
            "temperature": 0.0,
            "top_p": 0.95,
            "chat_template_kwargs": {"enable_thinking": False},
            "command_timeout_seconds": 600,
            "container_timeout": "2h",
            "task_timeout_seconds": 7200,
            "network": "none",
        },
        "models": {
            "q4": {
                "gpu": 0,
                "endpoint": "http://127.0.0.1:8080/v1",
                "served_model": "qwen3.8-27b-q4-gpukv128",
                "path": str(q4_path),
                "sha256": sha256_file(q4_path),
            },
            "q5": {
                "gpu": 1,
                "endpoint": "http://127.0.0.1:8081/v1",
                "served_model": "qwen3.8-27b-q5-gpukv128",
                "path": str(q5_path),
                "sha256": sha256_file(q5_path),
            },
        },
        "tasks": [
            {
                "order": order,
                "instance_id": row["instance_id"],
                "repo": row["repo"],
                "base_commit": row["base_commit"],
                "problem_statement_chars": len(row["problem_statement"]),
                "created_at": row["created_at"],
                "version": row["version"],
            }
            for order, row in enumerate(selected)
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(selected)} paired tasks)")
    print(f"manifest_sha256 {payload['manifest_sha256']}")


if __name__ == "__main__":
    main()
