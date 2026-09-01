#!/usr/bin/env python3
"""Create an official-format gold-patch prediction file for a frozen manifest."""

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


ROOT = Path(os.environ.get("SWEBENCH_ROOT", Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=ROOT / "swebench/gold/verified40-evaluator.json",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    dataset = manifest["dataset"]["name"]
    split = manifest["dataset"]["split"]
    rows = {row["instance_id"]: row for row in load_dataset(dataset, split=split)}
    evaluator_rows = {
        row["instance_id"]: row
        for row in load_dataset("SWE-bench/SWE-bench", split=split)
    }
    predictions = []
    selected_evaluator_rows = []
    for task in manifest["tasks"]:
        row = rows[task["instance_id"]]
        evaluator_row = evaluator_rows[task["instance_id"]]
        if row["base_commit"] != task["base_commit"]:
            raise RuntimeError(f"base commit changed for {task['instance_id']}")
        for field in ("base_commit", "patch", "test_patch", "problem_statement", "version"):
            if row[field] != evaluator_row[field]:
                raise RuntimeError(f"{field} differs between datasets for {task['instance_id']}")
        for field in ("image", "eval_script", "log_parser", "eval_type"):
            if not evaluator_row.get(field):
                raise RuntimeError(f"evaluator field {field} is missing for {task['instance_id']}")
        selected_evaluator_rows.append(evaluator_row)
        predictions.append(
            {
                "model_name_or_path": "gold-patch",
                "instance_id": task["instance_id"],
                "model_patch": row["patch"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row) for row in predictions) + "\n", encoding="utf-8")
    args.dataset_output.write_text(json.dumps(selected_evaluator_rows, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(predictions)} gold patches)")
    print(f"wrote {args.dataset_output} ({len(selected_evaluator_rows)} evaluator rows)")


if __name__ == "__main__":
    main()
