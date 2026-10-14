#!/usr/bin/env python3
"""Official SWE-bench evaluation with offline, resource-bounded containers.

Only container creation is adapted; official patch application, eval scripts,
log parsing and resolution criteria remain untouched. Runs in a fresh process.
"""
import argparse
import os


def cached_container(test_spec, client, run_id, logger):
    # Fail instead of pulling or building: all 40 task images must be cached.
    client.images.get(test_spec.image)
    return client.containers.create(
        image=test_spec.image, name=f"sweb.eval.{test_spec.instance_id.lower()}.{run_id}",
        user="root", detach=True, command="tail -f /dev/null",
        network_mode="none", nano_cpus=2_000_000_000, mem_limit="6g",
        labels={"paired-swe.eval": run_id},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    args = parser.parse_args()
    os.environ.update(HF_DATASETS_OFFLINE="1", HF_HUB_OFFLINE="1")
    from swebench.harness import run_evaluation as official
    official.create_container = cached_container
    official.main(dataset_name=args.dataset, split="test", instance_ids=args.ids,
                  predictions_path=args.predictions, max_workers=2, open_file_limit=4096,
                  run_id=args.run_id, timeout=1800, rewrite_reports=False, modal=False, report_dir="reports")


if __name__ == "__main__":
    main()
