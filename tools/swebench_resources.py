#!/usr/bin/env python3
"""Persist host and GPU resource samples for a headless SWE-bench run."""

import argparse
import csv
import os
import signal
import subprocess
import time
from pathlib import Path

import psutil


STOP = False


def stop(_signum, _frame):
    global STOP
    STOP = True


def gpu_rows() -> list[list[str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,power.limit,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return [next(csv.reader([line], skipinitialspace=True)) for line in result.stdout.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    path = args.run_dir / "resources.csv"
    new_file = not path.exists() or path.stat().st_size == 0
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    with path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        if new_file:
            writer.writerow(
                [
                    "timestamp",
                    "gpu_index",
                    "power_limit_w",
                    "memory_used_mib",
                    "memory_total_mib",
                    "power_draw_w",
                    "temperature_c",
                    "host_memory_used_bytes",
                    "host_memory_available_bytes",
                    "swap_used_bytes",
                ]
            )
        while not STOP:
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            try:
                rows = gpu_rows()
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                rows = []
            if not rows:
                writer.writerow([timestamp, "", "", "", "", "", "", memory.used, memory.available, swap.used])
            else:
                for row in rows:
                    writer.writerow([timestamp, *row, memory.used, memory.available, swap.used])
            output.flush()
            os.fsync(output.fileno())
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
