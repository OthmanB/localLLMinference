#!/usr/bin/env python3
"""Write periodic llama.cpp, GPU, and host-memory telemetry as JSONL."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone


GPU_FIELDS = (
    "index",
    "name",
    "power.limit",
    "power.draw",
    "fan.speed",
    "temperature.gpu",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "clocks.sm",
)


def numeric(value: str) -> int | float | None:
    value = value.strip()
    if value in {"", "N/A", "Not Supported"}:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def gpu_snapshot(gpu: int) -> dict:
    query = ",".join(GPU_FIELDS)
    try:
        output = subprocess.check_output(
            [
                "/usr/bin/nvidia-smi",
                "-i",
                str(gpu),
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        row = next(csv.reader(io.StringIO(output)))
        if len(row) != len(GPU_FIELDS):
            raise RuntimeError(f"unexpected nvidia-smi field count: {len(row)}")
        return {
            field.replace(".", "_"): (row[index].strip() if field == "name" else numeric(row[index]))
            for index, field in enumerate(GPU_FIELDS)
        }
    except (OSError, RuntimeError, StopIteration, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def meminfo() -> dict[str, object]:
    values: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                key, separator, value = line.partition(":")
                if separator:
                    fields = value.split()
                    if fields and fields[0].isdigit():
                        values[key] = int(fields[0])
    except OSError as error:
        return {"error": str(error)}
    return {
        key: values[key]
        for key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")
        if key in values
    }


def health_snapshot(url: str) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return {"status": response.status}
    except urllib.error.HTTPError as error:
        return {"status": error.code}
    except (OSError, urllib.error.URLError) as error:
        return {"status": None, "error": str(error)}


def snapshot(url: str, gpu: int) -> dict:
    load = os.getloadavg()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "health": health_snapshot(url),
        "gpu": gpu_snapshot(gpu),
        "host": {"load1": load[0], "load5": load[1], "load15": load[2], "meminfo_kib": meminfo()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080/health")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--interval", type=float, default=15.0)
    args = parser.parse_args()
    if args.gpu < 0 or args.interval <= 0:
        parser.error("--gpu must be non-negative and --interval must be positive")

    next_sample = time.monotonic()
    while True:
        print(json.dumps(snapshot(args.url, args.gpu), separators=(",", ":")), flush=True)
        next_sample += args.interval
        time.sleep(max(0.0, next_sample - time.monotonic()))


if __name__ == "__main__":
    raise SystemExit(main())
