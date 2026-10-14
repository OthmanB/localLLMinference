#!/usr/bin/env python3
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path


ROOT = Path(sys.argv[1])


def number(value):
    value = value.strip().replace("%", "")
    if value in {"N/A", "Not Supported", "[Not Supported]", ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def gpu_rows(path):
    rows = []
    marker = None
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("sample_epoch="):
            marker = number(line.split("=", 1)[1])
            continue
        if not line or line.startswith("bash:"):
            continue
        fields = next(csv.reader([line], skipinitialspace=True), [])
        if len(fields) < 11 or not fields[1].strip().isdigit():
            continue
        rows.append(
            {
                "sample_epoch": marker,
                "gpu": int(fields[1].strip()),
                "timestamp": fields[0].strip(),
                "uuid": fields[3].strip(),
                "memory_used_mib": number(fields[5]),
                "utilization_gpu_pct": number(fields[6]),
                "power_w": number(fields[7]),
                "temperature_c": number(fields[8]),
                "clock_sm_mhz": number(fields[9]),
                "clock_mem_mhz": number(fields[10]),
            }
        )
    return rows


def host_rows(path):
    rows = []
    marker = None
    lines = path.read_text(errors="replace").splitlines() if path.exists() else []
    for index, line in enumerate(lines):
        if line.startswith("sample_epoch="):
            marker = number(line.split("=", 1)[1])
            continue
        fields = line.split()
        if marker is not None and fields and fields[0].isdigit() and len(fields) >= 8:
            rows.append(
                {
                    "sample_epoch": marker,
                    "pid": int(fields[0]),
                    "cpu_pct": number(fields[4]),
                    "rss_kib": number(fields[6]),
                    "vsz_kib": number(fields[7]),
                }
            )
        if line.startswith("Mem:"):
            fields = line.split()
            if len(fields) >= 7:
                rows.append(
                    {
                        "sample_epoch": marker,
                        "host_memory_total_bytes": number(fields[1]),
                        "host_memory_used_bytes": number(fields[2]),
                        "host_memory_free_bytes": number(fields[3]),
                        "host_memory_available_bytes": number(fields[6]),
                    }
                )
    return rows


def values(rows, key):
    return [row[key] for row in rows if row.get(key) is not None]


def stats(rows, key):
    data = values(rows, key)
    if not data:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(data),
        "min": min(data),
        "mean": statistics.fmean(data),
        "max": max(data),
    }


def energy(rows, start, end, power_key="power_w"):
    points = sorted(
        (row["sample_epoch"], row[power_key])
        for row in rows
        if row.get("sample_epoch") is not None
        and start <= row["sample_epoch"] <= end
        and row.get(power_key) is not None
    )
    if not points:
        return {"sample_points": 0, "watt_hours": None, "method": "no samples"}
    if len(points) == 1:
        return {
            "sample_points": 1,
            "watt_hours": points[0][1] * (end - start) / 3600.0,
            "method": "single sample times wall interval",
        }
    total_ws = 0.0
    for (left_t, left_p), (right_t, right_p) in zip(points, points[1:]):
        total_ws += (right_t - left_t) * (left_p + right_p) / 2.0
    return {
        "sample_points": len(points),
        "watt_hours": total_ws / 3600.0,
        "method": "trapezoid between in-window samples",
    }


def interval_rows(rows, start, end):
    return [row for row in rows if row.get("sample_epoch") is not None and start <= row["sample_epoch"] <= end]


events = {}
event_path = ROOT / "events.tsv"
for line in event_path.read_text().splitlines():
    fields = line.split("\t")
    if len(fields) == 2:
        events[fields[0]] = float(fields[1])

gpu_samples = gpu_rows(ROOT / "gpu-samples.csv")
host_samples = host_rows(ROOT / "host-samples.log")
before = gpu_rows(ROOT / "gpu-before.csv")
after = gpu_rows(ROOT / "gpu-after-stop.csv")

run_rows = []
for summary_path in sorted(ROOT.glob("depth*-summary.json")):
    if "-cold-" in summary_path.name or "-warm-" in summary_path.name:
        continue
    summary = json.loads(summary_path.read_text())
    for request in summary["requests"]:
        label = request["label"]
        phase = request["phase"]
        start = events.get("%s_%s_start" % (label, phase))
        end = events.get("%s_%s_end" % (label, phase))
        if start is None or end is None:
            continue
        request_gpu = {gpu: interval_rows([row for row in gpu_samples if row["gpu"] == gpu], start, end) for gpu in (0, 1, 2)}
        request_host = interval_rows(host_samples, start, end)
        target_gpu = request_gpu[1]
        idle_power = statistics.fmean(values([row for row in before if row["gpu"] == 1], "power_w")) if [row for row in before if row["gpu"] == 1] else None
        target_energy = energy(target_gpu, start, end)
        if idle_power is not None:
            incremental_rows = [dict(row, power_w=max(0.0, row["power_w"] - idle_power)) for row in target_gpu]
            incremental_energy = energy(incremental_rows, start, end)
        else:
            incremental_energy = {"sample_points": 0, "watt_hours": None, "method": "no idle baseline"}
        timings = request.get("timings", {})
        row = {
            "label": label,
            "phase": phase,
            "target_token_depth": request["target_token_depth"],
            "actual_tokenize_count": request["actual_tokenize_count"],
            "http_status": request["status"],
            "wall_seconds": request["wall_seconds"],
            "ttft_estimate_ms": (timings.get("prompt_ms", 0) + timings.get("predicted_per_token_ms", 0)),
            "prompt_n": timings.get("prompt_n"),
            "prompt_ms": timings.get("prompt_ms"),
            "prefill_tokens_per_second": timings.get("prompt_per_second"),
            "decode_n": timings.get("predicted_n"),
            "decode_ms": timings.get("predicted_ms"),
            "decode_tokens_per_second": timings.get("predicted_per_second"),
            "cache_n": timings.get("cache_n"),
            "tokens_cached": request.get("tokens_cached"),
            "draft_n": timings.get("draft_n"),
            "draft_n_accepted": timings.get("draft_n_accepted"),
            "draft_acceptance": (timings.get("draft_n_accepted", 0) / timings["draft_n"] if timings.get("draft_n") else None),
            "truncated": request.get("truncated"),
            "stop_type": request.get("stop_type"),
            "gpu1_vram_mib": stats(target_gpu, "memory_used_mib"),
            "gpu1_utilization_pct": stats(target_gpu, "utilization_gpu_pct"),
            "gpu1_power_w": stats(target_gpu, "power_w"),
            "gpu1_temperature_c": stats(target_gpu, "temperature_c"),
            "gpu1_clock_sm_mhz": stats(target_gpu, "clock_sm_mhz"),
            "gpu1_clock_mem_mhz": stats(target_gpu, "clock_mem_mhz"),
            "gpu1_energy_wh": target_energy,
            "gpu1_incremental_energy_wh": incremental_energy,
            "gpu0_max_memory_mib": stats(request_gpu[0], "memory_used_mib")["max"],
            "gpu0_max_utilization_pct": stats(request_gpu[0], "utilization_gpu_pct")["max"],
            "gpu2_max_memory_mib": stats(request_gpu[2], "memory_used_mib")["max"],
            "gpu2_max_utilization_pct": stats(request_gpu[2], "utilization_gpu_pct")["max"],
            "host_process_cpu_pct": stats([row for row in request_host if "cpu_pct" in row], "cpu_pct"),
            "host_process_rss_kib": stats([row for row in request_host if "rss_kib" in row], "rss_kib"),
            "host_memory_used_bytes": stats([row for row in request_host if "host_memory_used_bytes" in row], "host_memory_used_bytes"),
            "host_memory_available_bytes": stats([row for row in request_host if "host_memory_available_bytes" in row], "host_memory_available_bytes"),
            "gpu_sample_points": {str(gpu): len(request_gpu[gpu]) for gpu in (0, 1, 2)},
        }
        run_rows.append(row)

proof = {
    "before": before,
    "after_stop": after,
    "during_request_maxima": {
        str(gpu): {
            "max_memory_mib": stats([row for row in gpu_samples if row["gpu"] == gpu], "memory_used_mib")["max"],
            "max_utilization_pct": stats([row for row in gpu_samples if row["gpu"] == gpu], "utilization_gpu_pct")["max"],
        }
        for gpu in (0, 1, 2)
    },
}

output = {
    "events": events,
    "requests": run_rows,
    "gpu_isolation": proof,
    "notes": [
        "TTFT estimate is prompt_ms plus one average decode-token interval; the non-streaming benchmark requests do not expose first-byte timing.",
        "Energy is sampled nvidia-smi power integrated over request intervals; incremental energy subtracts GPU1 idle baseline power.",
        "GPU rows are physical indices from nvidia-smi with CUDA_VISIBLE_DEVICES unset for the sampler.",
    ],
}
(ROOT / "measurements.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
(ROOT / "gpu-isolation-proof.json").write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")

csv_fields = [
    "label", "phase", "target_token_depth", "actual_tokenize_count", "http_status", "wall_seconds", "ttft_estimate_ms",
    "prompt_n", "prompt_ms", "prefill_tokens_per_second", "decode_n", "decode_ms", "decode_tokens_per_second", "cache_n",
    "tokens_cached", "draft_n", "draft_n_accepted", "draft_acceptance", "truncated", "stop_type", "gpu1_vram_mib",
    "gpu1_utilization_pct", "gpu1_power_w", "gpu1_temperature_c", "gpu1_clock_sm_mhz", "gpu1_clock_mem_mhz",
    "gpu1_energy_wh", "gpu1_incremental_energy_wh", "gpu0_max_memory_mib", "gpu0_max_utilization_pct", "gpu2_max_memory_mib",
    "gpu2_max_utilization_pct", "host_process_cpu_pct", "host_process_rss_kib", "host_memory_used_bytes",
    "host_memory_available_bytes", "gpu_sample_points",
]
with (ROOT / "measurements.csv").open("w", newline="") as output_csv:
    writer = csv.DictWriter(output_csv, fieldnames=csv_fields)
    writer.writeheader()
    for row in run_rows:
        writer.writerow(row)

print(json.dumps(output, indent=2, sort_keys=True))
