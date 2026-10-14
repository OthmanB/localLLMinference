import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


path = Path(__file__).with_name("telemetry.csv")
rows = defaultdict(list)
with path.open(newline="") as handle:
    for row in csv.DictReader(handle):
        row["time"] = datetime.strptime(row["timestamp"].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
        for field in ("memory_used_mib", "power_w", "temperature_c", "clocks_sm_mhz", "utilization_pct"):
            row[field] = float(row[field].strip())
        rows[row["index"].strip()].append(row)

summary = {}
for index, samples in rows.items():
    samples.sort(key=lambda row: row["time"])
    energy = 0.0
    for previous, current in zip(samples, samples[1:]):
        energy += (previous["power_w"] + current["power_w"]) * 0.5 * (current["time"] - previous["time"])
    summary[index] = {
        "uuid": samples[0]["uuid"].strip(),
        "samples": len(samples),
        "duration_seconds": samples[-1]["time"] - samples[0]["time"],
        "memory_used_mib_max": max(row["memory_used_mib"] for row in samples),
        "power_w_max": max(row["power_w"] for row in samples),
        "power_w_mean": sum(row["power_w"] for row in samples) / len(samples),
        "temperature_c_max": max(row["temperature_c"] for row in samples),
        "clocks_sm_mhz_max": max(row["clocks_sm_mhz"] for row in samples),
        "utilization_pct_max": max(row["utilization_pct"] for row in samples),
        "sampled_energy_joules": energy,
    }
(path.with_name("telemetry-summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
