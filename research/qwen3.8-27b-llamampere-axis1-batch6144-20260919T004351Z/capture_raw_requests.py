#!/usr/bin/env python3
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUEST_KEYS = (
    "n_predict",
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "seed",
    "ignore_eos",
    "stream",
    "cache_prompt",
)
UNITS = [
    "Record class code: def normalize_value(value: int, offset: int) -> int: return value + offset.\n",
    "Record class neutral: The stable observation is blue cedar seven, with no external action.\n",
    "Record class code: if total < 0: total = 0; return min(total, 1000003).\n",
    "Record class neutral: Preserve ordering, preserve exact integers, and do not invent records.\n",
    "Record class code: result = {\"name\": \"sample\", \"enabled\": True, \"count\": 3}.\n",
    "Record class neutral: This sentence is a fixed synthetic datum for long-context measurement.\n",
    "Record class code: for item in values: checksum = (checksum * 31 + item) % 1000000007.\n",
    "Record class neutral: The expected final response is short and must not quote the records.\n",
]
CYCLE = "".join(UNITS)


def token_count(path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        values[key] = int(value)
    return values


for count_path in sorted(ROOT.glob("depth*-token-count.txt")):
    label = count_path.name.removesuffix("-token-count.txt")
    counts = token_count(count_path)
    settings_path = ROOT / (label + "-request-settings.json")
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    depth = settings["target_token_depth"]
    prefix = (
        "Synthetic Phase 1 coding and neutral prefill workload. "
        "Read every record in order and retain the final instruction. "
        "The records are deliberately repetitive but include code-shaped and neutral text. "
        "Depth label: %s.\n\n" % label
    )
    suffix = (
        "\nFinal instruction: after reading the complete context, output exactly one line "
        "with the label %s and the integer checksum 3817. Do not repeat the records.\n" % label
    )
    prompt = prefix + CYCLE * counts["cycles"] + suffix
    prompt_path = ROOT / (label + "-prompt.txt")
    prompt_path.write_text(prompt, encoding="utf-8")
    payload = {key: settings[key] for key in REQUEST_KEYS}
    payload["prompt"] = prompt
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    for phase in ("cold", "warm"):
        (ROOT / (label + "-" + phase + "-request.json")).write_bytes(raw)
    settings["prompt_file"] = str(prompt_path)
    settings_path.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
