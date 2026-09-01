#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT=${SWEBENCH_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
readonly PYTHON=${SWEBENCH_PYTHON:-python3}
readonly MANIFEST=${ROOT}/swebench/manifest-q4-q5-40.json
readonly PREDICTIONS=${ROOT}/swebench/gold/verified40.jsonl
readonly EVAL_DATASET=${ROOT}/swebench/gold/verified40-evaluator.json
readonly RUN_ID=gold-verified40-89b534d4
readonly REPORT_DIR=${ROOT}/swebench/gold/${RUN_ID}
readonly MARKER=${ROOT}/swebench/gold/verified40.validation.json

if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
    printf '%s\n' "Docker must be installed and running before gold validation." >&2
    exit 1
fi

"${PYTHON}" "${ROOT}/tools/prepare_swebench_gold.py" \
    --manifest "${MANIFEST}" \
    --output "${PREDICTIONS}" \
    --dataset-output "${EVAL_DATASET}"

mkdir -p "${REPORT_DIR}"
"${PYTHON}" -m swebench.harness.run_evaluation \
    --dataset_name "${EVAL_DATASET}" \
    --split test \
    --predictions_path "${PREDICTIONS}" \
    --max_workers 2 \
    --timeout 1800 \
    --run_id "${RUN_ID}" \
    --report_dir "${REPORT_DIR}"

"${PYTHON}" - "${MANIFEST}" "${REPORT_DIR}" "${MARKER}" "${EVAL_DATASET}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path, report_dir, marker_path, eval_dataset = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
reports = sorted(report_dir.glob("*.json"))
if len(reports) != 1:
    raise SystemExit(f"expected one gold report, found {len(reports)}")
report = json.loads(reports[0].read_text(encoding="utf-8"))
required = {
    "total_instances": 40,
    "submitted_instances": 40,
    "completed_instances": 40,
    "resolved_instances": 40,
    "unresolved_instances": 0,
    "infra_failure_instances": 0,
    "error_instances": 0,
    "empty_patch_instances": 0,
}
for key, expected in required.items():
    if report.get(key) != expected:
        raise SystemExit(f"gold validation failed: {key}={report.get(key)!r}, expected {expected}")
marker = {
    "manifest_sha256": manifest["manifest_sha256"],
    "run_id": "gold-verified40-89b534d4",
    "report": str(reports[0]),
    "evaluator_dataset": str(eval_dataset),
    "validated_tasks": 40,
}
marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
print(f"GOLD VALIDATION OK: {reports[0]}")
PY
