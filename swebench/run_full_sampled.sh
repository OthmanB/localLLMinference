#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT=${SWEBENCH_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
export SWEBENCH_RUN_DIR=${ROOT}/swebench/runs/qwen38-q4-q5-verified40-r2-sampled-89b534d4
export SWEBENCH_MANIFEST=${ROOT}/swebench/manifest-q4-q5-40.json
export SWEBENCH_RUNTIME_CONFIG=${ROOT}/swebench/configs/runtime-sampled-nonthinking.yaml

exec "${ROOT}/swebench/run_headless.sh" "$@"
