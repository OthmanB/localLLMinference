#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT=${SWEBENCH_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
readonly PYTHON=${SWEBENCH_PYTHON:-python3}
readonly CONTROLLER=${ROOT}/tools/swebench_controller.py
readonly RESOURCES=${ROOT}/tools/swebench_resources.py
readonly MANIFEST=${SWEBENCH_MANIFEST:-${ROOT}/swebench/manifest-q4-q5-40.json}
readonly RUN_DIR=${SWEBENCH_RUN_DIR:-${ROOT}/swebench/runs/qwen38-q4-q5-verified40-r2-stock-89b534d4}
readonly LLAMA_SERVER_BIN=${LLAMA_SERVER_BIN:-llama-server}
readonly Q4_MODEL=${SWEBENCH_Q4_MODEL:-${ROOT}/models/qwen3.8-27b-q4-k-m-gguf/Qwen3.8-27B-UD-Q4_K_M.gguf}
readonly Q5_MODEL=${SWEBENCH_Q5_MODEL:-${ROOT}/models/qwen3.8-27b-q5-k-m-gguf/Qwen3.8-27B-UD-Q5_K_M.gguf}

usage() {
    printf '%s\n' \
        "Usage: $0 {init|start|resume|status|stop}" \
        "  init    Create the durable SQLite run directory" \
        "  start   Start Docker-gated 128k servers and the paired controller" \
        "  resume  Reuse healthy servers or start them, then resume the ledger" \
        "  status  Show durable task counts and heartbeats" \
        "  stop    Stop the controller and the temporary 128k model servers"
}

require_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        printf '%s\n' "Docker is required. Install it before starting the experiment." >&2
        exit 1
    fi
    if ! docker info >/dev/null 2>&1; then
        printf '%s\n' "Docker is installed but its daemon is unavailable." >&2
        exit 1
    fi
}

stop_conflicting_services() {
    systemctl --user stop \
        llama-qwen3.8-q4-corrected32.service \
        llama-qwen3.8-q5-gpu1-32.service \
        llama-qwen3.8-gpukv64.service \
        swebench-q4-128k.service \
        swebench-q5-128k.service >/dev/null 2>&1 || true
    ft daemon stop --url http://127.0.0.1:1900 >/dev/null 2>&1 || true
}

start_server_units() {
    stop_conflicting_services
    systemd-run --user --unit=swebench-q4-128k.service \
        --property=Restart=on-failure --property=RestartSec=5 --collect \
        env CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID GGML_CUDA_ENABLE_UNIFIED_MEMORY=0 \
        "${LLAMA_SERVER_BIN}" \
        --model "${Q4_MODEL}" \
        --alias qwen3.8-27b-q4-gpukv128 --host 127.0.0.1 --port 8080 \
        --ctx-size 131072 --parallel 1 --n-gpu-layers 99 --kv-offload \
        --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on \
        --threads 32 --threads-batch 32 --batch-size 2048 --ubatch-size 512 \
        --fit off --metrics --perf
    systemd-run --user --unit=swebench-q5-128k.service \
        --property=Restart=on-failure --property=RestartSec=5 --collect \
        env CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID GGML_CUDA_ENABLE_UNIFIED_MEMORY=0 \
        "${LLAMA_SERVER_BIN}" \
        --model "${Q5_MODEL}" \
        --alias qwen3.8-27b-q5-gpukv128 --host 127.0.0.1 --port 8081 \
        --ctx-size 131072 --parallel 1 --n-gpu-layers 99 --kv-offload \
        --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on \
        --threads 32 --threads-batch 32 --batch-size 2048 --ubatch-size 512 \
        --fit off --metrics --perf
}

wait_for_servers() {
    for _ in $(seq 1 180); do
        if curl --fail --silent http://127.0.0.1:8080/health >/dev/null 2>&1 \
            && curl --fail --silent http://127.0.0.1:8081/health >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    printf '%s\n' "Timed out waiting for the two 128k llama.cpp servers." >&2
    exit 1
}

servers_ready() {
    "${PYTHON}" -c 'import json,sys,urllib.request
checks=(("http://127.0.0.1:8080/v1/models", "qwen3.8-27b-q4-gpukv128"), ("http://127.0.0.1:8081/v1/models", "qwen3.8-27b-q5-gpukv128"))
try:
    for url,expected in checks:
        with urllib.request.urlopen(url, timeout=10) as response:
            ids={item.get("id") for item in json.load(response).get("data", [])}
        if expected not in ids:
            raise SystemExit(1)
except Exception:
    raise SystemExit(1)
raise SystemExit(0)'
}

start_controller() {
    mkdir -p "${RUN_DIR}"
    readonly resources_log=${RUN_DIR}/resources-monitor.log
    readonly resources_pid=${RUN_DIR}/resources.pid
    readonly log_path=${RUN_DIR}/controller.log
    readonly pid_path=${RUN_DIR}/controller.pid
    if [[ ! -s ${resources_pid} ]] || ! kill -0 "$(<"${resources_pid}")" 2>/dev/null; then
        setsid nohup "${PYTHON}" "${RESOURCES}" --run-dir "${RUN_DIR}" \
            >>"${resources_log}" 2>&1 < /dev/null &
        printf '%s\n' "$!" > "${resources_pid}"
    fi
    if [[ -s ${pid_path} ]] && kill -0 "$(<"${pid_path}")" 2>/dev/null; then
        printf '%s\n' "controller already running: $(<"${pid_path}")"
        return 0
    fi
    setsid nohup "${PYTHON}" "${CONTROLLER}" run-both --run-dir "${RUN_DIR}" \
        >>"${log_path}" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "${pid_path}"
    printf '%s\n' "controller started: $(<"${pid_path}")"
}

init_run() {
    local init_args=("${PYTHON}" "${CONTROLLER}" init --manifest "${MANIFEST}" --run-dir "${RUN_DIR}")
    if [[ -n ${SWEBENCH_TASK_IDS:-} ]]; then
        local task_id
        local -a task_ids
        read -r -a task_ids <<< "${SWEBENCH_TASK_IDS}"
        for task_id in "${task_ids[@]}"; do
            init_args+=(--task-id "${task_id}")
        done
    fi
    "${init_args[@]}"
}

case "${1:-}" in
    init)
        init_run
        ;;
    start)
        require_docker
        init_run 2>/dev/null || true
        "${PYTHON}" "${CONTROLLER}" preflight --run-dir "${RUN_DIR}" --skip-endpoints
        start_server_units
        wait_for_servers
        "${PYTHON}" "${CONTROLLER}" preflight --run-dir "${RUN_DIR}"
        start_controller
        ;;
    resume)
        require_docker
        "${PYTHON}" "${CONTROLLER}" preflight --run-dir "${RUN_DIR}" --skip-endpoints
        if ! servers_ready; then
            start_server_units
            wait_for_servers
        fi
        "${PYTHON}" "${CONTROLLER}" preflight --run-dir "${RUN_DIR}"
        start_controller
        ;;
    status)
        "${PYTHON}" "${CONTROLLER}" status --run-dir "${RUN_DIR}"
        ;;
    stop)
        if [[ -s ${RUN_DIR}/controller.pid ]]; then
            kill "$(<"${RUN_DIR}/controller.pid")" 2>/dev/null || true
            rm -f "${RUN_DIR}/controller.pid"
        fi
        if [[ -s ${RUN_DIR}/resources.pid ]]; then
            kill "$(<"${RUN_DIR}/resources.pid")" 2>/dev/null || true
            rm -f "${RUN_DIR}/resources.pid"
        fi
        systemctl --user stop swebench-q4-128k.service swebench-q5-128k.service >/dev/null 2>&1 || true
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
