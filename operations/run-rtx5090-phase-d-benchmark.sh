#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT=/home/michel/LLMs-tests/localLLMinference
readonly BENCH_ROOT=/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs
readonly SERVICE=llama-qwen3.8-q4-native.service
readonly MONITOR=ai-qwen3.8-q4-native-monitor.service
readonly ALIAS=qwen3.8-27b-q4-gpukv-native
readonly RUN_PREFIX=${AI_SERVER_PHASE_D_RUN_PREFIX:-phase-d}
readonly RUN_ID="${RUN_PREFIX}-${1:?engine is required}-$(date -u +%Y-%m-%dT%H%M%SZ)"
readonly OUTPUT_DIR="${BENCH_ROOT}/${RUN_ID}"
readonly BENCH_USER=michel
readonly MODEL=${AI_SERVER_PHASE_D_MODEL:-/home/michel/LLMs-tests/rtx5090-qwen38-bench/models/Qwen3.8-27B-NVFP4}
readonly GGUF_MODEL=/home/michel/LLMs-tests/rtx5090-qwen38-bench/models/Qwen3.8-27B-UD-Q4_K_M.gguf
readonly LLAMA=/home/michel/LLMs-tests/llama.cpp/build/bin/llama-server
readonly VLLM=${AI_SERVER_PHASE_D_VLLM:-/home/michel/LLMs-tests/.phase2-vllm-venv/bin/vllm}
readonly SGLANG=/home/michel/LLMs-tests/.phase2-sglang-venv/bin/sglang
readonly PYTHON=/home/michel/LLMs-tests/.phase2-vllm-venv/bin/python
readonly ALLOWED_GPU0_PID=${AI_SERVER_PHASE_D_ALLOWED_GPU0_PID:?set the approved unrelated GPU 0 compute PID}

if [[ ${EUID} -ne 0 ]]; then
    printf '%s\n' "Run with sudo: AI_SERVER_PHASE_D_ALLOWED_GPU0_PID=<pid> sudo -E ${0} <llama-replicas|vllm|sglang>" >&2
    exit 1
fi

case "$1" in
    llama-replicas|vllm|sglang) ;;
    *) printf '%s\n' "engine must be llama-replicas, vllm, or sglang" >&2; exit 1 ;;
esac

if ! [[ ${ALLOWED_GPU0_PID} =~ ^[1-9][0-9]*$ ]] || ! kill -0 "${ALLOWED_GPU0_PID}" 2>/dev/null; then
    printf '%s\n' "approved GPU 0 PID is not a live process: ${ALLOWED_GPU0_PID}" >&2
    exit 1
fi
if [[ ! -d ${MODEL} || ! -f ${MODEL}/config.json ]]; then
    printf '%s\n' "ModelOpt model directory is not usable: ${MODEL}" >&2
    exit 1
fi

install -d -m 0750 -o "${BENCH_USER}" -g "${BENCH_USER}" "${BENCH_ROOT}"
install -d -m 0750 -o "${BENCH_USER}" -g "${BENCH_USER}" "${OUTPUT_DIR}"
exec > >(tee -a "${OUTPUT_DIR}/maintenance-window.log") 2>&1

was_active=0
if systemctl is-active --quiet "${SERVICE}"; then
    was_active=1
fi
monitor_was_active=0
if systemctl is-active --quiet "${MONITOR}"; then
    monitor_was_active=1
fi

candidate_pid=
stop_candidate() {
    if [[ -z ${candidate_pid} ]] || ! kill -0 "${candidate_pid}" 2>/dev/null; then
        return
    fi
    printf '%s\n' "Stopping interrupted candidate process group ${candidate_pid}."
    kill -TERM -- "-${candidate_pid}" || true
    for _ in $(seq 1 30); do
        if ! kill -0 "${candidate_pid}" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    if kill -0 "${candidate_pid}" 2>/dev/null; then
        printf '%s\n' "Force-stopping candidate process group ${candidate_pid}." >&2
        kill -KILL -- "-${candidate_pid}" || true
    fi
    wait "${candidate_pid}" 2>/dev/null || true
    candidate_pid=
}

restore_service() {
    local status=$?
    trap - EXIT INT TERM
    stop_candidate
    if [[ ${was_active} -eq 1 ]]; then
        printf '%s\n' "Restoring ${SERVICE}."
        systemctl reset-failed "${SERVICE}" || true
        systemctl start "${SERVICE}" || true
    else
        systemctl stop "${SERVICE}" || true
    fi
    if [[ ${monitor_was_active} -eq 1 ]]; then
        systemctl start "${MONITOR}" || true
    else
        systemctl stop "${MONITOR}" || true
    fi
    chown -R "${BENCH_USER}:${BENCH_USER}" "${OUTPUT_DIR}"
    exit "${status}"
}
trap restore_service EXIT INT TERM

printf '%s\n' "Phase D maintenance window begins: $(date --iso-8601=seconds)"
printf '%s\n' "Recording reference unit, topology, and pre-stop GPU ownership."
systemctl cat "${SERVICE}" > "${OUTPUT_DIR}/reference-unit.txt"
systemctl show "${SERVICE}" --property=Environment --property=ExecStart --property=ActiveState > "${OUTPUT_DIR}/reference-unit-state.txt"
nvidia-smi topo -m > "${OUTPUT_DIR}/gpu-topology.txt"
nvidia-smi topo -p2p r > "${OUTPUT_DIR}/gpu-peer-read.txt"
nvidia-smi topo -p2p w > "${OUTPUT_DIR}/gpu-peer-write.txt"
nvidia-smi > "${OUTPUT_DIR}/gpu-before.txt"

if [[ ${was_active} -eq 1 ]]; then
    printf '%s\n' "Verifying rollback before benchmark by stopping and restarting ${SERVICE}."
    systemctl stop "${SERVICE}"
    systemctl is-active --quiet "${SERVICE}" && { printf '%s\n' "service did not stop" >&2; exit 1; }
    # Deliberate stop/start probes must not consume the production unit's burst budget.
    systemctl reset-failed "${SERVICE}"
    systemctl start "${SERVICE}"
    for _ in $(seq 1 900); do
        models=$(curl --silent --show-error --fail --max-time 5 http://127.0.0.1:8080/v1/models || true)
        if [[ ${models} == *"${ALIAS}"* ]]; then
            printf '%s\n' "Rollback verification passed."
            break
        fi
        sleep 1
    done
    [[ ${models:-} == *"${ALIAS}"* ]] || { printf '%s\n' "rollback readiness failed" >&2; exit 1; }
    printf '%s\n' "Stopping ${SERVICE} for temporary Phase D benchmark ownership."
    systemctl stop "${SERVICE}"
fi

systemctl is-active --quiet "${SERVICE}" && { printf '%s\n' "reference service remains active" >&2; exit 1; }
nvidia-smi > "${OUTPUT_DIR}/gpu-after-reference-stop.txt"

setsid runuser -u "${BENCH_USER}" -- env PYTHONUNBUFFERED=1 \
    "${PYTHON}" "${ROOT}/tools/rtx5090_phase_d.py" \
    --engine "$1" \
    --output-dir "${OUTPUT_DIR}/candidate" \
    --model "${MODEL}" \
    --gguf-model "${GGUF_MODEL}" \
    --llama "${LLAMA}" \
    --vllm "${VLLM}" \
    --sglang "${SGLANG}" \
    --allow-preexisting-compute-pid "${ALLOWED_GPU0_PID}" \
    "${@:2}" &
candidate_pid=$!
wait "${candidate_pid}"
candidate_pid=

printf '%s\n' "Phase D candidate run complete: ${OUTPUT_DIR}/candidate/results.json"
