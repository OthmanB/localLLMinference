#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT=/home/michel/LLMs-tests/localLLMinference
readonly RUN_ROOT=/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs
readonly RUN_ID="phase-e-kv-fp8-calibration-$(date -u +%Y-%m-%dT%H%M%SZ)"
readonly OUTPUT_DIR="${RUN_ROOT}/${RUN_ID}"
readonly SERVICE=llama-qwen3.8-q4-native.service
readonly MONITOR=ai-qwen3.8-q4-native-monitor.service
readonly ALIAS=qwen3.8-27b-q4-gpukv-native
readonly BENCH_USER=michel
readonly SOURCE=/home/michel/LLMs-tests/rtx5090-qwen38-bench/models/Qwen3.8-27B-BF16-e13a4f0e
readonly MODELOPT_ROOT=/home/michel/LLMs-tests/TensorRT-Model-Optimizer-87c9f8cf
readonly PYTHON=/home/michel/LLMs-tests/.phasee-modelopt-venv/bin/python
readonly RECIPE="${ROOT}/operations/hardware/rtx5090/modelopt/qwen3.8-27b-w4a16-nvfp4-fp8-attn-kv-fp8-calibrated.yaml"
readonly SOURCE_CHECKSUMS="${ROOT}/operations/hardware/rtx5090/modelopt/qwen3.8-27b-bf16-e13a4f0e.sha256"
readonly THERMAL_STOP_C=85
readonly ALLOWED_GPU0_PID=${AI_SERVER_PHASE_E_ALLOWED_GPU0_PID:?set the approved unrelated GPU 0 compute PID}

if [[ ${EUID} -ne 0 ]]; then
    printf '%s\n' "Run with sudo: AI_SERVER_PHASE_E_CONFIRM=calibrate AI_SERVER_PHASE_E_ALLOWED_GPU0_PID=<pid> sudo -E ${0}" >&2
    exit 1
fi
if [[ ${AI_SERVER_PHASE_E_CONFIRM:-} != calibrate ]]; then
    printf '%s\n' "set AI_SERVER_PHASE_E_CONFIRM=calibrate to run the two-GPU calibration window" >&2
    exit 1
fi
if ! [[ ${ALLOWED_GPU0_PID} =~ ^[1-9][0-9]*$ ]] || ! kill -0 "${ALLOWED_GPU0_PID}" 2>/dev/null; then
    printf '%s\n' "approved GPU 0 PID is not a live process: ${ALLOWED_GPU0_PID}" >&2
    exit 1
fi
for path in "${SOURCE}" "${MODELOPT_ROOT}" "${RECIPE}" "${SOURCE_CHECKSUMS}" "${PYTHON}"; do
    [[ -e ${path} ]] || { printf 'missing Phase E artifact: %s\n' "${path}" >&2; exit 1; }
done

install -d -m 0750 -o "${BENCH_USER}" -g "${BENCH_USER}" "${RUN_ROOT}"
install -d -m 0750 -o "${BENCH_USER}" -g "${BENCH_USER}" "${OUTPUT_DIR}"
exec > >(tee -a "${OUTPUT_DIR}/maintenance-window.log") 2>&1

was_active=0
systemctl is-active --quiet "${SERVICE}" && was_active=1
monitor_was_active=0
systemctl is-active --quiet "${MONITOR}" && monitor_was_active=1
calibration_pid=0
thermal_guard_pid=0

restore_service() {
    local status=$?
    trap - EXIT INT TERM
    if [[ ${thermal_guard_pid} -gt 0 ]]; then
        kill "${thermal_guard_pid}" 2>/dev/null || true
        wait "${thermal_guard_pid}" 2>/dev/null || true
    fi
    if [[ ${calibration_pid} -gt 0 ]] && kill -0 "${calibration_pid}" 2>/dev/null; then
        kill -- "-${calibration_pid}" 2>/dev/null || true
    fi
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

thermal_guard() {
    while kill -0 "${calibration_pid}" 2>/dev/null; do
        nvidia-smi --query-gpu=index,temperature.gpu,power.draw,memory.used --format=csv,noheader >> "${OUTPUT_DIR}/gpu-telemetry.csv" || true
        while IFS= read -r temperature; do
            if [[ ${temperature} =~ ^[0-9]+$ ]] && (( 10#${temperature} >= THERMAL_STOP_C )); then
                printf 'Thermal stop: GPU reached %s C (limit %s C).\n' "${temperature}" "${THERMAL_STOP_C}"
                kill -- "-${calibration_pid}" 2>/dev/null || true
                return
            fi
        done < <(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null || true)
        sleep 1
    done
}

printf '%s\n' "Phase E calibrated FP8-KV maintenance window begins: $(date --iso-8601=seconds)"
printf '%s\n' "Recording deployed reference, source identity, and topology."
systemctl cat "${SERVICE}" > "${OUTPUT_DIR}/reference-unit.txt"
systemctl show "${SERVICE}" --property=Environment --property=ExecStart --property=ActiveState > "${OUTPUT_DIR}/reference-unit-state.txt"
nvidia-smi topo -m > "${OUTPUT_DIR}/gpu-topology.txt"
nvidia-smi > "${OUTPUT_DIR}/gpu-before.txt"
(cd "${SOURCE}" && sha256sum --check "${SOURCE_CHECKSUMS}") > "${OUTPUT_DIR}/source-integrity.txt"
sha256sum "${SOURCE}/config.json" "${SOURCE}/model-00001-of-00018.safetensors" "${SOURCE}/model-00018-of-00018.safetensors" > "${OUTPUT_DIR}/source-checksums.txt"
sha256sum "${RECIPE}" > "${OUTPUT_DIR}/recipe-checksum.txt"
git -C "${MODELOPT_ROOT}" rev-parse HEAD > "${OUTPUT_DIR}/modelopt-commit.txt"

if [[ ${was_active} -eq 1 ]]; then
    printf '%s\n' "Verifying rollback before calibration by restarting ${SERVICE}."
    systemctl stop "${SERVICE}"
    systemctl reset-failed "${SERVICE}"
    systemctl start "${SERVICE}"
    for _ in $(seq 1 900); do
        models=$(curl --silent --show-error --fail --max-time 5 http://127.0.0.1:8080/v1/models || true)
        [[ ${models} == *"${ALIAS}"* ]] && break
        sleep 1
    done
    [[ ${models:-} == *"${ALIAS}"* ]] || { printf '%s\n' "rollback readiness failed" >&2; exit 1; }
    printf '%s\n' "Rollback verification passed. Stopping reference for Phase E calibration."
    systemctl stop "${SERVICE}"
fi

systemctl is-active --quiet "${SERVICE}" && { printf '%s\n' "reference service remains active" >&2; exit 1; }
mapfile -t compute_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sort -u)
for pid in "${compute_pids[@]}"; do
    [[ -z ${pid} || ${pid} == "${ALLOWED_GPU0_PID}" ]] || {
        printf 'unexpected compute PID after reference stop: %s\n' "${pid}" >&2
        exit 1
    }
done
nvidia-smi > "${OUTPUT_DIR}/gpu-after-reference-stop.txt"

printf '%s\n' "Starting ModelOpt PTQ with the original 1,024 x 512-token calibration corpus."
setsid runuser -u "${BENCH_USER}" -- env \
    CUDA_VISIBLE_DEVICES=0,1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    PYTHONPATH="${MODELOPT_ROOT}" \
    PYTHONUNBUFFERED=1 \
    "${PYTHON}" "${MODELOPT_ROOT}/examples/llm_ptq/hf_ptq.py" \
    --pyt_ckpt_path "${SOURCE}" \
    --recipe "${RECIPE}" \
    --batch_size 1 \
    --calib_size 1024 \
    --calib_seq 512 \
    --dataset cnn_dailymail \
    --export_path "${OUTPUT_DIR}/export" \
    --use_seq_device_map \
    --gpu_max_mem_percentage 0.80 \
    --trust_remote_code \
    --skip_generate \
    > "${OUTPUT_DIR}/calibration.log" 2>&1 &
calibration_pid=$!
thermal_guard &
thermal_guard_pid=$!
wait "${calibration_pid}"
calibration_pid=0
kill "${thermal_guard_pid}" 2>/dev/null || true
wait "${thermal_guard_pid}" 2>/dev/null || true
thermal_guard_pid=0

printf '%s\n' "ModelOpt PTQ completed: ${OUTPUT_DIR}/export"
