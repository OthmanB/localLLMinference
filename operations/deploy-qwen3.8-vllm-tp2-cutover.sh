#!/usr/bin/env bash
# Stage or cut over the approved Qwen3.8 vLLM TP2 deployment. Run only as root.
set -Eeuo pipefail

PATH=/usr/sbin:/usr/bin:/sbin:/bin

readonly APPROVED_ROOT=/home/michel/LLMs-tests/localLLMinference
readonly SCRIPT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
readonly SYSTEMD_SOURCE_DIR="${SCRIPT_ROOT}/operations/systemd"
readonly CONFIG_SOURCE_DIR="${SCRIPT_ROOT}/operations/config"
readonly SYSTEMD_TARGET_DIR=/etc/systemd/system
readonly ENV_TARGET_DIR=/etc/ai-server
readonly BACKUP_ROOT=/var/lib/ai-server
readonly BACKUP_PREFIX=qwen3.8-vllm-tp2-cutover
readonly LOCK_FILE=/run/lock/ai-qwen3.8-vllm-tp2-cutover.lock

readonly SERVICE_USER=michel
readonly MODEL_ALIAS=qwen3.8-27b-q4-gpukv-native
readonly MODEL_DIR=/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-e-kv-fp8-calibration-2026-09-20T052906Z/export
readonly MODEL_CONFIG="${MODEL_DIR}/config.json"
readonly MODEL_CONFIG_SHA256=731d9ea745aa5c171822e4e64b4a863ae3269620a43ca2bec8a1e42c13a89b68
readonly VLLM_BIN=/home/michel/LLMs-tests/.phase-f1p5-vllm-nightly/bin/vllm
readonly PYTHON_BIN=/home/michel/LLMs-tests/.phase-f1p5-vllm-nightly/bin/python
readonly GATEWAY_SOURCE=/home/michel/LLMs-tests/localLLMinference/lan-inference-gateway/src
readonly GATEWAY_ROOT=/home/michel/LLMs-tests/localLLMinference/lan-inference-gateway
readonly MONITOR_SCRIPT=/home/michel/LLMs-tests/localLLMinference/tools/qwen_gpu_monitor.py
readonly VLLM_VERSION=0.29.1rc1.dev438+g01f1f58f1
readonly TORCH_VERSION=2.13.0+cu132
readonly CUDA_VERSION=13.2
readonly BACKEND_PORT=18081
readonly GATEWAY_PORT=8080

readonly LEGACY_SERVICE=llama-qwen3.8-q4-native.service
readonly LEGACY_MONITOR=ai-qwen3.8-q4-native-monitor.service
readonly BACKEND_UNIT=ai-qwen3.8-vllm-tp2.service
readonly GATEWAY_UNIT=ai-qwen3.8-vllm-tp2-gateway.service
readonly MONITOR_TEMPLATE=ai-qwen3.8-vllm-tp2-monitor
readonly MONITOR_GPU0_UNIT="${MONITOR_TEMPLATE}@0.service"
readonly MONITOR_GPU1_UNIT="${MONITOR_TEMPLATE}@1.service"

BACKUP_DIR=
legacy_service_enabled=
legacy_service_active=
legacy_monitor_enabled=
legacy_monitor_active=
activation_in_progress=0
rollback_started=0
activation_started_at=

log() {
    printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

die() {
    log "ERROR: $*" >&2
    exit 1
}

on_exit() {
    local status=$?
    trap - EXIT INT TERM
    if (( status != 0 && activation_in_progress == 1 && rollback_started == 0 )); then
        rollback_started=1
        set +e
        log "Activation failed; restoring the legacy Qwen service from ${BACKUP_DIR}." >&2
        stop_candidate_units
        wait_for_ports_released || log "WARNING: candidate ports did not release before legacy restore." >&2
        restore_legacy_from_state || log "ERROR: automatic legacy restoration failed." >&2
    fi
    exit "${status}"
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

usage() {
    printf '%s\n' "Usage: $0 [--activate|--rollback]"
    printf '%s\n' 'Without an option, templates are staged and verified but no service is activated.'
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command is unavailable: $1"
}

unit_fragment() {
    systemctl show --property=FragmentPath --value "$1"
}

require_unit() {
    local fragment
    fragment=$(unit_fragment "$1") || die "Unable to inspect unit: $1"
    [[ -n ${fragment} && -f ${fragment} ]] || die "Required installed unit is missing: $1"
}

unit_enabled() {
    if systemctl is-enabled --quiet "$1" 2>/dev/null; then
        printf '1'
    else
        printf '0'
    fi
}

unit_active() {
    if systemctl is-active --quiet "$1"; then
        printf '1'
    else
        printf '0'
    fi
}

validate_commands() {
    local command
    for command in curl date flock id install journalctl mktemp nvidia-smi sha256sum ss systemctl systemd-analyze; do
        require_command "${command}"
    done
}

validate_legacy_units() {
    require_unit "${LEGACY_SERVICE}"
    require_unit "${LEGACY_MONITOR}"
}

validate_deployment_inputs() {
    local checksum vllm_python_version gateway_python_version vllm_version torch_version cuda_version binary_version

    validate_commands
    [[ ${SCRIPT_ROOT} == "${APPROVED_ROOT}" ]] || die "This cutover must run from ${APPROVED_ROOT}, not ${SCRIPT_ROOT}"
    id "${SERVICE_USER}" >/dev/null || die "Required service user is missing: ${SERVICE_USER}"

    for path in \
        "${SYSTEMD_SOURCE_DIR}/${BACKEND_UNIT}" \
        "${SYSTEMD_SOURCE_DIR}/${GATEWAY_UNIT}" \
        "${SYSTEMD_SOURCE_DIR}/${MONITOR_TEMPLATE}@.service" \
        "${CONFIG_SOURCE_DIR}/ai-qwen3.8-vllm-tp2.env" \
        "${CONFIG_SOURCE_DIR}/ai-qwen3.8-vllm-tp2-gateway.env" \
        "${MODEL_CONFIG}" \
        "${GATEWAY_SOURCE}/lan_inference_gateway/app.py" \
        "${MONITOR_SCRIPT}"; do
        [[ -r ${path} ]] || die "Required readable path is missing: ${path}"
    done
    [[ -x ${VLLM_BIN} ]] || die "Qualified vLLM executable is missing: ${VLLM_BIN}"
    [[ -x ${PYTHON_BIN} ]] || die "Qualified Python executable is missing: ${PYTHON_BIN}"
    [[ -d ${GATEWAY_ROOT} ]] || die "Gateway root is missing: ${GATEWAY_ROOT}"

    checksum=$(sha256sum "${MODEL_CONFIG}") || die "Cannot hash model config: ${MODEL_CONFIG}"
    [[ ${checksum%% *} == "${MODEL_CONFIG_SHA256}" ]] || {
        die "Model config SHA-256 does not match the approved export."
    }

    vllm_python_version=$("${PYTHON_BIN}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))') || {
        die "Cannot determine the qualified vLLM Python version."
    }
    gateway_python_version=$(PYTHONPATH="${GATEWAY_SOURCE}" "${PYTHON_BIN}" -c 'import sys, uvicorn, lan_inference_gateway.app; print(".".join(map(str, sys.version_info[:3])))') || {
        die "Gateway Python, uvicorn, or local gateway source is unusable."
    }
    [[ ${vllm_python_version} == "${gateway_python_version}" ]] || {
        die "vLLM and gateway Python versions differ: ${vllm_python_version} vs ${gateway_python_version}"
    }
    vllm_version=$("${PYTHON_BIN}" -c 'import vllm; print(vllm.__version__)') || die "Cannot import vLLM."
    torch_version=$("${PYTHON_BIN}" -c 'import torch; print(torch.__version__)') || die "Cannot import torch."
    cuda_version=$("${PYTHON_BIN}" -c 'import torch; print(torch.version.cuda)') || die "Cannot determine torch CUDA version."
    binary_version=$("${VLLM_BIN}" --version) || die "Cannot query the vLLM executable version."
    [[ ${vllm_version} == "${VLLM_VERSION}" ]] || die "Unexpected vLLM version: ${vllm_version}"
    [[ ${binary_version} == *"${VLLM_VERSION}"* ]] || die "vLLM executable does not report ${VLLM_VERSION}"
    [[ ${torch_version} == "${TORCH_VERSION}" ]] || die "Unexpected torch version: ${torch_version}"
    [[ ${cuda_version} == "${CUDA_VERSION}" ]] || die "Unexpected CUDA version: ${cuda_version}"

    validate_legacy_units
    log "Validated Python ${vllm_python_version}, vLLM ${vllm_version}, torch ${torch_version}, CUDA ${cuda_version}, and the approved model hash."
}

backup_legacy_state() {
    local fragment

    umask 077
    install -d -m 0700 -o root -g root "${BACKUP_ROOT}"
    BACKUP_DIR="${BACKUP_ROOT}/${BACKUP_PREFIX}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    install -d -m 0700 -o root -g root "${BACKUP_DIR}"

    legacy_service_enabled=$(unit_enabled "${LEGACY_SERVICE}")
    legacy_service_active=$(unit_active "${LEGACY_SERVICE}")
    legacy_monitor_enabled=$(unit_enabled "${LEGACY_MONITOR}")
    legacy_monitor_active=$(unit_active "${LEGACY_MONITOR}")

    for unit in "${LEGACY_SERVICE}" "${LEGACY_MONITOR}"; do
        fragment=$(unit_fragment "${unit}")
        install -m 0600 -o root -g root "${fragment}" "${BACKUP_DIR}/${unit}"
        systemctl cat "${unit}" > "${BACKUP_DIR}/${unit}.systemctl-cat"
        systemctl show "${unit}" --property=ActiveState --property=SubState --property=UnitFileState --property=FragmentPath > "${BACKUP_DIR}/${unit}.state"
        systemctl is-enabled "${unit}" > "${BACKUP_DIR}/${unit}.enabled" 2>&1 || true
        systemctl is-active "${unit}" > "${BACKUP_DIR}/${unit}.active" 2>&1 || true
    done

    printf 'legacy_service_enabled=%s\nlegacy_service_active=%s\nlegacy_monitor_enabled=%s\nlegacy_monitor_active=%s\n' \
        "${legacy_service_enabled}" "${legacy_service_active}" "${legacy_monitor_enabled}" "${legacy_monitor_active}" \
        > "${BACKUP_DIR}/legacy-state.env"
    curl --silent --show-error --max-time 10 -D "${BACKUP_DIR}/legacy-health.headers" \
        -o "${BACKUP_DIR}/legacy-health.body" "http://127.0.0.1:${GATEWAY_PORT}/health" || true
    curl --silent --show-error --max-time 10 -D "${BACKUP_DIR}/legacy-models.headers" \
        -o "${BACKUP_DIR}/legacy-models.body" "http://127.0.0.1:${GATEWAY_PORT}/v1/models" || true
    nvidia-smi > "${BACKUP_DIR}/gpu-before.txt" 2>&1 || true
    nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv,noheader > "${BACKUP_DIR}/gpu-compute-before.csv" 2>&1 || true
    log "Backed up legacy units, enablement, endpoints, and GPU ownership to ${BACKUP_DIR}."
}

stage_templates() {
    install -d -m 0750 -o root -g root "${ENV_TARGET_DIR}"
    install -m 0644 -o root -g root "${SYSTEMD_SOURCE_DIR}/${BACKEND_UNIT}" "${SYSTEMD_TARGET_DIR}/${BACKEND_UNIT}"
    install -m 0644 -o root -g root "${SYSTEMD_SOURCE_DIR}/${GATEWAY_UNIT}" "${SYSTEMD_TARGET_DIR}/${GATEWAY_UNIT}"
    install -m 0644 -o root -g root "${SYSTEMD_SOURCE_DIR}/${MONITOR_TEMPLATE}@.service" "${SYSTEMD_TARGET_DIR}/${MONITOR_TEMPLATE}@.service"
    install -m 0600 -o root -g root "${CONFIG_SOURCE_DIR}/ai-qwen3.8-vllm-tp2.env" "${ENV_TARGET_DIR}/ai-qwen3.8-vllm-tp2.env"
    install -m 0600 -o root -g root "${CONFIG_SOURCE_DIR}/ai-qwen3.8-vllm-tp2-gateway.env" "${ENV_TARGET_DIR}/ai-qwen3.8-vllm-tp2-gateway.env"

    systemctl daemon-reload
    systemd-analyze verify "${BACKEND_UNIT}" "${GATEWAY_UNIT}" "${MONITOR_GPU0_UNIT}" "${MONITOR_GPU1_UNIT}"
    log 'Candidate templates and root-only environment files are staged and verified.'
}

load_backup_state() {
    local line key value
    legacy_service_enabled=
    legacy_service_active=
    legacy_monitor_enabled=
    legacy_monitor_active=
    [[ -r ${BACKUP_DIR}/legacy-state.env ]] || die "Backup state is missing: ${BACKUP_DIR}/legacy-state.env"

    while IFS= read -r line; do
        key=${line%%=*}
        value=${line#*=}
        case "${key}" in
            legacy_service_enabled) legacy_service_enabled=${value} ;;
            legacy_service_active) legacy_service_active=${value} ;;
            legacy_monitor_enabled) legacy_monitor_enabled=${value} ;;
            legacy_monitor_active) legacy_monitor_active=${value} ;;
        esac
    done < "${BACKUP_DIR}/legacy-state.env"

    [[ ${legacy_service_enabled} =~ ^[01]$ && ${legacy_service_active} =~ ^[01]$ && ${legacy_monitor_enabled} =~ ^[01]$ && ${legacy_monitor_active} =~ ^[01]$ ]] || {
        die "Backup state is invalid: ${BACKUP_DIR}/legacy-state.env"
    }
}

latest_activation_backup() {
    local candidate latest=
    shopt -s nullglob
    for candidate in "${BACKUP_ROOT}"/"${BACKUP_PREFIX}"-*; do
        [[ -d ${candidate} && -f ${candidate}/activation-attempt ]] || continue
        if [[ -z ${latest} || ${candidate} > ${latest} ]]; then
            latest=${candidate}
        fi
    done
    shopt -u nullglob
    printf '%s' "${latest}"
}

port_is_free() {
    local listeners
    listeners=$(ss -H -ltn "sport = :$1" || true)
    [[ -z ${listeners//[[:space:]]/} ]]
}

wait_for_ports_released() {
    local attempt
    for ((attempt = 1; attempt <= 30; attempt++)); do
        if port_is_free "${BACKEND_PORT}" && port_is_free "${GATEWAY_PORT}"; then
            return 0
        fi
        sleep 1
    done
    return 1
}

models_include_alias() {
    local url body
    url=$1
    body=$(curl --fail --silent --max-time 5 "${url}") || return 1
    printf '%s' "${body}" | "${PYTHON_BIN}" -c '
import json
import sys

alias = sys.argv[1]
payload = json.load(sys.stdin)
raise SystemExit(0 if any(item.get("id") == alias for item in payload.get("data", [])) else 1)
' "${MODEL_ALIAS}"
}

private_listener_is_loopback_only() {
    local listeners
    listeners=$(ss -H -ltn "sport = :${BACKEND_PORT}" || true)
    [[ ${listeners} == *"127.0.0.1:${BACKEND_PORT}"* ]] || return 1
    [[ ${listeners} != *"0.0.0.0:${BACKEND_PORT}"* ]] || return 1
    [[ ${listeners} != *"[::]:${BACKEND_PORT}"* ]]
}

backend_ready() {
    local metrics
    systemctl is-active --quiet "${BACKEND_UNIT}" || return 1
    curl --fail --silent --max-time 5 "http://127.0.0.1:${BACKEND_PORT}/health" >/dev/null || return 1
    models_include_alias "http://127.0.0.1:${BACKEND_PORT}/v1/models" || return 1
    metrics=$(curl --fail --silent --max-time 5 "http://127.0.0.1:${BACKEND_PORT}/metrics") || return 1
    [[ ${metrics} == *'vllm:num_requests_running'*'model_name="qwen3.8-27b-q4-gpukv-native"'* ]] || return 1
    [[ ${metrics} == *'vllm:num_requests_waiting'*'model_name="qwen3.8-27b-q4-gpukv-native"'* ]] || return 1
    private_listener_is_loopback_only
}

gateway_ready() {
    systemctl is-active --quiet "${GATEWAY_UNIT}" || return 1
    curl --fail --silent --max-time 5 "http://127.0.0.1:${GATEWAY_PORT}/healthz" >/dev/null || return 1
    curl --fail --silent --max-time 5 "http://127.0.0.1:${GATEWAY_PORT}/readyz" >/dev/null || return 1
    models_include_alias "http://127.0.0.1:${GATEWAY_PORT}/v1/models"
}

legacy_ready() {
    systemctl is-active --quiet "${LEGACY_SERVICE}" || return 1
    curl --fail --silent --max-time 5 "http://127.0.0.1:${GATEWAY_PORT}/health" >/dev/null || return 1
    models_include_alias "http://127.0.0.1:${GATEWAY_PORT}/v1/models"
}

wait_for() {
    local seconds=$1 description=$2 attempt
    shift 2
    for ((attempt = 1; attempt <= seconds; attempt++)); do
        if "$@"; then
            log "${description} is ready."
            return 0
        fi
        sleep 1
    done
    log "ERROR: timed out waiting for ${description}." >&2
    return 1
}

assert_expected_preexisting_gpu_processes() {
    local gpu pid allowed_pid=${AI_SERVER_QWEN38_ALLOWED_GPU0_PID:-}

    if [[ -n ${allowed_pid} ]]; then
        [[ ${allowed_pid} =~ ^[1-9][0-9]*$ ]] || die 'AI_SERVER_QWEN38_ALLOWED_GPU0_PID must be a live positive PID.'
        kill -0 "${allowed_pid}" 2>/dev/null || die "Approved GPU 0 PID is not live: ${allowed_pid}"
    fi

    for gpu in 0 1; do
        while IFS= read -r pid; do
            pid=${pid//[[:space:]]/}
            [[ -z ${pid} || ${pid} == N/A ]] && continue
            if [[ ${gpu} == 0 && -n ${allowed_pid} && ${pid} == "${allowed_pid}" ]]; then
                continue
            fi
            die "Unexpected compute PID on physical GPU ${gpu} after legacy shutdown: ${pid}"
        done < <(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
    done
}

stop_candidate_units() {
    local unit
    for unit in "${GATEWAY_UNIT}" "${MONITOR_GPU0_UNIT}" "${MONITOR_GPU1_UNIT}" "${BACKEND_UNIT}"; do
        systemctl disable --now "${unit}" >/dev/null 2>&1 || true
        systemctl reset-failed "${unit}" >/dev/null 2>&1 || true
    done
}

apply_enablement() {
    local unit=$1 enabled=$2
    if [[ ${enabled} == 1 ]]; then
        systemctl enable "${unit}" >/dev/null
    else
        systemctl disable "${unit}" >/dev/null
    fi
}

restore_legacy_from_state() {
    apply_enablement "${LEGACY_SERVICE}" "${legacy_service_enabled}"
    apply_enablement "${LEGACY_MONITOR}" "${legacy_monitor_enabled}"
    systemctl reset-failed "${LEGACY_SERVICE}" "${LEGACY_MONITOR}" >/dev/null 2>&1 || true

    if [[ ${legacy_service_active} == 1 ]]; then
        systemctl start "${LEGACY_SERVICE}"
        wait_for 900 'legacy Qwen backend' legacy_ready
    else
        systemctl stop "${LEGACY_SERVICE}" >/dev/null 2>&1 || true
    fi
    if [[ ${legacy_monitor_active} == 1 ]]; then
        systemctl start "${LEGACY_MONITOR}"
        systemctl is-active --quiet "${LEGACY_MONITOR}"
    else
        systemctl stop "${LEGACY_MONITOR}" >/dev/null 2>&1 || true
    fi
}

verify_public_chat_completion() {
    local response
    response=$(curl --fail --silent --show-error --max-time 600 \
        -H 'Content-Type: application/json' \
        --data "{\"model\":\"${MODEL_ALIAS}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK.\"}],\"max_tokens\":8,\"temperature\":0}" \
        "http://127.0.0.1:${GATEWAY_PORT}/v1/chat/completions") || return 1
    printf '%s' "${response}" | "${PYTHON_BIN}" -c '
import json
import sys

alias = sys.argv[1]
payload = json.load(sys.stdin)
choices = payload.get("choices")
if payload.get("model") != alias or not isinstance(choices, list) or not choices:
    raise SystemExit(1)
raise SystemExit(0)
' "${MODEL_ALIAS}"
}

verify_public_streaming_chat_completion() {
    local response
    response=$(curl --fail --silent --show-error --max-time 600 \
        -H 'Content-Type: application/json' \
        --data "{\"model\":\"${MODEL_ALIAS}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK.\"}],\"max_tokens\":8,\"temperature\":0,\"stream\":true}" \
        "http://127.0.0.1:${GATEWAY_PORT}/v1/chat/completions") || return 1
    [[ ${response} == *'data:'* && ${response} == *'[DONE]'* ]]
}

verify_public_c3_graph() {
    local directory index result journal
    local -a pids=()

    directory=$(mktemp -d)
    for index in 1 2 3; do
        curl --fail --silent --show-error --max-time 600 \
            -H 'Content-Type: application/json' \
            --data "{\"model\":\"${MODEL_ALIAS}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with C3-${index}.\"}],\"max_tokens\":128,\"temperature\":0}" \
            "http://127.0.0.1:${GATEWAY_PORT}/v1/chat/completions" \
            > "${directory}/${index}.json" &
        pids+=("$!")
    done
    for index in "${!pids[@]}"; do
        if ! wait "${pids[${index}]}"; then
            rm -rf "${directory}"
            return 1
        fi
    done
    for result in "${directory}"/*.json; do
        if ! "${PYTHON_BIN}" -c '
import json
import sys

alias = sys.argv[1]
payload = json.load(sys.stdin)
choices = payload.get("choices")
raise SystemExit(0 if payload.get("model") == alias and isinstance(choices, list) and choices else 1)
' "${MODEL_ALIAS}" < "${result}"; then
            rm -rf "${directory}"
            return 1
        fi
    done
    rm -rf "${directory}"

    # vLLM emits CUDAGraph statistics at a ten-second interval.
    sleep 12
    journal=$(journalctl --no-pager --unit "${BACKEND_UNIT}" --since "${activation_started_at}") || return 1
    [[ ${journal} == *'Using Triton/FLA GDN prefill kernel'* ]] || return 1
    [[ ${journal} == *'CG Capture: mode=FULL'*'num_reqs=3'* ]] || return 1
    [[ ${journal} == *'**CUDAGraph Stats:**'*'| 3'*'FULL'* ]]
}

verify_public_long_admission() {
    local headers lowercase metrics

    headers=$("${PYTHON_BIN}" -c '
import json

print(json.dumps({
    "model": "qwen3.8-27b-q4-gpukv-native",
    "messages": [{"role": "user", "content": "x " * 200000}],
    "max_tokens": 1,
    "temperature": 0,
}, separators=(",", ":")))
' | curl --fail --silent --show-error --max-time 600 \
        -D - -o /dev/null \
        -H 'Content-Type: application/json' \
        --data-binary @- \
        "http://127.0.0.1:${GATEWAY_PORT}/v1/chat/completions") || return 1
    lowercase=${headers,,}
    [[ ${lowercase} == *'x-gateway-request-id:'* ]] || return 1
    [[ ${lowercase} == *'x-inference-admission: c2'* ]] || return 1
    [[ ${lowercase} == *'x-inference-gateway-admission-wait-ms:'* ]] || return 1
    metrics=$(curl --fail --silent --max-time 5 "http://127.0.0.1:${GATEWAY_PORT}/metrics") || return 1
    [[ ${metrics} == *'ai_gateway_cold_admission_queue_depth'*'ai_gateway_cold_admission_inflight'* ]]
}

verify_temperatures() {
    nvidia-smi --query-gpu=index,temperature.gpu --format=csv,noheader,nounits | "${PYTHON_BIN}" -c '
import csv
import sys

for row in csv.reader(sys.stdin):
    if len(row) != 2 or float(row[1]) >= 85:
        raise SystemExit(1)
'
}

activate() {
    [[ ${legacy_service_enabled} == 1 && ${legacy_service_active} == 1 ]] || {
        die "Refusing activation: ${LEGACY_SERVICE} was not enabled and active in the backup."
    }
    printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${BACKUP_DIR}/activation-attempt"
    activation_in_progress=1
    activation_started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    log "Stopping and disabling legacy units ${LEGACY_MONITOR} and ${LEGACY_SERVICE}."
    systemctl disable "${LEGACY_MONITOR}" "${LEGACY_SERVICE}"
    systemctl stop "${LEGACY_MONITOR}" "${LEGACY_SERVICE}"
    ! systemctl is-active --quiet "${LEGACY_SERVICE}"
    ! systemctl is-active --quiet "${LEGACY_MONITOR}"
    port_is_free "${BACKEND_PORT}" || die "Private candidate port is already occupied: ${BACKEND_PORT}"
    assert_expected_preexisting_gpu_processes

    log "Starting private vLLM backend on 127.0.0.1:${BACKEND_PORT}."
    # An installed but never-started unit can be garbage-collected by systemd.
    # That is not a failed candidate and must not abort the guarded cutover.
    systemctl reset-failed "${BACKEND_UNIT}" >/dev/null 2>&1 || true
    systemctl enable "${BACKEND_UNIT}"
    systemctl start "${BACKEND_UNIT}"
    wait_for 900 'private vLLM backend health, model alias, metrics, and loopback listener' backend_ready

    log "Starting telemetry monitors and public gateway on 0.0.0.0:${GATEWAY_PORT}."
    systemctl enable "${MONITOR_GPU0_UNIT}" "${MONITOR_GPU1_UNIT}" "${GATEWAY_UNIT}"
    systemctl start "${MONITOR_GPU0_UNIT}" "${MONITOR_GPU1_UNIT}"
    systemctl is-active --quiet "${MONITOR_GPU0_UNIT}"
    systemctl is-active --quiet "${MONITOR_GPU1_UNIT}"
    systemctl start "${GATEWAY_UNIT}"
    wait_for 60 'public gateway health, readyz, and model alias' gateway_ready
    verify_public_chat_completion || die 'Public non-streaming chat completion validation failed.'
    verify_public_streaming_chat_completion || die 'Public streaming chat completion validation failed.'
    verify_public_c3_graph || die 'Public C=3 CUDA-graph and Triton GDN validation failed.'
    verify_public_long_admission || die 'Public long-request C<=2 admission validation failed.'
    verify_temperatures || die 'GPU temperature reached the 85 C operational limit during validation.'

    printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${BACKUP_DIR}/activation-complete"
    activation_in_progress=0
    log "Cutover completed. Legacy units remain installed and are disabled as the rollback target."
}

rollback() {
    validate_commands
    validate_legacy_units
    BACKUP_DIR=$(latest_activation_backup)
    [[ -n ${BACKUP_DIR} ]] || die "No activation backup was found under ${BACKUP_ROOT}."
    load_backup_state
    [[ ${legacy_service_enabled} == 1 && ${legacy_service_active} == 1 ]] || {
        die "Refusing rollback: backup does not describe an enabled, active legacy backend."
    }

    log "Rolling back candidate units using ${BACKUP_DIR}."
    stop_candidate_units
    wait_for_ports_released || die 'Candidate ports did not release during rollback.'
    restore_legacy_from_state || die 'Legacy service restoration failed.'
    log 'Rollback completed and legacy health and model identity were verified.'
}

main() {
    local mode=${1:-stage}
    [[ ${EUID} -eq 0 ]] || die 'Run this script under sudo as root.'
    require_command flock
    [[ -d ${LOCK_FILE%/*} ]] || die "Required lock directory is missing: ${LOCK_FILE%/*}"
    exec 9>"${LOCK_FILE}"
    flock -n 9 || die 'Another Qwen3.8 vLLM TP2 cutover command is already running.'
    case "${mode}" in
        stage)
            [[ $# -eq 0 ]] || { usage >&2; exit 2; }
            validate_deployment_inputs
            backup_legacy_state
            stage_templates
            log 'No services were started. Re-run with --activate to perform the cutover.'
            ;;
        --activate)
            [[ $# -eq 1 ]] || { usage >&2; exit 2; }
            validate_deployment_inputs
            backup_legacy_state
            stage_templates
            activate
            ;;
        --rollback)
            [[ $# -eq 1 ]] || { usage >&2; exit 2; }
            rollback
            ;;
        -h|--help)
            usage
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
}

main "$@"
