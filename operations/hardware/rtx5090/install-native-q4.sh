#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ${EUID} -ne 0 ]]; then
    printf '%s\n' 'Run this installer as root.' >&2
    exit 1
fi

if [[ ${AI_SERVER_RTX5090_CONFIRM:-} != install ]]; then
    printf '%s\n' 'Refusing to install without AI_SERVER_RTX5090_CONFIRM=install.' >&2
    exit 1
fi

readonly ROOT=${AI_SERVER_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}
readonly PROFILE_ROOT=${ROOT}/operations/hardware/rtx5090
readonly SERVICE_USER=${AI_SERVER_USER:-${SUDO_USER:-$(id -un)}}
readonly LLAMA_CPP_ROOT=${AI_SERVER_LLAMA_CPP_ROOT:-/home/michel/LLMs-tests/llama.cpp}
readonly LLAMA_SERVER_BIN=${AI_SERVER_LLAMA_SERVER_BIN:-${LLAMA_CPP_ROOT}/build/bin/llama-server}
readonly MODEL_PATH=${AI_SERVER_MODEL_PATH:-/home/michel/LLMs-tests/rtx5090-qwen38-bench/models/Qwen3.8-27B-UD-Q4_K_M.gguf}
readonly PYTHON=${AI_SERVER_PYTHON:-/usr/bin/python3}
readonly LOG_DIR=${AI_SERVER_LOG_DIR:-/var/log/ai-server}
readonly GPU_INDEX=${AI_SERVER_RTX5090_GPU:-0}
readonly PORT=${AI_SERVER_RTX5090_PORT:-8080}
readonly POLICY_UNIT=ai-rtx5090-gpu0-policy.service
readonly MODEL_UNIT=ai-rtx5090-qwen3.8-q4-native.service
readonly MONITOR_UNIT=ai-rtx5090-monitor.service

[[ ${GPU_INDEX} =~ ^[0-9]+$ ]] || {
    printf 'AI_SERVER_RTX5090_GPU must be a non-negative integer: %s\n' "${GPU_INDEX}" >&2
    exit 1
}
[[ ${PORT} =~ ^[0-9]+$ && PORT -ge 1 && PORT -le 65535 ]] || {
    printf 'AI_SERVER_RTX5090_PORT must be between 1 and 65535: %s\n' "${PORT}" >&2
    exit 1
}

exec 9>/run/ai-rtx5090-install.lock
if ! flock -n 9; then
    printf '%s\n' 'Another RTX 5090 installer instance is already running.' >&2
    exit 1
fi

for command in systemctl install sed flock nvidia-smi ss; do
    command -v "${command}" >/dev/null || {
        printf 'Required command is missing: %s\n' "${command}" >&2
        exit 1
    }
done

id "${SERVICE_USER}" >/dev/null
[[ -x ${PYTHON} ]] || { printf 'Python is not executable: %s\n' "${PYTHON}" >&2; exit 1; }
[[ -x ${LLAMA_SERVER_BIN} ]] || { printf 'llama-server is not executable: %s\n' "${LLAMA_SERVER_BIN}" >&2; exit 1; }
[[ -r ${MODEL_PATH} ]] || { printf 'Model is not readable: %s\n' "${MODEL_PATH}" >&2; exit 1; }
[[ -f ${ROOT}/tools/nvidia_fan_control.py ]] || {
    printf 'Fan-control helper is missing under repository root: %s\n' "${ROOT}" >&2
    exit 1
}

gpu_info=$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=name,memory.total --format=csv,noheader,nounits)
IFS=',' read -r gpu_name gpu_memory <<< "${gpu_info}"
gpu_name=${gpu_name//[[:space:]]/ }
gpu_memory=${gpu_memory//[!0-9]/}
[[ ${gpu_name} == *'RTX 5090'* ]] || {
    printf 'Selected GPU is not an RTX 5090: %s\n' "${gpu_name}" >&2
    exit 1
}
(( gpu_memory >= 30000 )) || {
    printf 'Selected GPU has too little memory for this profile: %s MiB\n' "${gpu_memory}" >&2
    exit 1
}

port_is_free() {
    local listeners
    listeners=$(ss -H -ltn "sport = :${PORT}" || true)
    [[ -z ${listeners//[[:space:]]/} ]]
}

unit_is_inactive() {
    local unit=$1
    ! systemctl is-active --quiet "${unit}" 2>/dev/null &&
        ! systemctl is-enabled --quiet "${unit}" 2>/dev/null
}

compute_apps=$(nvidia-smi -i "${GPU_INDEX}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
[[ -z ${compute_apps//[[:space:]]/} ]] || {
    printf 'Physical GPU %s is already used by compute processes: %s\n' "${GPU_INDEX}" "${compute_apps}" >&2
    exit 1
}
port_is_free || {
    printf 'Port %s is already occupied. Refusing to guess its owner.\n' "${PORT}" >&2
    exit 1
}

for unit in \
    nvidia-power-limit.service \
    nvidia-fan-control.service \
    llama-muse-glimmer-30b-131k.service \
    llama-qwen3.8-q4-192k.service \
    llama-qwen3.8-q4-tensor-262k.service; do
    unit_is_inactive "${unit}" || {
        printf 'Conflicting unit is active or enabled: %s\n' "${unit}" >&2
        printf '%s\n' 'Stop and explicitly resolve ownership before using this profile.' >&2
        exit 1
    }
done

render_unit() {
    local source=$1
    local target=$2
    local rendered
    rendered=$(mktemp)
    sed \
        -e "s|@SERVICE_USER@|${SERVICE_USER}|g" \
        -e "s|@REPO_ROOT@|${ROOT}|g" \
        -e "s|@LLAMA_CPP_ROOT@|${LLAMA_CPP_ROOT}|g" \
        -e "s|@LLAMA_SERVER_BIN@|${LLAMA_SERVER_BIN}|g" \
        -e "s|@MODEL_PATH@|${MODEL_PATH}|g" \
        -e "s|@PYTHON@|${PYTHON}|g" \
        -e "s|@LOG_DIR@|${LOG_DIR}|g" \
        -e "s|@GPU_INDEX@|${GPU_INDEX}|g" \
        -e "s|@PORT@|${PORT}|g" \
        "${source}" > "${rendered}"
    install -m 0644 "${rendered}" "${target}"
    rm -f "${rendered}"
}

install -d -m 0750 "${LOG_DIR}"
chown "${SERVICE_USER}:${SERVICE_USER}" "${LOG_DIR}"
render_unit "${PROFILE_ROOT}/systemd/ai-rtx5090-gpu0-policy.service" "/etc/systemd/system/${POLICY_UNIT}"
render_unit "${PROFILE_ROOT}/systemd/ai-rtx5090-qwen3.8-q4-native.service" "/etc/systemd/system/${MODEL_UNIT}"
render_unit "${PROFILE_ROOT}/systemd/ai-rtx5090-monitor.service" "/etc/systemd/system/${MONITOR_UNIT}"

logrotate_tmp=$(mktemp)
sed \
    -e "s|@LOG_DIR@|${LOG_DIR}|g" \
    -e "s|@SERVICE_USER@|${SERVICE_USER}|g" \
    "${PROFILE_ROOT}/logrotate/ai-rtx5090-qwen3.8" > "${logrotate_tmp}"
install -m 0644 "${logrotate_tmp}" /etc/logrotate.d/ai-rtx5090-qwen3.8
rm -f "${logrotate_tmp}"

systemctl daemon-reload
systemctl enable --now "${POLICY_UNIT}"
systemctl enable --now "${MODEL_UNIT}"
systemctl enable --now "${MONITOR_UNIT}"

ready=0
for _ in {1..180}; do
    if "${PYTHON}" - "${PORT}" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request

port = sys.argv[1]
with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
    payload = json.load(response)
models = payload.get("data", [])
if not any(item.get("id") == "qwen3.8-27b-q4-gpukv-native-rtx5090" for item in models):
    raise SystemExit(1)
PY
    then
        ready=1
        break
    fi
    if ! systemctl is-active --quiet "${MODEL_UNIT}"; then
        systemctl --no-pager --full status "${MODEL_UNIT}" || true
        exit 1
    fi
    sleep 1
done

if [[ ${ready} -ne 1 ]]; then
    printf '%s\n' 'The RTX 5090 model service did not become identity-ready within 180 seconds.' >&2
    systemctl --no-pager --full status "${MODEL_UNIT}" || true
    exit 1
fi

nvidia-smi --query-gpu=index,name,power.limit,fan.speed,temperature.gpu,memory.used,memory.total --format=csv
systemctl --no-pager --full status "${POLICY_UNIT}" "${MODEL_UNIT}" "${MONITOR_UNIT}"
