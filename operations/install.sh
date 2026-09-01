#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ${EUID} -ne 0 ]]; then
    printf '%s\n' 'Run this installer as root: sudo /path/to/localLLMinference/operations/install.sh' >&2
    exit 1
fi

exec 9>/run/ai-server-install.lock
if ! flock -n 9; then
    printf '%s\n' 'Another installer instance is already running.' >&2
    exit 1
fi

readonly ROOT=${AI_SERVER_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
readonly OPS=${ROOT}/operations
readonly ETC=/etc/ai-server
readonly SERVICE_USER=${AI_SERVER_USER:-${SUDO_USER:-}}
readonly LAN_SUBNET=${AI_SERVER_LAN_SUBNET:-}
readonly PROMETHEUS_IP=${AI_SERVER_PROMETHEUS_IP:-}
readonly LLAMA_CPP_ROOT=${AI_SERVER_LLAMA_CPP_ROOT:-}
readonly LLAMA_SERVER_BIN=${AI_SERVER_LLAMA_SERVER_BIN:-${LLAMA_CPP_ROOT}/build/bin/llama-server}
readonly MODEL_PATH=${AI_SERVER_MODEL_PATH:-${ROOT}/models/qwen3.8-27b-q4-k-m-gguf/Qwen3.8-27B-UD-Q4_K_M.gguf}
readonly PYTHON=${AI_SERVER_PYTHON:-/usr/bin/python3}

if [[ -z ${SERVICE_USER} || -z ${LAN_SUBNET} || -z ${PROMETHEUS_IP} || -z ${LLAMA_CPP_ROOT} ]]; then
    printf '%s\n' 'Set AI_SERVER_USER, AI_SERVER_LAN_SUBNET, AI_SERVER_PROMETHEUS_IP, and AI_SERVER_LLAMA_CPP_ROOT before installation.' >&2
    exit 1
fi

readonly USER_ID=$(id -u "${SERVICE_USER}")

render_unit() {
    local source=$1
    local target=$2
    local rendered
    rendered=$(mktemp)
    sed \
        -e "s#@SERVICE_USER@#${SERVICE_USER}#g" \
        -e "s#@REPO_ROOT@#${ROOT}#g" \
        -e "s#@LLAMA_CPP_ROOT@#${LLAMA_CPP_ROOT}#g" \
        -e "s#@LLAMA_SERVER_BIN@#${LLAMA_SERVER_BIN}#g" \
        -e "s#@MODEL_PATH@#${MODEL_PATH}#g" \
        -e "s#@PYTHON@#${PYTHON}#g" \
        "${source}" > "${rendered}"
    install -m 0644 "${rendered}" "${target}"
    rm -f "${rendered}"
}

install -d -m 0750 -o root -g "${SERVICE_USER}" "${ETC}"
install -m 0644 "${OPS}/systemd/nvidia-power-limit.service" /etc/systemd/system/nvidia-power-limit.service
render_unit "${OPS}/systemd/nvidia-fan-control.service" /etc/systemd/system/nvidia-fan-control.service
render_unit "${OPS}/systemd/llama-qwen3.8-q4-192k.service" /etc/systemd/system/llama-qwen3.8-q4-192k.service
render_unit "${OPS}/systemd/lan-inference-gateway.service" /etc/systemd/system/lan-inference-gateway.service
render_unit "${OPS}/systemd/ai-metrics-exporter.service" /etc/systemd/system/ai-metrics-exporter.service

gateway_env=${ETC}/lan-inference-gateway.env
if [[ ! -e ${gateway_env} ]]; then
    token=$(openssl rand -hex 32)
    umask 077
    printf '%s\n' \
        "LAN_GATEWAY_CLIENT_TOKEN=${token}" \
        'LAN_INFERENCE_CLIENT_API_KEY_ENV=LAN_GATEWAY_CLIENT_TOKEN' \
        'LAN_INFERENCE_REQUEST_TIMEOUT_SECONDS=3600' \
        'LAN_INFERENCE_BACKENDS=[{"name":"q4-192k","base_url":"http://127.0.0.1:8080","models":["qwen3.8-27b-q4-gpukv192"],"health_path":"/health"}]' \
        > "${gateway_env}"
    chown root:root "${gateway_env}"
    chmod 0600 "${gateway_env}"
fi

metrics_env=${ETC}/ai-metrics-exporter.env
if [[ ! -e ${metrics_env} ]]; then
    umask 077
    printf '%s\n' \
        'AI_METRICS_BACKENDS={"q4":{"url":"http://127.0.0.1:8080/metrics","model":"qwen3.8-27b-q4-gpukv192","gpu":"0"}}' \
        > "${metrics_env}"
    chown root:root "${metrics_env}"
    chmod 0600 "${metrics_env}"
fi

systemctl daemon-reload
systemctl enable nvidia-power-limit.service
systemctl restart nvidia-power-limit.service
systemctl enable nvidia-fan-control.service
if ! systemctl restart nvidia-fan-control.service; then
    printf '%s\n' 'Warning: fixed fan control could not be applied; automatic GPU fan control remains active.' >&2
fi

if [[ -S /run/user/${USER_ID}/bus ]]; then
        runuser -u "${SERVICE_USER}" -- env XDG_RUNTIME_DIR=/run/user/${USER_ID} \
        systemctl --user disable --now llama-qwen3.8-gpukv64.service llama-qwen3.8-longctx.service || true
fi

restart_gateway() {
    if systemctl restart lan-inference-gateway.service; then
        return 0
    fi
    printf '%s\n' 'Warning: gateway restart exceeded its graceful stop timeout; checking its final state.' >&2
    systemctl reset-failed lan-inference-gateway.service || true
    if ! systemctl is-active --quiet lan-inference-gateway.service; then
        systemctl start lan-inference-gateway.service
    fi
}

systemctl enable llama-qwen3.8-q4-192k.service
systemctl enable ai-metrics-exporter.service
systemctl enable lan-inference-gateway.service
systemctl restart llama-qwen3.8-q4-192k.service
systemctl restart ai-metrics-exporter.service
restart_gateway

ufw allow from "${LAN_SUBNET}" to any port 8088 proto tcp
ufw allow from "${PROMETHEUS_IP}" to any port 9108 proto tcp
ufw deny 8080/tcp
if ip link show tailscale0 >/dev/null 2>&1; then
    ufw allow in on tailscale0 to any port 8088 proto tcp
fi

nvidia-smi --query-gpu=index,power.limit --format=csv
systemctl --no-pager --full status llama-qwen3.8-q4-192k.service lan-inference-gateway.service ai-metrics-exporter.service
