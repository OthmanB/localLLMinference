#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT=/home/obenomar/localLLMinference
readonly OPS=${ROOT}/operations
readonly ETC=/etc/ai-server
readonly USER_ID=$(id -u obenomar)
readonly LAN_SUBNET=${AI_SERVER_LAN_SUBNET:-}
readonly PROMETHEUS_IP=${AI_SERVER_PROMETHEUS_IP:-}
readonly GATEWAY_BACKENDS='LAN_INFERENCE_BACKENDS=[{"name":"qwen3.8-27b-q4-tensor-262k","base_url":"http://127.0.0.1:8080","models":["qwen3.8-27b-q4-tensor262k"],"health_path":"/health"},{"name":"muse-glimmer-30b-131k","base_url":"http://127.0.0.1:8082","models":["muse-glimmer-30b-kquant17"],"health_path":"/health"}]'
readonly BACKUP_DIR=/var/backups/ai-server/$(date -u +%Y%m%dT%H%M%SZ)

if [[ ${EUID} -ne 0 ]]; then
    printf '%s\n' 'Run this installer as root: sudo /home/obenomar/localLLMinference/operations/install.sh' >&2
    exit 1
fi

if [[ -z ${LAN_SUBNET} || -z ${PROMETHEUS_IP} ]]; then
    printf '%s\n' 'Set AI_SERVER_LAN_SUBNET and AI_SERVER_PROMETHEUS_IP before installation.' >&2
    exit 1
fi

exec 9>/run/ai-server-install.lock
if ! flock -n 9; then
    printf '%s\n' 'Another installer instance is already running.' >&2
    exit 1
fi

install -d -m 0700 -o root -g root "${BACKUP_DIR}"
for path in \
    /etc/ai-server/lan-inference-gateway.env \
    /etc/ai-server/ai-metrics-exporter.env \
    /etc/systemd/system/llama-qwen3.8-q4-192k.service \
    /etc/systemd/system/freetoken-qwen3.8-flash-next-262k.service \
    /etc/systemd/system/freetoken-flash-next-candidate.service \
    /etc/systemd/system/llama-qwen3.8-q4-tensor-262k.service \
    /etc/systemd/system/llama-muse-glimmer-30b-131k.service \
    /etc/systemd/system/lan-inference-gateway.service \
    /etc/systemd/system/ai-metrics-exporter.service \
    /etc/systemd/system/ai-cpu-power-profiler.service \
    /etc/ai-server/cpu-power-profiler.json; do
    if [[ -e ${path} ]]; then
        install -m 0600 "${path}" "${BACKUP_DIR}/$(basename "${path}")"
    fi
done
printf 'Backed up existing deployment files to %s\n' "${BACKUP_DIR}"

install -d -m 0750 -o root -g obenomar "${ETC}"
install -m 0644 "${OPS}/systemd/nvidia-power-limit.service" /etc/systemd/system/nvidia-power-limit.service
install -m 0644 "${OPS}/systemd/nvidia-fan-control.service" /etc/systemd/system/nvidia-fan-control.service
install -m 0644 "${OPS}/systemd/llama-qwen3.8-q4-tensor-262k.service" /etc/systemd/system/llama-qwen3.8-q4-tensor-262k.service
install -m 0644 "${OPS}/systemd/llama-muse-glimmer-30b-131k.service" /etc/systemd/system/llama-muse-glimmer-30b-131k.service
install -m 0644 "${OPS}/systemd/lan-inference-gateway.service" /etc/systemd/system/lan-inference-gateway.service
install -m 0644 "${OPS}/systemd/ai-metrics-exporter.service" /etc/systemd/system/ai-metrics-exporter.service
install -m 0644 "${OPS}/systemd/ai-cpu-power-profiler.service" /etc/systemd/system/ai-cpu-power-profiler.service

if [[ -f /etc/udev/rules.d/99-ai-powercap.rules ]]; then
    rm -f /etc/udev/rules.d/99-ai-powercap.rules
    udevadm control --reload-rules
fi

gateway_env=${ETC}/lan-inference-gateway.env
if [[ ! -e ${gateway_env} ]]; then
    token=$(openssl rand -hex 32)
    umask 077
    printf '%s\n' \
        "LAN_GATEWAY_CLIENT_TOKEN=${token}" \
        'LAN_INFERENCE_CLIENT_API_KEY_ENV=LAN_GATEWAY_CLIENT_TOKEN' \
        'LAN_INFERENCE_REQUEST_TIMEOUT_SECONDS=3600' \
        "${GATEWAY_BACKENDS}" \
        > "${gateway_env}"
    chown root:root "${gateway_env}"
    chmod 0600 "${gateway_env}"
fi

gateway_tmp=$(mktemp)
trap 'rm -f "${gateway_tmp}"' EXIT
awk -v replacement="${GATEWAY_BACKENDS}" '
    /^LAN_INFERENCE_BACKENDS=/ { print replacement; found = 1; next }
    { print }
    END { if (!found) print replacement }
' "${gateway_env}" > "${gateway_tmp}"
install -m 0600 -o root -g root "${gateway_tmp}" "${gateway_env}"

metrics_env=${ETC}/ai-metrics-exporter.env
install -m 0600 -o root -g root "${OPS}/config/ai-metrics-exporter.env.example" "${metrics_env}"
cost_config=${ETC}/ai-cost-accounting.json
pricing_config=${ETC}/ai-api-pricing.json
if [[ ! -e ${cost_config} ]]; then
    install -m 0640 -o root -g obenomar "${OPS}/config/ai-cost-accounting.json.example" "${cost_config}"
fi
if [[ ! -e ${pricing_config} ]]; then
    install -m 0640 -o root -g obenomar "${OPS}/config/ai-api-pricing.json" "${pricing_config}"
fi
profiler_config=${ETC}/cpu-power-profiler.json
if [[ ! -e ${profiler_config} ]]; then
    install -m 0644 "${OPS}/config/cpu-power-profiler.json.example" "${profiler_config}"
fi

systemctl daemon-reload
systemctl enable nvidia-power-limit.service
systemctl restart nvidia-power-limit.service
systemctl enable nvidia-fan-control.service
if ! systemctl restart nvidia-fan-control.service; then
    printf '%s\n' 'Warning: fixed fan control could not be applied; automatic GPU fan control remains active.' >&2
fi

if [[ -S /run/user/${USER_ID}/bus ]]; then
    runuser -u obenomar -- env XDG_RUNTIME_DIR=/run/user/${USER_ID} \
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

systemctl disable --now freetoken-flash-next-candidate.service || true
systemctl disable --now freetoken-qwen3.8-flash-next-262k.service || true
systemctl disable --now llama-qwen3.8-q4-192k.service || true
rm -f /etc/systemd/system/freetoken-flash-next-candidate.service /etc/systemd/system/freetoken-qwen3.8-flash-next-262k.service /etc/systemd/system/llama-qwen3.8-q4-192k.service
systemctl daemon-reload
systemctl enable llama-qwen3.8-q4-tensor-262k.service
systemctl enable llama-muse-glimmer-30b-131k.service
systemctl enable ai-metrics-exporter.service
systemctl enable ai-cpu-power-profiler.service
systemctl enable lan-inference-gateway.service
systemctl restart llama-qwen3.8-q4-tensor-262k.service
systemctl restart llama-muse-glimmer-30b-131k.service
systemctl restart ai-metrics-exporter.service
systemctl restart ai-cpu-power-profiler.service
restart_gateway

ufw allow from "${LAN_SUBNET}" to any port 8088 proto tcp
ufw allow from "${PROMETHEUS_IP}" to any port 9108 proto tcp
ufw allow from "${PROMETHEUS_IP}" to any port 9109 proto tcp
ufw deny 8080/tcp
if ip link show tailscale0 >/dev/null 2>&1; then
    ufw allow in on tailscale0 to any port 8088 proto tcp
fi

nvidia-smi --query-gpu=index,power.limit --format=csv
systemctl --no-pager --full status nvidia-fan-control.service llama-qwen3.8-q4-tensor-262k.service llama-muse-glimmer-30b-131k.service lan-inference-gateway.service ai-metrics-exporter.service ai-cpu-power-profiler.service
