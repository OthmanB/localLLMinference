#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT=/home/obenomar/localLLMinference
readonly OPS=${ROOT}/operations
readonly ETC=/etc/ai-server
readonly USER_ID=$(id -u obenomar)
readonly LAN_SUBNET=${AI_SERVER_LAN_SUBNET:-}
readonly PROMETHEUS_IP=${AI_SERVER_PROMETHEUS_IP:-}
readonly BACKUP_DIR=/var/backups/ai-server/$(date -u +%Y%m%dT%H%M%SZ)
readonly RELEASE_SOURCE=${ROOT}/research/llamAmpere-v0.3-36a6bca81
readonly RELEASE_DEST=/opt/ai-server/llamampere/36a6bca817
readonly MODEL_SOURCE=${ROOT}/research/qwen3.8-27b-atx-iq4_xs-m-gguf
readonly MODEL_DEST=/opt/ai-server/models/qwen3.8-27b-atx-iq4_xs_m

if [[ ${EUID} -ne 0 ]]; then
    printf '%s\n' 'Run this installer as root.' >&2
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

for script in "${OPS}"/*.sh; do
    bash -n "${script}"
done

install -d -m 0700 -o root -g root "${BACKUP_DIR}"
for path in \
    /etc/ai-server/lan-inference-gateway.env \
    /etc/ai-server/ai-metrics-exporter.env \
    /etc/ai-server/qwen-serving-profile \
    /etc/systemd/system/llama-qwen3.8-q4-192k.service \
    /etc/systemd/system/freetoken-qwen3.8-flash-next-262k.service \
    /etc/systemd/system/freetoken-flash-next-candidate.service \
    /etc/systemd/system/llama-qwen3.8-q4-tensor-262k.service \
    /etc/systemd/system/llamampere-qwen3.8-atx-iq4xs-m-gpu1.service \
    /etc/systemd/system/llamampere-qwen3.8-atx-iq4xs-m-gpu2.service \
    /etc/systemd/system/llama-muse-glimmer-30b-131k.service \
    /etc/systemd/system/lan-inference-gateway.service \
    /etc/systemd/system/ai-metrics-exporter.service \
    /etc/systemd/system/ai-cpu-power-profiler.service \
    /etc/ai-server/cpu-power-profiler.json; do
    if [[ -e ${path} ]]; then
        install -m 0600 -o root -g root "${path}" "${BACKUP_DIR}/$(basename "${path}")"
    fi
done
printf 'Backed up existing deployment files to %s\n' "${BACKUP_DIR}"

install -d -m 0750 -o root -g obenomar "${ETC}"
install -d -m 0750 -o root -g root "${ETC}/profiles"
install -d -m 0755 -o root -g root /usr/local/sbin
install -d -m 0755 -o root -g root /opt/ai-server /opt/ai-server/models /opt/ai-server/llamampere
install -d -m 0755 -o obenomar -g obenomar /var/log/ai-server

if [[ ! -d ${RELEASE_DEST} ]]; then
    [[ -d ${RELEASE_SOURCE}/build-sm86 ]] || { printf 'Missing llamAmpere source release: %s\n' "${RELEASE_SOURCE}" >&2; exit 1; }
    cp -a "${RELEASE_SOURCE}" "${RELEASE_DEST}"
    chown -R root:root "${RELEASE_DEST}"
    find "${RELEASE_DEST}" -type d -exec chmod 0755 {} +
    find "${RELEASE_DEST}" -type f -exec chmod a-w {} +
    chmod 0755 "${RELEASE_DEST}/build-sm86/bin/llama-server"
fi
if [[ ! -d ${MODEL_DEST} ]]; then
    [[ -f ${MODEL_SOURCE}/Qwen3.8-27B-ATX-4-XS.gguf ]] || { printf 'Missing ATX model source: %s\n' "${MODEL_SOURCE}" >&2; exit 1; }
    cp -a "${MODEL_SOURCE}" "${MODEL_DEST}"
    chown -R root:root "${MODEL_DEST}"
    find "${MODEL_DEST}" -type d -exec chmod 0755 {} +
    find "${MODEL_DEST}" -type f -exec chmod 0644 {} +
fi

install -m 0644 "${OPS}/systemd/nvidia-power-limit.service" /etc/systemd/system/nvidia-power-limit.service
install -m 0644 "${OPS}/systemd/nvidia-fan-control.service" /etc/systemd/system/nvidia-fan-control.service
install -m 0644 "${OPS}/systemd/llama-qwen3.8-q4-tensor-262k.service" /etc/systemd/system/llama-qwen3.8-q4-tensor-262k.service
install -m 0644 "${OPS}/systemd/llamampere-qwen3.8-atx-iq4xs-m-gpu1.service" /etc/systemd/system/llamampere-qwen3.8-atx-iq4xs-m-gpu1.service
install -m 0644 "${OPS}/systemd/llamampere-qwen3.8-atx-iq4xs-m-gpu2.service" /etc/systemd/system/llamampere-qwen3.8-atx-iq4xs-m-gpu2.service
install -m 0644 "${OPS}/systemd/llama-muse-glimmer-30b-131k.service" /etc/systemd/system/llama-muse-glimmer-30b-131k.service
install -m 0644 "${OPS}/systemd/lan-inference-gateway.service" /etc/systemd/system/lan-inference-gateway.service
install -m 0644 "${OPS}/systemd/ai-metrics-exporter.service" /etc/systemd/system/ai-metrics-exporter.service
install -m 0644 "${OPS}/systemd/ai-cpu-power-profiler.service" /etc/systemd/system/ai-cpu-power-profiler.service
install -d -m 0755 -o root -g root /etc/tmpfiles.d /etc/logrotate.d
install -m 0644 -o root -g root "${OPS}/config/tmpfiles-ai-server.conf" /etc/tmpfiles.d/ai-server.conf
install -m 0644 -o root -g root "${OPS}/config/logrotate-ai-server" /etc/logrotate.d/ai-server
install -m 0755 "${OPS}/verify-llamampere-artifacts.sh" /usr/local/sbin/ai-verify-llamampere-artifacts
install -m 0755 "${OPS}/qwen-profile-preflight.sh" /usr/local/sbin/ai-qwen-profile-preflight
install -m 0755 "${OPS}/qwen-profile-switch.sh" /usr/local/sbin/ai-qwen-profile-switch
for profile in atx-dual stock-q4-tensor; do
    install -m 0640 -o root -g obenomar "${OPS}/config/profiles/${profile}.gateway.env" "${ETC}/profiles/${profile}.gateway.env"
    install -m 0640 -o root -g obenomar "${OPS}/config/profiles/${profile}.metrics.env" "${ETC}/profiles/${profile}.metrics.env"
done
"/usr/local/sbin/ai-verify-llamampere-artifacts"

if [[ -f /etc/udev/rules.d/99-ai-powercap.rules ]]; then
    rm -f /etc/udev/rules.d/99-ai-powercap.rules
    udevadm control --reload-rules
fi

gateway_env=${ETC}/lan-inference-gateway.env
if [[ ! -e ${gateway_env} ]]; then
    gateway_tmp=$(mktemp)
    trap 'rm -f "${gateway_tmp}"' EXIT
    {
        printf 'LAN_GATEWAY_CLIENT_TOKEN=%s\n' "$(openssl rand -hex 32)"
        cat "${OPS}/config/profiles/stock-q4-tensor.gateway.env"
    } > "${gateway_tmp}"
    install -m 0600 -o root -g root "${gateway_tmp}" "${gateway_env}"
fi

metrics_env=${ETC}/ai-metrics-exporter.env
if [[ ! -e ${metrics_env} ]]; then
    install -m 0640 -o root -g obenomar "${OPS}/config/profiles/stock-q4-tensor.metrics.env" "${metrics_env}"
fi
if [[ ! -e ${ETC}/qwen-serving-profile ]]; then
    printf '%s\n' stock-q4-tensor > "${ETC}/qwen-serving-profile"
    chown root:root "${ETC}/qwen-serving-profile"
    chmod 0644 "${ETC}/qwen-serving-profile"
fi

if [[ ! -e ${ETC}/ai-cost-accounting.json ]]; then
    install -m 0640 -o root -g obenomar "${OPS}/config/ai-cost-accounting.json.example" "${ETC}/ai-cost-accounting.json"
fi
if [[ ! -e ${ETC}/ai-api-pricing.json ]]; then
    install -m 0640 -o root -g obenomar "${OPS}/config/ai-api-pricing.json" "${ETC}/ai-api-pricing.json"
fi
profiler_config=${ETC}/cpu-power-profiler.json
if [[ ! -e ${profiler_config} ]]; then
    install -m 0644 "${OPS}/config/cpu-power-profiler.json.example" "${profiler_config}"
fi

systemd-tmpfiles --create /etc/tmpfiles.d/ai-server.conf
systemctl daemon-reload
systemctl enable nvidia-power-limit.service
systemctl restart nvidia-power-limit.service
systemctl enable nvidia-fan-control.service
if ! systemctl restart nvidia-fan-control.service; then
    printf '%s\n' 'Warning: fixed fan control could not be applied; automatic GPU fan control remains active.' >&2
fi
systemctl enable ai-cpu-power-profiler.service
systemctl restart ai-cpu-power-profiler.service

if [[ -S /run/user/${USER_ID}/bus ]]; then
    runuser -u obenomar -- env XDG_RUNTIME_DIR=/run/user/${USER_ID} \
        systemctl --user disable --now llama-qwen3.8-gpukv64.service llama-qwen3.8-longctx.service || true
fi

systemctl disable --now freetoken-flash-next-candidate.service || true
systemctl disable --now freetoken-qwen3.8-flash-next-262k.service || true
systemctl disable --now llama-qwen3.8-q4-192k.service || true
rm -f /etc/systemd/system/freetoken-flash-next-candidate.service /etc/systemd/system/freetoken-qwen3.8-flash-next-262k.service /etc/systemd/system/llama-qwen3.8-q4-192k.service
systemctl daemon-reload

if command -v ufw >/dev/null 2>&1; then
    ufw allow from "${LAN_SUBNET}" to any port 8088 proto tcp
    ufw allow from "${PROMETHEUS_IP}" to any port 9108 proto tcp
    ufw allow from "${PROMETHEUS_IP}" to any port 9109 proto tcp
    ufw deny 8080/tcp
    ufw deny 8081/tcp
    if ip link show tailscale0 >/dev/null 2>&1; then
        ufw allow in on tailscale0 to any port 8088 proto tcp
    fi
fi

printf '%s\n' 'Installed services, profiles, artifact verifier, and profile switcher.'
printf '%s\n' 'Select the active Qwen profile with: sudo /usr/local/sbin/ai-qwen-profile-switch {atx-dual|stock-q4-tensor}'
