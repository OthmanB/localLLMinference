#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT=/home/obenomar/localLLMinference
readonly OPS=${ROOT}/operations
readonly ETC=/etc/ai-server
readonly USER_ID=$(id -u obenomar)
readonly GATEWAY_ENV=${ETC}/lan-inference-gateway.env
readonly METRICS_ENV=${ETC}/ai-metrics-exporter.env
readonly COST_CONFIG=${ETC}/ai-cost-accounting.json
readonly PRICING_CONFIG=${ETC}/ai-api-pricing.json
readonly GATEWAY_BACKENDS='LAN_INFERENCE_BACKENDS=[{"name":"qwen3.8-27b-q4","base_url":"http://127.0.0.1:8080","models":["qwen3.8-27b-q4-gpukv192"],"health_path":"/health"},{"name":"flash-next-262k","base_url":"http://127.0.0.1:1901","models":["qwen3.8-flash-next-nvfp4-262k"],"health_path":"/health"},{"name":"muse-glimmer-30b-131k","base_url":"http://127.0.0.1:8082","models":["muse-glimmer-30b-kquant17"],"health_path":"/health"}]'

if [[ ${EUID} -ne 0 ]]; then
    printf '%s\n' "Run as root: sudo ${OPS}/promote-muse-glimmer-30b-131k.sh" >&2
    exit 1
fi

if [[ ! -f ${GATEWAY_ENV} ]]; then
    printf '%s\n' "Missing ${GATEWAY_ENV}; create it from config/lan-inference-gateway.env.example first." >&2
    exit 1
fi

install -m 0644 "${OPS}/systemd/llama-muse-glimmer-30b-131k.service" /etc/systemd/system/llama-muse-glimmer-30b-131k.service
install -m 0644 "${OPS}/systemd/lan-inference-gateway.service" /etc/systemd/system/lan-inference-gateway.service
install -m 0644 "${OPS}/systemd/ai-metrics-exporter.service" /etc/systemd/system/ai-metrics-exporter.service
install -m 0600 -o root -g root "${OPS}/config/ai-metrics-exporter.env.example" "${METRICS_ENV}"
if [[ ! -e ${COST_CONFIG} ]]; then
    install -m 0640 -o root -g obenomar "${OPS}/config/ai-cost-accounting.json.example" "${COST_CONFIG}"
fi
if [[ ! -e ${PRICING_CONFIG} ]]; then
    install -m 0640 -o root -g obenomar "${OPS}/config/ai-api-pricing.json" "${PRICING_CONFIG}"
fi

tmp=$(mktemp)
trap 'rm -f "${tmp}"' EXIT
awk -v replacement="${GATEWAY_BACKENDS}" '
    /^LAN_INFERENCE_BACKENDS=/ { print replacement; found = 1; next }
    { print }
    END { if (!found) print replacement }
' "${GATEWAY_ENV}" > "${tmp}"
install -m 0600 -o root -g root "${tmp}" "${GATEWAY_ENV}"

systemctl daemon-reload
if [[ -S /run/user/${USER_ID}/bus ]]; then
    runuser -u obenomar -- env XDG_RUNTIME_DIR=/run/user/${USER_ID} \
        systemctl --user stop muse-glimmer-gpu2-131k.service || true
fi

systemctl enable --now llama-muse-glimmer-30b-131k.service
printf '%s\n' "Waiting for Muse Glimmer to finish loading (up to 450 seconds)..."
for attempt in {1..90}; do
    if curl --fail --silent http://127.0.0.1:8082/health >/dev/null 2>&1; then
        printf '%s\n' "Muse Glimmer is serving."
        break
    fi
    if (( attempt % 12 == 0 )); then
        printf 'Muse is still loading after %d seconds...\n' "$((attempt * 5))"
    fi
    sleep 5
    if (( attempt == 90 )); then
        printf '%s\n' "Muse Glimmer did not become ready within 450 seconds." >&2
        systemctl status llama-muse-glimmer-30b-131k.service --no-pager >&2 || true
        exit 1
    fi
done

systemctl enable --now ai-metrics-exporter.service lan-inference-gateway.service
systemctl restart ai-metrics-exporter.service lan-inference-gateway.service

systemctl is-active --quiet llama-muse-glimmer-30b-131k.service
systemctl is-active --quiet ai-metrics-exporter.service
systemctl is-active --quiet lan-inference-gateway.service
printf '%s\n' "Muse promotion complete."
