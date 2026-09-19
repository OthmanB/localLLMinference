#!/usr/bin/env bash
set -Eeuo pipefail

readonly ETC=/etc/ai-server
readonly PROFILE_DIR=${ETC}/profiles
readonly GATEWAY_ENV=${ETC}/lan-inference-gateway.env
readonly METRICS_ENV=${ETC}/ai-metrics-exporter.env
readonly PROFILE_MARKER=${ETC}/qwen-serving-profile
readonly BACKUP_ROOT=/var/backups/ai-server/profile-switch
readonly LOCK_PATH=/run/ai-server-qwen-profile.lock
readonly ATX_GPU1=llamampere-qwen3.8-atx-iq4xs-m-gpu1.service
readonly ATX_GPU2=llamampere-qwen3.8-atx-iq4xs-m-gpu2.service
readonly STOCK=llama-qwen3.8-q4-tensor-262k.service
readonly GATEWAY=lan-inference-gateway.service
readonly EXPORTER=ai-metrics-exporter.service

if [[ ${EUID} -ne 0 ]]; then
    printf '%s\n' 'Run this profile switch as root.' >&2
    exit 1
fi
if [[ $# -ne 1 || ($1 != atx-dual && $1 != stock-q4-tensor) ]]; then
    printf '%s\n' 'Usage: ai-qwen-profile-switch {atx-dual|stock-q4-tensor}' >&2
    exit 2
fi
readonly TARGET=$1

exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
    printf '%s\n' 'Another Qwen profile switch is already running.' >&2
    exit 1
fi

[[ -r ${PROFILE_DIR}/${TARGET}.gateway.env ]] || { printf 'Missing gateway profile: %s\n' "${TARGET}" >&2; exit 1; }
[[ -r ${PROFILE_DIR}/${TARGET}.metrics.env ]] || { printf 'Missing metrics profile: %s\n' "${TARGET}" >&2; exit 1; }
[[ -r ${GATEWAY_ENV} ]] || { printf 'Missing gateway environment: %s\n' "${GATEWAY_ENV}" >&2; exit 1; }

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
backup_dir=${BACKUP_ROOT}/${timestamp}
install -d -m 0700 -o root -g root "${backup_dir}"
install -m 0600 -o root -g root "${GATEWAY_ENV}" "${backup_dir}/lan-inference-gateway.env"
if [[ -e ${METRICS_ENV} ]]; then
    install -m 0600 -o root -g root "${METRICS_ENV}" "${backup_dir}/ai-metrics-exporter.env"
fi
if [[ -e ${PROFILE_MARKER} ]]; then
    install -m 0644 -o root -g root "${PROFILE_MARKER}" "${backup_dir}/qwen-serving-profile"
fi
printf 'Profile switch backup: %s\n' "${backup_dir}"

old_profile=$(cat "${PROFILE_MARKER}" 2>/dev/null || true)
if [[ ${old_profile} != atx-dual && ${old_profile} != stock-q4-tensor ]]; then
    printf '%s\n' 'No valid current Qwen profile is selected.' >&2
    exit 1
fi
old_gateway_backup=${backup_dir}/lan-inference-gateway.env
old_metrics_backup=${backup_dir}/ai-metrics-exporter.env
switch_failed=1
rollback_done=0

restore_profile_enablement() {
    if [[ ${old_profile} == atx-dual ]]; then
        systemctl disable "${STOCK}" >/dev/null 2>&1 || true
        systemctl enable "${ATX_GPU1}" "${ATX_GPU2}" >/dev/null 2>&1 || true
    else
        systemctl disable "${ATX_GPU1}" "${ATX_GPU2}" >/dev/null 2>&1 || true
        systemctl enable "${STOCK}" >/dev/null 2>&1 || true
    fi
}

restore_previous_configuration() {
    if [[ -e ${old_gateway_backup} ]]; then
        install -m 0600 -o root -g root "${old_gateway_backup}" "${GATEWAY_ENV}"
    fi
    if [[ -e ${old_metrics_backup} ]]; then
        install -m 0600 -o root -g root "${old_metrics_backup}" "${METRICS_ENV}"
    else
        rm -f "${METRICS_ENV}"
    fi
    if [[ -n ${old_profile} ]]; then
        printf '%s\n' "${old_profile}" > "${PROFILE_MARKER}"
    else
        rm -f "${PROFILE_MARKER}"
    fi
    restore_profile_enablement
}

on_error() {
    if (( switch_failed && !rollback_done )); then
        rollback_done=1
        systemctl stop "${GATEWAY}" "${EXPORTER}" "${ATX_GPU1}" "${ATX_GPU2}" "${STOCK}" >/dev/null 2>&1 || true
        restore_previous_configuration
        printf '%s\n' 'Profile switch failed closed. New model services and gateway remain stopped.' >&2
    fi
}

on_signal() {
    local code=$1
    on_error
    exit "${code}"
}

fail_switch() {
    printf '%s\n' "$1" >&2
    on_error
    exit 1
}

trap on_error ERR

port_is_free() {
    local port=$1
    local listeners
    if ! listeners=$(ss -H -ltn "sport = :${port}"); then
        return 1
    fi
    [[ -z ${listeners//[[:space:]]/} ]]
}

gpu_is_free() {
    local gpu=$1
    local pids
    if ! pids=$(nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null); then
        return 1
    fi
    [[ -z ${pids//[[:space:]]/} ]]
}

wait_url() {
    local url=$1
    for _ in {1..90}; do
        if curl --fail --silent --show-error --max-time 5 "${url}" >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done
    printf 'Endpoint did not become ready: %s\n' "${url}" >&2
    return 1
}

ensure_llamampere_log_directory() {
    install -d -m 0755 -o obenomar -g obenomar /var/log/ai-server
}

token_line=$(grep '^LAN_GATEWAY_CLIENT_TOKEN=' "${GATEWAY_ENV}" || true)
if [[ -z ${token_line} ]]; then
    token_line="LAN_GATEWAY_CLIENT_TOKEN=$(openssl rand -hex 32)"
fi

gateway_tmp=$(mktemp)
metrics_tmp=$(mktemp)
marker_tmp=$(mktemp)
trap 'rm -f "${gateway_tmp}" "${metrics_tmp}" "${marker_tmp}"' EXIT
trap 'on_signal 130' INT
trap 'on_signal 143' TERM
{
    printf '%s\n' "${token_line}"
    grep -v '^LAN_GATEWAY_CLIENT_TOKEN=' "${PROFILE_DIR}/${TARGET}.gateway.env"
} > "${gateway_tmp}"
install -m 0600 -o root -g root "${gateway_tmp}" "${GATEWAY_ENV}"
install -m 0600 -o root -g root "${PROFILE_DIR}/${TARGET}.metrics.env" "${metrics_tmp}"
install -m 0600 -o root -g root "${metrics_tmp}" "${METRICS_ENV}"

systemctl stop "${GATEWAY}" "${EXPORTER}" >/dev/null 2>&1 || true
systemctl stop "${ATX_GPU1}" "${ATX_GPU2}" "${STOCK}" >/dev/null 2>&1 || true

if [[ ${TARGET} == atx-dual ]]; then
    systemctl disable "${STOCK}" >/dev/null 2>&1 || true
    systemctl enable "${ATX_GPU1}" "${ATX_GPU2}" >/dev/null
else
    systemctl disable "${ATX_GPU1}" "${ATX_GPU2}" >/dev/null 2>&1 || true
    systemctl enable "${STOCK}" >/dev/null
fi

systemctl daemon-reload
port_is_free 8080 || fail_switch 'Port 8080 is still occupied.'
port_is_free 8081 || fail_switch 'Port 8081 is still occupied.'
gpu_is_free 1 || fail_switch 'Physical GPU 1 is still occupied.'
gpu_is_free 2 || fail_switch 'Physical GPU 2 is still occupied.'

systemctl start llama-muse-glimmer-30b-131k.service
wait_url http://127.0.0.1:8082/health
if [[ ${TARGET} == atx-dual ]]; then
    ensure_llamampere_log_directory
    systemctl start "${ATX_GPU1}" "${ATX_GPU2}"
    wait_url http://127.0.0.1:8080/health
    wait_url http://127.0.0.1:8081/health
else
    systemctl start "${STOCK}"
    wait_url http://127.0.0.1:8080/health
fi

# Make the preflight checks match the model services before starting dependents.
printf '%s\n' "${TARGET}" > "${marker_tmp}"
install -m 0644 -o root -g root "${marker_tmp}" "${PROFILE_MARKER}"

systemctl enable "${EXPORTER}" "${GATEWAY}" >/dev/null
systemctl restart "${EXPORTER}"
systemctl is-active --quiet "${EXPORTER}"
systemctl restart "${GATEWAY}"
systemctl is-active --quiet "${GATEWAY}"
wait_url http://127.0.0.1:8088/healthz
curl --fail --silent --show-error --max-time 10 -H "Authorization: Bearer ${token_line#*=}" http://127.0.0.1:8088/readyz >/dev/null
curl --fail --silent --show-error --max-time 10 -H "Authorization: Bearer ${token_line#*=}" http://127.0.0.1:8088/v1/models >/dev/null

switch_failed=0
printf 'Active Qwen profile: %s\n' "${TARGET}"
