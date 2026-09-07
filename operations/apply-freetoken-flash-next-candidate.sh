#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT=/home/obenomar/localLLMinference
readonly UNIT=freetoken-flash-next-candidate.service
readonly SOURCE=${ROOT}/operations/${UNIT}

if [[ ${EUID} -ne 0 ]]; then
    printf '%s\n' "Run as root: sudo ${ROOT}/operations/apply-freetoken-flash-next-candidate.sh" >&2
    exit 1
fi

install -m 0644 "${SOURCE}" "/etc/systemd/system/${UNIT}"
systemctl daemon-reload
printf '%s\n' "Restarting ${UNIT} with the requested cache geometry..."
systemctl restart "${UNIT}"
printf '%s\n' "Waiting for FreeToken to finish loading (up to 450 seconds)..."

for attempt in {1..90}; do
    status=$(curl --fail --silent http://127.0.0.1:1901/health 2>/dev/null || true)
    if [[ ${status} == *'"status":"ok"'* && ${status} == *'"maintenance":"serving"'* ]]; then
        systemctl is-active --quiet "${UNIT}"
        printf '%s\n' "FreeToken is serving."
        exit 0
    fi
    if (( attempt % 12 == 0 )); then
        printf 'Still loading after %d seconds...\n' "$((attempt * 5))"
    fi
    sleep 5
done

printf '%s\n' "FreeToken did not become healthy within 450 seconds." >&2
systemctl status "${UNIT}" --no-pager >&2 || true
exit 1
