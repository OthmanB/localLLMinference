#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT=/home/obenomar/localLLMinference
readonly OPS=${ROOT}/operations
readonly ETC=/etc/ai-server
readonly RULE=/etc/udev/rules.d/99-ai-powercap.rules
readonly PROFILER_CONFIG=${ETC}/cpu-power-profiler.json

if [[ ${EUID} -ne 0 ]]; then
    printf '%s\n' "Run as root: sudo ${OPS}/install-cpu-power-profiler.sh" >&2
    exit 1
fi

if [[ -f ${RULE} ]]; then
    # udev MODE= only affects /dev nodes, never sysfs attribute permissions.
    # The profiler unit fixes powercap readability with root ExecStartPre lines.
    rm -f "${RULE}"
    udevadm control --reload-rules
fi

if ! id obenomar >/dev/null 2>&1; then
    printf '%s\n' "Expected user obenomar does not exist; adjust the profiler unit User= first." >&2
    exit 1
fi

install -m 0644 "${OPS}/systemd/ai-cpu-power-profiler.service" /etc/systemd/system/ai-cpu-power-profiler.service
if [[ ! -e ${PROFILER_CONFIG} ]]; then
    install -m 0644 "${OPS}/config/cpu-power-profiler.json.example" "${PROFILER_CONFIG}"
fi
systemctl daemon-reload
systemctl restart ai-cpu-power-profiler.service 2>/dev/null || systemctl enable --now ai-cpu-power-profiler.service
sleep 12
systemctl --no-pager --full status ai-cpu-power-profiler.service | head -n 8 || true
ls -l /sys/devices/virtual/powercap/intel-rapl/intel-rapl:0/energy_uj || true
printf '%s\n' "Verify with: curl -s http://127.0.0.1:9109/metrics | grep ai_cpu_profiler_rapl_available (expect 1)"
