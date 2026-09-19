#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT=/home/obenomar/localLLMinference
readonly RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
readonly OUTPUT_DIR=${ROOT}/research/qwen3.8-27b-q4-dflash-${RUN_ID}
readonly SERVICE=llama-qwen3.8-q4-tensor-262k.service
readonly DEPENDENT_SERVICES=(
    ai-metrics-exporter.service
    lan-inference-gateway.service
)

if [[ ${EUID} -ne 0 ]]; then
    printf '%s\n' "Run as root through systemd-run: sudo systemd-run --unit=qwen-q4-dflash-benchmark --collect --wait ${ROOT}/operations/run-qwen-q4-dflash-benchmark.sh" >&2
    exit 1
fi

install -d -m 0750 -o obenomar -g obenomar "${OUTPUT_DIR}"
exec > >(tee -a "${OUTPUT_DIR}/wrapper.log") 2>&1

declare -A WAS_ACTIVE
for service in "${SERVICE}" "${DEPENDENT_SERVICES[@]}"; do
    if systemctl is-active --quiet "${service}"; then
        WAS_ACTIVE["${service}"]=1
    else
        WAS_ACTIVE["${service}"]=0
    fi
done

restore_service() {
    local status=$?
    trap - EXIT INT TERM
    printf '%s\n' "Restoring recorded service states."
    if [[ ${WAS_ACTIVE["${SERVICE}"]} -eq 1 ]]; then
        systemctl start "${SERVICE}" || true
    else
        systemctl stop "${SERVICE}" || true
    fi
    for service in "${DEPENDENT_SERVICES[@]}"; do
        if [[ ${WAS_ACTIVE["${service}"]} -eq 1 ]]; then
            systemctl start "${service}" || true
        else
            systemctl stop "${service}" || true
        fi
    done
    chown -R obenomar:obenomar "${OUTPUT_DIR}"
    exit "${status}"
}
trap restore_service EXIT INT TERM

printf '%s\n' "Output directory: ${OUTPUT_DIR}"
printf '%s\n' "Stopping the production Qwen tensor service; dependent services will be restored afterward."
systemctl stop "${SERVICE}"

runuser -u obenomar -- env PYTHONUNBUFFERED=1 \
    /usr/bin/python3 "${ROOT}/tools/qwen_q4_dflash_benchmark.py" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"

printf '%s\n' "Benchmark completed. Results: ${OUTPUT_DIR}/results.json"
