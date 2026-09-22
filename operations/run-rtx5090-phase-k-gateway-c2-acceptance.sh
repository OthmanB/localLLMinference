#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT=/home/michel/LLMs-tests/localLLMinference
readonly GATEWAY_ROOT="${ROOT}/lan-inference-gateway"
readonly RUNNER="${ROOT}/operations/run-rtx5090-phase-d-benchmark.sh"
readonly MODEL=/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-e-kv-fp8-calibration-2026-09-20T052906Z/export
readonly VLLM=${AI_SERVER_PHASE_K_VLLM:-/home/michel/LLMs-tests/.phase-f1p5-vllm-nightly/bin/vllm}
readonly PYTHON=/home/michel/LLMs-tests/.phase-f1p5-vllm-nightly/bin/python
readonly ALLOWED_GPU0_PID=${AI_SERVER_PHASE_D_ALLOWED_GPU0_PID:?set the approved unrelated GPU 0 compute PID}
readonly BACKEND_PORT=${AI_SERVER_PHASE_K_BACKEND_PORT:-18081}
readonly GATEWAY_PORT=${AI_SERVER_PHASE_K_GATEWAY_PORT:-18080}
readonly REQUEST_TIMEOUT=${AI_SERVER_PHASE_K_REQUEST_TIMEOUT:-600}
readonly WORKLOAD_MODE=${AI_SERVER_PHASE_K_WORKLOAD_MODE:-c2}
readonly MODEL_ALIAS=qwen3.8-27b-nvfp4-vllm-tp2-phase-d
readonly GATEWAY_LOG=/tmp/rtx5090-phase-k-gateway-c2.log
readonly GATEWAY_BACKENDS="[{\"name\":\"rtx5090-qualification\",\"base_url\":\"http://127.0.0.1:${BACKEND_PORT}\",\"models\":[\"${MODEL_ALIAS}\"],\"health_path\":\"/health\",\"cold_admission\":{\"metrics_url\":\"http://127.0.0.1:${BACKEND_PORT}/metrics\",\"target_model\":\"${MODEL_ALIAS}\",\"max_active_leases\":3,\"dispatch_running_limit\":2,\"cold_min_input_tokens\":100000,\"poll_interval_seconds\":1.0,\"queue_depth\":3,\"queue_timeout_seconds\":900.0,\"upstream_timeout_seconds\":600.0}}]"

case "${WORKLOAD_MODE}" in
    c2) readonly WORKLOAD_FLAG=--cached-c3-c2-admission-only ;;
    native) readonly WORKLOAD_FLAG=--cached-c3-native-admission ;;
    *) printf '%s\n' "AI_SERVER_PHASE_K_WORKLOAD_MODE must be c2 or native" >&2; exit 1 ;;
esac

if [[ ${EUID} -ne 0 ]]; then
    printf '%s\n' "Run with sudo: AI_SERVER_PHASE_D_ALLOWED_GPU0_PID=<pid> sudo -E ${0}" >&2
    exit 1
fi
if ! [[ ${REQUEST_TIMEOUT} =~ ^[1-9][0-9]*$ ]] || [[ ${REQUEST_TIMEOUT} -gt 600 ]] || [[ ! -x ${RUNNER} || ! -x ${VLLM} || ! -x ${PYTHON} || ! -f ${MODEL}/config.json ]]; then
    printf '%s\n' "Phase K gateway timeout, runner, Python, nightly vLLM, or calibrated model is not usable" >&2
    exit 1
fi

gateway_pid=
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n ${gateway_pid} ]] && kill -0 "${gateway_pid}" 2>/dev/null; then
        kill -TERM -- "-${gateway_pid}" 2>/dev/null || kill -TERM "${gateway_pid}" 2>/dev/null || true
        wait "${gateway_pid}" 2>/dev/null || true
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

setsid runuser -u michel -- env \
        LAN_INFERENCE_BACKENDS="${GATEWAY_BACKENDS}" \
        LAN_INFERENCE_DEFAULT_BACKEND=rtx5090-qualification \
        LAN_INFERENCE_REQUEST_TIMEOUT_SECONDS=600 \
        PYTHONPATH="${GATEWAY_ROOT}/src" \
        "${PYTHON}" -m uvicorn lan_inference_gateway.app:app --host 127.0.0.1 --port "${GATEWAY_PORT}" \
        >"${GATEWAY_LOG}" 2>&1 &
gateway_pid=$!

for _ in $(seq 1 30); do
    if curl --silent --show-error --fail --max-time 2 "http://127.0.0.1:${GATEWAY_PORT}/healthz" >/dev/null; then
        break
    fi
    sleep 1
done
curl --silent --show-error --fail --max-time 2 "http://127.0.0.1:${GATEWAY_PORT}/healthz" >/dev/null

AI_SERVER_PHASE_D_ALLOWED_GPU0_PID=${ALLOWED_GPU0_PID} \
AI_SERVER_PHASE_D_MODEL=${MODEL} \
AI_SERVER_PHASE_D_RUN_PREFIX="phase-k-gateway-${WORKLOAD_MODE}-acceptance" \
AI_SERVER_PHASE_D_VLLM=${VLLM} \
    "${RUNNER}" vllm \
        "${WORKLOAD_FLAG}" \
        --port "${BACKEND_PORT}" \
        --request-port "${GATEWAY_PORT}" \
        --metrics-port "${BACKEND_PORT}" \
        --context-tokens 262144 \
        --gpu-memory-utilization 0.90 \
        --max-num-seqs 3 \
        --cached-c3-base-prompt-tokens 196000 \
        --cached-c3-probe-tokens 256 \
        --cached-c3-agent-append-tokens 1024 2048 4096 \
        --cached-c3-agent-output-tokens 2048 \
        --cached-c3-cold-prompt-tokens 196000 \
        --cached-c3-cold-output-tokens 256 \
        --cached-c3-max-workload-seconds 900 \
        --startup-timeout 900 \
        --request-timeout "${REQUEST_TIMEOUT}" \
        --vllm-attention-config '{"backend":"TRITON_ATTN","use_trtllm_attention":false}' \
        --vllm-enable-prefix-caching \
        --vllm-enable-cuda-graph \
        --vllm-cudagraph-metrics \
        --vllm-cudagraph-capture-sizes 1 3 \
        --vllm-compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --vllm-gdn-prefill-backend triton \
        --no-vllm-breakable-cudagraph \
        --vllm-logging-level DEBUG \
        --vllm-enable-chunked-prefill \
        --vllm-max-num-batched-tokens 4096 \
        --vllm-long-prefill-token-threshold 256
