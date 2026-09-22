#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT=/home/michel/LLMs-tests/localLLMinference
readonly GATEWAY_ROOT="${ROOT}/lan-inference-gateway"
readonly RUNNER="${ROOT}/operations/run-rtx5090-phase-d-benchmark.sh"
readonly EVALUATOR="${ROOT}/tools/gsm8k_recovery_eval.py"
readonly MODEL=/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-e-kv-fp8-calibration-2026-09-20T052906Z/export
readonly VLLM=${AI_SERVER_PHASE_E_VLLM:-/home/michel/LLMs-tests/.phase-f1p5-vllm-nightly/bin/vllm}
readonly GATEWAY_PYTHON=${AI_SERVER_PHASE_E_GATEWAY_PYTHON:-/home/michel/LLMs-tests/.phase-f1p5-vllm-nightly/bin/python}
readonly EVAL_PYTHON=${AI_SERVER_PHASE_E_GSM8K_PYTHON:-/tmp/phase-e-sgl-eval-venv/bin/python}
readonly ALLOWED_GPU0_PID=${AI_SERVER_PHASE_D_ALLOWED_GPU0_PID:?set the approved unrelated GPU 0 compute PID}
readonly BACKEND_PORT=${AI_SERVER_PHASE_E_GSM8K_PORT:-18081}
readonly GATEWAY_PORT=${AI_SERVER_PHASE_E_GSM8K_GATEWAY_PORT:-18080}
readonly EXAMPLES=${AI_SERVER_PHASE_E_GSM8K_EXAMPLES:-256}
readonly REQUEST_TIMEOUT=${AI_SERVER_PHASE_E_GSM8K_REQUEST_TIMEOUT:-3600}
readonly RUN_TIMEOUT=${AI_SERVER_PHASE_E_GSM8K_RUN_TIMEOUT:-21600}
readonly MODEL_ALIAS=qwen3.8-27b-nvfp4-vllm-tp2-phase-d
readonly GATEWAY_LOG=/tmp/rtx5090-phase-e-gsm8k-gateway.log
# Admit every evaluator request so the gateway preserves the qualified C=3 ceiling.
readonly GATEWAY_BACKENDS="[{\"name\":\"rtx5090-gsm8k\",\"base_url\":\"http://127.0.0.1:${BACKEND_PORT}\",\"models\":[\"${MODEL_ALIAS}\"],\"health_path\":\"/health\",\"cold_admission\":{\"metrics_url\":\"http://127.0.0.1:${BACKEND_PORT}/metrics\",\"target_model\":\"${MODEL_ALIAS}\",\"max_active_leases\":3,\"dispatch_running_limit\":2,\"cold_min_input_tokens\":1,\"poll_interval_seconds\":1.0,\"queue_depth\":3,\"queue_timeout_seconds\":${REQUEST_TIMEOUT}.0,\"upstream_timeout_seconds\":${REQUEST_TIMEOUT}.0}}]"

if [[ ${EUID} -ne 0 ]]; then
    printf '%s\n' "Run with sudo: AI_SERVER_PHASE_D_ALLOWED_GPU0_PID=<pid> sudo -E ${0}" >&2
    exit 1
fi
if ! [[ ${EXAMPLES} =~ ^[1-9][0-9]*$ && ${REQUEST_TIMEOUT} =~ ^[1-9][0-9]*$ && ${RUN_TIMEOUT} =~ ^[1-9][0-9]*$ && ${BACKEND_PORT} =~ ^[1-9][0-9]*$ && ${GATEWAY_PORT} =~ ^[1-9][0-9]*$ && ${BACKEND_PORT} != "${GATEWAY_PORT}" && -x ${RUNNER} && -x ${EVALUATOR} && -x ${VLLM} && -x ${GATEWAY_PYTHON} && -x ${EVAL_PYTHON} && -f ${MODEL}/config.json ]]; then
    printf '%s\n' "The examples, ports, guarded runner, evaluator, vLLM, gateway Python, evaluator Python, or timeouts are not usable" >&2
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
        LAN_INFERENCE_DEFAULT_BACKEND=rtx5090-gsm8k \
        LAN_INFERENCE_REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT}" \
        PYTHONPATH="${GATEWAY_ROOT}/src" \
        "${GATEWAY_PYTHON}" -m uvicorn lan_inference_gateway.app:app --host 127.0.0.1 --port "${GATEWAY_PORT}" \
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
AI_SERVER_PHASE_D_RUN_PREFIX="phase-e-gsm8k-recovery-${EXAMPLES}" \
AI_SERVER_PHASE_D_VLLM=${VLLM} \
    "${RUNNER}" vllm \
        --port "${BACKEND_PORT}" \
        --request-port "${GATEWAY_PORT}" \
        --metrics-port "${BACKEND_PORT}" \
        --recovery-gsm8k-executable "${EVALUATOR}" \
        --recovery-gsm8k-python "${EVAL_PYTHON}" \
        --recovery-gsm8k-examples "${EXAMPLES}" \
        --recovery-gsm8k-max-tokens 4096 \
        --recovery-gsm8k-continuation-max-tokens 4096 \
        --recovery-gsm8k-concurrency 3 \
        --recovery-gsm8k-request-timeout "${REQUEST_TIMEOUT}" \
        --recovery-gsm8k-run-timeout "${RUN_TIMEOUT}" \
        --context-tokens 262144 \
        --gpu-memory-utilization 0.90 \
        --max-num-seqs 3 \
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
