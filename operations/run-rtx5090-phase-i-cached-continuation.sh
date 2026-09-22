#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT=/home/michel/LLMs-tests/localLLMinference
readonly RUNNER="${ROOT}/operations/run-rtx5090-phase-d-benchmark.sh"
readonly MODEL=/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-e-kv-fp8-calibration-2026-09-20T052906Z/export
readonly VLLM=${AI_SERVER_PHASE_I_VLLM:-/home/michel/LLMs-tests/.phase-f1p5-vllm-nightly/bin/vllm}
readonly ALLOWED_GPU0_PID=${AI_SERVER_PHASE_D_ALLOWED_GPU0_PID:?set the approved unrelated GPU 0 compute PID}
readonly APPEND_TOKENS=${AI_SERVER_PHASE_I_APPEND_TOKENS:-2048}
readonly LONG_PREFILL_THRESHOLD=${AI_SERVER_PHASE_I_LONG_PREFILL_THRESHOLD:-256}

if [[ ${EUID} -ne 0 ]]; then
    printf '%s\n' "Run with sudo: AI_SERVER_PHASE_D_ALLOWED_GPU0_PID=<pid> sudo -E ${0}" >&2
    exit 1
fi
if [[ ! ${APPEND_TOKENS} =~ ^[1-9][0-9]*$ ]] || [[ ${APPEND_TOKENS} -gt 4096 ]] || ! [[ ${LONG_PREFILL_THRESHOLD} =~ ^[1-9][0-9]*$ ]] || [[ ! -x ${RUNNER} || ! -x ${VLLM} || ! -f ${MODEL}/config.json ]]; then
    printf '%s\n' "Phase I append/threshold, runner, nightly vLLM, or calibrated model is not usable" >&2
    exit 1
fi

AI_SERVER_PHASE_D_ALLOWED_GPU0_PID=${ALLOWED_GPU0_PID} \
AI_SERVER_PHASE_D_MODEL=${MODEL} \
AI_SERVER_PHASE_D_RUN_PREFIX="phase-i-graph-b0-cached-continuation-append${APPEND_TOKENS}-threshold${LONG_PREFILL_THRESHOLD}" \
AI_SERVER_PHASE_D_VLLM=${VLLM} \
    "${RUNNER}" vllm \
        --cached-continuation-interference-only \
        --context-tokens 262144 \
        --gpu-memory-utilization 0.90 \
        --max-num-seqs 3 \
        --cached-active-prompt-tokens 196000 \
        --cached-active-output-tokens 8192 \
        --cached-base-prompt-tokens 196000 \
        --cached-continuation-append-tokens "${APPEND_TOKENS}" \
        --interference-established-decode-seconds 20 \
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
        --vllm-long-prefill-token-threshold "${LONG_PREFILL_THRESHOLD}"
