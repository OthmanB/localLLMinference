#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT=/home/michel/LLMs-tests/localLLMinference
readonly RUNNER="${ROOT}/operations/run-rtx5090-phase-d-benchmark.sh"
readonly MODEL=/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-e-kv-fp8-calibration-2026-09-20T052906Z/export
readonly VLLM=${AI_SERVER_PHASE_H_VLLM:-/home/michel/LLMs-tests/.phase-f1p5-vllm-nightly/bin/vllm}
readonly ALLOWED_GPU0_PID=${AI_SERVER_PHASE_D_ALLOWED_GPU0_PID:?set the approved unrelated GPU 0 compute PID}
readonly BUDGET=${AI_SERVER_PHASE_H_MAX_NUM_BATCHED_TOKENS:?set the selected Phase H token budget}
readonly LONG_PREFILL_THRESHOLD=${AI_SERVER_PHASE_H_LONG_PREFILL_TOKEN_THRESHOLD:?set the selected long-prefill threshold}

if [[ ${EUID} -ne 0 ]]; then
    printf '%s\n' "Run with sudo: AI_SERVER_PHASE_D_ALLOWED_GPU0_PID=<pid> AI_SERVER_PHASE_H_MAX_NUM_BATCHED_TOKENS=<budget> AI_SERVER_PHASE_H_LONG_PREFILL_TOKEN_THRESHOLD=<threshold> sudo -E ${0}" >&2
    exit 1
fi
if [[ ! ${BUDGET} =~ ^[1-9][0-9]*$ ]] || ! [[ ${LONG_PREFILL_THRESHOLD} =~ ^[1-9][0-9]*$ ]] || [[ ! -x ${RUNNER} || ! -x ${VLLM} || ! -f ${MODEL}/config.json ]]; then
    printf '%s\n' "Phase H budget, runner, nightly vLLM, or calibrated model is not usable" >&2
    exit 1
fi

AI_SERVER_PHASE_D_ALLOWED_GPU0_PID=${ALLOWED_GPU0_PID} \
AI_SERVER_PHASE_D_MODEL=${MODEL} \
AI_SERVER_PHASE_D_RUN_PREFIX="phase-h-graph-b0-decode-under-prefill-budget${BUDGET}-threshold${LONG_PREFILL_THRESHOLD}" \
AI_SERVER_PHASE_D_VLLM=${VLLM} \
    "${RUNNER}" vllm \
        --interference-only \
        --context-tokens 262144 \
        --gpu-memory-utilization 0.90 \
        --max-num-seqs 3 \
        --interference-decode-prompt-tokens 196000 \
        --interference-decode-tokens 8192 \
        --interference-prefill-prompt-tokens 196000 \
        --interference-prefill-tokens 1024 \
        --interference-established-decode-seconds 20 \
        --vllm-attention-config '{"backend":"TRITON_ATTN","use_trtllm_attention":false}' \
        --no-vllm-enable-prefix-caching \
        --vllm-enable-cuda-graph \
        --vllm-cudagraph-metrics \
        --vllm-cudagraph-capture-sizes 1 3 \
        --vllm-compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --vllm-gdn-prefill-backend triton \
        --no-vllm-breakable-cudagraph \
        --vllm-logging-level DEBUG \
        --vllm-enable-chunked-prefill \
        --vllm-max-num-batched-tokens "${BUDGET}" \
        --vllm-long-prefill-token-threshold "${LONG_PREFILL_THRESHOLD}"
