#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT=/home/michel/LLMs-tests/localLLMinference
readonly RUNNER="${ROOT}/operations/run-rtx5090-phase-d-benchmark.sh"
readonly MODEL=/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-e-kv-fp8-calibration-2026-09-20T052906Z/export
readonly VLLM=${AI_SERVER_PHASE_K_VLLM:-/home/michel/LLMs-tests/.phase-f1p5-vllm-nightly/bin/vllm}
readonly ALLOWED_GPU0_PID=${AI_SERVER_PHASE_D_ALLOWED_GPU0_PID:?set the approved unrelated GPU 0 compute PID}
readonly REQUEST_TIMEOUT=${AI_SERVER_PHASE_K_REQUEST_TIMEOUT:-600}

if [[ ${EUID} -ne 0 ]]; then
    printf '%s\n' "Run with sudo: AI_SERVER_PHASE_D_ALLOWED_GPU0_PID=<pid> sudo -E ${0}" >&2
    exit 1
fi
if ! [[ ${REQUEST_TIMEOUT} =~ ^[1-9][0-9]*$ ]] || [[ ${REQUEST_TIMEOUT} -gt 600 ]] || [[ ! -x ${RUNNER} || ! -x ${VLLM} || ! -f ${MODEL}/config.json ]]; then
    printf '%s\n' "Phase K timeout, runner, nightly vLLM, or calibrated model is not usable" >&2
    exit 1
fi

AI_SERVER_PHASE_D_ALLOWED_GPU0_PID=${ALLOWED_GPU0_PID} \
AI_SERVER_PHASE_D_MODEL=${MODEL} \
AI_SERVER_PHASE_D_RUN_PREFIX="phase-k-c2-admission" \
AI_SERVER_PHASE_D_VLLM=${VLLM} \
    "${RUNNER}" vllm \
        --cached-c3-c2-admission-only \
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
