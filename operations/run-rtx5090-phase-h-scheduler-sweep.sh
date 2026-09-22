#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT=/home/michel/LLMs-tests/localLLMinference
readonly RUNNER="${ROOT}/operations/run-rtx5090-phase-d-benchmark.sh"
readonly MODEL=/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-e-kv-fp8-calibration-2026-09-20T052906Z/export
readonly VLLM=${AI_SERVER_PHASE_H_VLLM:-/home/michel/LLMs-tests/.phase-f1p5-vllm-nightly/bin/vllm}
readonly ALLOWED_GPU0_PID=${AI_SERVER_PHASE_D_ALLOWED_GPU0_PID:?set the approved unrelated GPU 0 compute PID}

if [[ ${EUID} -ne 0 ]]; then
    printf '%s\n' "Run with sudo: AI_SERVER_PHASE_D_ALLOWED_GPU0_PID=<pid> sudo -E ${0}" >&2
    exit 1
fi
if [[ ! -x ${RUNNER} || ! -x ${VLLM} || ! -f ${MODEL}/config.json ]]; then
    printf '%s\n' "Phase H runner, nightly vLLM, or calibrated model is not usable" >&2
    exit 1
fi

readonly -a GRAPH_B0_C3=(
    --decode-only
    --no-decode-warm-prefixes
    --context-tokens 262144
    --gpu-memory-utilization 0.90
    --max-num-seqs 3
    --decode-prompt-tokens 196000
    --decode-output-tokens 4096
    --decode-concurrency 3
    --vllm-attention-config '{"backend":"TRITON_ATTN","use_trtllm_attention":false}'
    --no-vllm-enable-prefix-caching
    --vllm-enable-cuda-graph
    --vllm-cudagraph-metrics
    --vllm-cudagraph-capture-sizes 1 3
    --vllm-compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --vllm-gdn-prefill-backend triton
    --no-vllm-breakable-cudagraph
    --vllm-logging-level DEBUG
    --vllm-enable-chunked-prefill
)

for budget in 2048 4096 8192 16384; do
    AI_SERVER_PHASE_D_ALLOWED_GPU0_PID=${ALLOWED_GPU0_PID} \
    AI_SERVER_PHASE_D_MODEL=${MODEL} \
    AI_SERVER_PHASE_D_RUN_PREFIX="phase-h-graph-b0-c3-budget${budget}" \
    AI_SERVER_PHASE_D_VLLM=${VLLM} \
        "${RUNNER}" vllm "${GRAPH_B0_C3[@]}" --vllm-max-num-batched-tokens "${budget}"
done
