#!/usr/bin/env bash
# Explicitly configured benchmark launcher; service checks are read-only.
set -Eeuo pipefail
root=${HYPERQWEN_ROOT:?Set HYPERQWEN_ROOT to the prepared pinned HyperQwen checkout}
mode=${1:-}
case "$mode" in batch|mtp|dflash|dflash-wide|int4-dflash) ;; *) printf 'Usage: %s {batch|mtp|dflash|dflash-wide|int4-dflash}\n' "$0" >&2; exit 2 ;; esac

for unit in llamampere-qwen3.8-atx-iq4xs-m-gpu1.service llamampere-qwen3.8-atx-iq4xs-m-gpu2.service; do
    [[ $(systemctl is-active "$unit") == active ]] || { printf 'Protected service unavailable: %s\n' "$unit" >&2; exit 1; }
done
[[ $(systemctl is-active llama-muse-glimmer-30b-131k.service) != active ]] || {
    printf 'Muse still owns GPU0; a manual handoff is required.\n' >&2
    exit 1
}
owners=$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z ${owners//[[:space:]]/} ]] || { printf 'GPU0 still has compute owners: %s\n' "$owners" >&2; exit 1; }
used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits)
[[ $used =~ ^[0-9]+$ && $used -le 1024 ]] || {
    printf 'GPU0 reports %s MiB used; refusing launch.\n' "$used" >&2; exit 1
}
[[ -z $(ss -H -ltn 'sport = :18120') ]] || { printf 'Port 18120 is already bound.\n' >&2; exit 1; }
cd "$root"
[[ ! -e "$root/api_key.txt" ]] || { printf 'Trial requires keyless loopback; unexpected api_key.txt exists.\n' >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID HOST=127.0.0.1
export CUDA_HOME="$root/venv/lib/python3.12/site-packages/nvidia/cu13"
export MODEL="$root/models/Qwen3.8-27B-W4A16-AutoRound-fast"
export PORT=18120 MAX_LEN=262144 MAX_SEQS=1 REQ_METRICS=0
export PREFIX_CACHE=0 EXTRA_ARGS=--no-enable-prefix-caching
unset VLLM_API_KEY
if [[ $mode == batch ]]; then
    export KV=kvarn INT8_ACT= GPU_UTIL=0.93
    export EXTRA_ARGS='--no-enable-prefix-caching --enable-per-request-metrics --enable-force-include-usage'
    exec bash batch/start_qwen.sh
fi
if [[ $mode == int4-dflash ]]; then
    export DRAFT="$root/models/Qwen3.8-27B-DFlash2-W4A16"
    export SPEC=dflash2 DFLASH_TOKENS=7 GPU_UTIL=0.95 REQ_METRICS=1
    exec bash single-user/alternative.sh
fi
export CTX=huge SPEC=mtp GPU_UTIL=0.93 REQ_METRICS=1
if [[ $mode == dflash || $mode == dflash-wide ]]; then
    export SPEC=dflash2 DFLASH_TOKENS=7
fi
if [[ $mode == dflash-wide ]]; then
    export DFLASH_TOKENS=15 KV_MEM=5920000000 CUDAGRAPH_MODE=PIECEWISE
fi
exec bash single-user/start_qwen.sh
