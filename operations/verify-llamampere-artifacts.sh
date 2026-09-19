#!/usr/bin/env bash
set -Eeuo pipefail

readonly RELEASE=/opt/ai-server/llamampere/36a6bca817
readonly BINARY=${RELEASE}/build-sm86/bin/llama-server
readonly MODEL=/opt/ai-server/models/qwen3.8-27b-atx-iq4_xs_m/Qwen3.8-27B-ATX-4-XS.gguf
readonly VOCAB=${RELEASE}/docs/mtp-vocab/atx_65536.txt

readonly BINARY_SHA256=bcee5b8939763da8f87e87d77325142d9b2d4b686036bb4450d3c4754d77ec46
readonly MODEL_SHA256=5cf05ad901dcaa76f41db13a5629146ed882219339377a80e37b12a8528d963b
readonly VOCAB_SHA256=8405ff0f8970da24b72d68c9e61fa4309fdcf3719ecc1d7626f0ae8bd2a8e326

verify_hash() {
    local path=$1
    local expected=$2
    local label=$3
    local actual

    [[ -r ${path} ]] || { printf 'Missing unreadable artifact: %s\n' "${path}" >&2; return 1; }
    actual=$(sha256sum "${path}" | cut -d' ' -f1)
    [[ ${actual} == "${expected}" ]] || {
        printf '%s hash mismatch: expected %s, got %s\n' "${label}" "${expected}" "${actual}" >&2
        return 1
    }
}

verify_hash "${BINARY}" "${BINARY_SHA256}" runtime
verify_hash "${MODEL}" "${MODEL_SHA256}" model
verify_hash "${VOCAB}" "${VOCAB_SHA256}" vocabulary
printf '%s\n' 'llamAmpere runtime, model, and vocabulary hashes verified.'
