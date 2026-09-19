#!/usr/bin/env bash
set -Eeuo pipefail

readonly PROFILE_FILE=/etc/ai-server/qwen-serving-profile
readonly PROFILE=$(cat "${PROFILE_FILE}" 2>/dev/null || true)

[[ ${PROFILE} == atx-dual || ${PROFILE} == stock-q4-tensor ]] || {
    printf '%s\n' 'No valid active Qwen profile is selected.' >&2
    exit 1
}

# Backend health is intentionally reported by /readyz rather than blocking
# systemd startup. Model loading can take several minutes, and a failed Muse
# backend must not prevent the gateway from serving health and diagnostics.
printf 'Qwen serving profile is valid: %s\n' "${PROFILE}"
