#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT=/home/obenomar/localLLMinference
readonly SWITCH=/usr/local/sbin/ai-qwen-profile-switch

if [[ ${EUID} -ne 0 ]]; then
    printf '%s\n' "Run as root: sudo ${ROOT}/operations/promote-muse-glimmer-30b-131k.sh" >&2
    exit 1
fi

if [[ ! -x ${SWITCH} ]]; then
    printf '%s\n' "${SWITCH} is not installed. Run operations/install.sh first." >&2
    exit 1
fi

printf '%s\n' 'Muse is already part of both supported Qwen profiles and is not promoted by restarting stock Qwen.'
