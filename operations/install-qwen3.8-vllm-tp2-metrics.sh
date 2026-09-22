#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT=/home/michel/LLMs-tests/localLLMinference
readonly ENV=/etc/ai-server/ai-qwen3.8-vllm-tp2-metrics.env
readonly UNIT=ai-qwen3.8-vllm-tp2-metrics.service

[[ ${EUID} -eq 0 ]] || { printf '%s\n' "Run with sudo: $0" >&2; exit 1; }
[[ -x /usr/bin/python3 && -x /usr/bin/tailscale ]] || { printf '%s\n' 'Python or Tailscale is unavailable.' >&2; exit 1; }
systemctl is-active --quiet ai-qwen3.8-vllm-tp2.service || { printf '%s\n' 'The vLLM backend is not active.' >&2; exit 1; }

listen_host=$(/usr/bin/tailscale ip -4)
[[ -n ${listen_host} ]] || { printf '%s\n' 'No Tailscale IPv4 address is available.' >&2; exit 1; }
install -d -m 0750 -o root -g root /etc/ai-server
install -m 0644 -o root -g root "${ROOT}/operations/systemd/${UNIT}" "/etc/systemd/system/${UNIT}"
umask 077
printf '%s\n' "AI_SERVING_PROFILE=rtx5090-vllm-tp2" "AI_METRICS_HOST_ID=monstera-rtx5090" "AI_METRICS_LISTEN_HOST=${listen_host}" "AI_GATEWAY_METRICS_URL=http://127.0.0.1:8080/metrics" 'AI_METRICS_BACKENDS={"qwen3_8_vllm_tp2":{"url":"http://127.0.0.1:18081/metrics","model":"qwen3.8-27b-q4-gpukv-native","gpus":["0","1"],"runtime":"vllm","cached_input_mode":"observed"}}' > "${ENV}"
chown root:root "${ENV}"
chmod 0600 "${ENV}"
systemctl daemon-reload
systemctl enable "${UNIT}"
systemctl restart "${UNIT}"
for _ in {1..10}; do
  if curl --fail --silent "http://${listen_host}:9108/metrics" >/dev/null; then
    printf '%s\n' "Exporter is listening on ${listen_host}:9108 for Tailscale scraping."
    exit 0
  fi
  sleep 1
done
systemctl --no-pager status "${UNIT}" >&2
exit 1
