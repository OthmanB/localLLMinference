# RTX 5090 Hardware Profile

This is an opt-in standalone profile for a host with an RTX 5090 at physical
GPU 0. It is separate from the canonical three-GPU RTX 3090 profile and from
the `atx-dual` gateway profile.

## Status

- Profile type: standalone llama.cpp reference profile.
- Model: Qwen3.8-27B UD Q4_K_M GGUF.
- Context setting: 262,144 tokens.
- Attention KV: Q8_0 K/V.
- Runtime: stock/local llama.cpp; the binary and commit are host inputs.
- Endpoint: loopback `127.0.0.1:8080`.
- Gateway: not installed or configured by this profile.
- Validation: Phase B/C/D research is complete. The operator-approved 85 C
  Q4/Q8 two-replica sustained rerun passed at a 79 C GPU 0 peak. Production
  approval remains open.

This profile is not a production recommendation and is not part of the current
Qwen profile switcher. It exists to preserve the local RTX 5090 deployment
knowledge without adding a third value to a switcher that assumes physical
GPUs 1 and 2 on the canonical host.

## Known Hardware Scope

The assessed workstation has two RTX 5090 cards, approximately 32 GiB each,
with an asymmetric PCIe topology: GPU 0 is x16 and GPU 1 is x8 at its upstream
port. NVIDIA reports no usable GPU peer reads or writes between the cards. Do
not infer a transparent 64 GiB pool or assume that tensor parallelism wins.

This standalone profile intentionally uses one GPU. Two-GPU serving candidates
belong in the dated research plan and must be benchmarked separately.

## Model and Runtime Inputs

The installer defaults are host-specific examples and must be reviewed before
use:

```text
AI_SERVER_USER
AI_SERVER_LLAMA_CPP_ROOT
AI_SERVER_LLAMA_SERVER_BIN
AI_SERVER_MODEL_PATH
AI_SERVER_PYTHON
AI_SERVER_LOG_DIR
AI_SERVER_RTX5090_GPU
AI_SERVER_RTX5090_PORT
```

The model file must match the manifest entry in `models/README.md`. The runtime
binary must be identified by its source commit or a separately recorded build
hash before production use.

## Installation

The installer is deliberately guarded. It does not disable unrelated systemd
units, overwrite the canonical profile switcher, or accept a health response
from an unknown listener as model readiness.

It refuses to proceed when the selected GPU is not an RTX 5090, when another
compute process owns it, when port 8080 is occupied, or when the canonical
global GPU policy or a known GPU 0 model service is active or enabled.

Review the files and set the explicit confirmation value only after approving
the resource ownership and power policy:

```bash
sudo env \
  AI_SERVER_RTX5090_CONFIRM=install \
  AI_SERVER_MODEL_PATH=/path/to/Qwen3.8-27B-UD-Q4_K_M.gguf \
  /path/to/localLLMinference/operations/hardware/rtx5090/install-native-q4.sh
```

The script installs and starts only the three RTX 5090 profile units. It does
not run as part of `operations/install.sh` and is not invoked by
`qwen-profile-switch.sh`.

The profile service names are intentionally distinct:

```text
ai-rtx5090-gpu0-policy.service
ai-rtx5090-qwen3.8-q4-native.service
ai-rtx5090-monitor.service
```

## Runtime Details

- GPU selection is explicit and defaults to physical GPU 0.
- Model server and monitor use loopback only.
- The profile uses 500 W and a requested 95% fan speed, subject to board,
  cooling, and power-supply approval.
- The historical 262k test reached 79 C against an 80 C stop threshold. The
  cooling issue is considered resolved by the current host policy; the Phase 1
  controlled run peaked at 56 C with the 95% fan request. The operator-approved
  controlled stop threshold is 85 C, not an expected operating condition.
- The service does not set `GGML_CUDA_ENABLE_UNIFIED_MEMORY`. In the assessed
  llama.cpp revision, setting it to `0` still enables managed allocation because
  the runtime checks only whether the variable exists.
- Host prompt-cache snapshots and system RAM use remain possible even with GPU
  KV offload. They are distinct from CUDA managed memory.

## Logs and Monitoring

The profile uses the configured log directory and a profile-specific rule:

```text
qwen3.8-q4-native-rtx5090.server.log
qwen3.8-q4-native-rtx5090.metrics.jsonl
qwen3.8-q4-native-rtx5090.monitor.log
```

The JSONL monitor is lightweight troubleshooting telemetry. It does not replace
the canonical Prometheus exporter, token accounting, gateway metrics, or Grafana
dashboard.

## Stop and Remove

The installer does not provide an automatic rollback because this profile is
not part of the canonical profile switcher. Stop only these units, then remove
their installed files after confirming no requests are active:

```bash
sudo systemctl disable --now \
  ai-rtx5090-monitor.service \
  ai-rtx5090-qwen3.8-q4-native.service \
  ai-rtx5090-gpu0-policy.service
```

Do not stop or disable `nvidia-power-limit.service`,
`nvidia-fan-control.service`, Muse, or a canonical Qwen service as a side effect
of using this profile. If those services own the resource, resolve the topology
explicitly first.

## Validation Required Before Production Use

- Verify the exact GPU UUID, board, power limit, fan behavior, and PSU margin.
- Re-run the baseline with managed-memory variable absent.
- Verify normal EOS, thinking, tool output, structured output, and long-context
  retrieval.
- Measure cold and warm prefill, decode, TTFT, ITL, and sustained temperature.
- Test one and two slots with latency-qualified concurrency criteria.
- Keep the endpoint loopback-only or put it behind the authenticated gateway.
- Record the runtime commit, model hash, host state, and raw telemetry.

The controlled Phase 1 baseline artifact is outside this repository at:
`/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase1-q4-gpu1-managed-memory-absent-2026-09-20-final/`.
