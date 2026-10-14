# Qwen3.8-27B Q4 Dual-GPU Layer Campaign

Date: 2026-09-09

## Goal

Determine whether Qwen3.8-27B Q4_K_M gains enough long-context decode speed
and 262,144-token VRAM headroom from GPU1+GPU2 to replace the current
single-GPU Q4 service. The first campaign uses one full-capability agent only:
`--parallel 1`.

The primary decision metric is decode speed. Extra context capacity and faster
prefill are valuable only if the two-GPU profile remains operationally fast
enough to justify activating a second 300 W-capped RTX 3090.

## Scope

- Do not change the gateway, OpenCode routing, model aliases, or client config.
- Stop `freetoken-qwen3.8-flash-next-262k.service` for the campaign only. This
  is reversible by restarting that service; no routing removal occurs now.
- Stop the Q4 and Muse services while their GPUs are benchmarked.
- Keep GPU0 idle during the campaign. Do not move Muse until a Q4 candidate
  has passed all decision gates.
- On a future approved cutover, run Q4 on GPU1+GPU2, move Muse to GPU0, and
  decommission Flash as a separate reversible service/routing change.

Stopping Flash frees GPU0 VRAM and reduces its process, CPU, and host-memory
pressure. It does not add VRAM to the Q4 candidate, which uses GPU1+GPU2.

## Hardware Facts

- GPU1 and GPU2 are the closest pair (`PHB` topology); GPU0 to GPU1 is `NODE`.
- CUDA peer-to-peer is unsupported for every pair. Do not set
  `GGML_CUDA_P2P`.
- The current Q4 GGUF is
  `models/qwen3.8-27b-q4-k-m-gguf/Qwen3.8-27B-UD-Q4_K_M.gguf` (16 GiB).
- The current one-GPU server uses Q8 K/V, Flash Attention, `--parallel 1`, and
  `--ctx-size 196608`.
- Prior one-GPU Q4 measurements at 300 W reached 15.43 tok/s at 196,608 and
  10.16 tok/s at 262,144. Native context passed but had only about 452 MiB peak
  VRAM reserve.

## Layer Candidate

The first candidate uses pipeline/layer split because it supports Q8 KV and
Flash Attention while distributing both layers and their associated KV cache.

```text
CUDA_VISIBLE_DEVICES=1,2
CUDA_DEVICE_ORDER=PCI_BUS_ID
GGML_CUDA_ENABLE_UNIFIED_MEMORY=0

llama-server \
  --model <existing-Q4-GGUF> \
  --device CUDA0,CUDA1 \
  --split-mode layer \
  --tensor-split 1,1 \
  --n-gpu-layers all \
  --ctx-size 262144 \
  --parallel 1 \
  --kv-offload \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --flash-attn on \
  --threads 32 \
  --threads-batch 32 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --fit off \
  --metrics --perf --no-ui \
  --reasoning auto --reasoning-effort medium
```

`CUDA_VISIBLE_DEVICES=1,2` renumbers the selected physical GPUs to CUDA0 and
CUDA1 for the test process. The benchmark runner samples physical GPU1 and
GPU2 through `nvidia-smi`.

Layer split is expected to improve capacity and potentially prefill. It does
not guarantee lower single-request decode latency because layers remain serial
and this machine has no supported direct P2P path. The result, not an estimate,
decides whether it is production-worthy.

## Comparable Measurements

Run each request against a fresh server configured with `--ctx-size 262144`.
This measures the intended production allocation rather than a smaller
temporary allocation.

| Stage | Physical GPUs | Split | Prompt target | Completion | Purpose |
|---|---|---|---:|---:|---|
| Baseline | GPU1 | none | 4k | 128 | Short-context decode anchor |
| Baseline | GPU1 | none | 196k | 128 | Current practical long-context anchor |
| Baseline | GPU1 | none | 261.7k | 128 | Native-context capacity anchor |
| Candidate | GPU1+GPU2 | layer | 4k | 128 | Short-context decode comparison |
| Candidate | GPU1+GPU2 | layer | 196k | 128 | Main speed decision |
| Candidate | GPU1+GPU2 | layer | 261.7k | 128 | Native-context capacity decision |

Prompts are deterministic repeated text, adjusted through `/tokenize`, with
thinking disabled, EOS ignored, and 128 generated tokens. That preserves the
prior long-context methodology while avoiding reasoning-length variation.

The first campaign holds every point at a 262,144 allocation, so its baseline
measures the native-context penalty rather than the deployed 196,608 profile.
After it completes, run a separate GPU1-only deployed-profile baseline with:

```bash
sudo systemd-run \
  --unit=qwen-q4-single-gpu-196k-baseline \
  --collect \
  --no-block \
  ${EXPERIMENT_ROOT}/operations/run-qwen-q4-dual-gpu-layer-benchmark.sh \
  --ctx-size 196608 \
  --modes baseline \
  --prompt-tokens 4096 196224
```

This uses the same build, Q8 KV, Flash Attention, physical GPU1, request shape,
and 300 W cap as the layer campaign while matching the live Q4 allocation. It
is the authoritative performance comparison for the production decision.

For every point record:

- Server-returned prefill and decode tok/s, wall time, and complete timing data.
- Startup log, model residency, allocated context, and any NCCL warning.
- Per-GPU peak VRAM, peak/mean watts, temperature, and utilization for all
  participating GPUs.
- CPU RAPL power and energy from the CPU profiler.
- Integrated GPU+CPU joules and joules per generated token.
- Errors and service restoration status.

## Decision Gates

1. The candidate must keep all Q4 layers and KV on GPU1+GPU2; no CPU fallback
   or unified-memory fallback is a scored result.
2. The 262,144 point must generate all 128 tokens without OOM or service error,
   with materially more VRAM reserve than the one-GPU result.
3. The candidate must improve, not merely match, the one-GPU 196k decode result
   by enough to justify its measured extra energy. A capacity-only gain does not
   meet the primary goal of maximizing speed.
4. No thermal, correctness, or sustained-power issue may appear in the server
   log or measurements.

If layer split fails the speed gate, retain the one-GPU Q4 service and do not
move Muse or change routing.

## Tensor Follow-up

No download is required for tensor mode. F16/BF16 KV is a runtime cache type;
the existing Q4 GGUF remains the model file.

Tensor split is deferred until the layer result is available. It may reduce
single-request decode latency, but requires Flash Attention enabled and F16 or
BF16 KV (Q8 KV is unsupported). Start only at 196k, without `--fit`, and only
attempt 262k after the startup log confirms a healthy memory reserve. Do not
enable P2P on this topology. Treat a missing-NCCL warning or degraded decode as
a negative result.

## Concurrency Deferred

Do not test `--parallel 2` in the first campaign. The strict requirement is at
least 196,608 tokens for each agent, which requires a 393,216-token total pool
and an explicit server-log verification of 196,608 context per slot. A 327,680
shared pool does not meet that requirement.

## Headless Execution and Recovery

The campaign must run through a root `systemd-run` transient unit, not an SSH
terminal. `tools/qwen_q4_dual_gpu_layer_benchmark.py` writes a timestamped
result JSON after every completed point and retains separate server logs and
raw GPU/CPU samples. A network drop therefore loses at most the in-flight
point.

Before launching, the wrapper records whether Flash, Q4, and Muse were active.
It stops them, runs the temporary benchmark servers as `<operator>`, and restores
the recorded service state on exit or error. No gateway/OpenCode file changes
are part of this campaign.

The implemented root wrapper is
`operations/run-qwen-q4-dual-gpu-layer-benchmark.sh`. Launch it from an
authenticated terminal with:

```bash
sudo systemd-run \
  --unit=qwen-q4-dual-gpu-layer-benchmark \
  --collect \
  --no-block \
  ${EXPERIMENT_ROOT}/operations/run-qwen-q4-dual-gpu-layer-benchmark.sh
```

Follow it after reconnecting with:

```bash
systemctl status qwen-q4-dual-gpu-layer-benchmark.service
journalctl -fu qwen-q4-dual-gpu-layer-benchmark.service
```

The wrapper records the original state of Flash, Q4, and Muse, stops them for
the test, then restores those states on normal exit, error, or termination. No
gateway/OpenCode files are changed. The existing `tools/context_speed_sweep.py`
is the implementation base; the dual-GPU runner extends its fresh-server
cleanup and per-point persistence behavior with selected-GPU and CPU-RAPL
sampling.

## Post-Campaign Actions

- Publish a result record with the exact commands, logs, and metrics.
- Request explicit approval before changing the Q4 systemd service.
- If approved, update monitoring/accounting so Q4 activity and energy include
  both GPU1 and GPU2 before the production cutover.
- Move Muse to GPU0 only after the successful Q4 decision and separate service
  validation.
