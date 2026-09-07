# Large MoE Three-GPU Probe Plan

Date: 2026-09-04

## Objective

Test two complementary large GGUF MoE candidates on all three RTX 3090s:

1. DeepSeek-V4-Flash-0731 `UD-IQ3_XXS` from Unsloth, four GGUF shards totaling
   approximately 104 GB. The first shard is a 5.3 MB metadata/index shard;
   shards 2 and 3 are approximately 49.9 GB and 49.3 GB, and shard 4 is 5.0 GB.
2. MiniMax-M2.5 `IQ3_S` from marksverdhei, one GGUF of approximately 98.7 GB.

This is an initial runtime and capacity probe, not a claim that either model is
ready for production or that publisher benchmark scores transfer to these exact
GGUFs.

## Current Hardware Gate

- Three RTX 3090 GPUs, 24,576 MiB each; GPU 0 is currently the Qwen production
  service and must be stopped before an all-three-GPU test.
- Live PCIe state: GPU 0 Gen3 x4, GPUs 1 and 2 Gen3 x8. GPU 0 is therefore the
  communication-constrained card.
- Host RAM: 141 GiB total; swap is 8 GiB. The completed long runs used roughly
  1.8-2.5 GiB of swap, so swap activity is included in the performance results.
- Root disk: approximately 505 GiB available after migration. The WDC 500 GB
  SATA disk is formatted ext4 and mounted at `/mnt/model-store`, with
  approximately 246 GiB available after storing both models. The fstab entry is
  `UUID=2e8702db-bd37-42f2-a4aa-51c398cd5111 /mnt/model-store ext4 defaults,nofail 0 2`.
- Current power limits are GPU 0 = 300 W, GPU 1 = 275 W, GPU 2 = 350 W. Do not
  silently normalize these during the first probe; record them. A later power
  comparison must use a deliberate common cap.
- No NVLink; topology reports GPUs 1 and 2 on a PHB path and GPU 0 on NODE paths.

## Runtime Gate

The installed llama.cpp binary exposes `--cpu-moe`, `--n-cpu-moe`,
`--tensor-split`, `--fit`, `--fit-target`, `--fit-ctx`, and `--no-kv-offload`.
Current llama.cpp documentation lists `deepseek4` and `minimax-m2` GGUF
architectures and documents CPU expert placement. The exact checkpoint still
requires a load test and output sanity check.

Use explicit placement with `--fit off`; do not use `--fit` as the scored
configuration. Start with `--n-cpu-moe` values chosen to leave model weights and
runtime buffers inside VRAM, then vary the count only if startup proves safe.
The initial harness records startup logs, per-GPU allocation, temperature, power,
PCIe state, host memory, and generation timings.

Initial command shape:

```text
CUDA_VISIBLE_DEVICES=0,1,2 llama-server \
  --n-gpu-layers 99 --fit off --tensor-split 0.5,1,1 \
  --n-cpu-moe N --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn on --ctx-size 8192
```

The `0.5,1,1` split is a starting hypothesis because GPU 0 is Gen3 x4, not a
performance conclusion. Compare with `1,1,1` only after a safe load exists.

## DeepSeek Follow-up Performance Sweep

The initial `0.5,1,1` / `--n-cpu-moe 24` configuration is a capacity-safe
baseline, not a utilization optimum. Its logs show about 53 GiB of CPU-mapped
weights and a serial layer pipeline: GPU 2 and then GPU 0 are saturated during
decode while GPU 1 is mostly idle. This is expected for `--split-mode layer`,
but the uneven work means it merits tuning.

The fixed prompt/decode sweep completed with these load-safe candidates. Cases
that loaded successfully but caused swap activity are retained and marked in
the JSON rather than treated as equivalent measurements:

1. `--split-mode layer --tensor-split 1,1,1 --n-cpu-moe 24`
2. `--split-mode layer --tensor-split 1,1,1 --n-cpu-moe 20`
3. `--split-mode layer --tensor-split 1,1,1 --n-cpu-moe 16`
4. `--split-mode row --tensor-split 1,1,1 --main-gpu 1` comparisons at CPU-MoE
   values 24 and 20.

Do not infer performance from the initial 8k/32k/128k smoke values: each used
only a 17-token prompt and different short completions, so the small increase
from 6.64 to 7.62 tok/s is sample variation and warm page-cache state, not a
context-length benefit. Collect per-second `nvidia-smi dmon` telemetry during
each candidate instead.

`tools/deepseek_performance_sweep.py` implements this as a controlled 8k-context
test: approximately 4k model-token prefill followed by a fixed 256-token decode,
with per-second GPU utilization/power/clock samples and `/proc/vmstat` swap-I/O
counts. `tools/run_deepseek_performance_sweep.sh` is detached, waits for the
reviewer to exit, and holds `deepseek-test.lock` so download finalization cannot
move the active GGUF files.

Measured DeepSeek results on the fixed approximately 4k-prefill/256-decode
workload:

| Placement | Decode tok/s |
|---|---:|
| `0.5,1,1`, CPU-MoE 24 | 9.47 |
| `1,1,1`, CPU-MoE 24 | 11.50 |
| `1,1,1`, CPU-MoE 20 | 11.60 |
| `0.9,1,1`, CPU-MoE 20 | 12.77 |
| `1.1,1,1`, CPU-MoE 20 | 13.09 |
| `1.2,1,1`, CPU-MoE 20 | **13.26** |

The best measured two-GPU cases were 6.44 tok/s on GPUs 1+2 and 6.03 tok/s
on GPUs 0+2. Row split was rejected by the build/model with
`device CUDA0 does not support split buffers`.

MiniMax completed 8k, 32k, and 128k startup probes with all three GPUs and
CPU-MoE 20. Its short generation measured 1.96, 1.94, and 1.61 tok/s
respectively; the first two responses exhausted the reasoning budget, while
the 128k response returned valid content. On the fixed workload, the best
MiniMax result was 9.76 tok/s at `1.2,1,1`, CPU-MoE 20. CPU-MoE 24 reached
9.32 tok/s at that split and produced substantially more swap activity.

## DDR4-2933 No-Swap Follow-up

On 2026-09-05, host memory was increased and the supplied DMI output reported
`Configured Memory Speed: 2933 MT/s` for all eight DIMMs. Swap was disabled
again with `sudo swapoff -a`, and Qwen remained stopped. The exact llama.cpp
invocation and fixed approximately 4k-prefill/256-decode workload were reused;
only the output directory and telemetry handling differed.

Three repetitions produced:

| Run | Prefill tok/s | Decode tok/s | Workload wall s | pswpout |
|---:|---:|---:|---:|---:|
| 1 | 66.3385 | 8.2539 | 97.43 | 0 |
| 2 | 66.4701 | 12.2644 | 83.65 | 0 |
| 3 | 70.2291 | 12.3682 | 80.17 | 0 |

The median decode rate was **12.2644 tok/s**. Compared with the no-swap
DDR4-2666 median of `12.1067 tok/s`, this is **+1.302%**. It remains **7.508%
below** the original `13.26 tok/s` reference. The first run was a cold-start
outlier, so the median is retained rather than selecting a favorable run. The
small 2933-versus-2666 difference is not enough to establish a large
RAM-frequency benefit.

## Recent Build Expert Placement and Tensor Mode

An isolated current llama.cpp build at `4d917609` was compiled with CUDA 13.2
and NCCL `2.30.4`. No existing llama.cpp binary or Qwen service configuration
was changed. The earlier harness mistakenly enabled CUDA unified memory by
exporting `GGML_CUDA_ENABLE_UNIFIED_MEMORY=0`; new runs remove that variable
and use physical VRAM allocations.

Manual `--override-tensor` placement assigned DeepSeek4 expert layers `0-7` to
GPU0, `8-16` to GPU1, `17-25` to GPU2, and `26-42` to CPU. It passed three
deterministic output checks in every run, peaked around `21.9 / 21.2 / 21.3
GiB` of VRAM, generated no swap I/O, and measured three-run medians of
**17.7190 tok/s decode** and **76.5089 tok/s prefill**.

The same manual mapping at a 32k allocated context, with the fixed 4k-token
benchmark prompt, measured medians of **17.6697 tok/s decode** and **76.4502
tok/s prefill**. This is within run variation versus 8k and added only about
`30 / 98 / 94 MiB` of peak VRAM on GPUs 0/1/2. It is a 32k allocation result,
not a fully populated 32k-context performance result.

NCCL-backed `--split-mode tensor`, tested separately with equal tensor split,
CPU-MoE 22, and f16 K/V cache, also passed all checks. Its medians were
**13.5994 tok/s decode** and **128.1106 tok/s prefill**. Manual layer placement
is the low-concurrency decode choice on this x4/x8/x8, no-NVLink topology;
tensor mode is viable when prefill dominates.

## Execution Order

1. Finish and checksum the DeepSeek shards; verify all four files are present.
2. Download and checksum MiniMax IQ3_S.
3. Stop only the GPU 0 Qwen service for the all-three-GPU test; preserve its
   configuration and keep it stopped between the DeepSeek and MiniMax tests.
   Restart it only after both candidate tests are complete.
4. For each model, try one explicit hybrid placement at 8k with no prompt cache:
   DeepSeek `--n-cpu-moe 24`, MiniMax `--n-cpu-moe 20`. These are probe values,
   not assumed optimal placements.
5. If startup succeeds, send a short chat completion and inspect the output for
   template, repetition, truncation, and obvious corruption.
6. Record cold startup time, prefill tok/s, decode tok/s, completion finish
   reason, resident VRAM, CPU model buffers, RSS, swap, power, temperature, and
   PCIe width/gen.
7. Only for a successful 8k smoke test, repeat at 32k and 128k. A model that
   exceeds host RAM, swaps, or becomes unstable fails the capacity gate.
8. Run the direct `bash` / `submit_review` adversarial reviewer protocol only
    after runtime correctness is established. Use the same four hidden-key cases
    and the same 30-step budget used for IQ3_S. The DeepSeek attempt was
    operationally incomplete: after approximately 52 minutes and nine model
    turns it had not submitted case1, so no quality score was assigned.

## Acceptance Gates

- Load: all required GGUF shards discovered; no model-load error.
- Memory: no swap, no disk paging during decode, and host RSS plus GPU usage
  remain measurable with reserve.
- Runtime: one short natural-EOS response with valid chat formatting; no
  unsupported-template or tool-call parser failure.
- Throughput: report the measured result; the planning target is >=10 tok/s,
  while >=15 tok/s is the preferred occasional-planner threshold. Neither is a
  pass/fail quality gate by itself.
- Quality: do not infer reviewer capability from throughput. Use the separate
  repository-aware review screen and score hidden and independently verified
  objections.

## Artifacts

- Harness: `tools/large_moe_probe.py`
- Set `MODEL_STORE=/mnt/model-store/models` when the completed model directories
  have been moved there; the default remains the existing project `models/`
  directory.
- `tools/finalize_large_model_downloads.sh` waits for the exact expected sizes,
  verifies all five SHA-256 values, then waits for all `llama-server` processes
  to exit before moving both model directories to `/mnt/model-store/models`.
- `tools/large_moe_agentic_reviewer.py` runs the four-case direct
  `bash`/`submit_review` screen against DeepSeek at 128k with the same 30-step
  budget used for the Mistral comparison.
- `tools/run_deepseek_agentic_reviewer_bounded.sh` is the reduced 32k-context
  retry with a 4096-token response cap; it was stopped after the same long
  case1 behavior and its partial artifacts are in
  `research/large-moe-3gpu-probe-2026-09-04/deepseek-agentic-bounded/`.
- DeepSeek download directory:
  `/mnt/model-store/models/deepseek-v4-flash-0731-iq3-xxs-gguf/`
- MiniMax download directory: `/mnt/model-store/models/minimax-m2.5-iq3-s-gguf/`
- Probe output directory:
  `research/large-moe-3gpu-probe-2026-09-04/`
- Fine and MiniMax launchers:
  `tools/run_deepseek_fine_sweep.sh`, `tools/run_minimax_probe.sh`,
  `tools/run_minimax_performance_sweep.sh`, and
  `tools/run_minimax_validation_sweep.sh`.
- DDR4-2666 reference comparison:
  `research/large-moe-3gpu-probe-2026-09-04/ram2666-reference.md`.
- Corrected no-swap DDR4-2666 comparison:
  `research/large-moe-3gpu-probe-2026-09-04/ram2666-noswap-reference.md`.
- DDR4-2933 no-swap follow-up:
  `research/large-moe-3gpu-probe-2026-09-04/ram2933-noswap-reference.md`.
- Recent-build expert-placement and NCCL tensor experiment:
  `research/large-moe-3gpu-probe-2026-09-04/recent-build-expert-placement-and-tensor-nccl.md`.
- Follow-up optimization experiment plan:
  `research/deepseek-v4-flash-optimization-plan-2026-09-05.md`.

## Download Integrity

The source Hub metadata exposes LFS SHA-256 object IDs. Verify them after each
download before any model is loaded:

| File | SHA-256 |
|---|---|
| DeepSeek shard 1 | `dec1cee704800267d9d836d5a61aefc33705be939bbb3058fa9006d98191576d` |
| DeepSeek shard 2 | `3064d3c4c1d6363e9f9ad88e90a3e2c5fb2d6f7ae16ca72135c3ce6a5c984da5` |
| DeepSeek shard 3 | `2e9b2732eca7da8324f731653624a4f5c9846258926fd9f468cc703afb51a019` |
| DeepSeek shard 4 | `4ca79d8e5107dd1b9bb57b176a7c09948837425dee49f0f1dfd6547a3769fea7` |
| MiniMax IQ3_S | `86e03e372617f2335dbe490e1d2cf9d097e3623d027304e050e13b95aef46bbd` |
