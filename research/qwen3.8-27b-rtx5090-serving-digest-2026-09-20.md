# Qwen3.8-27B on 2x RTX 5090 — Single-File Serving Digest

Date: 2026-09-20
Purpose: self-contained summary of everything tested on this workstation, with
the measured numbers, the current blocker, and the untested improvement levers.
Intended to be shared as a single document. All numbers are local measurements
from the cited runs, not published recipe claims.

## 1. Hardware

| Component | Value |
|---|---|
| GPUs | 2x NVIDIA RTX 5090, 32 GiB each, SM120 (Blackwell, consumer) |
| GPU interconnect | **No NVLink. PCIe only. NVIDIA reports peer read/write unsupported.** Do not treat as a 64 GiB pool. |
| PCIe | GPU0 Gen5 x16, GPU1 Gen5 x8 (at upstream port); idle speed not a loaded-link verdict |
| CPU/RAM | AMD Threadripper PRO 7995WX (96c/192t), 672 GiB RAM (7/8 DIMMs, not full 8-channel) |
| Driver/toolkit | NVIDIA 610.43.03, CUDA toolkit 13.3 |
| Power caps | GPU0 500 W, GPU1 575 W (unequal — flagged as a confound, not yet equalized) |
| Thermal policy | 80 C historical cutoff; operator-approved 85 C defensive stop for the benchmark window |

## 2. Model

Qwen3.8-27B: hybrid architecture, 64 layers = 16 full-attention + 48 Gated
DeltaNet (linear attention) layers. Attention uses 4 KV heads x 256 head_dim.
Native context 262,144 tokens. Has an embedded MTP (multi-token prediction)
head (`mtp_num_hidden_layers=1`), usable for speculative decoding in vLLM
(`qwen3_5_mtp`) and llama.cpp (`--spec-type draft-mtp`).

Attention KV size per full 262,144-token context: BF16 ~16 GiB, FP8 ~8 GiB,
llama.cpp Q8_0 ~8.5 GiB. This is the binding memory constraint.

## 3. What is live right now

- Service: `llama-qwen3.8-q4-native.service`, llama.cpp commit
  `b96806d96061049a5b574269b049bf62d46` built for `sm_120a`.
- Model: `Qwen3.8-27B-UD-Q4_K_M.gguf` (Unsloth export), Q8_0 K/V, Flash
  Attention, all layers on **GPU 0 only. GPU 1 effectively unused.**
- One 262,144-token slot. **No speculative decoding; the embedded MTP tensors
  are not used.**
- Known open issues: live unit exports `GGML_CUDA_ENABLE_UNIFIED_MEMORY=0`
  (presence, not value, selects `cudaMallocManaged` in this revision — must be
  absent for the plain allocator) and binds `0.0.0.0:8080` unauthenticated.
  Neither has been fixed on the live unit yet.

## 4. Candidates tested (all 2026-09-20, guarded maintenance window, verified rollback)

Checkpoint for the TP2 engines: local RadixArk ModelOpt NVFP4/FP8 mixed
checkpoint (source Qwen BF16 rev `e13a4f0e35203116364e3b3f3f0c82f6ef1afd3c`),
`/home/michel/LLMs-tests/rtx5090-qwen38-bench/models/Qwen3.8-27B-NVFP4`.
Its FP8 KV metadata had **no calibrated scales**, so engines defaulted the KV
scale to 1.0 — a known quality risk. Phase E produced a calibrated export
(1,024 CNN/DailyMail examples, 512 tokens, 32 learned K/V scale tensors).

### 4.1 Headline results (250k = 250,016 prompt + 512 forced output tokens)

| Candidate | Single 250k wall | 2x 196k sessions | Max simultaneous near-native | KV pool | 249k retrieval | Thermal |
|---|---:|---|---|---:|---:|---|
| 2x llama.cpp Q4 replicas (1 GPU each) | 221.9 s | 149.9 / 153.7 s (wall 154.5 s) | 2 (1 per card, duplicated weights) | 1 slot/GPU | 212.2 s | 79 / 61 C |
| vLLM 0.29.0 TP2 (NVFP4, FP8 KV, eager, FlashInfer attn) | 126.8 s | 161.4 / 168.7 s (wall 169.6 s) | **4 passed** (wall 418.7 s, KV 95.2%) | 1,033,510 tok = 3.94 native windows | 96.7 s | 64 / 53 C |
| SGLang 0.5.20 TP2 (NVFP4, FP8 KV, chunked 4k, no CUDA graph) | 127.3 s | 160.7 / 160.7 s (wall 161.6 s) | 2 (3x 250k exceeds pool) | 569,951 tok = 2.17 native windows | 84.9 s | 64 / 54 C |

All three passed: normal-EOS check, forced `lookup_symbol` tool call, 249,041-
token buried-fact retrieval (exact marker), 300 s sustained load
(llama 58/58 reqs 29,696 tok; vLLM 18/18 9,216 tok; SGLang 14/14 7,168 tok),
and the 85 C stop was never triggered by the TP2 engines.

SGLang TP1 was rejected at Phase C (hybrid recurrent-state scheduler: only
~48k active tokens on one card at 196k context). vLLM TP1 at 196k needed
`gpu_memory_utilization=0.95` to fit and left ~1.8 GiB headroom — no room for
concurrency. PP2 was not run (no peer path to justify it).

### 4.2 Decode speed (measured, per request)

| Configuration | Decode @ 250k ctx | @ 196k | @ 8k short ctx |
|---|---:|---:|---:|
| llama.cpp Q4, 1x 5090 (Q8_0 KV) | **34.45 tok/s** (29.0 ms/tok) | 38.6 / 40.0 tok/s | **72.5 tok/s** (13.8 ms/tok) |
| vLLM TP2 Phase D | **~17 tok/s** (engine log: 17.4/17.3) | — | ~31–34 tok/s aggregate, 2 concurrent (≈15–17/user) |
| vLLM TP2 Phase E calibrated (Triton attn) | **~15–16 tok/s** (16.3/16.2/15.1) | — | similar |
| SGLang TP2 | **~13 tok/s** (12.2–13.2) | — | ~13–14/user at 2 concurrent |

Decode is KV-read bandwidth bound at long context: llama.cpp's rate roughly
halves from 8k (72.5) to 250k (34.5). The TP2 numbers additionally pay the
PCIe all-reduce per layer (no NVLink).

### 4.3 Prefill at the 250,016-token prompt

| Configuration | Prefill time | Rate |
|---|---:|---:|
| llama.cpp Q4 | 206.4 s (server log, exact) | **1,211 tok/s** |
| vLLM TP2 | GPU-side: whole prompt in one ~10 s engine window at **~25,000 tok/s** (24,998.8); 196k prompts at **~19,600 tok/s** per window | burst ~20–25k tok/s |
| SGLang TP2 | ~85 s total, 4,096-token chunks; per-chunk rate decays **5,544 -> 1,967 tok/s** as depth grows | **~2,900 tok/s effective** |

**Critical caveat on vLLM end-to-end:** the client-side wall for that same
250k+512 request was 126.85 s (Phase D) / 165.44 s (Phase E calibrated)
because the first long request after server start carries ~90–140 s of
pre-prefill overhead (Triton kernel JIT for the new long-sequence shapes —
confirmed by `jit_monitor` warnings in both server logs — plus tokenization
and scheduling). Subsequent 196k requests still showed ~50–100 s from
submission to prefill burst. So: vLLM burst prefill is ~20x llama.cpp, but
first-request end-to-end is only ~1.75x better (126.8 vs 221.9 s). Warmup
extension is an open tuning item.

64k-prefill-while-decoding interference test:

| | 64k prefill | Decode p95 ITL during it |
|---|---:|---:|
| llama Q4 | 2,390 tok/s (TTFT 34.88 s) | 15 ms |
| vLLM TP2 | 3,668 tok/s (TTFT 17.87 s) | 401 ms |
| SGLang TP2 | 4,699 tok/s (TTFT 13.95 s) | 224 ms |

Historical single-GPU Q4 data (2026-09-02, 500 W): 196,227 prompt -> 1,428.6
prefill / 38.70 decode tok/s; 261,765 -> 1,118.05 / 32.26. Phase B re-
validation (2026-09-20, managed-memory var absent): 261,763 -> 1,168.59 /
34.04, 25,494 MiB peak VRAM, 56 C.

## 5. What was chosen, and why

**Chosen: vLLM 0.29.0 TP2 with the calibrated FP8-KV ModelOpt NVFP4
checkpoint, Triton attention (`TRITON_ATTN`), non-FlashInfer sampler
(`VLLM_USE_FLASHINFER_SAMPLER=0`), eager mode, max-num-seqs 4.**

Reasons (measured, 2026-09-20):
1. Only candidate that ran **4 simultaneous near-native (250k) sessions**
   (3.94 native KV windows); llama replicas and SGLang cap at 2.
2. Fastest cold prefill (burst ~25k tok/s) and ~1.75x faster 250k end-to-end
   than the llama reference.
3. Coolest of the three (64/53 C vs 79/61 C for llama) at the 500 W caps.
4. The pinned FlashInfer JIT rejects SM120 (XQA path fails), so the
   TRITON_ATTN + non-FlashInfer-sampler flags are **required pinned settings**,
   not preferences.
5. SGLang TP2 is the "latency profile" (best decode ITL, fastest retrieval)
   but its pool only holds 2.17 native contexts (FP32 recurrent state = 7.59
   GB/rank) and it needed SIGKILL for clean shutdown.

TP2 exists because each 262k context alone costs ~8 GiB of FP8 KV; splitting
weights across both cards shares the KV pool instead of duplicating weights
per card. Cost: NCCL all-reduce over PCIe every layer.

## 6. What is blocking the chosen infrastructure

**A quality gate, not the infrastructure.** The calibrated vLLM TP2 export
passed every operational test (native context, 4x250k capacity, forced tool,
249k retrieval, prefix cache 0.37 s vs 2.57 s, clean restart/reload, 300 s
sustained, 76 C peak, 505.87 W peak, 2,928 MiB minimum free VRAM, no thermal
stop). But the 128-example GSM8K smoke:

| Metric | Result | Required |
|---|---:|---:|
| Accuracy | 95.3125% (122/128) | >= 96.5% |
| Stop rate | 97.6563% | 100% |
| Truncations | 3 | 0 |
| Request errors | 0 | 0 |

Root cause hypothesis: the production path is 4-bit NVFP4 weights + FP8 KV
cache (needed to fit 4 full contexts in 2x 32 GiB). The original checkpoint
had no calibrated KV scales (defaulted to 1.0); Phase E's calibrated export
fixed the scales but the quantized path still lands marginally below the
quality floor set by the old Q4_K_M/Q8_0 service, with a small truncation
regression. Caveats: 128 examples is not a statistical rejection (a full gate
is 1,319 examples, ~2.4 h, monopolizes both GPUs), and the harness was
reconstructed (original test assets not stored locally).

Before any canary: (1) full 1,319-example GSM8K window, (2) resolve
stop/truncation at the exact full-run prompt and output cap, (3)
representative coding-agent and multi-turn tool trajectories, (4) latency SLO
+ authenticated gateway pool + metrics. None installed. **The llama.cpp Q4
service remains the only live model.**

## 7. What was NOT tested — improvement levers (the interesting part)

1. **MTP speculative decoding — not enabled anywhere in the 5090 work.**
   - The live llama.cpp Q4 service has embedded MTP tensors but runs with
     speculative decoding OFF.
   - The Phase D/E vLLM TP2 runs had **no** `num_speculative_tokens` set.
     The official vLLM Qwen3.8-27B recipe for 2x RTX 5090 explicitly documents
     "working in-checkpoint MTP". Whether the local RadixArk ModelOpt NVFP4
     checkpoint (and Phase E calibrated export) contain the MTP tensors is
     unverified.
   - The plan's own tuning order (step 6: "Compare MTP off with 1-3
     speculative tokens. Measure acceptance, per-user latency, aggregate
     goodput") was never executed.
2. **DFlash2 drafter (llama.cpp): blocked on the 3090 rig, untested on 5090.**
   `Qwen3.8-27B-DFlash2-Q8_0.gguf` (2.06 GB) was downloaded and tested
   2026-09-12 on the two-RTX-3090 tensor-split target: fails with a
   multi-GPU target/draft graph limitation (`SPLIT_AXIS_0` assert;
   `output.weight` in a buffer that cannot run the operation), including an
   isolated upstream build `82d6bb2` with the single-device drafter fix
   `415e909`. The 5090 live service is **single-GPU** (all layers on GPU0),
   so that multi-GPU blocker may not apply — single-GPU DFlash2 on the 5090
   has simply never been tried. vLLM has a `dflash` method but requires an
   HF/safetensors drafter; only the GGUF is on disk.
3. **CUDA graphs / eager mode.** vLLM ran `--enforce-eager` (required while
   the FlashInfer XQA/SM120 path was broken) and SGLang ran
   `--disable-cuda-graph`. Both are known decode-rate suppressors; neither
   engine was re-benchmarked with graphs enabled on a working SM120 path.
4. **llamAmpere on the 5090.** llamAmpere is staged as the RTX 3090 profile;
   its SM86-oriented kernels/pinned binaries are not an automatic SM120 win,
   but it was never run on the 5090.
5. **Power-cap asymmetry** (500 W vs 575 W) and the failed `fans-max.service`
   were recorded as confounds, never equalized and re-measured.
6. **Prefill-chunk sizing** (plan suggests 2k/4k/8k sweep) and
   `max-num-seqs` tuning were never run; SGLang's 4k chunk choice was a
   starting point, not a measured optimum.
7. **First-request JIT overhead** (~90-140 s on the first long vLLM request)
   was observed, never eliminated (warmup extension).
8. SGLang BF16 recurrent-state experiment: capacity-only improvement, quality
   gate not run.
9. MTP acceptance rate at long context: unmeasured anywhere. Acceptance
   typically falls with context depth; at 196-262k it must be measured per
   context, not assumed from short-context numbers.

## 8. Comparison with the RTX 3090 rig (why "30 tok/s on a 5090" is below expectations)

RTX 3090 rig (2x 3090 24 GiB, 200 W caps, Gen3 x8):
- llamAmpere profile: `Qwen3.8-27B ATX IQ4_XS-M` GGUF, Q8_0 K / turbo3 V,
  `GGML_Q8_TURBO3_MMA_FUSED=1`, **`--spec-type draft-mtp --spec-draft-n-max 3`**
  (MTP3 speculative decoding), single 3090 per replica.
  Measured: **~65 tok/s decode per replica at 196k context**, ~130 tok/s
  aggregate across two independent replicas (2026-09-19 plan; 70-80 was
  explicitly labeled aspirational).
- Stock llama.cpp Q4_K_M two-GPU tensor-split production baseline
  (2026-09-16 plan): **~45 tok/s median decode at long context**, 364 tok/s
  median prefill, ~584 W both cards, 13.8 J per output token.

RTX 5090 rig, same model family:
- Live single-GPU llama.cpp Q4_K_M (no MTP): 34.45 tok/s @ 250k,
  38.6-40.0 @ 196k, 72.5 @ 8k.
- vLLM/SGLang TP2 (no MTP, eager/no-graph): 13-17 tok/s at 250k per stream.

Reading: a single *unoptimized* 5090 decodes at roughly half the rate of a
single *optimized* (llamAmpere + MTP3 + smaller IQ4_XS-M weights + turbo3 KV)
3090 at 196k, and the 5090 rig's second card is idle in the live service
whereas the 3090 rig uses both cards. The 5090's raw bandwidth advantage
(GDDR7 ~1.8 TB/s vs GDDR6X ~936 GB/s) is not currently visible in any
production number because: (a) no speculative decoding, (b) no tuned kernels/
quantization mix, (c) one card idle, (d) the TP2 numbers are degraded by
eager mode, no CUDA graphs, and PCIe all-reduce. **Speculative decoding (MTP
first, DFlash2 second) plus CUDA graphs is the largest untested lever on this
hardware.**

## 9. Source files and raw artifacts

Research files (this repo):
- `research/qwen3.8-27b-rtx5090-consolidation-and-serving-plan-2026-09-20.md`
  (master plan + Phase A/B records)
- `research/qwen3.8-27b-rtx5090-phase2-gpu1-feasibility-2026-09-20.md` (Phase C)
- `research/qwen3.8-27b-rtx5090-phase3-tp2-benchmark-window-2026-09-20.md` (Phase D)
- `research/qwen3.8-27b-rtx5090-phase-e-calibrated-validation-2026-09-20.md` (Phase E)
- `research/qwen3.8-27b-vllm-spec-decode-plan-2026-09-16.md` (spec-decode plan,
  PLANNED not started, targets the 3090 rig)
- `research/qwen3.8-27b-q4-dflash-blocked-2026-09-12.md` (DFlash2 blocker)
- `research/qwen3.8-27b-llamampere-production-upgrade-plan.md` (3090 MTP3
  profile, 2026-09-19)

Raw run artifacts:
```text
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-d-llama-replicas-2026-09-20T041507Z/candidate/   (results.json + 2 server logs)
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-d-vllm-2026-09-20T025445Z/candidate/             (results.json + server log)
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-d-sglang-2026-09-20T032421Z/candidate/           (results.json + server log)
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-d-vllm-2026-09-20T033904Z/   (3x250k capacity)
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-d-vllm-2026-09-20T034635Z/   (4x250k capacity)
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-d-vllm-2026-09-20T040052Z/   (vLLM retrieval re-run)
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-d-sglang-2026-09-20T040508Z/  (SGLang retrieval re-run)
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-e-calibrated-triton-nosampler-vllm-2026-09-20T055707Z/candidate/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-e-cache-restart-vllm-2026-09-20T061821Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-e-retrieval-vllm-2026-09-20T062127Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-e-gsm8k-smoke-vllm-2026-09-20T063256Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-e-kv-fp8-calibration-2026-09-20T052906Z/export/  (calibrated checkpoint)
```

## 10. Open questions (for discussion)

1. Does the local NVFP4 checkpoint (and the Phase E calibrated export) contain
   MTP tensors? If yes: what decode rate does `num_speculative_tokens` 1-3
   give at 8k / 196k / 250k on vLLM TP2, and what is the acceptance rate per
   context? Does it change the Phase E quality-smoke result?
2. Single-GPU llama.cpp on the 5090 with `--spec-type draft-mtp` (UD Q4_K_M
   has embedded MTP tensors) or DFlash2 (multi-GPU blocker may not apply to a
   single-GPU target): expected vs the 3090's 65 tok/s @ 196k?
3. Is a safetensors DFlash2 drafter published anywhere? (vLLM `dflash`
   method requires it.)
4. Re-measure vLLM TP2 with CUDA graphs once the SM120 XQA/FlashInfer path is
   validated (eager mode was a pinned workaround, not an optimum).
5. Should the live single-GPU Q4 service move to a two-GPU tensor-split
   llama.cpp (matching the 3090 rig's 45 tok/s stock profile) while the TP2
   candidate waits on its quality gate?
6. Equalize the 500/575 W power caps and fix `fans-max.service` before any
   further thermal-sensitive comparisons.
