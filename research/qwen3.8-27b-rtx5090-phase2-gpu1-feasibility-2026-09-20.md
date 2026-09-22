# Qwen3.8-27B RTX 5090 Phase 2 GPU 1 Feasibility

Date: 2026-09-20
Status: Phase C complete; two-GPU benchmark shortlist established

## Scope

This phase tested isolated TP1 vLLM and SGLang candidates on physical GPU 1.
The production llama.cpp service on GPU 0 stayed active throughout. Candidate
servers used private loopback ports, explicit `CUDA_VISIBLE_DEVICES=1`,
separate virtual environments, and temporary processes only.

The tested checkpoint was the local RadixArk ModelOpt mixed NVFP4/FP8
checkpoint:

```text
/home/michel/LLMs-tests/rtx5090-qwen38-bench/models/Qwen3.8-27B-NVFP4
source repository: Qwen/Qwen3.8-27B
source revision: e13a4f0e35203116364e3b3f3f0c82f6ef1afd3c
ModelOpt producer commit: 87c9f8cf83021957d1a1a575c90c9a4eaaf7ef0c
config.json: 7ff41ec6f96ad50efea3c92751cd261b63839d39936eb6e6ffc9066db8672740
model-00001: fbcdb5ba1cdda462b5f38592d071e772c4d398afea61a0aa9188b32d1a239a79
model-00002: db6146a5464fb0a891181b93c81593f0ca65c602eb14120a1c2b1b09bca11f85
model-00003: d3cfb92742e30c8b46564665791dbe0a86ed64cfc02b1275081530793c0c9581
```

The checkpoint's FP8 KV metadata has no calibrated scaling factors. Both
engines therefore warned that they defaulted FP8 KV scaling to `1.0`. Any
quality or production decision must treat this as an unresolved numerical
quality gate.

## Pinned Environments

Both environments use Python 3.12.14, CUDA 13.0 PyTorch, and FlashInfer
0.6.18:

| Engine | Environment | Key versions |
|---|---|---|
| vLLM | `/home/michel/LLMs-tests/.phase2-vllm-venv` | vLLM 0.29.0, torch 2.13.0+cu130, Transformers 5.17.0, Triton 3.7.1 |
| SGLang | `/home/michel/LLMs-tests/.phase2-sglang-venv` | SGLang 0.5.20, SGLang kernel 0.4.7, torch 2.13.0+cu130, Transformers 5.12.1, Triton 3.7.1, ModelOpt 0.46.1 |

## Results

| Candidate | Configuration | Result |
|---|---|---|
| SGLang TP1 smoke | 8,192 context, ModelOpt, FP8 E4M3 KV, static fraction 0.75, Qwen reasoning/tool parsers | Loaded; 24,996 MiB peak; bounded text response passed in 33.8 s; forced structured tool call passed |
| SGLang TP1 196k | 196,608 context, FP32 recurrent state | Startup passed, but scheduler reported only 48,214 total active tokens, one request, and 7.24 GB available GPU memory; reject as a native-context TP1 deployment |
| SGLang TP1 196k BF16 experiment | 196,608 context, BF16 recurrent state | Startup passed; 49,798 total active tokens, three request slots, 6.76 GB available GPU memory; slight capacity improvement only, not native context and not quality-approved |
| vLLM TP1 smoke | 8,192 context, eager, FP8 E4M3 KV, GPU utilization 0.75, Qwen reasoning/tool parsers | Loaded; 24,746 MiB peak; bounded text response and forced structured tool call passed |
| vLLM TP1 196k at 0.75 | 196,608 context, eager, FP8 E4M3 KV | Rejected before serving: only 0.8 GiB KV available versus 6.32 GiB required; estimated maximum length 15,680 tokens |
| vLLM TP1 196k at 0.95 | 196,608 context, eager, FP8 E4M3 KV | Startup passed; 30,824 MiB peak of 32,607 MiB; approximately 1.78 GiB observed headroom |
| vLLM TP1 189k request at 0.95 | 189,242 prompt tokens, 32 output tokens | Passed actual request in 68.8 s; 30,466 MiB peak; no context truncation or request error |

The long vLLM response was deliberately bounded and is a runtime feasibility
result, not a quality result. It used thinking-disabled generation. No
long-context quality, normal-EOS corpus, or sustained-load claim is made.

## Kernel and Parser Evidence

vLLM selected the following paths on SM120 during the successful startup:

- `FlashInferCutlassNvFp4LinearKernel` for NVFP4 GEMM.
- `FlashInferFP8ScaledMMLinearKernel` for ModelOpt FP8 linear layers.
- Triton/FLA GDN prefill and CUDA GDN decode.
- FlashInfer attention and sampling.

SGLang selected Triton GDN decode/extend/verify kernels and completed FlashInfer
autotuning. Both engines accepted `qwen3` reasoning and `qwen3_coder` tool
parser configuration. Each forced tool test returned a structured call with
valid JSON arguments for `lookup_symbol(Record)`.

SGLang emitted optional TorchCodec/FFmpeg import warnings for unused multimodal
processors. Text-only loading and requests passed; this dependency is still a
multimodal follow-up if image/video serving is considered.

## Decision

Phase C passes as a feasibility gate and produces this shortlist:

1. **Primary: vLLM TP2 with ModelOpt NVFP4 and FP8 KV.** TP1 is already able
   to execute a 189k request, but it uses almost the entire 32 GiB card at
   `gpu_memory_utilization=0.95`. TP2 must be tested for capacity, headroom,
   peer-transfer cost, and sustained behavior rather than assumed superior.
2. **Conditional: SGLang TP2 with ModelOpt NVFP4.** SGLang TP1 is parser- and
   kernel-correct but its hybrid recurrent-state scheduler cannot provide
   native context on one card. TP2 may recover weight and state margin; keep
   the BF16 recurrent-state setting as a separately quality-gated experiment.

Do not promote either candidate. The next phase requires a maintenance window,
verified rollback, and two-GPU comparison against the unchanged reference. It
must include real native-context requests, per-GPU telemetry, peer-transfer
behavior, repeated runs, and quality/tool-contract gates.

## Raw Artifacts

```text
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase2-sglang-tp1-smoke-2026-09-20/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase2-sglang-tp1-tool-2026-09-20/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase2-sglang-tp1-196k-startup-2026-09-20/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase2-sglang-tp1-196k-bf16-startup-2026-09-20/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase2-vllm-tp1-smoke-2026-09-20/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase2-vllm-tp1-tool-2026-09-20/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase2-vllm-tp1-196k-startup-2026-09-20/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase2-vllm-tp1-196k-util95-startup-2026-09-20/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase2-vllm-tp1-196k-request-2026-09-20/
```
