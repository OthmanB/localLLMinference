# Qwen3.8-27B RTX 5090 Phase D Two-GPU Benchmark Window

Date: 2026-09-20
Status: Phase D execution complete; no production promotion

## Scope and Safety

This maintenance window compared the unchanged Q4/Q8 llama.cpp design as two
independent replicas with ModelOpt NVFP4/FP8-KV vLLM and SGLang TP2. Every
candidate was temporary, loopback-only, and explicitly used physical GPUs 0
and 1. The wrapper captured topology and service state, stopped the reference
unit, verified it could start and serve its expected model ID, then restored
the original model and monitor state in an exit trap.

The unrelated local RAG process on GPU 0 (792 MiB reported VRAM) was recorded
and explicitly allowed but not stopped. It is a reproducibility constraint for
the GPU 0 measurements.

Rollback verification initially found that the live reference unit required the
nonexistent `nvidia-qwen3.8-gpu0-policy.service`. The real active policy is
`nvidia-qwen3.8-gpu-policy.service`. The installed compatibility dependency is
now preserved at `operations/systemd/nvidia-qwen3.8-gpu0-policy.service`; it
only requires and orders after the real policy. It does not change power, fan,
model, or llama.cpp command settings. The wrapper resets the reference unit's
`StartLimitBurst=3/300s` limiter only before its deliberate stop/start probes.

NVIDIA reports no GPU peer read or write support. Both TP2 engines disabled
custom peer all-reduce and used NCCL/PYNCCL fallback communication. PP2 was not
added: TP2 was measured, whereas the topology offers no peer path that would
justify a secondary distributed architecture at this stage.

## Common Configuration

| Field | Value |
|---|---|
| GPUs | 0 `GPU-dfca3a7f-54b9-cc0a-a031-81f19c4430cf`, 1 `GPU-8e28ecfd-fd2e-d821-49bf-70e324d8f9fe` |
| Context | 262,144 tokens |
| Long performance probe | 250,016 to 250,017 prompt tokens, 512 forced output tokens, thinking disabled |
| Capacity probe | Independent prompt prefixes, 196,017 or 250,017 prompt tokens, 512 forced output tokens |
| Quality retrieval | Exact marker at 120,001 content tokens followed by 128,993 tokens of distinct trailing context; normal EOS required |
| Stop condition | 85 C sampled on either GPU, explicitly approved after the initial 80 C reference result |
| Checkpoint for TP2 | Local ModelOpt NVFP4/FP8 checkpoint documented in the Phase C report |

The 512-token performance probes use `ignore_eos`; their output is not a
quality result. Normal-EOS, forced-tool, and retrieval requests are separate.

## Measured Results

| Candidate | Single 250k + 512 | Two 196k + 512 | Interference TTFT / p95 ITL | Five-minute sustained result | Thermal result |
|---|---:|---:|---:|---|---|
| Two Q4/Q8 llama.cpp replicas | 221.9 s | 149.9 s and 153.7 s, 154.5 s wall | Decode 6.68 s / 15 ms; 64k prefill 34.88 s / 18 ms | 58 of 58 requests, 29,696 completion tokens | 79 C GPU 0, 61 C GPU 1; no stop at approved 85 C |
| vLLM 0.29.0 TP2 | 126.8 s | 161.4 s and 168.7 s, 169.6 s wall | Decode 1.87 s / 401 ms; 64k prefill 17.87 s / 123 ms | 18 of 18 requests, 9,216 completion tokens | 64 C GPU 0, 53 C GPU 1 |
| SGLang 0.5.20 TP2 | 127.3 s | 160.7 s and 160.7 s, 161.6 s wall | Decode 1.80 s / 224 ms; 64k prefill 13.95 s / 196 ms | 14 of 14 requests, 7,168 completion tokens | 64 C GPU 0, 54 C GPU 1 |

All three candidates passed a normal-EOS `phase-d-ok` request, a forced
`lookup_symbol({"symbol":"Record"})` tool call, and the long-context retrieval
marker. Retrieval prompt usage was 249,041 tokens. End-to-end retrieval times
were 212.2 s for Q4/Q8 replicas, 96.7 s for vLLM TP2, and 84.9 s for SGLang
TP2.

The initial Q4/Q8 result reached the former 80 C defensive cutoff. The
operator-approved 85 C rerun completed the same suite with no stop, a 79 C GPU
0 peak, and clean temporary-process exits. This supersedes the initial thermal
result for Phase D acceptance while preserving it as historical evidence.

## Capacity

| Candidate | Measured capacity result | Engine allocation evidence |
|---|---|---|
| Two Q4/Q8 replicas | Two independent 196,017-token sessions and the five-minute sustained run passed | One slot per replica; duplicate weights retain the two-session limit |
| vLLM TP2 | Two 196,017-token sessions passed; three 250,017-token sessions passed in 319.5 s wall; four 250,017-token sessions passed in 418.7 s wall | 1,033,510 FP8-KV tokens, reported 3.94 native windows; four-session run reached 95.2% KV cache usage and 71 C GPU 0 |
| SGLang TP2 | Two 196,017-token sessions passed | 569,951 scheduler tokens, reported 2.17 native windows; three 250k sessions exceed the measured pool |

The four-session vLLM result is a capacity/stretch result, not an interactive
latency claim. Requests were initially queued while the scheduler chunked their
prefills; individual completion times were 388.9 to 417.3 seconds. It does
prove four independently prefixed near-native windows completed without OOM,
truncation, or thermal stop in this configuration.

## Engine Evidence and Decision

vLLM selected ModelOpt mixed quantization, TP2, FP8 E4M3 KV, native FlashInfer
attention, Triton/FLA GDN prefill, CUDA GDN decode, and PYNCCL all-reduce. Its
automatic SM120 XQA path failed in the pinned FlashInfer 0.6.18 worker runtime.
The passing configuration explicitly sets
`--attention-config '{"use_trtllm_attention": false}'`, which selects native
FlashInfer decode. This is a required pinned setting, not an optional tuning
preference.

SGLang selected ModelOpt mixed quantization, TP2, FP8 E4M3 KV, Triton GDN
prefill/decode, and NCCL fallback communication. Its default FP32 recurrent
state allocated 7.59 GB per rank and capped `max_running_requests` at 21. The
server required SIGKILL after successful tests, so cleanup behavior remains an
operational follow-up. The BF16 recurrent-state experiment is not promoted: it
has not passed an equivalent quality gate.

Both engines defaulted FP8 KV scaling factors to `1.0`, because the checkpoint
does not provide calibrated factors. Neither candidate is quality-approved for
production on that basis alone.

Phase D produces two distinct next-stage profiles:

1. **Capacity profile: vLLM TP2.** It is the only measured candidate with four
   simultaneous near-native contexts. Retain its eager mode and disabled
   TRTLLM/XQA setting until an upgraded compatible FlashInfer path is validated.
2. **Latency/interference profile: SGLang TP2.** It delivered the fastest
   retrieval and lower decode ITL during the measured interference test, but its
   current pool only supports two native contexts.

Do not enter Phase E yet. Required before a canary are calibrated FP8-KV quality
evidence, representative code/agent and cache-restoration tests, an explicit
latency SLO, SGLang graceful cleanup resolution, authenticated routing and
metrics, and a deliberate choice between the capacity and latency profiles.

Phase E update: the calibrated vLLM TP2 candidate completed cache/restart,
native-context, tool, retrieval, sustained-load, and bounded GSM8K validation.
It requires `TRITON_ATTN` and `VLLM_USE_FLASHINFER_SAMPLER=0` because the pinned
FlashInfer JIT rejects Blackwell SM120. The GSM8K smoke missed the existing
quality gate, so no canary has been installed. See the Phase E report.

## Artifacts

```text
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-d-llama-replicas-2026-09-20T023445Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-d-vllm-2026-09-20T025445Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-d-sglang-2026-09-20T032421Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-d-vllm-2026-09-20T033904Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-d-vllm-2026-09-20T034635Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-d-llama-replicas-2026-09-20T035636Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-d-vllm-2026-09-20T040052Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-d-sglang-2026-09-20T040508Z/
/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-d-llama-replicas-2026-09-20T041507Z/
```

The failed automatic-vLLM-XQA and initial SGLang model-ID readiness attempts are
preserved in adjacent timestamped run directories as diagnostic evidence.
