# Qwen3.8-27B Tensor 262k Optimization Results

## Recommendation

Use Qwen3.8-27B UD Q4_K_M in llama.cpp tensor split for the single-agent,
long-context coding profile. Keep the separate layer/Q8-KV profile for
maximum-capacity ingestion and strict two-slot work; do not replace it with the
tensor profile.

The selected single-agent command profile is:

```text
CUDA_VISIBLE_DEVICES=1,2
--split-mode tensor --tensor-split 1,1 --n-gpu-layers all
--ctx-size 262144 --parallel 1
--cache-type-k f16 --cache-type-v f16 --flash-attn on
--batch-size 2048 --ubatch-size 1024 --fit off
```

Use the NCCL defaults. Keep `GGML_CUDA_ENABLE_UNIFIED_MEMORY` absent and do
not enable `GGML_CUDA_P2P`. The GPUs run at 300 W; GPU2's permanent fan target
is 85%.

## Validated Performance

| Workload | Prefill tok/s | Decode tok/s | Total s | Peak MiB/GPU | Total kJ |
|---|---:|---:|---:|---:|---:|
| 196k input + 4,096 output | 856.45 | 36.95 | 341.07 | 16862 | 229.99 |
| 257k input + 4,096 output | 763.97 | 33.49 | 459.53 | 16862 | 308.56 |

At 196k plus 4,096 output, `2048/1024` improves the prior `2048/512` tensor
control by 4.9% prefill, 1.2% decode, 3.6% total latency, and 5.4% energy.

## Rejected Variants

- Q5_K_M: no measured coding-quality advantage over Q4, 4.0% slower decode at
  196k plus 4,096 output, and 0.4% slower end-to-end.
- Q8_0: fits at 22,122 MiB per GPU but is slower than Q4/Q5 at the coding
  workload and leaves only about 2.4 GiB headroom.
- Q6_K: downloaded but not benchmarked because the Q5 quality-benefit gate was
  not met and Q8 confirmed the adverse higher-precision latency trend.
- BF16 K/V: decode falls to 20.49 tok/s and adds 916 MiB per GPU.
- Forced NCCL `Simple`, `LL128`, and one-CTA/one-channel modes: all regress
  the 64k plus 1,024 control. NCCL defaults select Ring plus LL over two
  SHM/direct/direct channels on this PHB topology.

## Cache Behavior

With a synthetic 160k retained prefix and appended 512-token tool-result
payload, the next 161,040-token request reused 159,494 tokens. Prompt
evaluation fell from 173.478 seconds cold to 2.884 seconds warm, while decode
remained about 39 tok/s. This confirms prompt-cache mechanics only, not coding
agent task quality or semantic tool-call correctness.

## Artifacts

- Plan and raw-artifact index:
  `research/qwen3.8-27b-tensor-262k-optimization-plan-2026-09-09.md`
- Batch sweep:
  `research/qwen3.8-27b-q4-tensor-batch-sweep-20260909T140914Z/`
- Long controls:
  `research/qwen3.8-27b-q4-tensor-batch2048-ubatch1024-long-20260909T142018Z/`
  and `research/qwen3.8-27b-q4-tensor-batch2048-ubatch1024-native-20260909T143100Z/`
- Warm-prefix cache run:
  `research/qwen3.8-27b-q4-tensor-warm-cache-20260909T153433Z/`
