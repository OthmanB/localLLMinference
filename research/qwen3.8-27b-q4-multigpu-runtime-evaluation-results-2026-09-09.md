# Qwen3.8-27B Q4 Multi-GPU Runtime Evaluation Results

Date: 2026-09-09

## Decision

Do not promote or change production routing yet. Layer split is the only profile
that fits two native 262,144-token slots, but tensor split is the preferred
llama.cpp profile for one long-running coding/reasoning agent. Layer remains
the prompt-ingestion and maximum-capacity profile. Large 524k layer concurrency
is not fair; moderate-pool tensor concurrency is promising but does not meet
the strict two-native-slot capacity requirement. See
`qwen3.8-27b-q4-runtime-follow-up-results-2026-09-09.md` for validated
long-output, tensor-concurrency, and native-INT4 SGLang follow-up results.

## Method and Validity

- Physical llama.cpp campaign model: `Qwen3.8-27B-UD-Q4_K_M.gguf`; no
  alternative quantization was used for its layer/tensor comparison.
- GPUs: physical GPU1 and GPU2 (`CUDA_VISIBLE_DEVICES=1,2`), no CUDA P2P.
- llama.cpp: NCCL-enabled v0.4.0-dev build `4d91760` at
  `${RUNTIME_ROOT}/llama.cpp-recent-20260905/build/bin/llama-server`.
- Layer profile: layer split, Q8 K/V, Flash Attention on.
- Tensor profile: tensor split with `--tensor-split 1,1`, F16 K/V, and Flash
  Attention on, the combination required by this build.
- The physical campaign was run after removing the erroneous
  `GGML_CUDA_ENABLE_UNIFIED_MEMORY=0` export. The harness now also removes that
  variable from inherited child environments because its presence enables UVM.
- Initial physical prompts are fresh-server requests with 128 generated tokens,
  no prompt-cache reuse, and thinking disabled. The follow-up adds a 4,096-token
  validation for long-running coding-agent behavior.

Earlier campaigns that exported the UVM variable are retained as exploratory
artifacts only. This report uses the physical-VRAM campaign for comparisons.

## One-Agent Runtime Comparison

All points use a 262,144-token allocation and one slot. Peak VRAM is measured
by NVIDIA sampling; temperatures are per-request maxima.

| Prompt | Profile | Prefill tok/s | Decode tok/s | Wall s | Peak VRAM MiB (GPU1/GPU2) | Max temp C (GPU1/GPU2) |
|---:|---|---:|---:|---:|---:|---:|
| 4k | Layer, Q8 K/V | 1771.12 | 39.73 | 5.56 | 13,708 / 14,800 | 46 / 50 |
| 4k | Tensor, F16 K/V | 1231.21 | 50.92 | 5.86 | 16,470 / 16,470 | 72 / 80 |
| 196k | Layer, Q8 K/V | 1069.67 | 18.96 | 191.20 | 13,708 / 14,800 | 69 / 75 |
| 196k | Tensor, F16 K/V | 813.08 | 36.34 | 245.99 | 16,470 / 16,470 | 78 / 87 |
| 262k | Layer, Q8 K/V | 889.15 | 15.85 | 303.31 | 13,712 / 14,802 | 74 / 83 |
| 262k | Tensor, F16 K/V | 710.08 | 33.02 | 373.38 | 16,476 / 16,476 | 79 / 88 |

Tensor split supplies about twice the long-context decode rate while losing
20-24% prefill throughput, using about 1.7-2.8 GiB more VRAM on each GPU, and
pushing GPU2 to 88 C. It cannot provide the required two-native-slot pool with
F16 K/V, so layer is the strict-524k candidate. The initial 128-token wall-time
result is not representative of coding agents: the 4,096-token follow-up
validated tensor as 11.8% faster end-to-end at 196k.

Raw artifact:
`research/qwen3.8-27b-q4-physical-layer-tensor-20260909T085757Z/results.json`.

## SGLang Exact-GGUF Gate

SGLang 0.5.16 was installed in
`${HOME}/.venvs/sglang-0.5.16` and tested with the exact GGUF at TP=2,
8k context, and one request. It exited before GPU initialization with:

```
ValueError: GGUF model with architecture qwen35 is not supported yet.
```

No AWQ/GPTQ/HF conversion of the exact GGUF was attempted, so there is no
same-model SGLang throughput or prefix-cache comparison. The gate can be
revisited only when a compatible SGLang/Transformers GGUF loader is available.

A separate native-INT4 SGLang gate was run with a public Qwen3.8 AWQ-labelled
Transformers checkpoint. It loaded as `compressed-tensors` but produced repeated
`!` tokens in a trivial quality smoke while logging missing packed hybrid
linear-attention parameters. It is numerically invalid and was not benchmarked
at long context. This is a native-quantization compatibility result, not a
conclusion against SGLang scheduling. Details are in the follow-up result.

## Layer Strict 524k Two-Slot Test

The layer profile started with `--ctx-size 524288 --parallel 2`; its startup
log records `n_slots = 2, n_ctx_slot = 262144, kv_unified = 'false'`. Static
VRAM was 19,152 MiB on GPU1 and 20,238 MiB on GPU2. A short concurrent smoke
completed at 53.72 aggregate decode tok/s, confirming both slots receive work.

Two simultaneous 261,765-token requests then completed successfully without
OOM. Peak VRAM was 19,166 MiB on GPU1 and 20,252 MiB on GPU2, proving physical
capacity for two native slots. Capacity alone is not sufficient for promotion:
long prefill scheduling starves one active decode stream.

| Scenario | Stagger | Request decode tok/s | Aggregate decode tok/s | Wall s | GPU2 max C | Result |
|---|---:|---:|---:|---:|---:|---|
| 4k + 4k smoke | 0 s | 31.39 / 27.21 | 53.72 | 9.07 | 69 | Both progress |
| 262k + 262k strict maximum | 0 s | 0.36 / 15.85 | 0.71 | 658.64 | 84 | First stream starved |
| 196k + 196k production equivalent | 0 s | 0.56 / 15.83 | 1.10 | 421.03 | 83 | First stream starved |
| 196k + 64k main/subagent | 0 s | 2.42 / 16.10 | 4.74 | 243.80 | 82 | Main stream starved |
| 196k + 64k main/subagent | 5 s | 16.74 / 0.59 | 1.15 | 269.65 | 79 | Subagent stream starved |

A short stagger merely transfers starvation to the later request. Since both
the symmetric production case and the staggered asymmetric case fail the
steady-progress acceptance condition, the remaining staggered maximum and
shared-prefix permutations were not run. They would add heat and elapsed time
without changing the non-promotion result.

Tensor's single moderate two-slot check is different: with a 262k shared pool,
128k plus 64k input, and 1,024 outputs, both streams sustained 11.72 and 30.27
decode tok/s. That does not establish strict-524k tensor capacity, but it means
the severe layer behavior cannot be attributed solely to llama.cpp's generic
server scheduler.

Artifacts:

- `research/qwen3.8-27b-q4-524k-parallel2-smoke-20260909T091847Z/results.json`
- `research/qwen3.8-27b-q4-524k-parallel2-full-20260909T092155Z/results.json`
- `research/qwen3.8-27b-q4-524k-production196k-pair-20260909T095203Z/results.json`
- `research/qwen3.8-27b-q4-524k-main196k-sub64k-20260909T093641Z/results.json`
- `research/qwen3.8-27b-q4-524k-main196k-sub64k-stagger5-20260909T094455Z/results.json`

## Follow-Up Conditions

Before any production decision, complete workload routing between layer and
tensor, validate the selected profile's coding quality, complete metrics and
energy accounting, and repeat SGLang evaluation with a numerically correct
native Qwen3.8 quantization. Services, gateway routing, Muse placement, and
Flash remain unchanged.
