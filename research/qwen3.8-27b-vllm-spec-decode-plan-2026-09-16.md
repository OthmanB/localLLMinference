# Qwen3.8-27B vLLM / Speculative-Decode Evaluation Plan

Date: 2026-09-16
Status: PLANNED — not started
Scope: decide whether an alternate Qwen3.8-27B weight/build plus vLLM
speculative decoding can beat the current llama.cpp Q4_K_M production profile
on decode throughput without losing task quality.

## Objective

Test whether vLLM on the two serving RTX 3090s can reach the reported
>70 tok/s decode at long context (196k-262k) on Qwen3.8-27B, using either the
model's native MTP head or a DFlash2 drafter, while keeping quality within
noise of the current UD-Q4_K_M GGUF profile.

## Economic context (decided 2026-09-16)

- The rig was donated by a former employer, so there is no capex payback
  target. The relevant lens is marginal electricity per token plus the idle
  overhead of keeping the host on.
- Measured production window (JST 2026-09-13 07:20 -> 2026-09-15 12:20):
  - 3.28 M output tokens, 5.90 M processed input, 316.7 M cached input
  - decode median 45.3 tok/s, prefill median 364 tok/s, ~584 W on GPUs 1+2
  - active-GPU energy 12.42 kWh -> 418.6 JPY at 33.71 JPY/kWh
  - 13.8 J per output token -> 129 JPY per 1M output tokens
  - corrected API-equivalent for the same workload: 852 JPY off-peak,
    1,704 JPY peak (DeepSeek V4 Flash)
- A rise to 70/100 tok/s at the same power gives 8.9/6.2 J per output token
  (83/58 JPY per 1M output), roughly a 3-4.5x marginal advantage over
  off-peak flash. At rest the host still costs about 156 JPY/day (~193 W).
- Quality gate matters more than a small speed gain: Q4_K_M currently
  resolves 22/40 on the sampled SWE-bench Verified run (thinking disabled)
  and scores on the stored HumanEval runs.

## Current production baseline

| Item | Value |
|---|---|
| Runtime | llama.cpp build `llama.cpp-recent-20260905` |
| Model | Qwen3.8-27B UD-Q4_K_M GGUF, tensor split 1,1 on GPUs 1 and 2 |
| Context | 262,144 tokens, F16 K/V, Flash Attention, batch/ubatch 2048/1024 |
| Decode | ~45 tok/s median at long context |
| Active power | ~584 W for both GPUs (near the 2 x 300 W limit) |
| Energy per output token | 13.8 J |

## Known constraints and assets

1. **DFlash2 on llama.cpp is blocked.** Both the production build and an
   isolated upstream build fail with a multi-GPU target/draft graph
   limitation (`SPLIT_AXIS_0` assert; `output.weight` in a buffer that
   cannot run the operation). See
   `research/qwen3.8-27b-q4-dflash-blocked-2026-09-12.md`.
2. **The local DFlash2 asset is GGUF only**
   (`models/qwen3.8-27b-dflash2/Qwen3.8-27B-DFlash2-Q8_0.gguf`,
   2,056,414,816 bytes). vLLM supports a `dflash` speculative method, but it
   requires an HF/safetensors drafter; the GGUF cannot be used.
3. **MTP is the native, verified path.** `config.json` has
   `model_type=qwen3_5` and `mtp_num_hidden_layers=1`; vLLM maps it to
   `qwen3_5_mtp` with `num_speculative_tokens` and documents the method.
   The repo already parks this item in
   `research/three-rtx3090-141gib-local-model-plan.md` (lines 58, 159, 202,
   235).
4. **Alternate weights already on disk:**
   - `models/qwen3.8-27b-awq-int4/`: compressed-tensors pack-quantized
     4-bit, group size 32, 21.02 GB, MTP weights present, vision tower
     present. This is the primary vLLM candidate.
   - `models/qwen3.8-27b-nvfp4/`: Blackwell-class FP4, not viable on 3090.
   - GGUF Q5/Q6/Q8: llama.cpp alternatives, already quality-characterised.
5. **Environment:** vLLM is not installed. A SGLang 0.5.16 venv exists at
   `~/.venvs/sglang-0.5.16`. The SWE-bench controller already speaks
   `hosted_vllm/<model>` against an OpenAI-compatible endpoint, so the same
   quality harness can score a vLLM server.
6. **Memory math at 262k:** weights ~21 GB + F16 KV for 16 full-attention
   layers x 4 KV heads x 256 head_dim x 2 (K+V) x 2 bytes = ~16.8 GB
   -> ~38 GB before activations and CUDA graphs on 48 GB. fp8 KV on SM86 is
   unvalidated; plan for `max_num_seqs=1` and memory tuning, and test 196k
   before 262k.

## Steps

Estimates assume one operator; stop at any gate that fails.

### Gate 0 — feasibility (0.5 day)
- Check whether a safetensors DFlash2 drafter is published; if not, drop
  DFlash2 and proceed with MTP only.
- Pin a vLLM version that supports `qwen3_5`/`qwen3_5_mtp`,
  compressed-tensors w4a16, and SM86; record exact versions in the run log.
- Confirm the gateway can point at a vLLM backend without code changes
  (`lan-inference-gateway` already advertises vLLM support).

### Step 1 — vLLM baseline without speculative decoding (0.5-1 day)
- Create an isolated venv (do not reuse the SGLang venv).
- Serve the compressed-tensors 4-bit model with `tensor_parallel_size=2`,
  `max_model_len=262144`, serving on a loopback port parallel to production.
- Reproduce the existing benchmark matrix (8k / 196k / 257k prompt, fixed
  output, power sampled every 5 s, three repeats) using
  `tools/qwen_q4_dflash_benchmark.py` prompt construction as reference.
- Record prefill/decode, VRAM per GPU, wall power, and OOM margins.

### Step 2 — speculative decoding (1 day)
- Enable MTP with `num_speculative_tokens` 1 and 2; measure acceptance rate,
  decode at each context, and memory.
- If Gate 0 found an HF DFlash2 drafter, run the same matrix with
  `method=dflash` and compare.
- Reject any configuration that drops decode below the Step-1 baseline or
  pushes a GPU into OOM after repeated 262k requests.

### Step 3 — quality gate (1 day)
- Serve the chosen configuration and run the sampled SWE-bench Verified
  40-task set with thinking disabled, using
  `tools/swebench_controller.py` and a `hosted_vllm/` overlay; compare with
  the 22/40 Q4_K_M reference.
- Run the stored HumanEval/evalplus assets against the same prompt policy.
- Verify speculative decoding is lossless for the chosen sampler (fixed
  seed and temperature 0 comparison against the non-speculative run).

### Step 4 — economics and adoption decision (0.5 day)
- Recompute J per output token, JPY per 1M output at both tariffs, and the
  API-equivalent comparison for an equivalent 24-53 h window.
- If adopted, plan the migration items: systemd unit, gateway backend,
  exporter support for vLLM token counters (`vllm:prompt_tokens_total`,
  `vllm:generation_tokens_total`, request counters) and the throughput
  retention logic, plus a dashboard re-check.

## Decision rules

Adopt only if all of the following hold:
- decode at 196k and 262k is at least 1.5x the current ~45 tok/s;
- SWE-bench sampled result is within one task of the Q4_K_M reference;
- speculative decoding passes the losslessness check;
- repeated 262k requests keep >= 1 GiB free on both cards.

Otherwise keep the Q4_K_M llama.cpp production profile and revisit when
llama.cpp upstream supports DFlash2 with a multi-GPU target.

## Risks

- vLLM hybrid linear-attention kernels may be slower or unsupported on SM86;
  validate before drawing conclusions about MTP.
- Long-context KV pressure can force fp8 KV, whose quality effect must be
  measured separately.
- Speculative decoding acceptance typically falls with context; report
  acceptance per context, not only the headline decode rate.
- The exporter cannot parse vLLM metrics today; a production move requires a
  monitoring change, not only a serving change.

## Artifacts to produce

- Timestamped run directories under `research/` with server logs, benchmark
  JSON, power samples, and acceptance rates.
- A results note comparing vLLM/MTP/DFlash2 against the Q4_K_M baseline.
- If adopted: updated `operations/` units, gateway config, exporter parser,
  and cost-model notes.

## References

- `research/qwen3.8-27b-q4-dflash-blocked-2026-09-12.md`
- `research/qwen3.8-27b-q4-multigpu-runtime-evaluation-results-2026-09-09.md`
- `research/three-rtx3090-141gib-local-model-plan.md` (MTP row, line 202)
- vLLM MTP speculative decoding: https://docs.vllm.ai/en/stable/features/speculative_decoding/mtp/
- `tools/qwen_q4_dflash_benchmark.py`, `tools/swebench_controller.py`
