# Plan: Mistral-Small-4-119B-2603 IQ3 on GPU 1+2 — token/s + context sweep

Date: 2026-09-02
Status: **Phase 5 REVISED** (2026-09-04). Final report:
`research/mistral-small-4-119b-2603-context-sweep-2026-09-04.{md,json}`.
Purpose: Pick a model to fill the two free RTX 3090s (GPU 1 + GPU 2) as a
planner / reviewer / meta-reviewer, alongside Qwen3.8-27B Q4 on GPU 0.
This run measures **decode/prefill token/s** and **context capacity**
(8k/32k/128k/192k/256k) for Mistral-Small-4-119B-2603 at IQ3, then a small
blind reviewer-quality A/B.

---

## Confirmed facts (verified, not assumed)

### Model
- Base: `mistralai/Mistral-Small-4-119B-2603` (MoE, multimodal/VLM — text-only is fine here).
- GGUF architecture: **`mistral4`** (registered in this llama.cpp build;
  `src/models/mistral4.cpp` present, `LLM_ARCH_MISTRAL4` in `llama-arch.cpp`).
  -> **No llama.cpp rebuild needed.**
- text_config (from config.json):
  - `num_hidden_layers` = 36
  - `num_attention_heads` = 32, `num_key_value_heads` = 32
  - **MLA**: `kv_lora_rank` = 256, `qk_rope_head_dim` = 64, `q_lora_rank` = 1024,
    `head_dim` = 128, `qk_nope_head_dim` = 64, `v_head_dim` = 128
  - **MoE**: `n_routed_experts` = 128, `num_experts_per_tok` = 4,
    `n_shared_experts` = 1, `moe_intermediate_size` = 2048 -> ~6.5-7B active
    (matches "A6.5B").
  - `hidden_size` = 4096, `intermediate_size` = 12288, `vocab_size` = 131072
  - `max_position_embeddings` = 1,048,576, `rope_type` = yarn
    (`original_max_position_embeddings` = 8192, factor 128)

### Context tiers (decision)
- Mistral officially advertises/supports **256k** context. The 1M figure is only
  the raw YaRN RoPE ceiling, **not** a supported operating point.
- Test tiers: **8k, 32k, 128k, 192k, 256k**. All are within the supported 256k
  range -> **no YaRN/RoPE extension needed** for any tier.
  - 8k = practical decode ceiling / fixed multi-GPU overhead floor
  - 32k = ordinary working-context baseline
  - 128k = long, 192k = very long, 256k = supported limit

### Quantization (unsloth GGUF, single-file, imatrix)
| File | Size (bytes) | Size | Role |
|---|---:|---|---|
| `Mistral-Small-4-119B-2603-UD-IQ3_XXS.gguf` | 42,797,027,008 | 42.8 GB / 39.83 GiB | **max fit / speed probe** |
| `Mistral-Small-4-119B-2603-UD-IQ3_S.gguf` | 44,407,639,744 | 44.4 GB / 41.33 GiB | **preferred production candidate if it fits comfortably** |

- Repo: `unsloth/Mistral-Small-4-119B-2603-GGUF` (not gated).
- SHA-256 IQ3_XXS: `9c2b030b565125a456f7eb7adb05bd533f292cd24611fa445421ad9178d5b4c8`
- SHA-256 IQ3_S: `c56fdf063ad8c215c0e56c322383b3fb242189ce9047cdbb3e7dcad11380c1b8`
- mmproj (vision) NOT needed for the text benchmark.
- Download target: `~/localLLMinference/models/mistral-small-4-119b-2603-iq3-xxs-gguf/`
  and `.../mistral-small-4-119b-2603-iq3-s-gguf/` (mirror existing dir naming).
  Verify SHA-256 after download.

### Hardware / runtime
- 3x RTX 3090 (24 GB each). GPU 0 = Qwen3.8-27B Q4 (~22 GB used). GPU 1 + GPU 2 free.
- GPU 2 is the **display** card (Disp.A: On, ~9 MiB baseline). GPU 1 fully free.
- PCIe: Gen3 x8 links (no NVLink). Aggregate 48 GB, ~46-47 GB usable
  (per-card CUDA context/overhead ~0.5-1 GB each).
- RAM: 141 GiB (137 available). Disk: ~584 GB free.
- llama.cpp: `0.3.0-dev` build 1, commit `c841aee`, at
  `/home/obenomar/.local/share/llama.cpp/build/bin/llama-server`.
- **Power cap decision: ALL GPUs at 300 W** (set via `nvidia-smi -pl 300`, sudo).
  Current live caps at planning time: GPU0=300, GPU1=300, GPU2=350 -> normalize to 300.
- Concurrency: **single stream only** (`--parallel 1`), matching the existing harness.

### Existing comparable anchor (in our own data)
- `gpt-oss-120b` (116.8B / 5.1B active, MXFP4 = 60.8 GiB) ran on 2 cards at
  **4.6-6.6 tok/s @ 4k** with 40.5 GiB VRAM + 61.6 GiB host (FreeToken TP=2,
  hybrid expert offload). Same size class, but it was NOT fully resident.
  Mistral at ~6.5-7B active, IF fully resident, should be materially faster.

### Existing harness (base for the fork)
- `tools/context_speed_sweep.py`: spawns a fresh `llama-server` per (model, context)
  point, builds a deterministic repeated-token prompt to `ctx-384`, sends one
  128-token request (EOS ignored), records prefill/decode tok/s from server
  `timings`, samples nvidia-smi (power/temp/VRAM) + full server log.
  Hard-coded for Q4/Q5 single-GPU; will be forked, not rewritten.

---

## Fit analysis (estimate; the startup log is authoritative)

MLA keeps KV tiny: KV per token ~= (kv_lora_rank 256 + rope 64) x 36 layers x
~1 B (q8_0) ~= **11.5 KB/token**.

| Context | KV (q8_0) approx |
|---:|---:|
| 8k   | ~0.1 GB |
| 32k  | ~0.4 GB |
| 128k | ~1.5 GB |
| 192k | ~2.3 GB |
| 256k | ~3.0 GB |

Weights + KV (estimate only):
| Quant | Weights | +KV@128k | +KV@192k | +KV@256k | Expected on 2x24GB |
|---|---:|---:|---:|---:|---|
| IQ3_XXS | 42.8 GB | 44.3 | 45.1 | 45.8 | likely fully resident all tiers (borderline @256k) |
| IQ3_S   | 44.4 GB | 45.9 | 46.7 | 47.4 | borderline; may fail full residency @192k/256k |

**Do NOT treat the arithmetic as the verdict.** Real startup also needs CUDA
context, compute/scratch buffers, and asymmetric main-GPU allocations. The
llama-server startup log (offload split, `model buffer size`, `KV self size`,
per-card VRAM) is authoritative and gates every scored point (see residency rule).

---

## Adopted design changes (from review; all accepted)

1. **Residency rule (clean benchmark).** Every *scored* point runs
   `--n-gpu-layers 99 --fit off` and must **assert all 36 layers GPU-resident**
   from the startup log (off = 36/36, `CPU main buffer` ~0). If a tier cannot
   stay fully resident, record it as **`not-fully-resident`** (that tier fails
   the desired profile). Do **not** silently fall back to host-KV / CPU offload.
   `--fit` runs **only as a diagnostic** after a failure (documents the computed
   split; never a scored config).
2. **Quant priority reversed.** IQ3_XXS = establish max fit/speed. IQ3_S =
   preferred production candidate *if* it fits comfortably (reserve + >= 15 tok/s).
   Do not assume 119B params fully compensate for 3-bit quantization.
3. **Include 8k and 32k** baseline points (cheap, reveal fixed multi-GPU overhead
   and how much the Gen3 x8+x8 layer split costs).
4. **Reasoning.** Mistral exposes only `reasoning_effort in {none, high}`.
   Speed pass = `none` + forced 128 tokens (EOS ignored) for comparable timings.
   Review mode = `high` with natural EOS (characterizes the real review workload).
5. **Added: blind reviewer A/B** (Phase 4) — tiny/qualitative, not a marathon.

---

## Phases

### Phase 0 — Download + fit probe (no full benchmark)
1. Download both quants (IQ3_XXS + IQ3_S) into the two model dirs; verify SHA-256.
2. Cap all 3 GPUs to 300 W (`nvidia-smi -pl 300`, sudo).
3. Load-probe IQ3_XXS at 8k on GPU 1+2, `--n-gpu-layers 99 --fit off`,
   `CUDA_VISIBLE_DEVICES=1,2`. Confirm mistral4 loads; **assert full residency**;
   capture per-card VRAM + the exact log lines used for the residency assertion.
   Keep GPU 2 (display) load minimal during the test.
   - If 8k is already `not-fully-resident`, nothing higher fits -> record and run a
     `--fit` diagnostic for information.

### Phase 1 — Harness (`tools/mistral_context_sweep.py`, fork of context_speed_sweep.py)
- `MODELS["mistral-xxs"]` / `MODELS["mistral-s"]`: `gpu="1,2"`, ports 8082 / 8083,
  aliases, respective GGUF paths.
- Context list per execution order below.
- Server args: `--n-gpu-layers 99 --fit off --flash-attn on --cache-type-k q8_0
  --cache-type-v q8_0 --parallel 1 --batch-size 2048 --ubatch-size 512
  --threads 32 --threads-batch 32 --metrics --perf`.
  `--tensor-split` even by default (keep as a knob: bias if one card OOMs and the
  other has headroom).
- Remove Qwen-only `enable_thinking` / `reasoning_format`. Speed pass sends
  `chat_template_kwargs={"reasoning_effort":"none"}`, `max_tokens=128`,
  `ignore_eos=True`.
- Add a short **warmup** request before the timed request (cleaner decode tok/s).
- **Residency assertion** from the startup log gates whether a point is scored.
- Extend `gpu_snapshot()` to sample **both** cards (index 1 and 2): VRAM sum,
  power/temp max; also capture PCIe gen/width under load + host RSS + zero-swap.
- `not-fully-resident` points are recorded results (not crashes), with the log
  excerpt as evidence.

### Phase 2 — Throughput + context sweep (execution order)
```
IQ3_XXS:  8k -> 32k -> 128k -> 192k -> 256k
IQ3_S:    128k -> 192k -> 256k   (stop where residency fails)
```
Per scored point record: residency verdict, decode tok/s, prefill tok/s, wall time,
startup + peak VRAM per card, power draw, temperature, PCIe gen/width.

### Phase 3 — Review-mode point
- 128k with `reasoning_effort:"high"`, natural EOS, for each quant that fits.
  Characterizes the actual planner/reviewer/meta-reviewer workload vs the `none`
  speed baseline.

### Phase 4 — Blind reviewer A/B (qualitative)
- 3-5 genuinely hard "find the objection" cases (subtle-bug patches, flawed plans)
  from real code; hand-pick, sanity-check before scoring.
- One fixed adversarial reviewer prompt; `reasoning_effort:"high"`; blind model
  identity where practical.
- Compare IQ3_XXS vs IQ3_S vs the existing Qwen baseline
  (`quality-results/` has corrected Q4/Q5 outputs; `swebench/manifest-q4-q5-40.json`
  has real problem statements as a sourcing pool).
- Metric: valid objections found + objections the other model missed
  (NOT raw code-writing skill). Small and manual.

### Phase 5 — Analysis
- Write `research/mistral-small-4-119b-2603-context-sweep-<date>.md` + `.json`
  (mirror `qwen3.8-27b-context-runtime-benchmark-2026-08-30.md`).
- Verdict: residency per tier, tok/s-vs-context curve, power, A/B readout, and a
  go/no-go for the planner/reviewer/meta-reviewer role (and which quant, if any,
  is the production pick).

---

## Deliverables
- `tools/mistral_context_sweep.py`
- `research/mistral-small-4-119b-2603-context-sweep-<date>.md` + `.json`
- Fit verdict per tier, tok/s curve, power, KV residency, A/B readout, go/no-go.

## Open / non-blocking
- The 3-5 A/B cases: draft during execution, sanity-check with the user before scoring.

## Phase 0 result (executed 2026-09-02)
- **0.1 Download**: both quants downloaded to
  `models/mistral-small-4-119b-2603-iq3-{xxs,s}-gguf/`; **SHA-256 both PASS**
  (XXS `9c2b030b…5b4c8`, S `c56fdf06…80c1b8`).
- **0.2 Power caps**: all 3 GPUs at 300 W (user ran `sudo nvidia-smi -pl 300`).
- **0.3 8k residency probe** (IQ3_XXS, GPU 1+2, `--n-gpu-layers all --fit off
  --flash-attn on --cache-type-k/v q8_0 --parallel 1 --ctx-size 8192`):
  - `offloaded 37/37 layers to GPU` — **FULL GPU RESIDENCY** (no CPU weight offload).
  - Model buffers: `CUDA0 20924.75 MiB` (phys GPU1) + `CUDA1 19462.16 MiB`
    (phys GPU2) = 40,386.9 MiB + 420 MiB `CPU_Mapped` (mmap file map, not active
    CPU compute). Layer split 0-18->CUDA0, 19-36->CUDA1.
  - `flash_attn = enabled` (MLA + FA OK). KV q8_0 on GPU (0-18 CUDA0, 19-35 CUDA1).
  - Coherent generation; process RSS ~1 GiB (confirms no CPU offload).
  - Per-card VRAM at 8k: GPU1 21,424/24,576 MiB (3,152 free), GPU2 19,964/24,576
    MiB (4,612 free). Combined free ~7.6 GiB.
  - Note: real weights ~40.4 GiB on GPU (the "42.8 GB" was decimal GB), so there is
    more headroom than the conservative estimate implied.
- **Fit verdict (8k anchor)**: IQ3_XXS fits fully resident at 8k with ~7.6 GiB
  combined headroom; KV grows ~11.5 KB/token, so 256k adds ~2.8 GB total (~1.4 GB
  per card) — **expected to fit all tiers 8k->256k, 256k being the tightest**.
  Per-tier residency will be re-asserted from the log at each sweep point.
- Logs: `research/mistral-probe-8k-xxs.log` (verbose-3),
  `research/mistral-probe-8k-xxs-verbose.log` (verbose-5, residency lines).
- Qwen production server (GPU 0, port 8080) untouched throughout.

## Phase 1 result (executed 2026-09-02)
- Built `tools/mistral_context_sweep.py` (executable; fork of
  `tools/context_speed_sweep.py`). `py_compile` OK.
- Key changes vs the Qwen harness:
  - Targets `mistral-xxs` (port 8082) and `mistral-s` (port 8083), both GPU "1,2".
  - **No `--kv-offload`** (KV stays on GPU); `--n-gpu-layers 99`, `--fit off`,
    `--tensor-split 1,1` (even, CLI knob), `-lv 4` (minimal verbosity that still
    prints the offload summary).
  - Per-point **residency assertion** parsed from the server log:
    `offloaded N/M layers to GPU` + `CUDA*/CPU model buffer size`. A point that is
    not fully resident is recorded as `not_fully_resident` and **not scored**; the
    model's sweep stops at the first such point (higher contexts need more KV).
  - Short decode **warmup** before the timed request.
  - Mistral chat template: `chat_template_kwargs={"reasoning_effort":"none"}`,
    `max_tokens 128`, `ignore_eos` (Qwen `enable_thinking`/`reasoning_format` removed).
  - Multi-GPU `gpu_snapshot` (per-card VRAM, power/temp peaks, **PCIe gen/width
    under load**) + host `MemAvailable`/`SwapFree` + process RSS, sampled at 1 Hz.
- Validated (no model load): residency parser on the real 8k probe log ->
  `fully_resident=True 37/37 cuda 40386.9 MiB`; synthetic partial-offload -> False;
  multi-GPU/host/RSS snapshots correct. All server flags confirmed present in build
  c841aee.
- Default context order: XXS [8192, 32768, 131072, 196608, 262144];
  S [131072, 196608, 262144]. CLI: `--models`, `--contexts`, `--tensor-split`,
  `--request-timeout` (1800), `--startup-timeout` (360), `--verbosity` (4).

## Phase 2 result (executed 2026-09-02)
All 8 points **passed and fully GPU-resident** (37/37 layers, no CPU weight offload,
`cache_n=0` cold prefill, PCIe **Gen3 x8 under load** — split not degraded). No
`not_fully_resident` anywhere; both quants fit every tested tier. Output:
`research/mistral-small-4-119b-context-sweep/results.json` (+ per-point `*.server.log`).

| model | ctx | decode t/s | prefill t/s | prompt_n | peak VRAM g1/g2 MiB | max °C |
|---|---:|---:|---:|---:|---:|---:|
| xxs | 8k   | 92.9 | 2151 | 7,926    | 21,584 / 20,056 | 69 |
| xxs | 32k  | 78.1 | 1555 | 32,502   | 21,840 / 20,312 | 77 |
| xxs | 128k | 48.9 | 643  | 130,800  | 22,866 / 21,338 | 86 |
| xxs | 192k | 38.7 | 457  | 196,349  | 23,634 / 21,978 | 88 |
| xxs | 256k | 30.1 | 358  | 261,887  | 24,126 / 22,618 | 87 |
| s   | 128k | 48.7 | 636  | 130,809  | 23,762 / 21,978 | 87 |
| s   | 192k | 31.6 | 449  | 196,336  | 24,126 / 22,618 | 87 |
| s   | 256k | 24.2 | 348  | 261,885  | 24,126 / 23,258 | 85 |

Key findings:
- **Fit**: both quants fully resident at all tiers. GPU 1 (layers 0-18) is the tight
  card — peaks 24,126/24,576 MiB (≈450 MiB free) at 256k for both quants. No swap
  used (SwapFree stayed 8 GiB); process RSS ~1 GiB (no CPU offload).
- **Decode t/s vs context** (the planner/reviewer metric): falls ~3x from 8k->256k
  (xxs 92.9 -> 30.1). Even at 256k, 24-30 t/s is a usable reviewer speed; 128k is
  ~49 t/s (xxs) — a comfortable long-context working point.
- **Quant delta**: IQ3_S is ~15-20% slower than IQ3_XXS at equal context (e.g. 256k
  24.2 vs 30.1) for ~1.6 GB more weight and higher precision. S is NOT clearly
  better here; the speed gap is real.
- **Power/temp**: ~300 W (capped), peak 88 °C (under ~93 °C throttle). Sustainable.
- **PCIe**: Gen3 x8 under load at every point (the x8 link is not the bottleneck).

Harness fixes made during Phase 2 (recorded for reproducibility):
- Added `--no-cache-prompt` so the timed prefill is a true cold prefill (comparable to
  the Qwen baseline) and the template-overhead probe is cache-independent.
- Replaced the hard-coded `ctx-384` headroom with a runtime-measured chat-template
  overhead (`measure_template_overhead`, ~557 tokens for Mistral) + 128 output +
  128 margin, sized from `timings.prompt_n`. Without this the prompt overran ctx-size.

## Phase 3 result (executed 2026-09-03)
Review-mode point per quant: 128k with `reasoning_effort:"high"`, **natural EOS**,
full-text capture. Built `tools/mistral_review_sweep.py` (imports the Phase-2
harness; adds a review path: `--reasoning auto`, per-request `chat_template_kwargs
{reasoning_effort:"high"}`, natural EOS, and full `reasoning_content`/`content` +
token-split capture). Output: `research/mistral-small-4-119b-context-sweep/phase3/`
(`summary.json` + per-run `p3-*.json` + `.reasoning.txt`/`.answer.txt`/`.full.txt`).

**Reasoning mechanism (verified):** Mistral's chat template is *prompt-driven*, not a
standard thinking tag. It takes a `reasoning_effort` variable that only accepts
`"none"` or `"high"` (anything else, e.g. Qwen-style `"medium"`, **raises an
exception**). It injects `[MODEL_SETTINGS]{"reasoning_effort":"high"}`; the model
wraps thinking in `[THINK]…[/THINK]`, which llama.cpp's template parser auto-splits
into `message.reasoning_content`. Confirmed live on a `--reasoning auto` server.

**Prompt sizing (deviation, by necessity):** the Phase-2 prompt (130,800 tok) left
only ~272 tokens of output room — infeasible for a natural-EOS reasoning+answer (a
4k-ctx validation showed the model still reasoning when the context ran out). Used
**reserve 8192** → prompt ~120k (user decision). Wall is prefill-dominated
(~185-205 s, ~650 tok/s prefill, ~51 tok/s decode). All 11 runs fully resident
(37/37), no swap. GPU 1 max 81-85 °C under the **275 W** cap (post power-limit
revision) vs 86-88 °C at 300 W in Phase 2.

| quant | seed | temp | finish | wall s | comp | reasoning / answer | max °C |
|---|---|---|---|---:|---:|---|---:|
| xxs | 100  | 0.8 | stop | 204.5 | 847 | 478 / 366 | 84 |
| xxs | 7    | 0.8 | stop | 194.3 | 465 | 367 / 95  | 82 |
| xxs | 1337 | 0.8 | stop | 189.8 | 247 | 140 / 104 | 82 |
| xxs | 42   | 0.8 | stop | 187.9 | **15**  | 13 / **0** (premature EOS) | 84 |
| xxs | rand | 0.8 | stop | 185.0 | **34**  | 32 / **0** (premature EOS) | 81 |
| xxs | 1337 | 0.6 | stop | 196.4 | 412 | 228 / 181 | 84 |
| xxs | 7    | 0.6 | stop | 193.5 | 245 | 202 / 40  | 85 |
| xxs | 42   | 0.6 | stop | 186.9 | **34**  | 32 / **0** (premature EOS) | 84 |
| s   | 42   | 0.8 | stop | 197.0 | 427 | 264 / 160 | 84 |
| s   | 1337 | 0.8 | stop | 194.4 | 308 | 181 / 124 | 84 |
| s   | 7    | 0.8 | stop | 194.3 | 288 | 263 / 22  | 85 |

Key findings:
- **s (IQ3_S): reliable** — 0/3 premature EOS; always completes a coherent review
  (288-427 tok; e.g. seed 42 = 427 tok, 264 reasoning + 160 answer: structured
  objection + follow-up questions).
- **xxs (IQ3_XXS): unreliable at 128k** — **~40% premature-EOS** (3/8 overall, 2/5
  at temp 0.8): stops mid-thought at 15-34 tok with **no answer**. Seed 42 repeated
  at 0.8 and 0.6; a later raw 0.7 diagnostic also stopped mid-`[THINK]` without an
  answer. When it does complete it can be the most thorough
  (seed 100 = 847 tok).
- **Response-reliability verdict (128k, `reasoning_effort:"high"`):** IQ3_S is the
  dependable quant for natural-EOS long-context assistance; IQ3_XXS has a persistent
  early-stopping failure mode at long context. Both are prefill-bound
  (~3.3 min wall at ~120k).
- Representative full-text reviews: `phase3/mistral-xxs-seed100.full.txt` (xxs),
  `phase3/mistral-s-seed42.full.txt` (s). Premature-EOS example:
  `phase3/mistral-xxs-seed42.full.txt`.

## Phase 4 result (executed 2026-09-04)
Four genuinely flawed Qwen SWE-bench patches were used as blind review inputs:
Sphinx local-link checking, Seaborn large-number legend labels, SymPy second-
quantization LaTeX, and Django related-manager async methods. The candidate patches
were unresolved in the Qwen SWE-bench run; valid objections were derived from the
gold patch and test patch and stored in `phase4/scoring_key.json`. The same fixed
adversarial prompt was sent to both Mistral quants at `reasoning_effort:"high"`,
32k context, seed 1337, temperature 0.8. All eight Mistral reviews reached natural
EOS and ran fully resident (37/37 layers).

**Supervision policy revision:** Qwen is a SWE comparison target and the source of
the candidate patches, so it is not used as the Phase 4 supervisor. Per user
direction, Terra independently scored the Mistral answers against the fixed
gold-derived key. Earlier Qwen review requests are retained only as excluded
diagnostic artifacts and do not contribute to the score. The driver now defaults
to Mistral-only reviewers; a Qwen request requires explicit opt-in.

| reviewer | full valid objections | partial | weighted score | total |
|---|---:|---:|---:|---:|
| IQ3_XXS | 1 | 1 | 1.5 | 7 |
| IQ3_S | 2 | 0 | 2.0 | 7 |

Scoring details: `phase4/terra_scoring.json`. IQ3_XXS found the Sphinx
source-document limitation and partially recognized the default-off behavior. IQ3_S
found the same Sphinx limitation and correctly flagged the SymPy change as limited
to power paths rather than the creator printer. Neither quant found the Seaborn
core defect (labels remain offset-stripped), the SymPy tensor omission, or the
Django GenericRelation omission. Both generated false positives; XXS made several
confident but verifiably false claims (including an invalid-if-chain assertion and
an async-recursion claim). S was modestly better here, but neither quant is a
dependable adversarial reviewer without external verification.

## Phase 5 follow-up (executed 2026-09-04)
GPT review correctly noted that static-text scoring cannot alone establish
repository-aware reviewer viability. A new `tools/mistral_agentic_reviewer.py`
initially used mini-SWE-agent, but its marker-only submission protocol could not
capture a review file and those trajectories are excluded from quality evidence.
The harness was replaced with a direct Docker-backed `bash(command)` /
`submit_review(text)` loop, preserving normal tool-message history and removing
all marker/file semantics. With real SWE-bench repositories, 128k context, high
reasoning, 0.7 temperature, and a 30-step phased budget, IQ3_S submitted Seaborn
at step 30 and Django at step 28. The same protocol was then extended to the two
remaining preregistered cases: Sphinx submitted at step 23 but reviewed the base
bug rather than the candidate patch, while SymPy reached step 30 without a
submission. Terra scored the full agentic screen against the hidden key and checked
for additional repository-supported objections: 0/2 Sphinx, 0/2 Seaborn, 0/2 SymPy,
and 0/1 Django. The direct agentic result is therefore 0/7 with a 3/4 submission
rate, supporting the reviewer no-go independently of the earlier scaffold mismatch.
