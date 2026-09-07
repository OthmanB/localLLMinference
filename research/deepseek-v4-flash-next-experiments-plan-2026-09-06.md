# DeepSeek-V4-Flash Next Experiments Plan

Date: 2026-09-06

## Objective

First establish a released-runtime DeepSeek-V4 sparse-Flash-Attention baseline,
then make only narrowly gated DSpark/DFlash, cache, and memory changes.
The selected non-speculative reference is:

- Runtime: `moe-cache-v2-pr` commit `e3096b046bb809f7f80bc47801f6579aed1cbc60`.
- Static experts: `6/7/7` - CUDA0 layers 0-5, CUDA1 8-14, CUDA2 17-23,
  all remaining expert tensors on CPU.
- Cache: `--moe-cache 16000 --no-repack` and
  `GGML_CUDA_MOE_CACHE_RESERVE_MB=512`.
- Shared controls: three GPUs, 12 target and batch threads, Q8 K/V, Flash
  Attention, `--parallel 1`, no prompt-cache reuse, physical VRAM only, and
  Qwen stopped.

Its validated fully populated scores are 29.7709 tok/s at 128k and 29.2438
tok/s at 512k. The 512k result is the primary non-speculative comparison.

## Released v0.4.0 Baseline (Completed)

An isolated official `llama.cpp` v0.4.0 build at commit
`5266f24da75dc449bd56cbed7addb9c8e4a6a73e` is at
`/home/obenomar/.local/share/llama.cpp-v0.4.0-20260906/build/bin/llama-server`.
It was built Release with CUDA 13.2, SM86, NCCL, CUDA Graphs, and Flash
Attention. It is separate from both experimental source trees.

- The requested mixed F16-K/Q8-V control is unsupported for this DeepSeek-V4
  model in the release: context creation rejects different K and V cache types.
- The supported F16/F16 8k control with the pre-cache `8/9/9` static placement
  passed three sanity checks and 20 stability requests at 19.2550 tok/s decode
  and 133.9421 tok/s 4k prefill, with zero swap.
- The F16/F16 128k allocation gate passed, then a 125,992-token populated
  prompt passed at 127.1245 tok/s prefill and 18.6322 tok/s decode, with zero
  swap.
- The long-context log confirms `flash_attn = enabled` and native DSV4 raw,
  compressed CSA/HCA, and lightning-indexer KV caches. All three GPUs were
  Gen3 x8 during the 8k stability snapshot; post-run idle links downclocked to
  Gen1 while retaining x8 width.

Results are under
`research/large-moe-3gpu-probe-2026-09-04/v0.4.0-deepseek/f16k-f16v/`.
Do not compare this release score directly to the cache-fork Q8-K/V 6/7/7
score: cache precision, static placement, and MoE caching differ. Use it as
the no-cache released-runtime sparse-FA reference.

## Released v0.4.0 MoE-Cache Integration (Completed)

The cache fork was ported into the isolated
`/home/obenomar/.local/share/llama.cpp-v0.4.0-moe-cache-20260906` worktree on
branch `moe-cache-v0.4-integration`. The implementation preserves the v0.4.0
DeepSeek MMVQ/fusion path, server output limits, and draft/MTP fit behavior,
then adds a separate cache MMVQ dispatcher. It builds as
`build-moe-cache-cuda-sm86/bin/llama-server` with the same Release CUDA 13.2,
SM86, NCCL, CUDA Graphs, and Flash Attention settings. Neither source binary
used as input was modified.

The first cache gate uses the cache-fork `6/7/7` placement, F16/F16 KV,
`--moe-cache 16000 --no-repack`, and
`GGML_CUDA_MOE_CACHE_RESERVE_MB=512`. This is the supported release cache
precision; mixed F16-K/Q8-V remains unsupported.

- Matched 8k cache-off: 16.6716 tok/s decode and 109.1730 tok/s 4k prefill.
- Matched 8k cache-on: 29.0417 tok/s decode and 109.4368 tok/s 4k prefill,
  a 74.2% decode gain. Three warmups, 20 stability requests, and all output
  checks passed with zero swap.
- The 128k cache-on allocation gate retained nonzero pools on every GPU:
  5.57, 6.05, and 6.36 GiB granted after the 512 MiB reserve.
- Matched populated 128k cache-off: 16.6190 tok/s decode and 102.8035 tok/s
  prefill for 125,992 input tokens.
- Matched populated 128k cache-on: 28.3996 tok/s decode and 102.9826 tok/s
  prefill, a 70.9% decode gain. The cache reported 93.0-93.7% hits, zero
  fill, dispatch, or collect failures, and zero swap.
- The cache-on long-context log confirms Flash Attention and native DSV4 raw,
  compressed CSA/HCA, and lightning-indexer KV caches. Large prompt batches
  bypass the cache by design, so the cache changes decode rather than prefill.

Raw artifacts are under
`research/large-moe-3gpu-probe-2026-09-04/v0.4.0-moe-cache/f16k-f16v/`.
This promotes the isolated cache integration to the DSpark/DFlash startup
gate. It is not a deployment decision or a replacement for representative
quality tests.

## Released v0.4.0 DFlash Gate (Completed, Not Promoted)

The standard DFlash sidecar was tested on the cache-enabled v0.4.0 integration
with `--spec-type draft-dflash`, CPU draft placement, `--spec-draft-n-max 3`,
and `--spec-draft-p-min 0`. The release did not read the no-vocabulary
drafter's `tokenizer.ggml.mask_token_id`; the isolated integration adds the
small metadata fallback. The server then reported the expected
`mask_token_id=128799`.

At the original 512 MiB cache reserve, DFlash required an additional 513 MiB
CUDA2 compute buffer after cache allocation and could not start. A 1 GiB
reserve solved the capacity issue while retaining active target cache pools.

- The matching 8k non-speculative control at the 1 GiB reserve scored 29.3816
  tok/s decode.
- DFlash passed 8k output checks and 20 stability requests with zero swap. It
  drafted 925 tokens, accepted 714 (77.2%), averaged 2.31 accepted tokens per
  verification, and scored 30.9582 tok/s: a 5.4% matched gain.
- The 128k-context allocation screen started but did not pass speculation:
  it drafted 123 tokens, accepted 22 (17.9%), averaged 0.54 accepted tokens
  per verification, and scored 13.3977 tok/s. This was a 4k-prompt capacity
  screen, not a populated 128k score.

The 128k screen fails the required 40% acceptance and positive decode-gain
gate. Do not run a populated 128k DFlash benchmark, tune DFlash further, or
promote it to the 512k path. The completed DeepSeek reference remains the
non-speculative cache-enabled v0.4.0 runtime.

## PCIe Baseline Reset

The former GPU0 Gen3 x4 constraint is no longer applicable. All three cards
now negotiate x8 width; the reported Gen1 current speed while idle is normal
power management, not a throughput result.

Before tuning, recapture the current reference's PCIe trace because the prior
x4 trace is not comparable with the repaired topology.

1. During a warmed 1,024-token 8k decode, record `nvidia-smi dmon -s t -d 1
   -o DT` and a mid-decode link-width/generation query.
2. Require every active card to report Gen3 x8 during the decode. Stop and
   diagnose topology if a card falls below x8 under load.
3. Repeat this trace for every candidate promoted beyond a screen. Compare
   integrated RX plus TX bytes, burst timing, cache evictions, and decode
   throughput; do not optimize hit rate alone.

## Priority 1: DSpark/DFlash Metadata Fix

The standardized dflash artifact is the applicable one. Keep
`--spec-type draft-dflash`; do not call this artifact `draft-dspark`.

The current fork reproduces issue #26761 exactly: `common/speculative.cpp`
reads `llama_vocab_mask()` at line 970, which returns `-1` for this no-vocab
drafter even though GGUF metadata supplies `tokenizer.ggml.mask_token_id`.
The local log consequently prints `mask_token_id=-1` and repeated
`draft: llama_decode returned -1` warnings.

Create a new isolated source copy and build output for this experiment. Do not
alter the validated `llama.cpp-moe-cache-20260905` binary. Apply the issue's
small fallback only in the dflash constructor: if the vocab mask ID is negative,
read `tokenizer.ggml.mask_token_id` with `llama_model_meta_val_str()` and parse
it as a token ID. The known GGUF value is `128799`.

### Build And Startup Gate

1. Build Release CUDA/NCCL/SM86 with the same build options as the validated
   cache fork. Record source commit, patch diff, compiler, and binary path.
2. Run the target with the selected cache configuration and CPU draft sidecar:
   `--spec-draft-device none --spec-draft-ngl 0 --spec-draft-p-min 0`.
3. Require the server log to show `mask_token_id=128799`, cache enabled with
   nonzero target pools, and no `invalid token` or draft `llama_decode` warning.
4. Run the three deterministic output checks before any score. Stop on any
   loader, scheduler, draft-decode, output, UVM, or swap failure.

### 128k Speculation Gate

Run the first scored arm directly at 128k with a 126,000-token prefill, Q8 K/V,
and `n_max=3`. Compare it with the 29.7709 tok/s non-speculative 128k result.
Record drafted tokens, accepted tokens, verification steps, acceptance fraction,
accepted tokens per verification, decode tok/s, actual cache pools, GPU memory,
RSS, and PCIe RX/TX bytes.

The existing harness's stability loop runs before its long prefill and is not a
DSpark long-context safety gate. Before promoting DSpark, add a dedicated
post-prefill sequence runner that records RSS, GPU memory, and speculative
counters after every request. It must execute both:

- One 3,072-token decode after a fully populated 128k prefill.
- Twenty sequential 1,024-token requests after a 64k prefill, leaving headroom
  inside the 128k context so observed growth is not ordinary context overflow.

Promote only if the initial 128k arm has at least 40% acceptance, positive
drafted and accepted counters, no repeated draft errors, no abnormal per-request
memory growth versus the non-speculative control, and better end-to-end decode.
If it passes, sweep `--spec-draft-n-max 2`, `3`, and `4` with three fresh
1,024-token warmed runs each. Validate the best arm with the post-prefill
sequence above before considering a 512k speculation run.

## Priority 2: One Further Static Placement Step

Only after the DSpark gate is complete, test `5/6/6` once:

- CUDA0 expert layers 0-4.
- CUDA1 expert layers 8-13.
- CUDA2 expert layers 17-22.
- Fallback expert rule remains CPU and is last.

Use an interleaved `6/7/7`, `5/6/6`, `6/7/7` warm-screen sequence to control
for thermal drift. Every arm receives a fresh server, unique
`XDG_CACHE_HOME`, 4k prefill, three 256-token warmups, a 1,024-token score, and
a PCIe trace. Stop this branch immediately if `5/6/6` is below the bracketing
`6/7/7` median; do not sweep more CPU-resident placements. If it wins by at
least 0.3 tok/s and retains stable cache pools, run three confirmations and one
fully populated 128k gate before replacing the placement reference.

## Priority 3: Cache Churn Controls

Keep the selected placement and 512 MiB reserve fixed. These variables are
implementation controls read at scheduler creation, so every arm needs a fresh
server:

1. Establish a fresh control with adaptive admission and throttle 8.
2. Sweep `GGML_CUDA_MOE_CACHE_ADMIT_AFTER=8,16,32,64`, retaining
   `GGML_CUDA_MOE_CACHE_THROTTLE=8`.
3. Promote at most the top two admission settings whose 8k score is not below
   control. Confirm each three times with one-second PCIe RX/TX telemetry.
4. For the best admission setting only, sweep throttle `8,16,32`.

Score candidates by sustained decode tok/s first, then total PCIe bytes and
evictions/fills. A lower hit rate is acceptable if decode improves and traffic
or churn falls. Reject any arm with output failures, swap activity, cache
fill/dispatch/collection failures, capacity collapse, or a confirmed decode
regression of at least 0.3 tok/s. Promote at most one winner to 128k; run 512k
only if its populated 128k decode remains at least 25 tok/s.

## Priority 4: Sparse Flash-Attention Rebase

Do not rebase the validated cache binary in place. First identify and record the
exact upstream DeepSeek-V4 sparse-Flash-Attention merge commit and its Q8-KV
fallback behavior. The current guidance is that sparse FA benefits f16 KV but
Q8 falls onto a slower extreme-context path, so a blind rebase is not a speed
claim.

1. Create a separate integration branch from the cache fork and rebase or
   backport only the required sparse-FA commits.
2. Build an isolated binary and run cache-off and cache-on 8k smoke controls.
3. Compare the rebased Q8 configuration with the selected current Q8 reference
   at fully populated 128k before any 512k run. Stop the branch on a confirmed
   Q8 decode regression.
4. Attempt f16 KV only after a 128k physical-VRAM capacity preflight confirms
   nonzero cache pools and at least a 512 MiB reserve. Do not enable UVM to make
   it fit. If f16 cannot satisfy that gate, defer sparse FA rather than changing
   the long-context production candidate.

## Priority 5: Host Memory And Mmap

These are prefill-focused A/B tests, not expected primary decode gains at the
current 94-96% cache hit rates.

1. At current DDR4-2933, compare default mmap with `--no-mmap` at 128k using
   the selected non-speculative configuration. Record prefill, decode, RSS,
   `MemAvailable`, and swap activity.
2. Change one hardware setting at a time: 2933 baseline, then 3200, then 3333
   only if the prior setting completes independent memory-stability validation.
3. For every stable memory setting, repeat the same 128k mmap A/B. Do not begin
   with 512k or combine an unvalidated RAM overclock with a code/runtime change.
4. Treat a few-percent prefill change as plausible; require a repeatable decode
   improvement before changing the selected decode reference.

## Shared Rules

- Keep the Qwen service stopped, `SwapTotal=0`, no UVM environment variable,
  and no concurrent `llama-server` before every score.
- Preserve raw result JSON, server logs, link snapshots, and timestamped PCIe
  traces under a unique arm directory.
- Keep the original and experimental runtime binaries separate. No deployment
  decision follows from these narrow deterministic checks; use a representative
  task suite before promotion.

## Sources

- DSpark/DFlash no-vocab mask-token issue:
  `https://github.com/ggml-org/llama.cpp/issues/26761`
- Local cache controls:
  `llama.cpp-moe-cache-20260905/docs/backend/CUDA-MOE-CACHE.md`
- Local dflash constructor:
  `llama.cpp-moe-cache-20260905/common/speculative.cpp`
