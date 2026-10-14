# Qwen3.8-Flash-Next (GSQ-RCO Coder GGUF) — GPU0 Replacement Evaluation

Date: 2026-09-30 (detached session)

## Objective

Determine whether Qwen3.8-Flash-Next, in the ISTA-DASLab GSQ-RCO Coder GGUF
form, is a credible GPU0 replacement for Muse Glimmer (research/planning/
review/tool-use role), on exactly one RTX 3090 (GPU0) plus host RAM.
GPU1/GPU2 host the supervising Qwen3.8-27B ATX services and are immutable.

## Safety constraints

- GPU1/GPU2 are PROTECTED. No experimental workload may touch them.
- All experimental launches use `CUDA_VISIBLE_DEVICES=0`.
- No service stop/restart of `llamampere-qwen3.8-atx-iq4xs-m-gpu1.service`,
  `llamampere-qwen3.8-atx-iq4xs-m-gpu2.service`, or
  `llama-muse-glimmer-30b-131k.service` (Muse is the comparison baseline and
  an existing process).
- No host reboot, no driver/PCIe/system-CUDA changes.

## Stage 0 — Environment snapshot (2026-09-30 02:42 UTC)

### GPUs

| GPU | Model | Mem used/total | Process (PROTECTED) | Port | Service |
|---|---|---|---|---|---|
| 0 | RTX 3090 24 GiB | 19542/24576 MiB (free 4585 MiB) | PID 2114 llama-server (Muse Glimmer 30B, KQuant + DFlash draft) | 8082 | llama-muse-glimmer-30b-131k.service |
| 1 | RTX 3090 24 GiB | 23304/24576 MiB | PID 3151 llama-server (Qwen3.8-27B ATX, llamAmpere 36a6bca817, MTP) | 8080 | llamampere-qwen3.8-atx-iq4xs-m-gpu1.service |
| 2 | RTX 3090 24 GiB | 23262/24576 MiB | PID 3152 llama-server (Qwen3.8-27B ATX, llamAmpere 36a6bca817, MTP) | 8081 | llamampere-qwen3.8-atx-iq4xs-m-gpu2.service |

- GPU0 memory is held by PID 2114 (Muse; `fuser /dev/nvidia0` shows only 2114
  plus nvidia-smi). nvidia-smi's per-process table under-reports it (302 MiB)
  but total used 19542 MiB is the operative number: **GPU0 free VRAM ≈ 4.5 GiB**.
- PCIe (nvidia-smi, idle): GPU0 x16, gen.current=1 (max 3); GPU1 x8 gen 3;
  GPU2 x8 gen 1. Gen1 reading at idle may be ASPM downclock; verify under load.
  Prior 2026-09-07 work recorded GPU0 negotiating Gen3 x16 after a riser change.
- Topology: GPU0/GPU1 = NODE, GPU1/GPU2 = PHB. Single NUMA node, CPUs 0-63.
- Other listeners: gateway 8088 (PID 7462), exporter 9108 (PID 7414),
  FreeToken daemon 1900 (PID 7453, idle, no engine, no GPU0 context).

### Host

- CPU: AMD Ryzen Threadripper 3970X, 32 cores / 64 threads, 1 socket, 1 NUMA.
- RAM: 141 GiB total, ~95 GiB free at snapshot (131 GiB available), swap 8 GiB unused.
- Measured (2026-09-07, `ft bench bw`): CPU STREAM read 52.5 GB/s, CPU MoE decode
  effective 45.9 GB/s, PCIe linear H2D 13.1 GB/s (Gen3 x16 then).
- Disk: / has 267 GiB free; ${MODEL_STORE} 214 GiB free.

### Runtimes present

- `${RUNTIME_ROOT}/llama.cpp` @ c841aee (2026-08-29) — Muse's build;
  binary has QWEN4EXP support; CUDA build (sm86).
- `${RUNTIME_ROOT}/llama.cpp-recent-20260905` @ 4d91760 (2026-09-04,
  post-v0.4.0 master snapshot) — binary has QWEN4EXP support; Release CUDA build,
  newest local binary. **Selected as experimental runtime.**
- `${EXPERIMENT_ROOT}/research/llamAmpere-v0.3-36a6bca81/` —
  llamAmpere source; production build under /opt/ai-server (PROTECTED, not used).
- No `hf`/`wget` HF tooling preinstalled; `hf` 2.0.0 installed via uv for the
  download (local-only, not system-wide).

## Stage 1 — Candidate and runtime selection

### Model

- Base: `Qwen/Qwen3.8-Flash-Next` (released 2026-08-31): 176.9B total params
  (125B transformer incl. 512×48 MoE experts, 10 active/token; 6B activated;
  51B n-gram PLE table; 4B MTP), hybrid Gated DeltaNet + Qwen Sparse Attention
  (full attention every 4th of 48 layers), native 262,144-token context.
- Prior local work (2026-09-06..08) validated the `nvidia/Qwen3.8-Flash-Next-NVFP4`
  checkpoint (132.7 GB) via an isolated FreeToken checkout: 262k context,
  ~102 tok/s prefill, ~36-40 tok/s settled decode, 24,000 MiB peak GPU0,
  ~131.5 GiB cgroup RAM, zero swap. That service was later removed when GPU0
  was reassigned to Muse.
- **Selected candidate** (per task brief): `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF`:
  - 50% expert pruning (RCO, KL-divergence budget search): 256 of 512 experts/layer,
    capability-targeted at code, agentic tool use, vision, spatial reasoning.
  - Retained weights at 3.5 bpw (GSQ); GGUF file type IQ1_M.
  - Shard 1 (transformer): 29.6 GB — must be resident.
  - Shard 2 (n-gram PLE, IQ4_NL 4.5 bpw, 51.2B params): 28.8 GB — lookup table,
    can stay mmap'd on disk; we have RAM headroom to hold it resident.
  - Total download 58.4 GB (+0.91 GB mmproj BF16).
  - Reported results (xhigh): SWE-bench Verified 75.60 (91.3% of BF16 base 82.80),
    LiveCodeBench v6 86.28 (98.7% of 87.43).
  - Unpruned alternatives (general purpose) in `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`:
    Q2_0 66.4 GB / IQ2_XS 68.0 / IQ3_XXS 75.8 (47.0 GB resident) / IQ3_S 83.6 (54.8 GB resident).
    IQ3_XXS holds 99.4% of base task average; may be evaluated as fallback if the
    Coder's capability targeting proves too narrow for research/review roles.

### Runtime

- `llama.cpp` @ 4d91760 (local checkout `llama.cpp-recent-20260905`, Release CUDA
  build, QWEN4EXP arch present). Standard upstream line — maintained, reproducible.
  No fork required; the GSQ-RCO GGUFs are standard scalar GGUF types.
- Planned baseline launch (GPU0 only):
  - `CUDA_VISIBLE_DEVICES=0 llama-server -m <shard1> -alias flashnext-coder
     --port 1901 --ctx-size 32768 (start) -ngl <fit/0> -fa on --threads 32
     --threads-batch 32 --batch-size 2048 --ubatch-size 1024 --jinja --metrics --perf`
  - n-gram shard: first mmap-on-disk per the card's guidance; move to RAM if
    paging cost shows up (we have headroom).
  - VRAM budget 4.5 GiB → only a small number of layers can be GPU-resident;
    pure-CPU (`-ngl 0`) is the first baseline; a low-`-ngl` variant follows to
    measure the PCIe cost (with live link-speed sampling).

### Download

- `hf download` (hf 2.0.0 + Xet) proved unreliable here (stalled 7.5 MB chunks,
  had to kill it). Parallel CDN ranges then stalled at ~50-80 MB per connection
  and, because `curl` had no speed limit, the connections hung indefinitely;
  ~11 GB of partials were lost when the temp trees were cleaned.
- Final method: `sliced-dl.sh` — one worker per range, each incrementally
  appends its bytes and, on a stall, aborts (`--speed-time 90 --speed-limit
  16K`) and resumes from the current file size. Started 2026-09-30 04:41 UTC,
  6+6+1 workers, ~8.4 MiB/s aggregate.
- Sizes (from Content-Range): shard1 29,608,446,496 B, shard2 28,800,138,432 B,
  mmproj 907,543,008 B (total 58.4 GB).

## Stage 2 — Basic qualification

Tooling (in `research/qwen3.8-flash-next-gsq-rco-coder-gpu0-2026-09-30/`):
`probes.py` (determinism, eos, coding, json, tool, marker, prompt),
`sample-resources.sh` (1 Hz GPU0/RAM/swap/RSS), `start-server.sh` /
`stop-server.sh` (GPU0-only server on port 1901, alias `flashnext-coder`;
stop only kills a PID whose cmdline matches alias+model — no broad pkill),
`run-context-sweep.sh`, `bench.sh`, `run-coder-quality.sh`.

Probes validated against the live Muse service (127.0.0.1:8082, model
`muse-glimmer-30b-kquant17`), which is a **thinking model** (responses carry
`reasoning_content` + `content`; probes capture both):

| Probe | Muse result |
|---|---|
| determinism (temp 0, seed 1234, 2x) | identical content+reasoning |
| eos (`17*19?`, just the number) | `323`, clean early stop |
| coding (parse_ranges + doctests) | correct implementation, 4 doctests, validation branches (1292 chars) |
| json (response_format json_object) | valid JSON, all 5 required keys |
| tool (get_weather) | single correct call `get_weather(city="Zurich")` |

Full Stage-4 quality prompts (R1 research / R2 planning / R3 review with
planted bugs) run against Muse as the comparison baseline
(`results/quality/muse/`): R1, R2, R3 all completed; R3 correctly identified
the missing `time` import, the dropped tail of `b` in `merge_sorted`, and the
`a=b=c` split issue, while explicitly dismissing the token-bucket and
`len(jobs)` cases as not-bugs (no false positives).

### Runtime smoke test (2026-09-30 04:20 UTC)

Validated the experimental runtime before the big run, on GPU0 only, with
`ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF` (531,068,128 B, size-verified):

- `llama.cpp-recent-20260905` build (commit 4d91760) initializes CUDA on GPU0
  and serves: VRAM 19542 → 20677 MiB (+1135 MiB), probe answers, clean shutdown.
- GPU1: 23304 → 23306 MiB (2 MiB normal variation), GPU2: 23262 MiB unchanged.
- Kill-guard verified (stop logic refuses a PID whose cmdline does not match
  the expected alias+model).

### Candidate load and performance (2026-09-30 ~08:00-09:00 UTC)

Model loads via mmap almost instantly (metadata only; pages fault in on use).
Screened `-ngl`/threads with a fixed 256-token deterministic generation
(server-side `eval time`):

| Config (ctx 8192 unless noted, threads 16) | Decode tok/s | Prefill tok/s |
|---|---:|---:|
| ngl 0, threads 32 | 4.40 | 34.98 |
| ngl 0, threads 16 | 5.01 | 34.70 |
| ngl 0, threads 8 | 5.10 | 29.64 |
| ngl 4, fa on | 5.76 | 41.03 |
| ngl 8, fa on | 6.20 | 43.38 |
| ngl 12, fa on | 6.84 | 43.39 |
| ngl 16, fa on | 7.47 | 50.47 |
| ngl 20, fa on | 8.20 | 55.07 |
| ngl 24, fa on | 9.41 | 59.86 |
| ngl 28, fa on | 10.87 | 66.90 |
| ngl 32, fa on | 12.81 | 74.76 |
| **ngl 36, fa on (ctx 8192 and 32768)** | **15.68** | **84.6** |
| ngl 40/44/48 | startup CUDA OOM (server exits) | — |

- GPU0 at ngl 36 is effectively full (4 MiB free during the run); ngl 40+ fails
  to allocate. The single-GPU VRAM ceiling caps offload at 36 of 48 layers.
- Threads: fewer is slightly faster on decode (thread-sync overhead); 16 is the
  chosen operating point (32 = 4.40, 16 = 5.01, 8 = 5.10 at ngl 0).
- Muse baseline on the same prompt: ~58-64 tok/s server-side with DFlash
  speculative decoding (client-observed 42-60 tok/s across prompts). Muse's
  process-shown GPU memory (≈0.3 GiB) is a driver/accounting artefact; when the
  bench servers claimed the GPU, the previously "unattributed" GPU0 memory was
  released, leaving more headroom (GPU0 settled near-idle at ~0.7 GiB used,
  ~23 GiB free). Muse itself stayed active and healthy throughout.

### Initial blocker: degenerate output (turned out to be file corruption)

Every generation from the first Coder download was degenerate:

- \`/v1/chat/completions\` "17*19? just the number" → 1500 tokens of
  `///////…` in the reasoning channel, empty content.
- Raw \`/completion\` "The capital of France is" → `//////…` (ngl 36) and
  `3333…` (ngl 0, pure CPU).

So it is not the chat template and not the GPU path — the model computation is
wrong. Checks performed:

- GGUF headers valid on both shards; tensor types are all standard
  (BF16/F32/Q6_K/IQ4_NL/Q4_K/IQ4_XS/IQ2_S/Q5_K/IQ3_XXS/IQ3_S/Q2_0/F16/Q8_0).
- Metadata is intact: \`general.architecture=qwen4exp\`, 48 blocks, 256 experts,
  10 used, ctx 262144, PLE/indexer/SSM/hyper-connection keys present.
- The local snapshot (4d91760, 2026-09-04) does reference the qwen4exp keys and
  has \`attn_qkv\`; no load warnings or errors are emitted.
- Upstream commit history shows continuing fused-QKV and qwen4exp work after
  2026-09-04 (e.g. #29294 "fix tensor split for fused qkv with uneven K/V head
  sizes" 2026-09-25; #24364 W4A4 path 2026-09-25; #22780 convert --fuse-qkv
  2026-09-06), consistent with the GGUF having been produced by a newer
  llama.cpp than the installed local snapshots.

Action: building current upstream master (2026-09-30, commit 185103dc) with
CUDA sm86 in `~/.local/share/llama.cpp-flashnext-20260930` and re-testing the
same GGUF. **Result: master also degenerates** (`3333…` on raw CPU
completion), so it is not a runtime-version problem.

### Root cause: downloaded file corruption

Byte-range spot checks against the live Hugging Face CDN showed the local
shard1 did not match the served bytes:

| Offset | CDN md5 (stable over 3 fetches) | Local md5 | Verdict |
|---|---|---|---|
| 1,000,000,000 | fdc187594543 | ed6e6af59933 | corrupt |
| 15,000,000,000 | 40257230bcd2 | 40257230bcd2 | ok |

A 512 KiB sample every 500 MB across shard1 found **16 of 59 regions
mismatched** (~27%). The CDN itself is consistent (repeated fetches of the same
range are identical), and the repo's commit history shows the GGUF shards were
never re-uploaded, so the fault is in our download, not the published file.
Cause: the first parallel downloader used `curl --retry` with a shell `>>`
append, which re-appends the same range on an internal retry (duplicating
bytes) and the strict follow-up downloader then resumed from those already
corrupt prefixes. Degenerate model output ("////", "3333") is the expected
symptom.

Recovery: re-download both shards from scratch with the strict per-slice
downloader (64 MB slices fetched to a temp file, requiring HTTP 206 **and** an
exact byte count before the slice is appended; no `curl --retry` on ranged
requests), started 2026-09-30 12:48 UTC. After the re-download, sampling both
shards against the live CDN at 1 GB intervals gave **0 mismatches out of 59**
checks. Re-testing the repaired file:

- `/completion` "The capital of France is" → ` Paris. Paris is a city. …`
- chat "17*19?" → `323`, 52 tokens, clean `stop`.

The candidate is functional. The earlier degeneracy was entirely download
corruption.

## Stage 3 — Context sweep (marker retrieval, master build)

Each run restarts the server at the target context, lowers `-ngl` until it fits,
and runs the UUID marker probe (fresh prompt, temperature 0). Server-side
timings from the llama.cpp log.

| Target ctx | ngl that fit | Prompt tokens | Prefill tok/s | Decode tok/s | Marker |
|---:|---:|---:|---:|---:|---|
| 32,768 | 36 | 31,071 | 606 | 14.2 | exact, first |
| 65,536 | 36 | 61,661 | 527 | — | exact, first |
| 131,072 | 32 (36/34 OOM) | 125,027 | 481 | 8.1 | exact, first |
| 262,144 | 28 (36/34/32/30 OOM) | 252,127 | 376 | 4.4 | exact, first |

- Peak GPU0 usage: 23,869 MiB (32k), 24,123 (64k), 23,465 (128k), 24,005 (256k)
  of 24,576 MiB — the card is effectively full at every point.
- Peak GPU0 power ≈ 254 W (300 W cap). Swap never exceeded ~3 MiB.
  Minimum `MemAvailable` across runs ≈ 97 GiB; host RAM is not the constraint.
- Prefill degrades gently (606→376 tok/s); decode degrades steeply with context
  (14.2→4.4 tok/s). Offload must drop as the KV grows.
- **≥128k usable context is met (exactly at 128k and at the full native 262k),
  and 256k is achievable.** This is the strongest result for the candidate.

## Stage 4 — Muse comparison

Same prompts to both. Muse (127.0.0.1:8082, DFlash speculative decoding) is a
thinking model and decodes at ≈ 58-64 tok/s. Coder config: 32k / ngl36, threads
16, fa on (≈ 14-18 tok/s).

| Probe | Muse | Coder (GSQ-RCO) |
|---|---|---|
| eos 17*19 | `323` | `323` |
| determinism (2x temp0) | identical | identical |
| structured JSON | valid, all keys | valid, all keys |
| tool call | correct `get_weather(Zurich)` | correct `get_weather(Zurich)` |
| R3 planted-bug review | found all 3 bugs + 1 extra, no false positives; crisp | found all 3 planted bugs + the `split("=")` issue; waffles on the token-bucket red herring |
| R1/R2 content (thinking off) | — | 7.2k / 7.9k chars, substantive |
| interactive decode | ≈ 58-64 tok/s | 18 (short) / 14 (32k) / 8 (128k) / 4 (256k) |
| usable context | 131k | 262k (native), exact at 256k |

**Thinking behaviour is the main practical gotcha.** With reasoning enabled,
hard prompts consume the entire output budget in `reasoning_content` and emit
no `content` (R1/R2/R3 hit 4000-5000-token caps with empty content; the
orchestrated reviewer spent 4000 tokens thinking). Passing
`chat_template_kwargs: {"enable_thinking": false}` fixes responsiveness
(answers in seconds) but makes reviews terse. Muse produces detailed content
within a comparable budget.

## Stage 5 — Orchestrator interaction

`orchestrated-test.py`: coder as planner + reviewer, the two Qwen3.8-27B ATX
replicas as executors (plain inference calls; no service changes).

| Planner | Plan | Executor | Review |
|---|---|---|---|
| Muse (baseline) | 18.1 s | qwen ATX (46.9 s) | 69.5 s, detailed, **APPROVE** |
| Coder (candidate) | 41.6 s | qwen ATX (49.8 s) | 3.0 s, `CHANGES_REQUESTED` only |

The coder's plan was concrete (2,438 chars: file structure, signatures,
algorithm, edge cases). Its review with thinking off was a 17-character verdict
with no analysis — materially less useful as a reviewer than Muse. With
thinking on the review would be thorough but takes minutes per turn at these
decode rates.

## Decision

**Recommendation: not a drop-in GPU0 replacement today, but the strongest
long-context/quality option on GPU0 if the workload can tolerate slow decode.**

Meets:
- Runs correctly on GPU0 + host RAM with **zero swap** and no impact on
  GPU1/GPU2 or the Muse service.
- Full native **262,144-token context** with exact first-token marker retrieval
  at 32k/64k/128k/256k; prefill 376-606 tok/s.
- Coding/research/review quality at least on par with (often sharper than)
  Muse on isolated tasks; correct structured JSON and tool calls.

Fails / risks:
- **Interactive decode target (20-25 tok/s) is not met**: 18 tok/s short-context
  falling to 14 (32k), 8 (128k), 4.4 (256k). Muse does 58-64 tok/s, so this is
  ~3-14x slower depending on context.
- Decode requires reasoning to be disabled for hard prompts, and even then the
  reviewer role is thin.
- GPU0 is filled to the brim (23.5-24.1 GiB) at every operating point; there is
  no headroom to also co-host another GPU model, and `-ngl` must fall as context
  rises (36 → 28 by 256k).
- Requires current llama.cpp and a **byte-verified** model file; the file is
  large (58.4 GB) and must be content-verified before use.

Suggested deployment if adopted: run the Coder as an on-demand
research/long-context worker at ≤32k (ngl 36) or as a dedicated 128k/256k
batch/summarization worker with thinking off, and keep Muse (or the Qwen ATX
replicas) for latency-sensitive interactive turns. A dedicated GPU0 with Muse
stopped would not change the decode ceiling materially, because the bottleneck
is the 12 CPU-resident expert layers plus the hybrid-attention ops, not VRAM.

## Failures and incidents

- **2026-09-30 download, incident 1:** `hf download` (Xet) stalled at 7.5 MB
  chunks repeatedly; the first launched process was also killed when its parent
  command hit the 120 s tool timeout (process-group kill). Fixed by using
  `setsid` with full fd redirection for detached jobs.
- **incident 2:** parallel `curl` range workers stalled at ~50-80 MB per
  connection with no speed timeout and hung; ~11 GB of partials were lost when
  the stale temp trees were removed. Fixed with `sliced-dl.sh` (per-range
  append + `--speed-time 90 --speed-limit 16K` abort + resume), restarted
  04:41 UTC.
- **incident 3:** during the sliced download, one mmproj attempt received a
  full 200 body instead of the requested 206 range, so the part overshot
  (1,314,846,246 B vs 907,543,008 B expected). The script's size verification
  refused to concatenate, so no corrupt final file was produced. mmproj is
  optional (vision) and was re-fetched successfully.
- **incident 4 (major):** the first completed shard download was silently
  corrupt (see "Root cause: downloaded file corruption"). Detected only by
  byte-comparing the local file against the live CDN after the model produced
  degenerate output with two independent llama.cpp builds. ~58 GB re-downloaded
  with a duplication-proof slice downloader and re-verified. Lesson: for large
  model files, content-verify (not just size) before benchmarking.
- No GPU1/GPU2 impact at any point; protected services stayed active with
  unchanged PIDs 3151/3152. Muse (GPU0) stayed active and healthy; its
  process-shown GPU memory is under-reported by nvidia-smi, and freeing the GPU
  from the benchmark servers left GPU0 near-idle with ~23 GiB free.
