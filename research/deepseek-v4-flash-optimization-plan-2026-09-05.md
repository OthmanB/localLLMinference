# DeepSeek-V4-Flash Optimization Plan

Date: 2026-09-05

## Objective

Raise practical local DeepSeek-V4-Flash throughput beyond the established
physical-VRAM manual-placement baseline without sacrificing output correctness
or 128k-context viability.

This is an experiment plan, not a commitment to change the production Qwen
service or install an experimental runtime as production infrastructure.

## Established Baselines

The stock-baseline figures below use the isolated current llama.cpp build at
`4d917609`; expert-cache figures use the separate `moe-cache-v2-pr` build at
`e3096b046`. All use three RTX 3090s, swap disabled, and no CUDA
unified-memory fallback.

| Configuration | Prefill tok/s | Decode tok/s | Intended use |
|---|---:|---:|---|
| Manual layer placement, 8k, 4k prompt | 76.5089 | 17.7190 | Decode-first interactive work |
| Manual layer placement, 8k, 4k prompt, 12 threads | 76.3661 | 18.1895 | Selected decode-first interactive setting |
| Expert cache, 8k, 4k prompt, 12 threads, warmed | 76.0162 | 28.2630 | Warm decode confirmation |
| Expert cache, 128k populated, 126k prompt, warmed | 70.0668 | 27.7022 | Long-context decode candidate |
| Expert cache, 256k populated, 257k prompt, warmed | 66.3797 | 26.5615 | Above 25 tok/s decode |
| Expert cache, 512k populated, 519k prompt, warmed | 57.2680 | 24.1073 | Below 25 tok/s decode |
| Manual layer placement, 32k allocated, 4k prompt | 76.4502 | 17.6697 | 32k capacity allocation |
| NCCL tensor mode, 8k, 4k prompt | 128.1106 | 13.5994 | Fresh long prompt and short output |

Manual placement maps expert layers `0-7` to CUDA0, `8-16` to CUDA1,
`17-25` to CUDA2, and `26-42` to CPU. It uses approximately
`21.9 / 21.2 / 21.3 GiB` of actual VRAM across GPUs 0/1/2.

The earlier `GGML_CUDA_ENABLE_UNIFIED_MEMORY=0` setting enabled unified memory
because llama.cpp checks environment-variable presence. Do not use that setting
in new scored experiments. All candidates must use physical VRAM only.

## Workload Selection

For a prompt of P tokens and a generated response of G tokens:

```text
T = P / prefill_tps + G / decode_tps
```

Using the measured medians, manual layer placement becomes faster than tensor
mode when `G / P > 0.308`. With the approximately 4k-prompt / 256-token
workload, predicted and measured wall time favor tensor mode:

| Mode | Predicted wall time | Measured wall time |
|---|---:|---:|
| Manual layer placement | about 66.7 s | 68.4 s |
| NCCL tensor mode | about 50.0 s | 51.5 s |

Use manual layer placement for cached prefixes and decode-heavy interaction.
Use tensor mode for uncached large prompts followed by short answers. Add a
separate prompt-cache workload for agent loops because prior scored harnesses
used `--no-cache-prompt` deliberately and do not represent prefix reuse.

## Common Controls

Apply these rules to every candidate unless the candidate explicitly tests one
of them:

- Use the isolated llama.cpp build and record its commit and CMake options.
- Keep Qwen stopped and swap disabled.
- Use the same DeepSeek GGUF, `--parallel 1`, Flash Attention, and fixed
  approximately 4k-prefill / 256-token decode workload for screening.
- Start each candidate with a fresh server process and record startup time,
  host memory, swap counters, GPU telemetry, PCIe state, and server RSS.
- Require the existing arithmetic, factual, and exact-echo output sanity gate.
- For speculative decoding and expert cache, also record a multi-request
  stability run rather than relying only on fresh-process results.
- Compare cache on and off using the same binary.

Do one screening run per value. Repeat the two best values three times. Do not
interpret differences below observed run variation as improvements.

## Execution Order

| Priority | Experiment | Success signal | Stop condition |
|---:|---|---|---|
| 1 | CPU thread and affinity sweep | >=5% decode gain or stable lower CPU cost | No result exceeds baseline variation |
| 2 | CPU-resident DSpark | >=40% acceptance and better end-to-end decode | Acceptance below 40% |
| 3 | GPU0 DSpark | Better than CPU DSpark without OOM or leak | OOM, corruption, or unstable VRAM |
| 4 | Expert-cache fork | >=20 tok/s warm decode with stable output | Regresses cold path or fails stability gate |
| 5 | Power and one-layer placement tuning | Reproducible incremental gain | No benefit beyond variation |
| 6 | Hardware changes | Re-measure after DDR4-3200 or riser repair | Do not infer from hardware theory alone |

### 1. CPU Thread and Affinity Sweep

Screen the physical-VRAM manual placement at:

```text
--threads 12
--threads 16
--threads 24
--threads 32
--threads 40
```

The CPU-resident 17 expert layers can be RAM-bandwidth or synchronization
limited, so CPU utilization alone is not sufficient evidence that 32 threads
is optimal. After screening, test strict placement only for the best two:

```text
--cpu-strict 1 --prio 2 --poll 100
```

Determine physical-core and CCD topology before applying `--cpu-mask` or
`--cpu-range`; do not guess a Threadripper affinity layout. Compare strict
affinity against the same thread count without affinity.

Result: completed on 2026-09-05. The one-run screen placed 12 threads first,
then three confirmations measured a median `18.1895 tok/s` decode and
`76.3661 tok/s` prefill. This is a reproducible 2.655% decode improvement over
the 32-thread manual baseline, but below the 5% priority-1 success threshold.
Testing 12 threads with strict primary-thread ranges `0-31`, high priority,
and full polling regressed decode by 6.329%. Retain ordinary scheduler
placement with 12 threads. See
`large-moe-3gpu-probe-2026-09-04/recent-build-thread-sweep.md`.

### 2. CPU-Resident DSpark

First validate current binary spellings with `llama-server --help`. The planned
starting sidecar is the smaller approximately 6.97 GB Q2_K/Q8_0 DSpark drafter,
not the approximately 10.90 GB MXFP4 variant.

Expected starting shape:

```text
--spec-draft-model <DSpark-Q2_K-Q8_0.gguf>
--spec-type draft-dspark
--spec-draft-n-max 3
--spec-draft-p-min 0
--spec-draft-device none
--spec-draft-ngl 0
--spec-draft-type-k q8_0
--spec-draft-type-v q8_0
```

Record accepted draft tokens per target evaluation, acceptance fraction, draft
time, target verification time, end-to-end wall time, and steady decode rate.
If acceptance exceeds 50%, screen `--spec-draft-n-max 2,3,4`. CPU drafting may
contend with CPU-resident target experts for RAM bandwidth, so it must beat the
non-speculative baseline end to end, not merely draft tokens quickly.

Result: blocked on 2026-09-05 before serving a request. The legacy Q2_K/Q8_0
artifact is not loadable by this build because its
`deepseek_v4_flash_dspark_draft` architecture is unknown. The checksum-verified
standardized `dflash` artifact loads, but its CPU-only context aborts during
graph reservation because a CUDA2 `output.weight` cannot run operation `NONE`.
No acceptance or throughput score exists. See
`large-moe-3gpu-probe-2026-09-04/recent-build-dspark.md`.

Sources to validate before download and execution:

- [DSpark drafter artifacts](https://huggingface.co/alessandrobologna/DeepSeek-V4-Flash-0731-DSpark-Drafter-GGUF)
- [Unsloth DSpark discussion](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/discussions/36)

### 3. GPU0 DSpark

Only run this if CPU DSpark fails the acceptance or throughput gate. Free GPU0
space by changing target expert placement to `5 / 9 / 9` first:

| Expert layers | Target |
|---|---|
| 0-4 | CUDA0 |
| 8-16 | CUDA1 |
| 17-25 | CUDA2 |
| All others | CPU |

Then place the drafter on GPU0:

```text
--spec-draft-device CUDA0
--spec-draft-ngl all
```

If capacity allows, restore layer 5 and test `6 / 9 / 9`. Test at least 20
sequential requests with VRAM snapshots because fresh-server scoring can hide a
per-request DSpark leak. Reject the configuration on any output-gate failure,
growing VRAM footprint, or OOM.

Result: blocked on 2026-09-05. The standardized `dflash` sidecar also aborted
with the same CUDA2 scheduler assertion after target placement was reduced to
`5 / 9 / 9`; it was not an OOM. Do not run the 20-request stability test until
the runtime compatibility failure is resolved.

Relevant issue to recheck before execution:

- [DeepSeek-V4 DSpark VRAM leak report](https://github.com/ggml-org/llama.cpp/issues/27155)

### 4. Expert-Cache Fork

Expert caching has the highest upside but is experimental. Build leloch's
`moe-cache-v2-pr` branch in a separate directory and preserve the current
NCCL build unchanged.

Initial same-binary comparison:

```text
--moe-cache 0
```

against:

```text
--cpu-moe
--moe-cache 16000
```

If stable and capacity-safe, test `18000`. Measure cold first 256 tokens and
warmed final 512-1024 tokens separately. Use multiple representative coding
tasks so a repeated narrow prompt does not overstate cache warmup.

The promotion gate is >=20 tok/s warm decode, stable output, no unbounded VRAM
growth, and acceptable cold-prompt behavior. Community claims of roughly
21-52% decode gains are hypotheses for this hardware, not expected results.

Result: the isolated `moe-cache-v2-pr` build at `e3096b046` completed the 8k
gate on 2026-09-05. A matched, same-binary canonical-weight control measured
18.3137 tok/s median warm decode; a 16,000 MiB/device request with
`GGML_CUDA_MOE_CACHE_RESERVE_MB=1024` measured 28.2630 tok/s, a 54.327% gain.
The available VRAM only granted 1,444 / 2,005 / 1,887 MiB on CUDA0/1/2, so an
18,000 MiB request would not increase the actual pools. Three warmup requests,
three 1024-token scores, and a 20-request output stability run passed with
zero swap and unchanged GPU memory during the stability sequence. The cache
fork remains experimental and isolated; it still requires the populated 128k
context gate. See `large-moe-3gpu-probe-2026-09-04/recent-build-moe-cache-v2.md`.

Source to recheck before branch selection:

- [Expert-cache RFC and community measurements](https://github.com/ggml-org/llama.cpp/discussions/24528)
- [leloch moe-cache-v2-pr branch](https://github.com/leloch/llama.cpp/tree/moe-cache-v2-pr)

### 5. Incremental Tuning

Run only after the higher-upside software experiments:

- Raise GPU1 from 275 W to 350 W only as a deliberate, recorded power test.
- Test one additional resident expert layer only if VRAM reserve permits.
- Keep the same workload, build, output gate, and power telemetry for each
  comparison.

These are expected to be incremental, not the primary path to 20 tok/s.

### 6. Hardware Follow-up

DDR4-3200 cannot be assumed to scale decode linearly. The theoretical upper
bound from memory frequency alone is:

```text
17.719 * (3200 / 2933) = 19.33 tok/s
```

Actual end-to-end improvement may be much smaller. Re-run the physical-VRAM
manual baseline after any memory change. Repairing GPU0's x4 link is more
important for tensor mode than manual layer decode because tensor mode performs
repeated cross-GPU reductions.

## Production-Relevant Context Gate

Do not declare a candidate successful based only on 8k screening. Advance only
the top two candidates to this fully populated long-context gate:

```text
--ctx-size 131072
--cache-type-k q8_0
--cache-type-v q8_0
--flash-attn on
--parallel 1
--kv-unified
```

Populate approximately 126k-127k tokens and generate at least 1024 tokens.
Record allocation success, prefill rate, steady decode rate, host memory,
swap I/O, output sanity, and stability across multiple requests. The completed
32k runs allocate a 32k window but use a 4k prompt, so they do not satisfy this
gate.

Result: completed on 2026-09-05 for the isolated cache candidate. A 131,072
token unified-KV context accepted a 125,992-token prompt at 70.0668 tok/s and
then generated 1,024 warmed tokens at 27.7022 tok/s. All sanity and 20-request
stability checks passed, GPU memory was unchanged across the stability sequence,
and swap I/O was zero. The cache pool reduced to 1,282 / 1,701 / 1,605 MiB on
CUDA0/1/2 at the larger context, yet retained 71.5%, 77.0%, and 81.2% hit rates
without fill or dispatch failures. See
`large-moe-3gpu-probe-2026-09-04/recent-build-moe-cache-v2.md`.

Expanded result: fully populated 256k remained above the requested 25 tok/s
threshold at 26.5615 tok/s decode after a 256,984-token prefill. The matching
512k run allocated and populated 518,992 tokens but decoded at 24.1073 tok/s.
Both runs passed all output and stability checks with zero swap. The 512k
result is the current tested capacity boundary for >=25 tok/s warm decode.

## Decision Rule

Keep the three GPUs together as a local executor. Use the manual placement for
decode-heavy cached interaction and tensor mode for fresh long prompts with
short responses. If stock llama.cpp cannot sustain >20 tok/s in the populated
128k gate after the thread and DSpark experiments, prioritize expert caching
over further ordinary tensor-split tuning.

## References

- [llama.cpp server options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [llama.cpp multi-GPU guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)
- `research/large-moe-3gpu-probe-2026-09-04/recent-build-expert-placement-and-tensor-nccl.md`
