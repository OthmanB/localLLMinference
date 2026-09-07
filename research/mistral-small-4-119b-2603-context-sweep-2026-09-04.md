# Mistral-Small-4-119B-2603 IQ3 two-RTX-3090 context and review evaluation

Date: 2026-09-04

## Result

Both Mistral-Small-4-119B-2603 IQ3 quants fit fully GPU-resident on RTX 3090
GPUs 1 and 2 through the supported 256k context limit. At populated 256k,
IQ3_XXS sustained 30.1 decode tok/s and IQ3_S sustained 24.2 decode tok/s.
The limiting card was GPU 1, with 24,126 of 24,576 MiB used at 256k, leaving
about 450 MiB free. Neither the model weights nor KV cache spilled to CPU, and
no swap was consumed.

That capacity result does not make either quant a safe autonomous reviewer.
At the 128k long-context review point, IQ3_XXS prematurely stopped without an
answer in 3 of 8 runs. A raw, exact-template diagnostic at the official 0.7
temperature confirmed a further natural stop in the middle of `[THINK]`, with
no closing tag or answer. IQ3_S completed all 3 of 3 initial 128k review runs
and the comparable 0.7 raw diagnostic, making it the only dependable
long-context Mistral quant in this test.

Terra scoring of four blind, static-text reviews of real flawed SWE-bench
patches found only 1.5 of 7 valid objections for IQ3_XXS and 2.0 of 7 for
IQ3_S. This is a constrained quality baseline, not by itself an autonomous
agent evaluation. The initial mini-SWE-agent follow-up was invalid for this
purpose: its marker-only submission protocol could not capture a review file.
A replacement direct Docker loop with explicit `bash(command)` and
`submit_review(text)` tools attempted all four preregistered repository-aware
IQ3_S reviews. Three submitted and one hit the step limit. Terra scored the
screen at 0 of 7 hidden objections, with no additional valid objections and
several concrete false claims.

## Decision

| Role | Decision | Rationale |
|---|---|---|
| Long-context capacity service | Go for both quants | All 8 context points passed fully resident through 256k. |
| 128k planning or drafting pilot | Conditional IQ3_S only | IQ3_S completed 3 of 3 natural-EOS reviews at 128k. Planning quality itself was not separately benchmarked. |
| Autonomous reviewer or meta-reviewer | No-go | The valid four-case agentic screen found 0/7 hidden objections and no additional valid objection; one run also missed its submission deadline. |
| IQ3_XXS at 128k review context | No-go | 3 of 8 initial runs stopped at 15-34 completion tokens with no answer; a separate 0.7 raw run also stopped mid-thought. |
| 192k-256k interactive work | Capacity mode, not default | Decode remains usable at 24-39 tok/s, but cold prefill is 7-13 minutes and GPU 1 has little reserve at 256k. |

Use IQ3_S only as a non-authoritative long-context assistant whose output is
verified by tests and an independent supervisor. Do not allow either quant to
approve a patch, make a final planning decision, or provide the only review.

This report does not establish a direct Mistral-versus-Qwen SWE-bench coding
score. Qwen supplied the flawed candidate patches but was excluded from the
qualitative score; Terra was the independent scorer.

## Test system

- GPUs: three RTX 3090 cards with 24 GiB each. Mistral used GPU 1 and GPU 2;
  the existing Qwen service on GPU 0 was not part of the Mistral measurements.
- Runtime: llama.cpp build `c841aee`, `llama-server`, `--parallel 1`.
- Mistral model: `mistralai/Mistral-Small-4-119B-2603` GGUF, text-only.
- Quants: Unsloth imatrix `IQ3_XXS` (42.8 GB file) and `IQ3_S` (44.4 GB file).
- Contexts: 8k, 32k, 128k, 192k, and 256k, all inside Mistral's supported
  256k operating limit.
- Common residency controls: `--n-gpu-layers 99`, `--fit off`,
  `--tensor-split 1,1`, Q8 K/V cache, Flash Attention, and no CPU weight
  offload. Full residency required 37 of 37 layers on GPU.

## Method

The Phase 2 throughput sweep started a fresh server for each context point,
used a cache-independent prompt close to the allocated context, warmed up once,
and recorded prefill/decode rates. The harness also sampled VRAM, temperature,
PCIe state, and host-memory use.

Phase 3 used `reasoning_effort:"high"`, natural EOS, and a roughly 120k-token
prompt at the 128k allocation. An 8192-token output reserve was necessary:
it allowed a natural reasoning response rather than filling the context with the
prompt alone.

Phase 4 used four Qwen unresolved SWE-bench patches as blind review inputs.
Both Mistral quants reviewed every case at 32k context, seed 1337, temperature
0.8, and `reasoning_effort:"high"`. Terra compared the resulting answers to a
fixed gold-derived scoring key that was not shown to the models. Each valid
objection counted as 1, a partial objection as 0.5, and a miss as 0.

## Phase 2: throughput and capacity

All 8 points passed the 37-of-37 full-residency gate. PCIe remained Gen3 x8
under load and host swap stayed unused. The first GPU card holds the lower
layer range and is the VRAM constraint at high context.

| Quant | Context | Decode tok/s | Prefill tok/s | Prompt tokens | Peak VRAM GPU1/GPU2 MiB | Max C |
|---|---:|---:|---:|---:|---:|---:|
| IQ3_XXS | 8k | 92.9 | 2151 | 7,926 | 21,584 / 20,056 | 69 |
| IQ3_XXS | 32k | 78.1 | 1555 | 32,502 | 21,840 / 20,312 | 77 |
| IQ3_XXS | 128k | 48.9 | 643 | 130,800 | 22,866 / 21,338 | 86 |
| IQ3_XXS | 192k | 38.7 | 457 | 196,349 | 23,634 / 21,978 | 88 |
| IQ3_XXS | 256k | 30.1 | 358 | 261,887 | 24,126 / 22,618 | 87 |
| IQ3_S | 128k | 48.7 | 636 | 130,809 | 23,762 / 21,978 | 87 |
| IQ3_S | 192k | 31.6 | 449 | 196,336 | 24,126 / 22,618 | 87 |
| IQ3_S | 256k | 24.2 | 348 | 261,885 | 24,126 / 23,258 | 85 |

Decode declines smoothly with populated context. IQ3_XXS loses about 68% of
its 8k decode rate by 256k (92.9 to 30.1 tok/s). At the matched long-context
points, IQ3_S is about 15-20% slower than IQ3_XXS: 48.7 versus 48.9 tok/s at
128k, 31.6 versus 38.7 at 192k, and 24.2 versus 30.1 at 256k. The larger
IQ3_S weight file therefore has no throughput advantage, but is justified by
its better long-context review reliability.

Cold prefill, not decode, dominates a one-off long-context review. A 128k
prompt took about 206-208 seconds to prefill; 256k took about 736-759 seconds.
That makes 128k the practical default working point. The 192k and 256k points
are valid capacity modes when their 7-13 minute cold-prefill cost is acceptable.

Phase 2 was collected mostly under the original 300 W measurement setup. After
the power-limit revision, Phase 3 ran with GPU 1 capped at 275 W and reported
81-85 C without residency loss. The Phase 3 thermal result confirms
sustainable review operation but is not a replacement throughput sweep at
275 W.

## Phase 3: 128k review reliability

| Quant | Runs | Premature EOS | Complete response range | Wall time | Fully resident |
|---|---:|---:|---:|---:|---|
| IQ3_XXS | 8 | 3/8 | 245-847 completion tokens | 185-205 s | Yes |
| IQ3_S | 3 | 0/3 | 288-427 completion tokens | 194-197 s | Yes |

IQ3_XXS returned no visible answer in all initial premature-EOS cases. Seed 42
repeated at temperatures 0.8 and 0.6. The comparable raw 0.7 diagnostic made a
longer 152-token attempt but still stopped in `[THINK]` without an answer; 0.7
therefore did not produce a usable result. A one-shot retry may conceal a
failure in an interactive workflow but does not make it production-reliable.

IQ3_S always returned a coherent natural-EOS review in this small sample. Its
completion lengths were shorter than the most verbose successful XXS result,
but this reliability result does not establish autonomous repository-review
capability.

## Phase 4: independently supervised blind review

The test covers only reviewer behavior, not SWE-bench patch-generation skill.
The source patches were real Qwen unresolved SWE-bench outputs. The scoring key
was derived from each gold patch and test patch before Terra read the Mistral
reviews. All eight Mistral reviews completed at 32k with natural EOS and full
37-of-37 GPU residency; the 128k XXS premature-EOS mode did not appear at 32k.

| Case | Valid objection(s) | IQ3_XXS | IQ3_S |
|---|---|---|---|
| Sphinx local-link checking | Default-off check; source-document-only/status behavior | Partial V1, found V2 | Missed V1, found V2 |
| Seaborn legend offset | Labels remain offset-stripped; formatter must disable offset/scientific notation | Missed both | Missed both |
| SymPy second-quantization LaTex | Standalone creator path; Tensor path | Missed both | Found V1, missed V2 |
| Django related managers | GenericRelation manager remains unimplemented | Missed | Missed |
| Total | 7 valid objections | 1 found + 1 partial = 1.5/7 | 2 found = 2.0/7 |

The two quants found the obvious Sphinx source-document limitation. IQ3_S also
identified that the SymPy change was limited to power printers instead of the
creator printer. Neither model found the most central Seaborn correctness issue
or the deeper Django GenericRelation omission. Both emitted false positives;
XXS included several confident claims that are directly contradicted by the base
code.

### Repository-aware follow-up

IQ3_S was then run as a mini-SWE-agent reviewer against the real SWE-bench base
repositories, with the candidate patch mounted read-only, shell access, tool-call
smoke validation, 128k context, high reasoning, and the official 0.7 temperature.
Those initial mini-SWE-agent trajectories are not quality evidence: the marker
command emitted no review text to stdout, so the framework could only capture an
empty submission. The replacement harness uses no mini-SWE protocol. It starts the
same real Docker images directly and maintains normal OpenAI history with only
`bash(command)` and terminal `submit_review(text)` calls.

The same 30-step budget and phased instructions were applied to all four
preregistered cases. All ran fully resident at 128k, high reasoning, and 0.7
temperature. Seaborn submitted at step 30 and scored 0/2: it missed that
`format_ticks()` still emits offset-stripped labels and instead incorrectly
proposed a numeric offset in the title. Django submitted at step 28 and scored
0/1: it missed the unhandled GenericRelation manager and falsely claimed the
nested manager methods were standalone functions. Sphinx submitted at step 23
but described the base bug rather than the candidate patch, missing both the
default-false gate and source-document/status limitations for 0/2. SymPy ignored
the explicit step-28 through step-30 synthesis instructions, made no submission,
and received 0/2. No review supplied an additional repository-supported
objection. The direct agentic result is therefore 0/7 with a 3/4 submission rate,
not a scaffold failure.

## Limitations

- Every Phase 2 context point is a single timed run. It establishes operational
  behavior, not a confidence interval.
- Phase 3 has 8 XXS and 3 S samples. The repeated XXS seed-42 failure supports
  the reliability conclusion, but a larger sample would better estimate rate.
- The static and corrected agentic Phase 4 screens each cover four cases and
  seven hidden objections. The agentic screen remains an adverse-quality screen,
  not a broad reviewer benchmark.
- Gold-derived objections are not exhaustive. Terra also checked the submitted
  reviews for additional concrete repository-supported objections; none were found.
- Qwen did not supervise Phase 4 and this report does not compare Qwen and
  Mistral SWE-bench resolution rates. The qualitative result applies to Mistral
  review behavior on Qwen-generated flawed patches only.
- No full 256k natural-EOS review test was run. The 256k evidence is forced-
  length throughput and capacity, not long-form review reliability.

## Artifacts

- Phase 2 raw measurements: `research/mistral-small-4-119b-context-sweep/results.json`
- Phase 3 aggregate and full text: `research/mistral-small-4-119b-context-sweep/phase3/summary.json`
- Phase 4 cases and fixed key: `research/mistral-small-4-119b-context-sweep/phase4/cases.json` and `phase4/scoring_key.json`
- Phase 4 independent scoring: `research/mistral-small-4-119b-context-sweep/phase4/terra_scoring.json`
- Corrected agentic trajectories and scoring: `research/mistral-small-4-119b-context-sweep/phase4/agentic-review-2026-09-04-r4/`, `agentic-review-2026-09-04-r5/`, and `phase4/agentic_terra_scoring.json`
- Harnesses: `tools/mistral_context_sweep.py`, `tools/mistral_review_sweep.py`, `tools/mistral_reviewer_ab.py`, and `tools/mistral_agentic_reviewer.py`
