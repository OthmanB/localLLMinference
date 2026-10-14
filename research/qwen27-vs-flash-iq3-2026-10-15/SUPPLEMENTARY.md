# Portable supplementary benchmarks

`supplementary.py` operates on existing HTTP endpoints. It never starts, stops,
or reconfigures a server. Run these phases under the main orchestrator's exclusive
backend scheduling: they must not overlap SWE or another phase on these endpoints.
Each phase has its own lock; cross-phase exclusion is the orchestrator's job.

## Commands

From this source directory, set these operator-owned environment variables:

```bash
export STRATA_ROOT=/path/to/existing/strata-runtime
export STRATA_TOKENIZER=/path/to/existing/iq3_s/tokenizer
export STRATA_PYTHON="$STRATA_ROOT/.venv/bin/python"
export EVALPLUS_PYTHON=/path/to/existing/evalplus-venv/bin/python
export RUN_DIR=/path/to/private/benchmark-output
```

The public scripts contain no workstation-specific model/tokenizer/interpreter
paths. `STRATA_ROOT/tools/strata_tokenizer.py` and the extracted tokenizer's
`vocab.json`, `merges.txt`, `token_type.json` are read only. They are not copied or
modified; the fixture loader checks the shared contiguous vocabulary of 248320.

```bash
"$STRATA_PYTHON" supplementary.py speed --run-dir "$RUN_DIR" --overall-seconds 14400
"$STRATA_PYTHON" supplementary.py doc-quality --run-dir "$RUN_DIR" --overall-seconds 7200
"$STRATA_PYTHON" supplementary.py capacity --run-dir "$RUN_DIR" --overall-seconds 3600
"$EVALPLUS_PYTHON" supplementary.py evalplus --run-dir "$RUN_DIR" --overall-seconds 86400
```

These are separate invocations, not a parallel shell recipe. `speed` already
includes the document-quality tasks; `doc-quality` is an optional independent
document-only repeat, not necessary for the default suite. Default **per-process**
budget is 108000 seconds (30h); it is not a shared multi-phase allocation. The
orchestrator should pass its remaining allocation to `--overall-seconds`.

Defaults match the supplied deployment:

| Backend | Base URL | Alias | Allocation / quantization |
|---|---|---|---|
| qwen | `http://127.0.0.1:8080/v1` | `qwen3.8-27b-atx-iq4xs-m-262144-gpu1` | GPU1; **ATX IQ4_XS M** |
| qwen2 | `http://127.0.0.1:8081/v1` | `qwen3.8-27b-atx-iq4xs-m-262144-gpu2` | GPU2; **ATX IQ4_XS M** |
| flash | `http://127.0.0.1:18144/v1` | `qwen3.8-flash-next-iq3_s` | GPU0; **IQ3_S** |

This names Qwen's supplied exact ATX IQ4_XS M quantization rather than generic
"Q4". These are operator declarations, not independent GGUF inspection results.
Override using `QWEN_URL`, `QWEN_MODEL`, `QWEN2_URL`, `QWEN2_MODEL`, `FLASH_URL`,
`FLASH_MODEL`, or corresponding `--qwen-url`, `--qwen-model`, etc.

Every request is greedy (`temperature=0`, `top_p=1`, `n=1`), with
`reasoning_effort=none` and `chat_template_kwargs.enable_thinking=false`.
Unexpected reasoning is retained privately, and its first nonempty output counts
toward TTFT; `first_content_s` is separately recorded.

## Speed and document quality

Defaults: 3 fresh-tag repeats at body targets **4096, 32768, 131072, 245760**.
Whole, distinct synthetic AUX service documents populate the archive; no EOS
padding or chopped evidence documents. Actual measured body size is recorded
(normally within one AUX document below target, maximum 1%/256-token tolerance).
Relevant sources span approximately 5%, 23%, 45%, 72%, 89%, 97% of the archive.
They exercise revision precedence, draft rejection, service distractors,
cross-document arithmetic, three exact random markers, and exact source citations.
The quality grader accepts only a JSON object; code fences and wrong JSON types
are format failures. Evidence arrays must match exactly, with no extra citations.

For each size/repeat, `speed` performs two independent paired tasks:

1. A natural JSON ledger answer, graded for quality.
2. A natural complete Python LRU-cache answer with five tests, using the populated
   archive as background. This is the substantive coding continuation workload;
   it is a **fresh single-turn request**, not a cached multi-turn continuation.
   No model-dependent previous answer is inserted into paired inputs.

Both models receive byte-identical messages within each pair, concurrently with
one active request per backend. The system starts with the same fresh random tag
for the pair. Each task/repeat has a different tag. Output headroom is 4096 tokens
by default; `--max-tokens 8192` is supported and still fits the largest default
body within Qwen's 262144 context, reserving 1024 tokens for templates. Actual
server prompt usage plus output headroom is checked and recorded after each call.
Generation stops naturally: output lengths are **not** forced equal.

Default speed total: **48 requests / 24 paired tasks**. Document-only: **24
requests / 12 pairs**. With default request timeout 1800s, the speed request-time
upper bound is 24 × 1800s = 12h, limited by overall budget; fixture building adds
local CPU time. Planning estimate, not a new measurement: roughly 0.5–3h for speed
on these deployments, highly dependent on long-context Qwen prefill.

`--qwen-cache-prompt-false` adds the optional per-request Qwen field only when the
operator knows it is supported. It is off by default and never triggers a hidden
fallback/retry. Actual cache counts come from usage details, falling back to
engine `cache_n`; missing counts are **unknown**, never silently "uncached".
Summary distributions show both all responses and only zero-reported-cache
responses. Reuse, failures, caps, and pair identities remain visible.

## Timing definitions

- TTFT: request start to first nonempty content/reasoning delta. Empty role and
  usage-only events do not count.
- `usage.prompt_tokens`, `usage.completion_tokens`, cache counts: actual server
  counts; missing usage is not replaced with event counts or estimates.
- Engine prefill/decode rates and raw `timings`: server-reported, preserving the
  server's prompt/decode counts and millisecond denominators. Missing fields stay
  null; TTFT is not labelled engine prefill time.
- `output_tokens_per_wall_s`: usage completion count / complete client request
  wall time, including queue, prefill and trailing stream overhead.
- Client decode TPS is deliberately null: SSE chunks may contain multiple tokens;
  `(tokens-1)/(last-first chunk time)` would imply unjustified token boundaries.
- Summary medians and min/max ranges include timing sample counts, and paired
  Flash-minus-Qwen differences. Transport failures remain in attempted totals and
  outcome counts; their partial timings are excluded from timing distributions.
  Caps remain labelled and included among completed-response timings.

## Capacity

Three fresh rounds by default; each submits the same coding task with matching
paired slot tags, `max_tokens=2048`. Two Qwen requests target **two replicas on
GPU1+GPU2**; two concurrent Flash submissions target **one replica on GPU0 with
server parallel=1**, so its second request queues. Parallel=1 is an operator
assumption: the script does not change or introspect server configuration.

**12 requests total**, four submitted concurrently per round. Aggregate output
tokens / each backend group's makespan includes queue and prefill, while
per-request latency, usage and engine rates remain separate. Round medians/ranges
are reported. This is a hardware-allocation capacity comparison, not a fair
single-GPU speed ratio or an accuracy claim. Interrupted partial rounds resumed
across invocations have aggregate metrics null rather than a misleading makespan.
Planning estimate: 5–15min, default timeout upper bound 3 × 1800s = 1.5h.

## EvalPlus

Run under the existing **EvalPlus 0.3.1 interpreter**. HumanEval+ **164** tasks,
then MBPP+ **378**, each with one sample per backend: **1084 requests** total.
Tasks are serial per backend, with Qwen and Flash generating each task pair
concurrently. Prompts use only the public problem prompt and entry point, never
canonical solutions or additional hidden tests. Both models share the exact chat
template text and `sanitize(raw, entrypoint=task['entry_point'])` call. Output cap
is always 4096 for EvalPlus, independently of `--max-tokens`.

Generation requires existing cached datasets; its loaders explicitly reject
downloads. Store dataset/package version, authoritative official dataset hash,
task-prompt hash, request/message hashes, source hashes, exact raw SSE, decoded
raw output/reasoning, timing metadata and sanitized JSONL samples. Infrastructure
errors produce an empty solution and retain their task in the denominator.
Output-capped solutions are sanitized and tested as generated, with cap metadata.
Sanitizer failures likewise yield empty samples and explicit metadata. Completed
records (including errors) are never silently regenerated on resume.

After each full dataset's generation, run the official evaluator, separately for
each backend, using exactly:

```text
EVALPLUS_PYTHON -m evalplus.evaluate --dataset humaneval --samples ABS_SAMPLES.jsonl --parallel 2
EVALPLUS_PYTHON -m evalplus.evaluate --dataset mbpp --samples ABS_SAMPLES.jsonl --parallel 2
```

Base and base+plus pass@1 are calculated from the official per-task results,
requiring exactly one result per planned task and matching dataset hash. There
is no score on an incomplete generation set. `--generate-only` skips evaluation;
rerun without it to evaluate completed generation artifacts. Overall budget is
per invocation. EvalPlus default per-evaluation subprocess timeout is 7200s,
overridable with `--evaluation-seconds`. Planning estimate: **8–24h**, depending
on natural answer lengths and Qwen throughput, plus official evaluation time.
Worst-case request time exceeds 30h; the deadline produces resumable partial
artifacts, not a truncated-denominator score.

### Evaluation isolation

This implementation uses the requested local evaluator fallback. It does not
start a Docker image or download anything. It makes **no Docker isolation claim**.
The evaluator's cwd and TMPDIR are inside its run-private `work/`; parallelism is
2, CPU time is bounded, core dumps disabled, file descriptors limited, and BLAS
threads fixed at 1. The official EvalPlus generated-code guard remains intact.
A timeout kills the evaluator process group including descendants. **Host file
and network access are not isolated**; cwd/TMPDIR is not a filesystem sandbox.
No cached SWE-bench image compatibility has been assumed or tested. The supplied
venv's absolute paths/Python ABI make a blind read-only mount into an arbitrary
SWE image unreliable; an external container wrapper may instead be supplied as
`--evalplus-python` if independently validated by the main orchestrator.

## Persistence and artifact layout

```text
RUN_DIR/
  speed/                       # or doc-quality/, capacity/, evalplus/
    manifest.json              # configuration + source/tokenizer hashes
    status.json                # atomic latest persistent phase state
    progress.jsonl             # stage updates
    summary.json               # public-summary candidate, review before publishing
    requests.json              # speed/doc-quality response metadata without answer text
    round-N.json               # capacity only
    humaneval/dataset.json     # EvalPlus only; also mbpp/
    private/                   # chmod 0700; not a publication tree
      fixtures/*.json          # exact paired input, tags, expected document answer
      raw/*.sse                # verbatim HTTP stream/body, potentially partial
      records/*.json           # durable completed attempts, decoded content + usage
      evalplus/DATASET/BACKEND/
        samples.jsonl          # exactly one sanitized sample per attempted task
        metadata.json          # all task outcome + sanitizer/timing metadata
        evaluation.json        # exact command, isolation, return code, hashes, scores
        evaluation.log
        samples_eval_results.json
        work/tmp/
```

Entry-point umask is 0077. Private inputs/raw outputs should remain private.
Atomic JSON records are the resume authority. A dropped/incomplete SSE stream is
an infrastructure failure; HTTP failures and deadlines are distinct from JSON
format, answer correctness and output caps. Quality grading and transport/cap
outcomes are independent axes. `error_category`, `outcome`, and `grading.outcome`
are stable JSON fields. Exit 0 means planned work/evaluation is complete, **not**
that every inference succeeded; exit 2 means incomplete/deadline. Unexpected setup
errors exit nonzero with persistent infrastructure status when setup reached a
run context. Reusing a run-dir requires identical phase configuration/source
hashes; use a new directory after code changes. Timing budgets may change on
resume. SIGKILL may leave status at the last durable stage; never infer completion
from status alone. No retry option silently replaces failed samples.

## Offline checks

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_supplementary.py
PYTHONDONTWRITEBYTECODE=1 python3 supplementary.py --help
```

Tests use mocked HTTP response bytes, deterministic tokenizer fixtures and
synthetic official results. They do not connect to endpoints or execute model
samples. They cover SSE framing, token-count/timing denominators, cache unknowns,
caps vs incomplete streams, exact ledger grading, aggregate capacity makespan,
official denominator coverage, and no-regeneration/error-record resume.
