# Paired SWE-bench: Qwen 27B IQ4_XS-M vs Flash-Next IQ3_S

The real campaign started **2026-10-14 at 21:56:22 UTC**; the directory uses the
next-day study label. The campaign resumed as of **2026-10-15 02:18 UTC** after
review of an automatic Qwen process restart with unchanged service configuration.
[RESULTS.md](RESULTS.md) and [results-summary.json](results-summary.json) retain the
**2026-10-14 23:50:41 UTC** blocked partial snapshot (`smoke_complete` for SWE).
They are historical snapshots, not a live status feed or final campaign scores;
this source/recovery update does not regenerate them.

`paired_swe.py` uses existing servers when explicitly invoked with `run --swe-phase`;
it does not manage inference servers or probe health/models endpoints.
The campaign orchestrator, `run_campaign.py`, starts and stops **only its own
Flash process on GPU0**, checks the existing Qwen deployments read-only, and
performs startup provenance probes. It leaves Qwen service configurations unchanged
and can reconcile a bounded number of verified systemd automatic PID changes.
See [CAMPAIGN.md](CAMPAIGN.md) for setup, sequencing, deadlines and publication.

## Protocol

- **Primary:** one active agent per model, two total, on Qwen GPU1/8080 and
  Flash GPU0/18144. No GPU2 traffic in the primary. Endpoint, alias and LiteLLM
  provider/backend are configurable at initialization.
- Native shared context assumption: **262144**, with no scaling. Server setup,
  Flash reserve=1536, MTP=4, skipping off, and actual context configuration must
  be recorded by the main agent in its server/benchmark provenance. The client
  cannot certify server-side settings without probing. It does not alter them.
- Fixed 40 tasks from the existing SHA-validated manifest. Order starts with
  `pallets__flask-5014`, `astropy__astropy-13579`; remaining IDs are sorted then
  shuffled with Python `Random(20261015)`. Exactly the same order for both models.
- Stock mini-agent 2.4.6 SWE-bench instructions and stock tool-calling behavior,
  including all three patch-submission steps, are retained unchanged. Overrides:
  150 steps, 8192 output tokens/call, temperature 0.7, top_p 0.8, top_k 20,
  presence_penalty 1.5, reasoning_effort `none`, and
  `chat_template_kwargs.enable_thinking=false`. Tool-call parallelism and HTTP
  streaming are disabled. A model may still emit multiple commands in a single
  response; stock mini-agent executes them sequentially.
- Per-task process wall limit **1800 seconds**, command/request timeout 300s;
  Docker CPU=2, RAM=6 GiB, network=none, pull=never. Container expiry=1900s is an
  additional orphan fallback, not the task budget. Process cancellation and
  container removal add small teardown overhead to the wall limit.
- The two smoke tasks run for both models, then official hidden-test evaluation
  runs before the other 38 tasks. Valid tool activity is required on both smoke
  tasks per model, and protocol errors halt progress. **A smoke task need not
  resolve**: resolution is never used to select/retry samples. Nonempty smoke
  submissions require valid official reports before the gate opens.
- After generation, remaining nonempty submissions are evaluated per model,
  **sequential model harnesses with max_workers=2 total**, test timeout=1800s,
  offline, fresh UUID run IDs. Smoke reports are reused rather than reevaluated.
  Wrapper changes only official container creation to enforce cached images,
  network=none and resource limits. Official patch application, hidden scripts,
  test log parsing and `resolved` criteria are unchanged.
- Under `run_campaign.py`, generation cutoff is campaign start +24h and the
  overall envelope is campaign start +30h. The orchestrator propagates those
  global deadlines to the SWE ledger before smoke generation; they persist
  across restarts and include startup, smoke and supplementary work. A standalone
  controller invocation anchors them to its first `run` instead.
  Active tasks are cancelled at the generation cutoff, remaining tasks become
  `deadline`. Evaluation stops at the 30h cutoff, leaving missing reports explicit.
  Do not report an incomplete envelope as a complete hidden-test result.

## Files

| File | Purpose |
| --- | --- |
| `paired_swe.py` | CLI, snapshots, atomic ledger, paired scheduler, gate, evaluation, status |
| `instrumented_model.py` | Stock LiteLLM subclass recording actual request latency/usage without probes |
| `official_eval.py` | Resource-bounded offline container adapter around official SWE-bench harness |
| `test_paired_swe.py` | Offline failure accounting, report, metric, immutability and cancellation contracts |
| `run_campaign.py` | Global campaign deadlines, smoke pause, supplementary scheduling, own Flash process |
| `test_campaign_recovery.py` | Mocked unchanged-config recovery, config-change rejection and event-cap tests |
| `audit_document_grades.py` | Symmetric post-hoc marker-schema/citation-order audit; original grades retained |
| `supplementary.py` | Matched speed/document, capacity and EvalPlus phases |
| `supplementary_http.py` | SSE framing, request timing, actual usage and cache accounting |
| `supplementary_fixtures.py` | Authored synthetic archives, tokenizer loader and exact-answer grader |
| `test_supplementary.py` | Offline streaming, fixture, denominator and resume contracts |
| `publish_results.py` | Privacy-filtered aggregate reporting; Git publication is explicit opt-in |
| `PLAN.md`, `SUPPLEMENTARY.md`, `CAMPAIGN.md` | Protocol and reproduction guidance |
| `RESULTS.md`, `results-summary.json` | Timestamped running-status snapshot, not final scores |
| `PUBLICATION.json` | Separate study source/export hash inventory and omissions |

The ledger/trajectory approach is adapted from `tools/swebench_controller.py`.
The initializer copies the operator-selected helper into a private run for
provenance; it never executes it. Its historical endpoints/runtime are not active.

## Standalone controller commands

Run from the project root, using the prepared stock environment and cached
datasets/images. These commands are for an independently reserved reproduction
window; the active campaign already owns its endpoints.

```bash
BASE="$PWD/research/qwen27-vs-flash-iq3-2026-10-15"
PY="${SWEBENCH_VENV:-$HOME/.local/share/swebench-venv}/bin/python"
RUN="$BASE/runs/primary-single-instance"

"$PY" -B "$BASE/paired_swe.py" --help
"$PY" -B "$BASE/paired_swe.py" init --run-dir "$RUN"

# Smoke generation and official evaluation, then pause at smoke_complete.
"$PY" -B "$BASE/paired_swe.py" run --run-dir "$RUN" --swe-phase --smoke-only

# Continue the other 38 tasks and official evaluation in a reserved SWE phase.
"$PY" -B "$BASE/paired_swe.py" run --run-dir "$RUN" --swe-phase --resume

# Read-only polling; does not touch models, Docker, or services.
"$PY" -B "$BASE/paired_swe.py" status --run-dir "$RUN"

# After SIGINT/SIGTERM or infrastructure interruption:
"$PY" -B "$BASE/paired_swe.py" run --run-dir "$RUN" --swe-phase --resume
```

`init` validates the offline dataset fingerprint and every base commit, verifies
the 40-task evaluator matches, and freezes stock instructions, issue-only task
manifest, evaluator fixture, helper/controller source, order, model configs,
package versions and installed mini-agent/harness source hashes. It launches no
containers and makes no model requests. Manifest/evaluator paths can be supplied
with `--manifest`/`--evaluator`, or `SWEBENCH_MANIFEST`/`SWEBENCH_EVALUATOR`.
`--helper` controls the provenance helper path. Run directories must remain below
this research directory. The venv is the Python used for `init`, not an embedded
username or a system Python selected later.

Example **separately labelled** GPU2 secondary validation (run in its own reserved
phase; do not combine its speed with the primary):

```bash
"$PY" -B "$BASE/paired_swe.py" init \
  --run-dir "$BASE/runs/secondary-gpu2" --label secondary-validation-gpu2 \
  --qwen-endpoint http://127.0.0.1:8081/v1 \
  --qwen-alias qwen3.8-27b-atx-iq4xs-m-262144-gpu2
```

Defaults are Qwen `http://127.0.0.1:8080/v1`, alias
`qwen3.8-27b-atx-iq4xs-m-262144-gpu1`, and Flash
`http://127.0.0.1:18144/v1`, alias `qwen3.8-flash-next-iq3_s`.
Use `--qwen-endpoint`, `--qwen-alias`, `--qwen-backend` and corresponding `flash`
arguments. Endpoint/alias also accept `QWEN_ENDPOINT`, `QWEN_ALIAS`,
`FLASH_ENDPOINT`, `FLASH_ALIAS`. Default backend is LiteLLM `hosted_vllm` (an
OpenAI-compatible transport, not a requirement to run vLLM). Non-default backend
support for all sampling/reasoning parameters must pass the smoke gate.

## Artifacts, resume, and failure denominator

- `immutable-sha256.json`, `config.json`, `frozen/`: read-only snapshots and hashes.
  Installed dependency sources and current controller source are checked before
  each run/resume. Do not edit source/config after initializing a run. Restore
  frozen source to resume an older run, or initialize a separate labelled run.
- `ledger.json`: robust atomic JSON; write temporary file, fsync, rename, fsync
  directory. One process holds a nonblocking `flock`; worker commits share a
  mutex. `status.json` and `<model>/summary.json` refresh every 30 seconds and
  at task/phase transitions. Ledger is the authority if derivative summaries
  were interrupted during update.
- `<model>/<instance>/<attempt_uuid>/`: command, attempt config, `agent.log`,
  `trajectory.json`, and `requests.jsonl` (actual backend request timing, usage,
  finish reasons, errors). Earlier attempts are never overwritten. Latency is
  client elapsed HTTP-call time, **not TTFT**; task walltime includes environment
  startup. Missing usage is explicit rather than estimated. Token/call totals
  include retries; per-attempt records permit final-attempt-only analysis.
- `<model>/predictions.jsonl`: all 40 tasks, including empty patches for failure,
  interruption, not-yet-attempted, or deadline. Only a successful nonempty stock
  `Submitted` trajectory becomes a prediction. This file refreshes at task
  completion. Official evaluator uses separate submission-only prediction files.
- `<model>/evaluation/<fresh_uuid>/`: official harness command/log, prediction
  snapshot, reports, test logs. Reports are the only authority for resolution.
- **Denominator is always 40 per model.** Submitted is not resolved. Summaries
  separately expose official evaluated, resolved, failed, infrastructure-failed,
  unattempted/deadline, and not-officially-evaluated counts. `resolved_fraction_all_tasks`
  is a lower bound while official evaluation is incomplete, never a completed
  accuracy claim. Inspect `phase` and `not_officially_evaluated` alongside it.
- Only interrupted attempts and recognized transport/Docker/storage/server
  infrastructure failures resume; infrastructure gets at most one automatic
  retry per task. Step limits, time limits, context exhaustion, empty submission,
  bad requests and format failures are terminal. Smoke protocol failures halt
  the run; repeated `--resume` never resamples those failed tasks.
- Official false/true reports are terminal; missing reports are infrastructure
  failures and can be retried on explicit resume under a fresh run ID. A nonzero
  harness exit blocks further progress even if partial reports exist.
- SIGINT/SIGTERM sets a stop flag; each supervisor terminates its child process
  group (TERM then KILL), and removes only labelled attempt/evaluator containers.
  On restart it reconciles recorded process PID/start-time identities before
  removing orphan containers; reused PIDs are never signalled. SIGKILL requires
  resume for reconciliation. No service or GPU-management commands exist here.

The frozen evaluator contains **gold patches and hidden tests**. It is host-only,
not passed as a mini-agent argument or mounted in generation containers. Stock
mini-agent loads the frozen manifest's cached dataset and prompts only with
`problem_statement`; its Docker environment sees the base issue repo. The
sanitized task snapshot contains no gold/test patches or hints. Keep runtime
snapshots private; publish reviewed source and derived aggregate results rather
than `frozen/evaluator.json`, the private helper snapshot or machine-local config.

## Offline checks and runtime expectation

```bash
# From this directory:
  "${SWEBENCH_VENV:-$HOME/.local/share/swebench-venv}/bin/python" -B -m unittest -v test_paired_swe.py test_supplementary.py
```

The tests mock Docker, HTTP/backend and official report creation. They do not
launch inference, generation tasks or hidden tests. Optional tokenizer and
EvalPlus-cache checks depend on the selected environment; skipped checks remain
explicit. Development validation directories, frozen fixtures and run snapshots
are excluded from publication. The live campaign's cached datasets/images and
dependency checks do not imply those assets are present in this public checkout.

Without retries, generation has a theoretical maximum of ~20h/model, simultaneous
across models, plus setup/teardown and the smoke-evaluation pause. The 24h
generation deadline leaves limited retry headroom. Evaluation worst case is
another ~20h total (80 submissions, 2 workers, 30min each), so **30h cannot
guarantee evaluating every submission in the worst case**. Typical runtime depends
on actual task convergence and hidden-test durations; use the ledger to allocate
the envelope. The controller truncates explicitly instead of silently extending
deadlines or shrinking denominators. Cached images can still fail offline if an
evaluation script tries to install uncached dependencies; inspect official logs.
The retained partial report records SWE `smoke_complete`; the campaign then
resumed after the reviewed runtime interruption. Inspect each report's timestamp,
phase and official coverage rather than treating a submission or campaign resume
as a final task-resolution claim. See `CAMPAIGN.md` for the separately labelled
post-hoc document audit and its format-versus-factual-error limitations.
