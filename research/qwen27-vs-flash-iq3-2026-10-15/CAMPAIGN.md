# Campaign execution and public snapshot

The real campaign began **2026-10-14 21:56:22 UTC**. `2026-10-15` is the study
directory label, not the start date. The exported aggregate is timestamped
**2026-10-14 21:58:29 UTC**: campaign `swe-smoke`, controller `smoke_generation`,
one submitted Flask patch and one active Astropy task per model. Neither model
has an official evaluation report in that snapshot. Pending counts are not final
accuracy scores. The public files are a snapshot, not a live status feed.

## What owns the servers

`run_campaign.py` launches the prepared Strata server as its own process group,
restricted to **GPU0**, loopback port **18144**, native **262144** context,
int8 KV / 32768 resident cells, 1536 MiB reserve, MTP4 and expert skipping off.
Its local base configuration supplies model, pack, MTP and tokenizer paths.

Existing Qwen GPU1/GPU2 services are checked read-only for active state and
unchanged PIDs. No experimental Qwen server is launched. The orchestrator neither
stops/restarts those services nor changes their deployment, gateway or GPU
settings. It stops its own Flash process on completion/failure. Muse must already
be inactive for the operator's GPU0 handoff; this runner does not stop or restore
Muse. Resource monitoring can stop the campaign at 88 C or less than 8 GiB
available host RAM.

The standalone `paired_swe.py` and `supplementary.py` clients do not manage
inference servers. The orchestrator performs health/model-ID provenance probes
before agent traffic and schedules phases exclusively on their endpoints.

## Prerequisites

- Linux with the historical systemd service names configured in `run_campaign.py`,
  or an operator-reviewed adaptation for a separately labelled reproduction.
  The campaign runner has fixed loopback endpoints/service names; standalone
  clients expose endpoint/model overrides.
- Prepared Strata **0.1.44**, IQ3_S pack/native/PLE assets and original MTP assets;
  a local `STRATA_ROOT/strata-iq3_s.json`; no model/runtime assets are published.
- A SWE interpreter with **mini-swe-agent 2.4.6**, **swebench 5.0.2**,
  **datasets 5.0.1**, LiteLLM and their required dependencies. `init` records and
  verifies installed source hashes/versions. Docker must be usable and all 40
  selected task images already cached; generation/evaluation do not pull images.
- The frozen 40-task manifest with matching SHA-256, offline Verified dataset
  fingerprint and base commits, and its matching canonical evaluator dataset.
  The public repository's `swebench/manifest-q4-q5-40.json` is the historical
  task selection; its Q4/Q5 runtime fields do not configure this study's models.
- An **EvalPlus 0.3.1** interpreter and cached HumanEval+ **164** / MBPP+ **378**
  tasks. Dataset download is rejected by the supplementary harness. Official
  evaluation uses the local EvalPlus subprocess/guards, not Docker isolation.
- Extracted shared Strata tokenizer with contiguous vocabulary of **248320** and
  `STRATA_ROOT/tools/strata_tokenizer.py` for the synthetic document fixtures.

The existing public [`tools/prepare_swebench_gold.py`](../../tools/prepare_swebench_gold.py)
can reconstruct the host-only evaluator rows from prepared dataset caches:

```bash
export SWEBENCH_PYTHON=/path/to/swebench-venv/bin/python
export SWEBENCH_MANIFEST=/path/to/frozen-40-task-manifest.json
export SWEBENCH_EVALUATOR=/path/to/private/verified40-evaluator.json
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1

"$SWEBENCH_PYTHON" tools/prepare_swebench_gold.py \
  --manifest "$SWEBENCH_MANIFEST" \
  --output /path/to/private/gold-predictions.jsonl \
  --dataset-output "$SWEBENCH_EVALUATOR"
```

That helper requires both the Verified and canonical `SWE-bench/SWE-bench`
caches and cross-checks their fields. Gold predictions and evaluator rows contain
held-out material; keep them private and do not expose them to agent generation.
The gold helper creates data but does not launch this study. The older
`swebench/run_headless.sh` recipe belongs to the historical Q4/Q5 deployment;
use the study runner below for this campaign.

## Launch and phase order

Run from the public repository root in an independently reserved reproduction
window, after preparing assets and handing off GPU0:

```bash
BASE="$PWD/research/qwen27-vs-flash-iq3-2026-10-15"
export STRATA_ROOT=/path/to/prepared/strata-runtime
export SWEBENCH_PYTHON=/path/to/swebench-venv/bin/python
export EVALPLUS_PYTHON=/path/to/evalplus-venv/bin/python

"$STRATA_ROOT/.venv/bin/python" -B "$BASE/run_campaign.py" \
  --strata-root "$STRATA_ROOT" \
  --swe-python "$SWEBENCH_PYTHON" \
  --evalplus-python "$EVALPLUS_PYTHON" \
  --run-dir "$BASE/campaign"
```

`psutil` is needed by the campaign interpreter. The runner derives
`STRATA_TOKENIZER` from the prepared local server config. Raw campaign data should
live in an unpublished working checkout; the source tree is not a raw-artifact
publication target.

1. Start Flash and record runtime/model/service provenance.
2. Initialize the SWE ledger if absent. Run two paired smoke tasks, then official
   hidden-test evaluation with `--smoke-only`; pause at `smoke_complete`. Valid
   tool activity and official reports for nonempty submissions are required;
   smoke resolution is not a sample-selection gate.
3. Run matched speed/document work (up to 3h), separate capacity work (up to 30min),
   then paired HumanEval+/MBPP+ work (up to 5h). Allocations shrink against the
   remaining generation budget while reserving approximately 14h for SWE.
4. Resume the same SWE ledger for the other 38 tasks and official evaluation.

Pairs run concurrently **within** phases; suites do not compete on an endpoint.
Primary SWE uses Qwen GPU1 and Flash GPU0. GPU2 participates only in the separately
labelled capacity comparison: two Qwen replicas versus one queued Flash replica.

## Global deadline and resume

`campaign-origin.json` persists start +24h for generation and start +30h for the
overall envelope. The runner propagates both deadlines into a newly initialized
SWE ledger before smoke generation. Startup, smoke evaluation, supplements and
full SWE share that clock. The published start corresponds to a generation cutoff
of **2026-10-15 21:56:22 UTC** and overall cutoff of **2026-10-16 03:56:22 UTC**.
Resuming does not grant another 30h.

The campaign takes a nonblocking lock. After a controlled stop and confirmation
that its previous Flash process has exited, rerun the same command/run directory.
The runner reuses the persisted origin, checks the smoke gate, skips completed
supplementary phases, and adds `--resume` to the SWE controller as needed.
Incomplete supplementary phases can resume their existing records without
regenerating completed requests. Controller source/dependency/config hashes must
match the initialized run; changes require restoring its frozen source or a new
separately labelled run.

The SWE ledger is authoritative. Successful submissions and terminal model
failures are not resampled; only interrupted/infrastructure work is eligible for
the documented retry rules. Missing official reports remain explicit. No final
score should be inferred from a phase name alone; inspect evaluation coverage and
the fixed 40-task denominator.

Read-only controller polling:

```bash
"$SWEBENCH_PYTHON" -B "$BASE/paired_swe.py" status \
  --run-dir "$BASE/runs/primary-single-instance"
```

## Reviewed aggregate publication

`publish_results.py` reads a private campaign and writes only the report/aggregate
pair. To refresh the public snapshot without Git operations:

```bash
"$SWEBENCH_PYTHON" -B "$BASE/publish_results.py" \
  --study-root /path/to/private/study-directory \
  --public-repo /path/to/public-checkout
```

Do not run it with the public source directory as its study input unless that
directory actually holds the intended run: absent ledgers would render
`not-started`, replacing the meaningful running snapshot. `--wait-unit` optionally
waits for an operator-selected user unit. Git commit/push behavior requires the
separate `--commit-and-push` flag; it is not part of this export or the command
above.

[`PUBLICATION.json`](PUBLICATION.json) records this study's source/export hashes,
snapshot timestamp and excluded artifacts separately from the earlier 118-file
archive. Refresh its result-file hashes whenever reports are updated. No
machine-local loader configs, frozen/evaluator data, private helpers/run trees,
offline-validation directories, raw dataset outputs, model responses/patches or
telemetry are included in the public export.

### Aggregate report publication

The publisher reads classified failures from each attempt's `reason` field and
keeps explicit task states and evaluation coverage. It refreshes the study
manifest hashes alongside the result report. Optional `--commit-and-push` first
requires a clean public working tree, fast-forwards origin/main, reviews status,
diff and recent commits, and stages only the two reports and study manifest.
If another session has public-tree changes, it stops rather than staging them.
