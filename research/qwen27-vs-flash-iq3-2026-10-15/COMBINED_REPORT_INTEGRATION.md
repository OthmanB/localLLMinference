# Combined SWE reporting integration

Run from this study directory using the standard-library Python interpreter:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_combined_swe.py
PYTHONDONTWRITEBYTECODE=1 python3 combined_swe_report.py --study-root . --output-dir combined-report --bootstrap-samples 20000
```

The CLI accepts absolute paths as well. `--study-root` defaults to the script's
directory. `--output-dir` defaults to its private `combined-report` subdirectory;
output must stay under this study directory, outside live-run/cohort directories.
Bootstrap seed is fixed at 20261016; replicate count is configurable.

## Outputs and consumption contract

- `combined-report/COMBINED_SWE.md`: side-by-side original40, additional40,
  combined80 tables, provenance, evaluator status, protocol checks and caveats.
- `combined-report/combined-swe.json`: allowlisted structured report, schema
  version 1; includes task IDs/repositories, filtered states, booleans and numeric
  attempt metrics. No raw task ledger, responses, patches, gold tests, evaluator
  data, endpoint configuration or machine paths are exported.
- Output directory mode is 0700, output files 0600. These remain private artifacts.
- Standard output is one JSON object listing output basenames, report kind and
  integration gate. Successful **incomplete snapshots exit 0**; invalid manifests,
  overlap, duplicate IDs/JSON keys, mismatched ledger IDs or invalid input schema
  exit 2 without emitting new report files. Outputs are individually replaced
  atomically; consumers should read JSON as their single structured artifact.

For later main-publisher integration, invoke this CLI and consume the two files.
Do not interpret process success as completion. Inspect:

```text
integration_gate.ready_for_complete_comparison
integration_gate.blocking_conditions
cohorts.original40
cohorts.additional40
cohorts.combined80
evaluator_status
protocol
```

The gate requires zero unknown model/task outcomes, zero submitted patches
without official booleans, zero unfinished evaluation records, both controllers
complete, and all frozen-protocol comparisons verified equal. Historical failed
evaluation records remain visible but do not block if later results supersede
them. `report_kind` remains `unpublished_snapshot` and `final_claim` remains false,
even when this comparison gate becomes ready. Publication is a separate action.

`complete_success_score`, `wilson95`, `paired_difference_qwen_minus_flash`,
`paired_difference95`, and `mcnemar_exact_two_sided_p` are nullable. Never replace
null with zero. Per-model score/interval requires all that model's selected
outcomes; paired statistics require all paired outcomes. A bootstrap CI can also
be null if every repository/cohort stratum is a singleton. Progress lower/upper
bounds and known-pair cell counts remain available while incomplete.

Terminal generation `failed` with no patch counts as a protocol failure; it does
not count toward official scored coverage. Submitted patches require an official
boolean. Infrastructure exhaustion and deadlines remain unknown, not intrinsic
code failures. `deployment_submission_fraction`, failure categories, official
patch results and attempt costs expose deployment effects separately. A
submission-conditioned result is not claimed as unbiased intrinsic code quality.

Selection authority is the original run's frozen source manifest plus
`extra-cohort/manifest.json`, including verified body/file hashes, additional seed
20261016 and the original-manifest hash reference. Each manifest must contain
exactly 40 distinct IDs; cross-cohort overlap is rejected. When initialized, each
model ledger must contain exactly its cohort's IDs. The additional frozen source
manifest must match the already-selected public-safe manifest.

Outcome and resource authority is `tasks[model][instance_id]` in each run's
atomic `ledger.json`, including attempts and official `evaluation.resolved`.
The two live ledgers are read independently; this is not a globally synchronized
snapshot. Regenerate after evaluation updates. Status summaries and host-only
evaluator datasets are never consulted. Protocol comparisons use frozen file
hashes and equality of allowlisted config limits and inherited deadlines; absent
additional-run files are unverified, not inferred equal.

## Offline verification

The 10 tests cover submission versus resolution, unknown-pair accounting,
no-patch failure semantics, disjointness/duplicate rejection, Wilson edge and
known values, exact McNemar values, reproducible stratified bootstrap with
singleton handling, retry/missing-metric accounting, input pairing, an absent
additional run, and sensitive-field filtering. Synthetic fixtures are created
and removed under this study directory. No inference, services, GPU access,
runner imports or evaluator reads are needed.
