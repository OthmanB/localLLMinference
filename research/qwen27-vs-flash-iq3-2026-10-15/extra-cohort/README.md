# Additional, disjoint 40-task SWE cohort

This cohort augments the original 40-task run. It is separately labelled and has
**40 unique tasks, zero overlap with the original, 80 unique tasks in the union**.
The original run and its controller files are unchanged by this extension.

## Frozen selection

Manifest: `manifest.json`, SHA-256 (canonical body):

`0007c250d4f8b1c2519fc7e55e3b9a7d08f5cefc0d06ad1eee0c87fa529107ab`

Selection seed **20261016**. Candidates are the cached Verified500 minus the
original 40: **460 eligible tasks**. Only repository, ID, and input-visible issue
length affect selection. Original repository quotas are capped by availability;
remaining seats are assigned by descending `original_weight/(assigned+1)` with
lexicographic tie breaking. Within each repository, candidates are sorted by
issue length and ID, split into equal rank strata, and sampled using a seeded
repository-local RNG. Gold patches, hidden tests, model outputs, resolution
results and cached-image availability play no role in selection.

| Repository | New tasks |
| --- | ---: |
| django/django | 11 |
| sympy/sympy | 5 |
| matplotlib/matplotlib | 4 |
| sphinx-doc/sphinx | 4 |
| astropy/astropy | 3 |
| scikit-learn/scikit-learn | 3 |
| pydata/xarray | 3 |
| psf/requests | 2 |
| pylint-dev/pylint | 2 |
| pytest-dev/pytest | 2 |
| mwaskom/seaborn | 1 |
| pallets/flask | 0 |

Flask has no eligible unseen tasks; its seat is deterministically redistributed
to Django. Coverage: **11 repositories, 30 repository/version strata**. Selected
issue lengths: min **225**, median **1470.5**, max **7476** characters.

Smoke tasks, selected deterministically from different repositories:

1. `psf__requests-1724`
2. `django__django-11790`

The inherited paired scheduler places these first and uses its unchanged
20261015 order seed for the remaining 38 tasks. Instructions, submission protocol,
sampling, model limits, resource limits, retries, smoke gate and official scoring
are reused directly from the existing controller.

Dataset provenance:

- Verified: 500 rows, fingerprint `1fdfd21ba2621130`, cache revision
  `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`.
- Canonical SWE-bench: 2294 rows, fingerprint `a11218685458cbbb`, cache revision
  `c6fe717fd7a4c3ac1daa4055a4fd082c6a1d28a2`.
- Both cache dataset versions: `0.0.0`; installed datasets package: `5.0.1`.
- Selected canonical rows were crosschecked against Verified for repo, base
  commit, version, issue text, source patch and test patch; required evaluator
  fields and compatibility of canonical image names with stock mini-agent were
  also checked. Crosschecks happened after committing the metadata selection.

## Files and privacy

New source files in the parent research directory:

- `extra_prepare.py`: deterministic selection, cache crosschecks, sealed cohort,
  serial image preparation, unit launch and progress CLI.
- `extra_swe.py`: separate init/run/status wrapper around unchanged paired code.
- `extra_tests.py`: six offline tests for selection, strata, outcome independence,
  source guards, new smoke IDs, unchanged legacy source and inherited deadlines.

Public-safe metadata: `manifest.json`, `original-ids.json`, `images.json`,
`policy.json`. The policy contains protected-source checksums and absolute deadline
epochs, with no usernames or machine-local paths.

**HOST ONLY, NEVER PUBLISH:** `canonical-evaluator.HOST-ONLY.json` contains gold
patches and hidden tests. The cohort directory is mode 0700 and the evaluator is
mode 0400. The host-only seal and launch record, preparation logs/status, and
eventual run snapshots must also remain private. No evaluator file is mounted
into generation containers or passed to mini-agent. Review any publication export
explicitly; do not recursively publish this directory.

`seal.HOST-ONLY.json` hashes immutable cohort artifacts. `policy.json` additionally
guards all three inherited controller files and both new implementation files.
The wrapper freezes its own source and policy again inside each additional run.
Do not edit those sources after freeze. The wrapper changes only its own imported
controller process's `SMOKE` global, never an existing file or running process.

## Image preparation already launched

Unit: **`qwen27-flash-extra-images-20261015.service`**.

It is a detached user-systemd service with `Restart=on-failure`, a 5-minute retry
delay, control-group cancellation, low scheduling priority and private append
logs. User lingering is enabled, so it survives terminal logout. It is a
transient runtime unit, not a reboot-enabled installed service; after a host
reboot, invoke the launch command below to resume from the frozen cohort/cache.

Image preparation inspects all canonical image tags first, then pulls missing
images **one at a time**. Before each pull it requires root free space >=25 GiB.
It never prunes, deletes images, changes Docker settings, swaps tasks, calls
models, or manages inference services. Root space can fall during a single pull;
the next pull is blocked if the guard is breached. A failed/cancelled pull may
leave reusable Docker layers. Source/cohort checks run again on every worker
restart. Preparation stops at the inherited generation cutoff.

- `preparation.log`: worker actions, failures and diagnostics.
- `pull-output.log`: full Docker pull output.
- `preparation-status.json`: atomic heartbeat, worker/pull PIDs, per-image state,
  cache count, exact image IDs/repository digests, attempts and disk-space data.

From the project root:

```bash
BASE="$PWD/research/qwen27-vs-flash-iq3-2026-10-15"
PY="${SWEBENCH_VENV:-$HOME/.local/share/swebench-venv}/bin/python"

"$PY" -B "$BASE/extra_prepare.py" status
systemctl --user show qwen27-flash-extra-images-20261015.service \
  --property=ActiveState,SubState,MainPID,ExecMainStatus

# Read-only readiness check; fails if any of the fixed 40 images is missing:
"$PY" -B "$BASE/extra_prepare.py" check-images

# Only if a new launch is necessary after reboot/unit disappearance:
"$PY" -B "$BASE/extra_prepare.py" launch
```

Do not invoke `launch` while that unit exists. A worker failure automatically
retries its fixed selection; space/network failures are retained in the private
status and logs rather than replaced with easier or already-cached tasks.

## Main extension-orchestrator hooks

The additional run has **not** been initialized or launched. Initialization is
allowed only once all 40 required images are cached. Run only after the primary
has finished official evaluation and both model endpoints have been exclusively
reserved for this additional SWE phase. The wrapper requires primary `complete`.
The main orchestrator must arrange server availability if its existing sequence
stops Flash after the primary; this wrapper never starts/stops/restarts services.

```bash
BASE="$PWD/research/qwen27-vs-flash-iq3-2026-10-15"
PY="${SWEBENCH_VENV:-$HOME/.local/share/swebench-venv}/bin/python"
RUN="$BASE/runs/additional40-single-instance"

# Use a private umask for the inherited controller's run snapshots.
umask 077
"$PY" -B "$BASE/extra_swe.py" init --run-dir "$RUN"

# For the main orchestrator to launch later, after its primary-evaluation hook:
"$PY" -B "$BASE/extra_swe.py" run --run-dir "$RUN" --swe-phase

"$PY" -B "$BASE/extra_swe.py" status --run-dir "$RUN"
"$PY" -B "$BASE/extra_swe.py" run --run-dir "$RUN" --swe-phase --resume
```

The wrapper accepts portable `--cohort`, `--run-dir`, `--primary-run`, model
endpoint/alias/backend arguments and endpoint/alias environment variables.
`--smoke-only` provides an optional pause after the new smoke/evaluation gate.
Defaults target the same single Qwen GPU1 endpoint and Flash GPU0 endpoint as the
primary; GPU2 is not combined into the primary speed comparison.

Exact inherited cutoffs, frozen before image pulling and copied into the
additional ledger and immutable deadline snapshot:

- Generation: **2026-10-15 21:56:22.511018 UTC**.
- Entire experiment: **2026-10-16 03:56:22.511020 UTC**.

Initialization and resume never reset these clocks. The wrapper rejects a later
primary deadline that disagrees with the frozen policy. Failed/incomplete init
does not produce the wrapper's ready marker and cannot launch through this CLI.
Missing images prevent even creating the new run directory.

Forty additional capped tasks require up to ~20h of generation walltime with the
models running simultaneously, plus image preparation, the smoke-evaluation
pause, and final evaluation. The remaining generation window is shorter than
that worst case, so all 40 attempts/evaluations are not guaranteed. Deadline and
missing-report accounting remains explicit with denominator 40 for this cohort.
Report primary and additional cohorts separately, with any pooled 80-task result
clearly labelled and all incomplete/failed outcomes retained. Adding tasks can
reduce sampling variance but does not itself establish statistical significance.
