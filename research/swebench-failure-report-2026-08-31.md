# SWE-bench Verified Failure Report

Date: 2026-08-31 UTC
Audience: GPT-5.6 Sol
Run: `qwen38-q4-q5-verified40-r1-89b534d4`
Manifest SHA-256: `89b534d44bfcfb63f0d5810261ed7ced70611f4968a493f9f499911a17ccb6a1`

## Executive Conclusion

The run did not produce a valid SWE-bench score. The main failure is a
submission-protocol mismatch in the custom mini-SWE-agent prompt:

- The prompt told the agent to run only `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
- mini-SWE-agent captures the patch only from output appearing after that marker.
- The standard SWE-bench prompt requires the agent to print the patch after the
  marker, using `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt`.
- Therefore, agents that solved a task could still be recorded with an empty
  patch.

This is confirmed by a Q5 task that modified source code and ran 40 focused
tests successfully, but ended with `exit_status=Submitted` and an empty
submission.

There is also a secondary model/prompt problem: many tasks used all 100 allowed
agent calls without reaching the submission command. The models repeatedly
searched or rewrote code instead of finishing. The 100-call limit is lower than
the standard SWE-bench configuration's 250-call limit and the custom prompt
omitted much of the standard workflow guidance.

The benchmark has been stopped. No benchmark model servers, controller,
resource collector, or SWE containers are currently running. Both GPUs are
idle.

## Run Configuration

- 40 identical SWE-bench Verified test instances per model, 80 model-task runs.
- Q4 model on GPU 0, port 8080.
- Q5 model on GPU 1, port 8081.
- Qwen3.8-27B Q4_K_M and Q5_K_M GGUF models.
- 131072-token llama.cpp context.
- Q8 K/V cache.
- `max_tokens=8192`, temperature 0, top-p 0.95.
- Qwen thinking disabled.
- Docker environment with `--rm --network none`.
- Command timeout 600 seconds.
- Agent wall-time limit 7200 seconds.
- Custom agent step limit 100.
- mini-SWE-agent 2.4.6.
- Dataset cache fingerprint and gold 40-task validation were verified before
  the run.

The task commands show that the intended model endpoints and Docker isolation
were actually used. For example, a trajectory records:

```text
docker run -d --name minisweagent-... -w /testbed --rm --network none ... sleep 2h
```

The Q4 and Q5 trajectories also contain valid parsed bash tool calls and normal
zero-return-code tool outputs. There is no evidence that the inference API was
unreachable or that CUDA failed during these task runs.

## Observed Results

Before the host outage, the durable ledger contained:

| Model | Failed | Running at outage | Queued | Completed |
|---|---:|---:|---:|---:|
| Q4 | 2 | 1 | 37 | 0 |
| Q5 | 3 | 1 | 36 | 0 |

The two running tasks were reconciled and retried after reboot. The run was
then stopped after the systematic no-submission pattern became clear. At stop:

| Model | Failed | Interrupted | Queued | Completed | Infrastructure failures |
|---|---:|---:|---:|---:|---:|
| Q4 | 30 | 1 | 9 | 0 | 0 |
| Q5 | 17 | 1 | 22 | 0 | 0 |

Across the 47 failed rows:

- 28 ended with `exit_status=LimitsExceeded` and
  `error_class=agent_no_submission`.
- 19 ended with `exit_status=Submitted` but still had an empty submission and
  were classified as `agent_no_submission` by the controller.
- No row reached the controller's `completed` state.
- No row was classified as `infra_failed`.

The `Submitted` status in this report does not mean that a usable patch was
stored. It only means that the mini-SWE-agent completion marker was detected.

## Primary Root Cause: Empty Patch Extraction

### Custom prompt

The frozen run prompt is in:

`swebench/configs/common.yaml`

Its submission instructions at lines 18-23 are effectively:

```text
When the fix is complete, run exactly:

echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

Do not run that command together with another command. Stop after submitting.
```

The controller invokes `mini-extra swebench-single` with `common.yaml` and a
model overlay. Because explicit config files were supplied, the built-in
SWE-bench configuration was not included. The agent log confirms that only the
custom common file and Q4/Q5 overlay were loaded.

### Standard mini-SWE-agent behavior

The installed standard configuration is:

`<SWEBENCH_VENV>/lib/python3.12/site-packages/minisweagent/config/benchmarks/swebench.yaml`

Its submission workflow at lines 75-101 requires:

```text
git diff -- path/to/file1 path/to/file2 > patch.txt
...
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
```

The installed Docker environment implements completion at:

`<SWEBENCH_VENV>/lib/python3.12/site-packages/minisweagent/environments/docker.py:140-150`

The relevant logic is:

```python
lines = output.get("output", "").lstrip().splitlines(keepends=True)
if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
    submission = "".join(lines[1:])
    raise Submitted(... submission ...)
```

With the custom marker-only command, `lines[1:]` is empty. The environment
therefore raises `Submitted` with `submission=""`.

The controller then deliberately rejects that result at:

`tools/swebench_controller.py:493-503`

because it requires both `exit_status == "Submitted"` and a non-empty
submission. It correctly prevents an empty prediction from being considered a
successful submission, but reports the root cause only as
`agent_no_submission`.

## Direct Proof From a Q5 Task

Task:

`q5/astropy__astropy-13579`

Evidence file:

`swebench/runs/qwen38-q4-q5-verified40-r1-89b534d4/q5/astropy__astropy-13579/agent.log`

The log shows that the agent:

1. Inspected the Astropy source.
2. Created a source diff in `astropy/wcs/wcsapi/wrappers/sliced_wcs.py`.
3. Ran the original issue reproduction and edge cases.
4. Ran the focused test file and obtained:

```text
============================== 40 passed in 0.68s ==============================
```

5. Issued the marker-only command:

```text
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```

The trajectory then records:

```json
{
  "exit_status": "Submitted",
  "submission": ""
}
```

The trajectory also records an `action was not executed` observation after the
completion signal. This is the mini-SWE-agent control-flow result after the
environment raises its `Submitted` exception; it is not evidence that the
preceding source edit or test command failed.

This task is enough to prove that the run's zero usable submissions cannot be
interpreted as a zero model capability score.

## Secondary Problem: Agent Looping and Low Step Limit

The custom prompt sets `step_limit: 100` in `common.yaml`. The standard
SWE-bench configuration uses 250 steps.

Representative Q4 trajectory:

`q4/astropy__astropy-14096/astropy__astropy-14096.traj.json`

- 100 API calls.
- Maximum recorded context: 33144 tokens.
- No completion marker.
- The final many steps repeatedly attempted variations of the same source edit.
- The final agent message explicitly says it is stuck in a loop.

Representative Q5 trajectory:

`q5/django__django-16100/django__django-16100.traj.json`

- 100 API calls on the resumed attempt.
- Maximum recorded context: 39144 tokens.
- No completion marker.
- The final steps repeatedly searched commit history using different `git log`
  ranges.

Thus, the inference service was producing valid tool calls, but the agent often
failed to converge. This is a real model/agent effectiveness problem, but it is
separate from the submission extraction bug. Some of the 19 marker-detected
rows may also contain valid edits, but their containers were removed and their
marker output contained no patch, so those edits were not recoverable.

## Controller Stop Bug Found During Diagnosis

The original controller worker loop was:

```python
while True:
    task = claim_task(connection, model)
    ...
```

On stop, the signal handler set `STOP_REQUESTED` and interrupted the active
task. The worker then immediately claimed the same `interrupted` row again,
because `claim_task` intentionally includes interrupted rows for normal
resume. This repeated until the controller was forcibly killed.

Evidence:

- Q4 `scikit-learn__scikit-learn-13328` reached attempt 1418.
- Q5 `mwaskom__seaborn-3187` reached attempt 1911.
- The event table contains repeated start/finish pairs for those same tasks at
  the stop time.

The worker loop was corrected to:

```python
while not STOP_REQUESTED:
```

in:

`tools/swebench_controller.py`

The two inflated ledger counters were restored from 1418 and 1911 to their
real value of 1. The event history was retained. The controller was then killed
and the orphaned Docker container was removed. This stop-path bug did not cause
the original zero-submission pattern, but it would corrupt future resume
accounting if left unfixed.

## Host Outage Context

The physical outage is independent evidence and should not be confused with
the task failures.

- The host became physically offline and required a PSU switch cycle before
  the power button worked.
- The prior boot journal stopped abruptly around 15:10 UTC, approximately
  00:10 JST.
- There was no clean shutdown record, kernel panic, OOM-killer event, NVIDIA
  Xid, thermal shutdown, or machine-check report.
- The likely cause remains a hardware-level hang, PSU protection latch, or
  motherboard/PCIe/GPU power fault.
- The paired run was successfully restarted once after the outage, but was
  later stopped intentionally after the harness failure was identified.

## Recommended Recovery Plan

Do not resume the current run as a scored experiment. Treat its ledger and
trajectories as diagnostic evidence only.

1. Restore the full standard SWE-bench submission instructions in the custom
   prompt, or include the standard `swebench.yaml` first and use a small overlay
   only for model endpoint and runtime settings.
2. Require the agent to create and inspect `patch.txt`, then submit with the
   exact command:

   ```bash
   echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
   ```

3. Increase the initial step limit to 250, or at least run a controlled pilot
   with 100 and 250 to measure convergence.
4. Add two Q4 and two Q5 smoke tasks before launching 40 tasks. The smoke test
   must assert all of the following:
   - trajectory `exit_status` is `Submitted`;
   - submission is non-empty;
   - submission starts with a valid git diff header;
   - `preds.jsonl` contains the same non-empty patch;
   - the patch can be passed to the SWE-bench evaluator.
5. Only after both smoke tests pass should a new clean 40-task paired run be
   started.
6. Keep the controller stop guard fix. Also make the controller classify an
   empty marker submission as `submission_protocol_error`, not as a generic
   model failure.
7. Because the host suffered a hard outage under sustained dual-GPU load, use
   additional PSU/UPS event monitoring or a lower-risk single-GPU pilot before
   another unattended dual-GPU run.

## Follow-up Stock-Config Smoke Test

The failed 40-task run was not resumed. A new smoke run was created at:

`swebench/runs/qwen38-q4-q5-smoke-r2`

The smoke run used:

- The installed stock `swebench.yaml`, copied byte-for-byte into the run
  directory.
- A small runtime overlay for endpoint-independent settings only.
- 128k context, Q8 K/V, thinking disabled, 8192 output tokens, temperature 0,
  `--network none`, and 250 agent steps.
- Two tasks per model: `pallets__flask-5014` first and
  `astropy__astropy-13579` second.
- A deliberate stop/resume test was performed in the preceding smoke run
  `qwen38-q4-q5-smoke-r1`; both active rows became interrupted and resumed at
  attempt 2 without the previous re-claim loop.

### Smoke results

| Model | Task | Controller result | Patch chars | Agent calls | Official evaluator |
|---|---|---|---:|---:|---|
| Q4 | `pallets__flask-5014` | completed | 433 | 11 | resolved |
| Q4 | `astropy__astropy-13579` | completed | 1919 | 56 | resolved |
| Q5 | `pallets__flask-5014` | completed | 456 | 10 | resolved |
| Q5 | `astropy__astropy-13579` | interrupted intentionally | 0 | 62 | not evaluated |

The official evaluator reports are:

- Q4: 2 submitted, 2 completed, 2 resolved, 0 empty patches, 0 errors.
- Q5 Flask: 1 submitted, 1 completed, 1 resolved, 0 empty patches, 0 errors.

This proves the stock submission protocol fixes the original harness defect for
both quantizations. The Q5 Astropy task was stopped after approximately 40
minutes while repeatedly generating the same reproduction script. Its last
recorded context was 70425 tokens, below the 131072-token server limit, so this
was not a context-window overflow.

The smoke launcher and runtime files are:

- `swebench/run_smoke.sh`
- `swebench/configs/runtime-stock.yaml`
- `swebench/runs/qwen38-q4-q5-smoke-r2/configs/swebench-stock.yaml`
- `swebench/runs/qwen38-q4-q5-smoke-r2/configs/runtime.yaml`

The new smoke run is stopped. Its ledger contains two completed Q4 rows, one
completed Q5 row, and one intentionally interrupted Q5 row. No model server,
controller, resource collector, or SWE container is active.

## Sampling-Policy Astropy Retry

The Astropy task was then rerun alone on both quantizations in a fresh run:

`swebench/runs/qwen38-q4-q5-astropy-sampled-r1`

Everything except sampling was kept unchanged:

- Stock mini-SWE SWE-bench configuration.
- 131072-token context and Q8 K/V.
- Thinking disabled.
- `max_tokens=8192`.
- 250 agent steps and 2-hour task timeout.
- Docker `--network none`.
- `temperature=0.7`.
- `top_p=0.8`.
- `top_k=20`.
- `presence_penalty=1.5`.

Results:

| Model | Agent calls | Max context | Wall time | Submission | Official evaluator |
|---|---:|---:|---:|---|---|
| Q4 | 45 | 31910 | 690.1 s | non-empty, 3058 chars | resolved |
| Q5 | 22 | 15171 | 270.0 s | non-empty, 1840 chars | resolved |

Under the previous greedy policy, the same Q5 Astropy task reached 62 calls,
approximately 70425 context tokens, and 2400 seconds while repeating the same
reproduction script and producing no patch. Under sampled non-thinking policy,
Q5 completed in 22 calls and its patch resolved the task. This strongly supports
the hypothesis that `temperature=0` was causing a deterministic non-convergent
trajectory for this agent workload.

The sampling runtime snapshot and launcher are:

- `swebench/configs/runtime-sampled-nonthinking.yaml`
- `swebench/run_astropy_sampling.sh`

The sampled run is stopped after evaluation; no benchmark process or container
is active and both GPUs are idle.

## Full Comparison Restarted

After the sampled Astropy retry resolved on both Q4 and Q5, the full paired
comparison was initialized in a new run directory:

`swebench/runs/qwen38-q4-q5-verified40-r2-sampled-89b534d4`

It uses the sampled runtime snapshot, stock mini-SWE configuration, 40 tasks per
model, and the same 128k/Q8/Docker-isolated setup. The launch preflight passed
for both endpoints. Initial status after launch:

- Q4: 1 running, 39 queued.
- Q5: 1 running, 39 queued.
- No failures or completions yet.
- Both GPUs were initially at the configured 250 W limit and resource logging
  was active.

During the run, both GPU power limits were raised to 300 W to test throughput.
The resource log confirms the change and shows both cards drawing approximately
299 W under load. At the latest check, Q4 had 10 completed tasks and Q5 had 9,
with one task running on each model and no failures.

The run is detached and continues independently of this SSH session. Its
runtime policy hash is recorded in `ledger.sqlite3`; the discarded original
run is not being resumed.

## Important Artifacts

- Run ledger:
  `swebench/runs/qwen38-q4-q5-verified40-r1-89b534d4/ledger.sqlite3`
- Controller log:
  `swebench/runs/qwen38-q4-q5-verified40-r1-89b534d4/controller.log`
- Resource log:
  `swebench/runs/qwen38-q4-q5-verified40-r1-89b534d4/resources.csv`
- Custom run prompt:
  `swebench/configs/common.yaml`
- Q4 overlay:
  `swebench/configs/q4-128k.yaml`
- Q5 overlay:
  `swebench/configs/q5-128k.yaml`
- Paired controller:
  `tools/swebench_controller.py`
- Headless launcher:
  `swebench/run_headless.sh`
- Direct proof of a solved-but-discarded Q5 task:
  `swebench/runs/qwen38-q4-q5-verified40-r1-89b534d4/q5/astropy__astropy-13579/agent.log`
