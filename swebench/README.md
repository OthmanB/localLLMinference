# Qwen3.8 Q4/Q5 SWE-bench comparison

This directory contains the frozen 40-task paired comparison setup. The same
40 SWE-bench Verified instances are run once with Q4 on GPU 0 and once with Q5
on GPU 1. The experiment is not started by preparing these files.

## Frozen setup

- Manifest: `manifest-q4-q5-40.json`
- Dataset: `princeton-nlp/SWE-bench_Verified`, `test` split
- Context: 131,072 tokens with Q8 K/V
- Model turns: 8,192 maximum output tokens, temperature 0, top-p 0.95
- Qwen thinking: disabled through `chat_template_kwargs`
- Container network: `none`
- Command timeout: 600 seconds
- Whole-task timeout: 2 hours
- Agent step limit: 250 (stock mini-SWE configuration)
- Container runtime: Docker

## Prerequisites

Install Docker interactively because this session cannot enter `sudo` passwords:

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
```

Reconnect after adding the user to the `docker` group, then verify:

```bash
docker run --rm hello-world
```

Set `SWEBENCH_ROOT` to the repository checkout and `SWEBENCH_PYTHON` to the
Python environment containing the locked dependencies. The examples below use
those variables so they work outside the original development host.

## Gold validation

Create the gold prediction file without exposing it to the agent containers:

```bash
export SWEBENCH_ROOT=/path/to/localLLMinference
export SWEBENCH_PYTHON=/path/to/swebench-venv/bin/python

"$SWEBENCH_PYTHON" "$SWEBENCH_ROOT/tools/prepare_swebench_gold.py" \
  --manifest "$SWEBENCH_ROOT/swebench/manifest-q4-q5-40.json" \
  --output "$SWEBENCH_ROOT/swebench/gold/verified40.jsonl"
```

Run the official harness on all 40 gold patches before model generation. The
wrapper cross-checks the raw Verified rows against the canonical SWE-bench
rows, creates the evaluator-compatible local dataset with image and test
metadata, verifies that all 40 resolve, and writes the marker required by the
headless launcher:

```bash
"$SWEBENCH_ROOT/swebench/run_gold_validation.sh"
```

Do not start the model run unless every selected gold task evaluates correctly.

## Headless launch

Run the smoke test before the full comparison:

```bash
"$SWEBENCH_ROOT/swebench/run_smoke.sh" init
"$SWEBENCH_ROOT/swebench/run_smoke.sh" start
```

The smoke launcher uses the stock mini-SWE SWE-bench configuration and runs
two selected tasks per model. It must produce non-empty patches and pass the
official evaluator before the full comparison is started.

After the smoke test and Docker/gold validation pass, initialize a new clean
full run and launch it:

```bash
"$SWEBENCH_ROOT/swebench/run_headless.sh" init
"$SWEBENCH_ROOT/swebench/run_headless.sh" start
```

The `start` command launches the two 128k llama.cpp services and detaches the
paired controller with `setsid` and `nohup`. Monitor or resume after an SSH
disconnect with:

```bash
"$SWEBENCH_ROOT/swebench/run_headless.sh" status
"$SWEBENCH_ROOT/swebench/run_headless.sh" resume
```

The run directory contains the SQLite/WAL ledger, controller log, durable
`resources.csv` GPU/RAM samples, immutable per-task trajectories, agent logs,
and per-model `preds.jsonl` files.

## Evaluation IDs

Use distinct immutable IDs for model evaluation. Never reuse an ID after
changing a generated patch:

```text
qwen38-q4-verified40-r1-89b534d4
qwen38-q5-verified40-r1-89b534d4
```

The full SWE-bench run remains gated on the paired subset result.
