# Qwen3.8-27B DFlash2 Test Blocked

Date: 2026-09-12

## Scope

The intended test kept the production target profile unchanged:

- Qwen3.8-27B UD Q4_K_M target
- GPUs 1 and 2, tensor split `1,1`
- 262,144-token context
- F16 K/V, Flash Attention
- batch/ubatch `2048/1024`
- official Qwen3.8-27B-DFlash2 Q8_0 drafter

The drafter was downloaded to:

`models/qwen3.8-27b-dflash2/Qwen3.8-27B-DFlash2-Q8_0.gguf`

Its size is 2,056,414,816 bytes.

## Result

The DFlash2 drafter is recognized and loaded far enough to report:

```text
adding speculative implementation 'draft-dflash'
n_max=7, n_min=0, p_min=0.00
block_size=8, mask_token_id=248070, n_extract=5, sample_from_anchor=true
```

The server cannot initialize the draft context with the current two-GPU Qwen
target setup. The approved full benchmark matrix was therefore not run.

### Installed llama.cpp snapshot

With the production September 5 build and inherited tensor split:

```text
GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0) failed
```

With the drafter forced to CPU:

```text
LLAMA_SPLIT_MODE_TENSOR needs >= 1 devices
```

### Isolated upstream build

An isolated CUDA build of upstream `82d6bb2` was tested. It includes the
single-device drafter split fix from commit `415e909`. With the drafter pinned
to `CUDA0`, initialization still fails:

```text
pre-allocated tensor (output.weight) in a buffer (CUDA1) that cannot run the operation (NONE)
```

The same failure occurs with a two-GPU layer-split target, so this is not only
the target tensor-split mode. It is a multi-GPU target/draft graph limitation
in the tested llama.cpp DFlash path, not a VRAM exhaustion failure.

## Production Safety

The production Qwen and Muse services were restored after each attempt.

- `llama-qwen3.8-q4-tensor-262k.service`: active
- `llama-muse-glimmer-30b-131k.service`: active
- Qwen health endpoint: passed
- Muse health endpoint: passed
- No production unit or model placement was changed

## Artifacts

Attempt logs and results are in the timestamped directories matching:

`research/qwen3.8-27b-q4-dflash-*`

The latest diagnostic run is:

`research/qwen3.8-27b-q4-dflash-20260912T004819Z/`

The benchmark harness is:

`tools/qwen_q4_dflash_benchmark.py`

The service-restoring wrapper is:

`operations/run-qwen-q4-dflash-benchmark.sh`

## Decision

Do not promote DFlash2 or run the throughput matrix on this runtime. A future
attempt needs a llama.cpp build that supports DFlash target/draft execution
across this multi-GPU Qwen configuration, or a different serving runtime.
