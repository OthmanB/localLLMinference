# Qwen3.8-Flash-Next NVFP4 Preflight

Date: 2026-09-06

## Target

- Checkpoint: `nvidia/Qwen3.8-Flash-Next-NVFP4` at revision
  `fc694b54fb0174e0913e6adf86691ef85a4ead47`.
- Architecture: `Qwen4ExpForConditionalGeneration` / `qwen4_exp`.
- Scope: text-only FreeToken experiment on one RTX 3090. Upstream FreeToken
  supports TP=1 only for this loader.

The NVIDIA model card documents Blackwell B200/B300 vLLM deployment. This host
uses FreeToken's RTX 30-series portable NVFP4 path instead, so this is a
compatibility and capacity experiment, not a supported NVIDIA deployment.

## Checkpoint And Storage

`hf download --dry-run` reported 25 files totaling 132.7 GB. The safetensors
metadata reports 119.6B tensor elements and 132,724,334,216 bytes of used Hub
storage. The transfer includes ten primary model shards and a 53.7 GB
`model-fp8-mtp-ple.safetensors` shard.

The full snapshot is downloaded at
`models/qwen3.8-flash-next-nvfp4`; a final dry run requires zero files. The
filesystem has 276 GiB free after download. Do not use optional FTW conversion
on this disk: direct-HF loading is sufficient for the initial test and avoids
requiring another checkpoint-sized margin.

## Runtime Compatibility

The installed FreeToken 0.1.2 daemon environment cannot load this checkpoint:
its registry lacks `Qwen4ExpForConditionalGeneration`.

An isolated current upstream checkout was created at
`/home/obenomar/.local/share/freetoken-flash-next-20260906`, commit
`af71ba43206e124f5ff6419b47ee36c6e9981078`. Its separate `.venv` includes the
CUDA 13 acceleration dependencies and does register the Qwen4Exp loader.

Its `tests/models/qwen4_exp/test_config.py` and
`tests/models/qwen4_exp/test_weight.py` passed: 30 tests in 3.97 seconds. The
loader accepts ModelOpt NVFP4 expert keys and discovers PLE tensors from the
safetensors index rather than relying on a repository-specific shard filename.

The NVIDIA ModelOpt config required an isolated parser fix. Its top-level
algorithm is `MIXED_PRECISION`, with explicit main-layer expert entries marked
`NVFP4`; the existing parser only recognized a top-level name containing
`fp4`. The candidate parser now derives each class from the authoritative
`quantized_layers` map. The focused suite passes 31 tests, and parsing the
downloaded checkpoint resolves routed experts as NVFP4 while dense, attention,
and head weights remain BF16. The candidate source is intentionally uncommitted
and has not been installed into the production FreeToken environment.

## Hardware And Memory Gate

- Host RAM: 141 GiB total, 134 GiB available at preflight, zero swap.
- GPU0: RTX 3090, 24 GiB, 300 W cap, idle after testing.
- Existing FreeToken daemon: active but idle, no loaded engine; it has
  `LimitMEMLOCK=infinity`.
- Interactive shell: `ulimit -l` is only 17.66 GiB.

FreeToken defaults to `--ple-backend disk`, which keeps the 47.7 GiB PLE table
on disk and uses only bounded pinned staging buffers. Its optional `pinned`
backend would require the complete table in locked host RAM and is excluded
from the initial test. The checkpoint also contains 56.25 GiB of U8 tensor
payload, dominated by packed NVFP4 routed-expert sources. The hybrid and CPU
paths lock those source banks, so the interactive limit is still insufficient.
No swap or unified-memory fallback is permitted.

Do not launch the isolated runtime directly from an interactive shell: its
lock limit cannot lock the expert banks. A separate systemd-owned candidate
service is required so the experiment inherits an unlimited memlock limit,
without changing the existing FreeToken daemon or the Qwen service. The
non-installed 4k candidate unit is
`operations/freetoken-flash-next-candidate.service`.

## Offload Calibration

An experiment-local profile was recorded at
`research/qwen3.8-flash-next-nvfp4-2026-09-06/cache/freetoken/benchbw.json` on
GPU0. It measured 33.7 GB/s CPU STREAM read and 6.5 GB/s PCIe H2D.

| Format | CPU MoE | PCIe gather | Selected backend | GPU fetch share |
|---|---:|---:|---|---:|
| NVFP4 | 48.7 GB/s | 6.5 GB/s | Hybrid | 13.6% |
| BF16 | 46.2 GB/s | 6.5 GB/s | Hybrid | 13.9% |

The initial launch should use the NVFP4 hybrid policy, one request, a 4k
context, and no MTP. Record host PSS, `VmLck`, zero swap, VRAM, power, TTFT,
and 128-token warmed decode before increasing context.

## Next Gate

Install and start the separate candidate service, then record host PSS,
`VmLck`, zero swap, VRAM, power, TTFT, and 128-token warmed decode. The
Qwen4Exp loader requires at least 512 unified GPU expert-cache slots; the
candidate reserves that minimum and disables the two-buffer prefill overlap.
Do not modify the existing FreeToken daemon or production Qwen service.
