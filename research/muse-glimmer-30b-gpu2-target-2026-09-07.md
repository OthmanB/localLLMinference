# Muse Glimmer 30B GPU2 Validation and Deployment

Model card: <https://huggingface.co/meta-models/Muse-Glimmer-30B>

Muse Glimmer 30B was validated on GPU2 without displacing the permanent GPU0
Flash Next or GPU1 Qwen3.8-27B Q4 services, then promoted as the permanent
third model service.

## Comparison Plan

The candidate targets one 131,072-token research slot; it does not target
parallel requests. Evaluate these cases independently on GPU2 while GPU0 Flash
Next and GPU1 Q4 remain online:

| Case | Runtime and artifacts | Status |
|---|---|---|
| A | Official llama.cpp K-Quant 17 GB, projector, and DFlash | Validated at one 131,072-token GPU2 slot |
| B | NVIDIA ModelOpt NVFP4 in FreeToken | Ineligible: unsupported mixed-precision format and no dense RAM offload |
| C | K-Quant Dynamic or BF16 with CPU/RAM offload | Fallback only; not a first GPU2 load |

Case A is `meta-models/Muse-Glimmer-30B-GGUF` with the 16,756,683,904-byte
`Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`, 1,400,328,928-byte projector, and
1,631,208,128-byte DFlash drafter. All three occupy about 18.43 GiB on disk.
The official card explicitly targets this configuration at 24 GB VRAM.

The relevant checkpoint is `nvidia/Muse-Glimmer-30B-NVFP4`, a 24.7 GB ModelOpt
mixed-precision export. It combines W4A16 NVFP4, FP8, and BF16 modules; the
vision encoder is BF16 and the KV cache is unquantized. Its tested runtime is
vLLM on Blackwell B200 hardware, not RTX 3090. FreeToken's Muse loader only
recognizes uniform `quant_method: compressed-tensors` W4A16 NVFP4 checkpoints;
the NVIDIA config declares `quant_method: modelopt` and `MIXED_PRECISION`, so
FreeToken does not select its NVFP4 loader for this artifact.

Independently, FreeToken's Muse configuration sets `num_experts=0`: Muse is
dense, not MoE. Its CPU/RAM offload controls serve routed MoE experts only and
are explicitly discarded for dense models. Therefore this checkpoint cannot
meet the one-131k-slot RTX 3090 target through FreeToken RAM offload. Do not
download or launch it for Case B.

K-Quant Dynamic with projector and drafter would require about 22.7 GiB of
artifacts before runtime overhead. BF16 is about 59.6 GB of weights. Neither is
a safe fully GPU-resident 24 GB configuration; RAM-offload feasibility must be
measured separately and must retain host-memory headroom for the two live
services.

## Case A Results

All case-A launches were bound to `CUDA_VISIBLE_DEVICES=2`, `127.0.0.1:8082`,
and one slot. GPU0 Flash Next and GPU1 Q4 remained active throughout.

- The Q4_K_M main model passed a 32k text marker smoke at 42.71 decode tok/s
  and 421.94 prefill tok/s, using 16,012 MiB of GPU2 memory.
- Adding the official projector passed an inline PNG data-URI image smoke with
  `IMAGE_OK`, using 17,608 MiB. Remote HTTPS `image_url` input is unavailable
  in the installed llama.cpp build because it lacks TLS image-fetch support.
- The official DFlash drafter auto-detected as `draft-dflash`. At 32k with the
  projector it used 20,080 MiB and returned the exact marker at 61.76 decode
  tok/s, with 57 of 63 draft tokens accepted (90.48%). The short baseline and
  DFlash requests were not token-identical, so this is a directional speed
  comparison rather than a formal benchmark.
- The DFlash-plus-projector server accepted `n_ctx_slot = 131072` without
  reducing context. A 120,106-token marker-retrieval prompt returned the exact
  marker and stopped normally: 909.05 prefill tok/s, 43.49 decode tok/s, and
  56 of 96 draft tokens accepted (58.33%).
- At 131k, the OpenAI-compatible API returned a structured forced
  `get_weather({"location":"Paris"})` tool call and passed a second inline
  image request with `IMAGE_131K_OK`.
- GPU2 used 20,904 MiB after the long-context request, leaving 3,672 MiB of
  VRAM. Host memory retained about 64 GiB available and swap remained disabled.

The transient validation unit `muse-glimmer-gpu2-131k.service` was replaced by
the permanent system service `llama-muse-glimmer-30b-131k.service`. It serves
GPU2 on loopback port 8082 as `muse-glimmer-30b-kquant17`, with the gateway and
metrics exporter configured as production consumers. GPU0 Flash Next and GPU1
Qwen remain isolated on their original GPUs and loopback ports.

## Case B Preflight

FreeToken has a native `MuseGlimmerForConditionalGeneration` registry entry,
text-only uniform compressed-tensors NVFP4 dispatch, and Muse reasoning/tool
parsers. `nvidia/Muse-Glimmer-30B-NVFP4` is not that format: it is a ModelOpt
`MIXED_PRECISION` export containing BF16, FP8, and packed NVFP4 weights. The
loader only enables its NVFP4 path for `quant_method: compressed-tensors`; it
has no ModelOpt mixed-precision loader. It is also not a viable RAM-offload
runtime because Muse has no routed experts and `--moe-backend offload`, `cpu`,
and `hybrid` are solely MoE-expert controls. No checkpoint was downloaded or
launched; Case B is closed for the defined GPU2 target.

## Evaluation Gates

1. Case A text-only smoke at 32k, 64k, 96k, then 131k context with one slot.
2. Case A image smoke using the projector, then DFlash A/B at the selected
   context. Retain the drafter only if it fits with a measured GPU margin and
   improves end-to-end decode.
3. Case B is closed: the NVIDIA ModelOpt mixed-precision artifact is not a
   FreeToken-supported checkpoint format, and FreeToken has no dense-weight
   CPU/RAM offload path.
4. Compare the accepted cases at the same 131k single-slot workload: exact
   marker retrieval, reasoning/tool-call formatting, image handling, prefill,
   decode, GPU peak, host memory, and swap.
5. Case A passed the gates and is deployed as the GPU2 service, gateway route,
   Prometheus backend, and Grafana label.
