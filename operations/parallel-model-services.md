# Parallel Model Services

The production topology reserves one RTX 3090 per model service:

| GPU | Service | Model | Private endpoint |
|---:|---|---|---|
| 0 | `freetoken-qwen3.8-flash-next-262k.service` | `qwen3.8-flash-next-nvfp4-262k` | `127.0.0.1:1901` |
| 1 | `llama-qwen3.8-q4-192k.service` | `qwen3.8-27b-q4-gpukv192` | `127.0.0.1:8080` |
| 2 | `llama-muse-glimmer-30b-131k.service` | `muse-glimmer-30b-kquant17` | `127.0.0.1:8082` |

The authenticated gateway on port 8088 publishes all three active model IDs.
The exporter on port 9108 scrapes all three model endpoints and labels every
model and GPU metric with the corresponding host ID, model ID, GPU UUID, and
GPU index. Do not bind a model server directly to the LAN.

Muse Glimmer passed the isolated GPU2 validation and is deployed with one
131,072-token slot, the official projector, and DFlash drafter. It must not
stop or reassign either GPU0 or GPU1 production service.
