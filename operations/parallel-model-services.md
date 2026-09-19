# Parallel Model Services

The staged production topology uses GPU0 for Muse and the selected Qwen profile
on GPUs1 and 2. The stock tensor-split service below is the manual rollback
profile, not a concurrent service:

| GPU | Service | Model | Private endpoint |
|---:|---|---|---|
| 0 | `llama-muse-glimmer-30b-131k.service` | `muse-glimmer-30b-kquant17` | `127.0.0.1:8082` |
| 1 | `llamampere-qwen3.8-atx-iq4xs-m-gpu1.service` | `qwen3.8-27b-atx-iq4xs-m-262144-gpu1` | `127.0.0.1:8080` |
| 2 | `llamampere-qwen3.8-atx-iq4xs-m-gpu2.service` | `qwen3.8-27b-atx-iq4xs-m-262144-gpu2` | `127.0.0.1:8081` |
| 1, 2 | `llama-qwen3.8-q4-tensor-262k.service` (rollback) | `qwen3.8-27b-q4-tensor262k` | `127.0.0.1:8080` |

The authenticated gateway on port 8088 publishes both active model IDs. The
exporter on port 9108 scrapes both model endpoints, labels GPU0 as Muse and
GPUs1 and 2 as Qwen, and labels every metric with the corresponding host ID,
model ID, GPU UUID, and GPU index. Qwen endpoint counters are scraped once and
are not duplicated for its two physical GPUs. Do not bind a model server
directly to the LAN.

Muse Glimmer passed isolated validation and is deployed with one
131,072-token slot, the official projector, and DFlash drafter. It must not
stop or reassign either GPU1 or GPU2 Qwen allocation.
