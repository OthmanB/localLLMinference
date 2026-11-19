# Parallel Model Services

The active production topology on the three-GPU RTX 3090 host is
`flash512-qwen-single`. The Muse, GPU1 llamAmpere, and stock tensor-split
services below are retired and must not run concurrently with Flash:

| GPU | Service | Model | Private endpoint | Status |
|---:|---|---|---|---|
| 0, 1 | `strata-flash-512k.service` | `qwen3.8-flash-next-iq3_s` | `127.0.0.1:8082` | active |
| 2 | `llamampere-qwen3.8-atx-iq4xs-m-gpu2.service` | `qwen3.8-27b-atx-iq4xs-m-262144-gpu2` | `127.0.0.1:8081` | active |
| 0 | `llama-muse-glimmer-30b-131k.service` | `muse-glimmer-30b-kquant17` | `127.0.0.1:8082` | retired |
| 1 | `llamampere-qwen3.8-atx-iq4xs-m-gpu1.service` | `qwen3.8-27b-atx-iq4xs-m-262144-gpu1` | `127.0.0.1:8080` | retired |
| 1, 2 | `llama-qwen3.8-q4-tensor-262k.service` | `qwen3.8-27b-q4-tensor262k` | `127.0.0.1:8080` | retired (manual restore only) |

The authenticated gateway on port 8088 publishes the two active model IDs plus
the single-member pool ID `qwen3.8-27b-atx-iq4xs-m-262144`. The exporter on
port 9108 scrapes both model endpoints, labels GPUs 0 and 1 as Flash and GPU 2
as Qwen, and labels every metric with the corresponding host ID, model ID, GPU
UUID, and GPU index. The unit files pin GPU UUIDs and the preflight checks
index/UUID identity, so a physical card swap refuses to start the wrong
allocation. The Flash unit declares `Conflicts=` against all retired units.

Do not bind a model server directly to the LAN. Historical Muse Glimmer
validation (131,072-token slot, official projector, DFlash drafter) is
recorded under `../research/`; it no longer holds GPU 0.
