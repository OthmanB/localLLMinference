# Strata Flash-Next native and 512k results (2026-10-14)

Single RTX 3090; Strata v0.1.44 `ed4773eb7c8f1f1531ba2bc70e0e1987de2661af`, CUDA 13.2, driver 595.91.07. Native Q4/IQ3 screens and the IQ3_S YaRN extension are separate measurements. Model loading is excluded from request TTFT.

## IQ3_S status — native comparison completed at 10:56 UTC

Both IQ3_S shards downloaded and were packed successfully; shared MTP reused.
The continuation unit exited successfully (status 0), state trials-complete at
2026-10-14T10:56:38Z. GPU0 is free; Muse remains inactive, and protected Qwen
services retain PIDs 3047/3048. No extension run has started yet.

| Window | Actual uncached input | IQ3_S prefill tok/s | IQ3_S decode tok/s | TTFT seconds |
|---:|---:|---:|---:|---:|
| 4096 | 3148 | 901.5 | 86.2 | 3.55 |
| 32768 | 31821 | 2267.1 | 84.2 | 14.12 |
| 131072 | 130123 | 2061.5 | 81.1 | 63.28 |
| 262144 | 261195 | 1915.7 | 78.6 | 136.62 |

All context runs have cache_n=0 and all three markers recovered. Each generated
768 answer tokens and finished at the cap; the requested one-line marker format
was not followed. Native prefill is ~48.3% faster and decode ~57.2% faster than
our initial Q4 run (1292.2 / 50.0); TTFT fell from 202.4 to 136.6 seconds.
Prompts follow the same construction with fresh random markers; single runs,
not a multi-repeat definitive quality/performance ranking.

IQ3_S short probes: coding natural stop at 100 tok/s; planning natural stop
at 87.4; review natural stop at 84.3; JSON valid keys; correct Zurich tool call;
tool error response acknowledged unavailability rather than inventing weather.
Revised state turn 3 explicitly removes caching, final checklist completed.
Other state turns reached their 768-token caps. Raw answers still need full
grading/code execution checks before declaring quality equivalent to Q4/Muse.


### Extension execution checkpoint

extend_iq3.py ran as strata-iq3-extension-20261014.service (MemorySwapMax=0).
It completed all five points successfully at 12:11 UTC. Native and scaled
controls, 384k and 512k all returned natural-EOS JSON document answers with
correct exact markers, latest approved policy, draft rejection, cross-document
arithmetic and known source citations. Continuations also finished naturally;
grade_extensions.py independently executed the LRU code against validation,
recency/eviction and 1500 randomized operations for each point: all passed.

| Window / scaling | Actual cold input | Uncached prefill tok/s | Document decode tok/s | Natural coding decode tok/s | TTFT seconds |
|---|---:|---:|---:|---:|---:|
| 262144 native | 253546 | 1907.0 | 61.7 | 83.3 | 133.19 |
| 32768 YaRN 2 | 24206 | 2286.3 | 67.2 | 84.5 | 10.64 |
| 262144 YaRN 2 | 253546 | 1897.0 | 62.5 | 78.4 | 133.89 |
| 393216 YaRN 2 | 384598 | 1761.8 | 60.5 | 81.7 | 218.63 |
| 524288 YaRN 2 | 515650 | 1629.9 | 58.7 | 73.7 | 316.82 |

Document replies used 269 tokens; coding replies used 979–1365 tokens. Coding
follow-ups intentionally reused the populated conversation cache; the table's
prefill/TTFT are the first, fully uncached document request (cache_n=0), not the
cached follow-up. No output was artificially padded and no answer reached its
4096-token cap. The 512k coding answer used 1177 tokens.

For 512k, the latest approved policy, rejected draft and late marker were
placed at approximately 371182, 458819 and 500074 body-token positions, beyond
the trained/native end. The varied synthetic AUX documents were unique, not
one repeated paragraph. This establishes controlled document/retrieval and
coding behavior, not broad agentic success or quality parity with the native
model on arbitrary documents.

512k measured minimum free VRAM 1169 MiB (~1.14 GiB), peak GPU0 used 22958 MiB,
peak summed process RSS 61.00 GB (~56.81 GiB), minimum MemAvailable 80.11 GB
(~74.60 GiB). Full int8 KV occupies 6.19 GiB pinned host RAM, with 32768 cells
per layer VRAM-resident; YaRN factor 2 and original context 262144 confirmed
in engine log. Experiment cgroup swap remained 0; host swap stayed at its
pre-existing 1.5 MiB baseline without growth.


### Completed extension result and recommendation

The fresh-seed 512k repeat finished successfully at 12:18:30 UTC. It used
515650 fully uncached input tokens and returned a natural-stop 267-token
document answer plus a natural-stop 1011-token coding continuation. All exact
facts/revision/citation/arithmetic checks passed again. The repeated LRU code
also passed independent validation and 1500 randomized operations.

Across the two populated 512k runs:

- Uncached prefill: 1629.9–1636.4 tok/s.
- TTFT: 315.57–316.82 seconds (~5.26–5.28 minutes); model-load time excluded.
- Document-answer decode: 58.7–59.8 tok/s (267–269 tokens, JSON facts/citations).
- Coding-continuation decode: 73.4–73.7 tok/s (1011–1177 tokens, cached history).
- Minimum GPU0 headroom: 1165–1169 MiB (~1.14 GiB); 1536 MiB configured reserve.
- Peak summed process RSS: 61.00–61.11 GB (~56.8–56.9 GiB).
- Experiment swap: 0. Existing host swap stayed at 1.5 MiB, no growth.
- Both natural completions and independent code grades passed; no output caps.

Recommendation: IQ3_S with YaRN factor 2 is a viable 524288-position long-context
profile on this single RTX 3090 for the workloads tested. It meets >40 decode
tok/s on both answer shapes while sustaining ~1.63k uncached prefill tok/s.
Native 262k remains a verified fallback and is faster to ingest shorter archives.
Extension is not native context, and these controlled document/code tests are
not a broad agent benchmark or a proof of identical accuracy for arbitrary
long documents. No production backend/service promotion was performed.
