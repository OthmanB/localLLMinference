# HyperQwen native 262k results (2026-10-02)

Single RTX 3090, 300 W, driver 595.91.07; HyperQwen commit `e1459c7631774f56de2f9425437d54e7e72ea688`, vLLM 0.30.0, FlashInfer 0.6.18.post1. Tests used pinned W4A16 targets and KVarN versus the ATX GGUF baseline; quantizations differ.

## Matched near-native comparison and decision

Every row below uses a 262,144-token model window on *physical GPU0*. The long-answer rows use the same text and uncached input of 261,336-261,347 tokens (llama.cpp raw tokenizer vs vLLM chat template differ by 11 tokens), with 360-391 output tokens and normal EOS. Short 22-token sentinel responses are **not** used to decide sustained decode. Power caps remained at 300 W, not HyperQwen's published 250 W.

| Runtime / configuration | Native capacity gate | Uncached prefill, tok/s | Sustained noncopy decode, tok/s | Code-copy decode, tok/s |
|---|---|---:|---:|---:|
| Production ATX llamAmpere MTP3 on GPU0 | 261,791-token native gate passed | **495.1** | **45.1** | **67.9** (extra sentinel) |
| HyperQwen vLLM batch, KVarN | 261,865, exact sentinels | ~599 | ~20-21 (only 22 output; screening) | not measured |
| HyperQwen vLLM MTP3, KVarN | 261,865, exact sentinels | ~576 | **18.9** | not measured |
| HyperQwen vLLM DFlash2 k7 + lookup, KVarN | 261,865, exact sentinels | ~588 | **15.8** | 36.5 (exact copy) |
| HyperQwen vLLM DFlash2 k15 + lookup, KVarN | 261,865, exact sentinels | ~585 | **13.5** | 37.9 (exact copy) |
| HyperQwen vLLM DFlash2 k7, int4 KV | 261,865, exact sentinels | **~206 (fails prefill gate)** | ~38 (22 output only) | not measured |
| ATX built-in ngram-map-k, no MTP | 261,359 near-native copy completed | 547.3 on copy | not measured | 59.8 (extra sentinel) |
| ATX ngram-map-k then adaptive MTP, max depth 6 | 261,336 near-native noncopy completed | 499.3 | **40.5** | not measured |

At native depth, vLLM's KVarN KV backend retains prefill but its long-context decode costs more than the speculative draft saves. The alternative int4 KV cache provides no cold-prefill solution. More lookup matches at 16-position verification did not translate into proportionate throughput (12.06 accepted/step, but only 37.9 tok/s on exact copying). The existing ATX stack already uses a calibrated 65,536-token MTP draft vocabulary and exact target verification; native-context code copying was faster on it than on any of the tested vLLM variants. Its built-in n-gram draft modes also did not beat its existing MTP3 setting on the matched task. No production service/profile/gateway changes are justified by these results.

**Outcome: no qualifying upgrade.** The required combination - native 262,144-token capacity, decode faster than ATX and prefill at least equivalent - was not achieved. Future work needs a measured native-context *verify/attention or KV* path that reduces target-step cost without losing the 495 tok/s prefill baseline; raising draft length alone is not enough at 262k. A different quantization or KV format also requires explicit quality validation rather than assuming speculative decoding preserves quantized target behavior.
