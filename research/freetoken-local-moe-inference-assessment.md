# FreeToken and local frontier-MoE inference on 2-3 RTX 3090 GPUs

Research snapshot: 2026-08-29  
Primary paper: [FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution, arXiv:2608.16157v1](https://arxiv.org/abs/2608.16157)  
Code: [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken)

## Executive conclusions

1. **FreeToken is a real, working open-source system, but it is an early one.** The repository is Apache-2.0, has installable v0.1.2 wheels, tests, an API server, and source implementations corresponding to the paper's central mechanisms. It is not only a paper stub. It is nevertheless a beta project created in July 2026, with a young compatibility surface and demanding dependencies: Linux, NVIDIA driver r580+, CUDA 13, PyTorch 2.11, and a CUDA 13 toolkit for JIT compilation.

2. **The individual ideas are mostly not new; the system composition and `q*` policy are the important contributions.** Layer streaming, CPU/GPU offload, LRU expert caches, pinned memory, prefix caches, and elastic memory pools all have prior art. FreeToken's meaningful contribution is combining them into a low-latency MoE runtime, especially the measured bandwidth-based allocation of misses between CPU execution and GPU cache fill, graph-resident heterogeneous execution, full-layer prefill overlap, and recurrent-state checkpoints placed at agent semantic boundaries.

3. **The Ollama comparison is directionally credible and reasonably fair for the tested Qwen configuration, but the exact multipliers are not yet independently established.** The authors aligned precision, weights, requests, and harnesses. On their RTX 3090 system, FreeToken delivered about 36.2 tok/s versus about 27.4 for KTransformers, 22.1 for llama.cpp, and 18.2 for Ollama. However, the paper omits engine commits, complete launch commands, tuning procedures, repetitions, and statistical uncertainty. Its 3090 was also attached to a server host with PCIe 4.0 x16, not to this machine's current Gen3 x8 links.

4. **AirLLM solves a different problem.** It makes an otherwise impossible model produce output by streaming layers, or in one implemented case individual experts, through one GPU. It does not pool multiple GPUs, has no AirLLM-specific paper, and provides almost no reproducible throughput evidence. Its published Kimi K3 experiment took about 292 seconds per token, or 0.0034 tok/s. AirLLM is a feasibility or inspection tool, not an interactive agent-serving alternative to FreeToken.

5. **The best compact quality candidates remain Qwen3.8-27B Q5_K_M and Ornith-1.5-35B-A3B Q4_K_M.** FreeToken now directly supports Qwen3.8-27B dense FP8/NVFP4 checkpoints as well as Qwen3.6/3.5 MoE. The 141 GiB RAM upgrade makes Qwen3.6-35B-A3B BF16 a physical FreeToken candidate and makes a gpt-oss-120b TP=2 experiment plausible, but neither is launch-ready until the current 17.66 GiB hard locked-memory limit is raised and CUDA 13 is installed. Ampere has no native NVFP4 tensor cores, so FreeToken uses its portable Triton or Marlin W4A16 path.

6. **The current bottleneck is not the number of GPUs.** Both GPUs report Gen3/x16 capability and currently idle at Gen1/x8, with no active NVLink. On this board, the bottom PCIE4 card's x8 link is expected electrically; the top PCIE1 card's x8 link under load is the configuration-dependent condition to investigate. The intended three-card topology is stable Gen3 x16 + Gen3 x16 + Gen3 x8, with reliable lane width preferred over an unreliable Gen4 attempt. The 200 W power limit is an intentional electrical-efficiency design choice, not an accidental throttle: the local load study measured 0.250 aggregate TFLOPS/W at 200 W versus 0.253 at 220 W. With the split-PSU arrangement, three 200 W cards plus 25% CPU load project to 1,059-1,089 W against the selected 1,100 W room-server budget; at 50% CPU they project to 1,157-1,167 W and are over budget. The paper's 3090 had 25.3 GB/s measured host-to-device bandwidth; Gen3 x8 has only 7.88 GB/s theoretical bandwidth. This can erase much of FreeToken's published advantage. Prioritize stable PCIe lane topology and the CUDA 13 toolkit; do not assume that a third GPU will multiply one-model token throughput.

7. **For near-frontier coding models such as MiniMax-M2.5 or DeepSeek-V4-Flash, target 256 GiB of fast, correctly populated quad-channel RAM.** Hugging Face reports Flash at 155.427 GiB used storage for 304.180B tensors. Its official configuration uses FP4 experts, FP8 weights, and BF16 ancillary tensors, so it is already a mixed-precision release rather than an uncompressed BF16 checkpoint. Flash is plausible only in the larger-RAM design with the stable intended PCIe lane topology; no published RTX 3090 result exists. A conservative expectation is single-digit to low-teens decode tok/s, not the 22-25 tok/s DeepSeek result obtained on an RTX 5090.

8. **A third 3090 is useful, but it is not the first point at which Qwen3-Coder-Next can be tested.** `llama.cpp --fit` can calculate a CPU/GPU hybrid placement for Q3/Q4 on the two installed cards; the observed 141 GiB total / 137 GiB available RAM increases its host-placement margin, but it remains a measured, nonresident path. Three cards make Q5_K_M the clean initial Coder-Next deployment and leave more runtime/context margin. FreeToken has tensor-parallel infrastructure, but support is model-specific: Qwen3.5/3.6 MoE remains TP=1, while gpt-oss source-level sharding supports TP=2 but not TP=3 with its 64 query heads.

9. **GLM-5.2 and Kimi K3 are not sensible targets for this platform.** GLM-5.2's NVFP4 checkpoint is about 433 GiB and the paper used 512 GiB host RAM plus a 96 GiB RTX PRO 6000. Kimi K3 is about 594 GiB and AirLLM's demonstrated rate is minutes per token. They may be technically launchable with extreme streaming, but they are not useful for an interactive coding pipeline.

## Confidence notation

| Label | Meaning |
|---|---|
| **M** | Measured and reported by the cited project or measured from this machine |
| **C** | Community-reported measurement, not reproduced or independently verified here |
| **E** | Estimate derived from measured bandwidth, checkpoint size, or a nearby published result |
| **U** | Unverified, unsupported, or no directly comparable measurement exists |

All FreeToken performance figures are author-reported. No result in this report should be treated as locally reproduced until the proposed benchmark suite is run.

## The machine actually available

The following was queried directly on 2026-08-29, except where an earlier observation is explicitly noted.

| Component | Observed state | Consequence |
|---|---|---|
| GPU 0 | MSI RTX 3090, 24 GiB | Ampere SM86, supported by FreeToken's RTX 30-series path |
| GPU 1 | Zotac RTX 3090, 24 GiB | Same capacity, but aggregate 48 GiB is usable only by engines/model paths that implement multi-GPU partitioning |
| PCIe | Both GPUs report max Gen3 / max x16; current idle state is Gen1 / x8 | Under load the generation should rise to Gen3. The bottom PCIE4 card is expected to operate at x8; investigate only the top PCIE1 card's x8 state. The intended final layout is stable Gen3 x16 + x16 + x8, not a required Gen4 configuration. |
| GPU topology | `PHB` between GPUs; NVLink links inactive | Peer traffic traverses the PCIe host bridge. Tensor parallel collectives will be much slower than NVLink. |
| GPU power | Both capped at 200 W by deliberate design; board maximum 350 W | The cap is near the local GEMM efficiency knee and respects UPS/room constraints. It can reduce compute-bound prefill performance versus unrestricted 3090 benchmarks, so all local token-rate estimates retain it rather than recommending a higher cap. See the local power study (kept outside the public repository). |
| CPU | 32 cores/64 threads, AVX2, one NUMA node | Many cores and four memory channels are useful for hybrid MoE, but it lacks the AVX-512/AMX paths used by some recent Intel KTransformers results. |
| RAM | 141 GiB total, 137 GiB available; 8 GiB swap enabled but unused | Physical capacity now covers the 128 GiB planning tier: Qwen3.6-35B-A3B BF16 and a cautious gpt-oss-120b TP=2 experiment. It remains below the safe DeepSeek-V4-Flash and MiniMax-M2.5 tier. Never permit swap during decode. |
| Locked memory | `ulimit -l` soft/hard: 18,521,516 KiB = 17.66 GiB | FreeToken pins host expert banks. This blocks Qwen3.6 BF16's 64.4 GB expert pool and Qwen3.8-Flash-Next's 47.7 GiB PLE table despite sufficient physical RAM. |
| Storage | SATA SSD | Fine for smaller models, but about 0.55 GB/s sequential class performance makes large-model startup slow. Source plus FTW conversion may require roughly twice checkpoint size. |
| NVIDIA driver | 595.84 | New enough for FreeToken's r580+ requirement |
| FreeToken / CUDA toolkit | FreeToken 0.1.2 is installed in `<FREETOKEN_VENV>`; Torch 2.11.0+cu130 detects both GPUs. `nvcc` is still absent. | The staged root installer adds CUDA toolkit 13.2 and a dedicated systemd service with unlimited `memlock`. FreeToken cannot JIT its kernels until that installer is run. |

### The PCIe finding is critical

The paper measured **25.3 GB/s** of pinned host-to-device expert transfer on its RTX 3090 PCIe 4.0 x16 system. A realistic Gen3 x8 transfer range is approximately **6-7.5 GB/s**, only 24-30% of that measurement. Current Gen1 at idle is normal power management. For this board layout, PCIE4 at Gen3 x8 is expected; the actionable issue is the top PCIE1 card remaining at x8 when the intended stable layout is Gen3 x16 + x16 + x8.

The CPU exposes enough lanes for the intended three-card layout, but the motherboard manual, slot wiring, BIOS bifurcation, and riser specification determine what is actually delivered. Validate every card under load after the third card and riser arrive. A stable Gen3 x16 link is preferred to an unreliable Gen4 link, and a Gen3 x8 PCIE4 link is expected rather than a fault to correct.

## What the 141 GiB RAM upgrade changes for FreeToken

The new RAM removes the prior physical-capacity barrier for the 128 GiB tier, but it does not bypass FreeToken's pinned-host-bank requirement. The current hard locked-memory limit is 17.66 GiB; changing it is a prerequisite for every candidate below that needs a larger pinned bank.

| Candidate | Physical-RAM assessment | FreeToken/runtime assessment | Decision before the third GPU |
|---|---|---|---|
| Qwen3.8-27B FP8/NVFP4 | Easily fits one GPU; additional RAM does not materially change this dense model's placement. | Current FreeToken support matrix lists Qwen3.8 dense checkpoints. `auto` resolves dense models to the GPU-resident fused backend. | Optional FreeToken-versus-llama.cpp comparison after CUDA 13 installation; not the reason for the RAM upgrade. |
| Qwen3.6-35B-A3B NVFP4 | Fits as before. Its approximately 16.4 GB expert source is near, but below, the current locked-memory ceiling. | Supported one-GPU Qwen MoE control. TP remains 1. | First FreeToken smoke test; record actual `VmLck` and do not assume the narrow lock margin is sufficient. |
| Qwen3.6-35B-A3B BF16 | The approximately 64.4 GB routed-expert pool now fits in physical RAM with useful headroom. | The present 17.66 GiB hard lock prevents pinning the expert bank. Qwen TP remains 1. | Highest-confidence FreeToken target after raising `memlock`; it is the closest direct RTX 3090 paper comparison. |
| gpt-oss-120b MXFP4 | The former 128 GiB host-RAM tier is now physically available, but actual per-rank host-bank and process RSS must be measured. | Source shards attention and MXFP4 expert intermediates for TP=2. TP=3 is invalid because 64 query heads are not divisible by 3. The current lock cap is still too small for an unmeasured expert bank. | Conditional two-GPU capacity experiment after Qwen BF16; retain 4k context first and stop on memory pressure or swap. |
| Qwen3.8-Flash-Next NVFP4 | The official NVFP4 artifact is about 126 GiB Hub used storage. Its 47.7 GiB FP8 PLE table plus routed-expert source banks make 141 GiB a conditional, tight host-memory experiment, not a safe fit. | FreeToken lists this architecture and pins the PLE table in a host bank. PLE alone exceeds the 17.66 GiB hard lock. Current public sources do not establish a two-3090 TP deployment for this loader. | Do not download or launch before raising `memlock` and obtaining a full FreeToken memory breakdown. Evaluate only after the Qwen BF16 and gpt-oss paths. |
| DeepSeek-V4-Flash / MiniMax-M2.5 | Still outside the safe 141 GiB tier. Flash alone has 155.427 GiB official used storage; MiniMax retains a roughly 112 GB estimated expert pool before practical margin. | No local 3090 result makes a swap-backed attempt useful. | Remain excluded pending a 256 GiB-class host-memory design. |

For a candidate with a larger host bank, set the service's hard `memlock` limit above the measured pinned-bank requirement before launching it, then confirm pinned usage and zero swap throughout a long decode. Do not set a large lock limit as a substitute for physical memory accounting.

## What FreeToken actually does

### Memory hierarchy

FreeToken uses a two-level expert hierarchy:

- Non-expert weights and runtime state remain on the GPU.
- The complete routed-expert pool remains in host RAM as the source of truth.
- Spare VRAM becomes a global LRU cache of complete `(layer, expert)` entries.
- Cache misses are either copied over PCIe and retained or evaluated directly from host RAM by the CPU.
- KV and recurrent-state memory share the GPU budget with the expert cache and can be resized at scheduler safe points.

It is therefore **not** using disk as steady-state virtual VRAM. The complete routed-expert pool must fit in host RAM for the fast path; non-expert weights remain on the GPU. Disk is involved at load/conversion time, not on every token.

### The `q*` allocation

For `m` unique expert misses in a layer, FreeToken measures:

- `B_P`: pinned host-to-device expert-transfer bandwidth;
- `B_H`: effective CPU expert-kernel bandwidth.

It fills approximately:

```text
q* = m * B_P / B_H
```

misses into the GPU cache and executes the rest from host RAM on the CPU. Both branches run concurrently. The intuition is that saturated PCIe consumes `B_P` of host bandwidth, leaving approximately `B_H - B_P` for CPU execution. At least one miss is filled so the cache continues to warm.

This is a good fit for a platform where host memory bandwidth exceeds PCIe bandwidth. The exact split cannot be inferred safely from specifications, which is why FreeToken provides:

```bash
ft bench bw --dtype nvfp4,bf16
```

### Prefill and agent-state handling

Long prompts route across nearly every expert in each layer, making MoE prefill effectively dense. FreeToken uses two full-layer buffers and streams layer `l+1` while the GPU computes layer `l`. If two complete expert layers do not fit, it falls back to on-demand loading.

For hybrid-attention models, recurrent state checkpoints are expensive and sparse. FreeToken places them at semantic boundaries such as tool calls, tool outputs, thinking blocks, and conversation turns. This lets an agent resume from a surviving boundary after context compaction instead of recomputing the full prefix.

### Code-to-paper correspondence

The public code contains concrete implementations and tests for the main claims:

| Paper mechanism | Repository evidence | Assessment |
|---|---|---|
| Bandwidth calibration and hybrid miss split | [`benchbw.py`](https://github.com/FlashML-org/FreeToken/blob/main/python/freetoken/moe/benchbw.py), [`cpu_offload.py`](https://github.com/FlashML-org/FreeToken/blob/main/python/freetoken/moe/cpu_offload.py), [`layers/moe.py`](https://github.com/FlashML-org/FreeToken/blob/main/python/freetoken/layers/moe.py) | Source-confirmed |
| Global LRU expert cache | [`offload_cache.py`](https://github.com/FlashML-org/FreeToken/blob/main/python/freetoken/moe/offload_cache.py), eviction/reload tests under [`tests/moe`](https://github.com/FlashML-org/FreeToken/tree/main/tests/moe) | Source-confirmed |
| Double-buffered prefill | CLI default plus NVFP4/MXFP4 overlap tests in [`tests/moe`](https://github.com/FlashML-org/FreeToken/tree/main/tests/moe) | Source-confirmed |
| Semantic recurrent-state checkpoints | Scheduler anchor handling and hybrid radix tests under [`tests/kvcache/radix`](https://github.com/FlashML-org/FreeToken/tree/main/tests/kvcache/radix) | Source-confirmed |
| Elastic expert/KV/Mamba/SWA pools | `ft ctl cache` rebuild endpoint and scheduler tests; see [CLI reference](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md) | Source-confirmed |
| FTW fast-load format | [`checkpoint/convert.py`](https://github.com/FlashML-org/FreeToken/blob/main/python/freetoken/checkpoint/convert.py) and `ft checkpoint` | Source-confirmed |
| Multi-GPU | Rank-aware launch/distributed code exists, but loaders opt in individually; Qwen3.5/3.6 MoE currently rejects TP > 1 | Partial/model-specific |

### Maturity and installation caveats

- The repository declares **Development Status: Beta** in [`pyproject.toml`](https://github.com/FlashML-org/FreeToken/blob/main/pyproject.toml).
- The first tagged release is v0.1.2, published 2026-08-19.
- Linux x86_64, driver r580+, Python 3.10+, PyTorch 2.11, CUDA 13, and `nvcc` are required by the current [installation guide](https://github.com/FlashML-org/FreeToken/blob/main/docs/install.md).
- The optimized NVFP4 Marlin path for SM80-SM99 depends on vLLM 0.14.x, while FreeToken core requires Transformers 5.5+ and that vLLM range pins Transformers below 5. The project explicitly recommends a dedicated environment for that path. Portable Triton is the fallback.
- The complete host expert bank must be pinned/device-mapped for fast offload. OS locked-memory limits and the driver's ability to register a very large region matter.
- FTW output is approximately checkpoint-sized. Some NVFP4 backend conversions require the full bank set in memory rather than streaming one completed layer at a time.

## Is it genuinely novel?

### Novelty verdict

**Yes as a serving-system design; no as an entirely new memory or MoE principle.**

| Component | Important prior art | FreeToken's added value |
|---|---|---|
| Disk/CPU/GPU offload | FlexGen, DeepSpeed ZeRO-Inference, EdgeMoE | Reorients the design toward interactive edge MoE and agent workloads rather than large-batch throughput |
| Expert caching | EdgeMoE, Mixtral-offloading, MoE-Infinity | One global dynamic LRU connected directly to exact CPU/GPU miss execution |
| Expert prefetch/prediction | MoE-Infinity, ProMoE, ExpertFlow, FineMoE | Does not depend on a perfect predictor; unavoidable misses are scheduled quantitatively |
| CPU expert execution | Fiddler, KTransformers, HybriMoE | Closed-form measured `q*` allocation and continued cache warming instead of static all-CPU/all-transfer decisions |
| Layer streaming | FlexGen, DeepSpeed | Full-layer MoE prefill overlap sharing storage with the decode cache |
| Prefix reuse | SGLang radix cache | Recurrent-state checkpoints placed where agent harnesses actually edit context |
| Elastic memory | eLLM, FluxMoE, vLLM-style paged memory | Runtime reallocation among expert, KV, Mamba, and sliding-window pools without reloading host experts |
| CUDA Graph execution | vLLM/SGLang/FlashInfer patterns | Dynamic cache decisions and the CPU branch represented inside a captured heterogeneous graph |

The strongest new idea is the combination of **compute where an absent expert already resides** and **copy only the bandwidth-balanced share that should become future cache hits**. The semantic recurrent-state placement is also a useful workload-specific contribution.

### Research caveats

- It is arXiv v1 and not peer-reviewed as of this snapshot.
- Only three models are evaluated despite a claim of more than 20 supported models.
- Only NVIDIA/CUDA systems are evaluated.
- The 3090, 4090, and 5090 comparison machines were rented dual-socket servers with CPU threads capped to emulate consumer hosts.
- The paper has no direct ablation of `q*` versus transfer-only, CPU-only, and alternative hybrid splits.
- Semantic anchors, elastic resizing, FTW startup, and CUDA Graph capture do not each receive an isolated quantitative ablation.
- One SWE-bench issue is used for the coding-agent experiments. This is a systems trajectory, not a broad model-quality evaluation.
- Energy, accuracy, total agent completion time, and quantization degradation are not reported.

## Is the comparison with Ollama fair?

### What was controlled well

- The same agent harnesses and nominal requests were used.
- Qwen3.6 was BF16 across engines in the main RTX 5090 comparison.
- Supporting engines consumed DeepSeek-V4's native MXFP4 expert blocks bit-exactly.
- Coding runs had to produce the reference gold patch; the email/calendar run had to complete all 13 turns.
- CPU thread caps and NUMA pinning are described.
- The paper reports decode throughput and TTFT separately rather than hiding latency in an aggregate score.

### What prevents a fully reproducible fairness judgment

- Engine versions/commits and full launch commands are absent.
- No tuning budget is described. Ollama is designed for simple static placement, while FreeToken is purpose-built for this workload.
- No repetitions, confidence intervals, or run-to-run variance are reported.
- Agent trajectories diverge, so per-request means summarize different paths and token counts. The paper deliberately avoids total wall-clock comparison.
- Unsupported configurations are marked as failures, which is operationally fair but not an algorithm-only comparison.
- The 4060 result uses Qwen NVFP4 while the desktop/server Qwen tests use BF16.
- The 3090 host is not representative of this exact CPU/PCIe topology, especially its PCIe bandwidth.

### Reported results

On the OpenCode/SWE workload in the cross-hardware figure:

| Hardware/model | FreeToken | KTransformers | llama.cpp | Ollama | Source status |
|---|---:|---:|---:|---:|---|
| RTX 4060 laptop, Qwen3.6 NVFP4 | 39.3 | Unsupported | ~22.3 | ~18.1 | FreeToken exact prose; bars approximate |
| RTX 3090, Qwen3.6 BF16 | ~36.2 | ~27.4 | ~22.1 | ~18.2 | Approximate graph reading |
| RTX 4090, Qwen3.6 BF16 | ~42.9 | ~31.8 | ~25.8 | ~14.1 | Approximate graph reading |
| RTX 5090 server, Qwen3.6 BF16 | ~76.7 | ~35.5 | ~41.1 | ~32.6 | Approximate graph reading |
| RTX 5090 desktop, Qwen3.6 BF16 | ~73.8 | ~34.8 | ~33.0 | ~24.9 | Approximate graph reading |

Across four RTX 5090 workloads, the paper reports FreeToken at **77-83 tok/s** for Qwen and **22-25 tok/s** for DeepSeek-V4. It claims a **1.8-2.3x** Qwen lead and **1.5-1.9x** DeepSeek lead over the strongest baseline in each workload. FreeToken kept its worst turn's TTFT below 44 seconds, while each baseline exceeded 150 seconds in at least one configuration.

### Verdict

There is no obvious apples-to-oranges precision trick in the primary Qwen comparison. The result is also physically plausible: Ollama/llama.cpp use static placement, whereas a dynamic cache can exploit token-to-token expert locality and FreeToken simultaneously uses CPU DRAM bandwidth and PCIe bandwidth. In the paper's route replay at equal cache capacity, Qwen miss rates were 16% for FreeToken LRU, 41% for KTransformers-style placement, and 62% for llama.cpp-style static placement.

The correct conclusion is therefore: **the directional advantage is credible, but the exact token rates and multipliers remain author-reported until reproduced on this system.**

## AirLLM comparison

### Paper status

There is no AirLLM-specific academic paper in the project. Its citation block cites [AirLLM as software](https://github.com/lyogavin/airllm#citing-airllm). The linked [arXiv:2212.09720](https://arxiv.org/abs/2212.09720), *The case for 4-bit precision*, motivates its optional block-wise weight compression; it does not describe or evaluate AirLLM. Another later paper named AirLLM concerns remote LoRA fine-tuning and is unrelated.

### Runtime behavior

For a dense model, every autoregressive token performs another full forward pass. AirLLM loads each layer shard into CPU memory, copies it to one GPU, executes it, and evicts it back to the meta device. The OS page cache can hide physical SSD reads only if enough RAM is available, but CPU-to-GPU movement and Python/module overhead remain.

Per-expert streaming is not a generic behavior for all supported MoE models. In the inspected code, only classes declaring an `expert_prefix` use it; Kimi K3 is the shipped implementation. Other MoE architectures can fall back to loading the complete decoder-layer shard, including all experts.

AirLLM accepts a single device such as `cuda:0`. It does not pool two or three GPUs.

### Published performance evidence

| Experiment | Peak VRAM | Startup | Decode | Assessment |
|---|---:|---:|---:|---|
| Qwen3.8-27B dense BF16 on RTX 3090 | 3.33 GiB | 2.5-minute split, 164-second initialization | Not reported | Demonstrates fit, not interactive performance |
| Kimi K3 2.8T MXFP4 on RTX 6000 Ada | 3.72 GiB | 900 seconds | 292 seconds/token, ~0.0034 tok/s | Technically impressive but operationally unusable for an agent |

AirLLM's older compression timing chart lacks enough information to reproduce a token rate. Its headline 70B/405B/671B VRAM figures should be read as capacity claims, not latency claims.

### When AirLLM is still useful

- Verifying that a checkpoint and tokenizer work.
- Inspecting a model that cannot fit in RAM or VRAM by any other route.
- Very low-frequency offline generation where one result is worth hours.
- Researching per-expert disk streaming.

It is not suitable as the main coding-agent inference server.

## RAM compression and virtual memory for LLM inference

These concepts still exist, but successful inference engines implement them with model-aware tiers rather than relying on transparent OS paging.

| Mechanism | What moves or compresses | Batch-1 MoE usefulness | Main problem |
|---|---|---|---|
| [CUDA Unified Memory oversubscription](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html) | GPU pages migrate or are remotely mapped between VRAM and system RAM | Usually poor unless the expert working set is stable and explicitly prefetched | GPU page faults, small-page migration, TLB pressure, PCIe thrashing |
| Linux swap | Anonymous pages move to SSD/HDD | Emergency OOM protection only | Repeated major faults make per-token inference non-interactive |
| [zram/zswap](https://docs.kernel.org/admin-guide/mm/zswap.html) | Anonymous swap pages are compressed in RAM or in a compressed pre-swap cache | Little value for packed model weights | CPU decompression and already-high entropy quantized bytes; clean mmap pages are normally dropped, not compressed |
| [Hugging Face Accelerate offload](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference) | Modules reside in CPU RAM or disk-backed maps and are staged by hooks | Functional, but generic and not expert-aware | Repeated disk/RAM and PCIe movement on every forward |
| [DeepSpeed ZeRO-Inference](https://www.deepspeed.ai/2022/09/09/zero-inference.html) | Full layers stream from CPU/NVMe | Good throughput with enough batch, poor interactive latency | Batch 1 has too little compute to hide layer transfer |
| [FlexGen](https://arxiv.org/abs/2303.06865) | Weights, activations, and KV are placed across GPU/CPU/disk; some can be 4-bit compressed | Not designed for low-latency batch 1 | Its headline throughput relies on very large effective batches |
| llama.cpp/Ollama | GGUF is memory-mapped; selected tensors/layers remain on GPU and the rest execute on CPU | Good while the quantized model fits RAM plus VRAM | CPU DRAM bandwidth; catastrophic major faults if the file is genuinely paged from disk each token |
| [KTransformers](https://github.com/kvcache-ai/ktransformers) | Hot experts on GPU, cold experts execute from CPU DRAM | Good for supported MoE and strong CPU kernels | DRAM bandwidth, CPU ISA, and synchronization |
| [MoE-Infinity](https://arxiv.org/abs/2401.14361) | Experts are prefetched/cached using activation traces | Designed for batch 1 | PCIe stalls after cache/prediction misses; its evaluated server was unsuitable for multi-turn agents |
| FreeToken | Complete experts in pinned RAM; dynamic GPU cache; CPU execution for selected misses | Designed for interactive batch 1 | Host RAM capacity, host bandwidth, PCIe, and model-specific support |
| AirLLM | Dense layers or selected Kimi experts stream from storage/RAM through one GPU | Fit-only | Storage/PCIe transfer is paid repeatedly per output token |

### Why quantization is preferred

Quantization reduces bytes at every tier simultaneously: download, disk, RAM, PCIe, VRAM, and often the bytes read by the compute kernel. A well-supported low-bit kernel computes directly from the packed representation, so capacity and speed improve together. Transparent compression instead adds a fault, decompression, and movement step to the critical path.

General-purpose RAM compression is especially weak after quantization. Packed 4-bit or FP8 weights tend to have relatively high entropy, so a second generic compressor finds limited redundancy. Specialized schemes such as GPTQ, AWQ, GGUF K-quants, MXFP4, and NVFP4 are effectively **model-aware memory compression with compute kernels designed around the compressed form**.

### Disk remains useful in three places

1. Cold checkpoint storage and FTW conversion.
2. Hierarchical **KV** caching when old context may be reused but is not touched every token.
3. Throughput-oriented offline inference where large batches hide I/O.

Disk is generally not viable as the source of expert weights repeatedly needed during interactive decode. On the current SATA SSD, a 149 GiB checkpoint has a best-case sequential read time of roughly 4.5 minutes before parsing, pinning, or warmup.

## Framework comparison for this machine

| Runtime | Best use here | Multi-GPU behavior | Oversized-model strategy | Main limitation |
|---|---|---|---|---|
| **FreeToken** | Supported frontier MoE, agent APIs, batch-1 latency | TP infrastructure exists but is model-specific; current Qwen3.5/3.6 MoE path rejects TP | Pinned host expert pool, global GPU LRU, CPU/GPU miss split | Young project, CUDA 13, large RAM, limited support matrix |
| **vLLM** | Fully resident models, concurrency, production-compatible API | TP, PP, and MoE expert parallelism; current docs recommend PP when GPUs lack NVLink | `cpu_offload_gb` and grouped prefetch exist, but generic offload is not FreeToken's expert cache | Collectives/PP bubbles over PCIe; some FP4 kernels require newer architectures |
| **Ollama** | Easiest GGUF deployment and quick model comparisons | Tries one GPU, then automatically spreads a model across available GPUs | Static GPU/CPU placement through llama.cpp | Less control and no semantic expert cache; performance depends heavily on model fit |
| **llama.cpp** | Flexible GGUF quantization, CPU+GPU and multi-GPU experiments, `llama-bench` | Tensor split, layer placement, and `--fit` automatic CPU/GPU placement | mmap plus CPU execution of nonresident tensors/experts; selectable `--cache-type-k` / `--cache-type-v` | Hybrid placement and cache quantization must be measured on this RAM/PCIe topology; CPU bandwidth remains a ceiling |
| **KTransformers** | Large supported MoE with a very strong CPU memory/ISA path | Primarily heterogeneous CPU/GPU rather than pooling consumer VRAM | Hot experts GPU, cold experts CPU | This Zen 2 CPU lacks AVX-512/AMX; results from recent Intel CPUs may not transfer |
| **SGLang** | Resident high-throughput serving and agent prefix reuse | Distributed serving supported | Hierarchical KV cache, not a general edge expert-offload replacement | Model weights generally need accelerator capacity |
| **AirLLM** | Make an otherwise impossible checkpoint emit output | One GPU only | Layer or selected-expert streaming | Seconds to minutes per token |
| **MoE-Infinity** | Research baseline for expert prefetch/cache | Not the practical focus | Host expert pool and predictive GPU cache | Paper found no usable multi-turn server/KV retention for its agent evaluation |

### Two versus three GPUs

For vLLM, the official [parallelism guide](https://docs.vllm.ai/en/stable/serving/parallelism_scaling.html) recommends:

- no distributed inference when a model fits one GPU;
- tensor parallelism when it fits a multi-GPU node with fast interconnect;
- pipeline parallelism for uneven splits or when GPUs lack NVLink.

Three GPUs are valid in principle, but model dimensions, layer count, expert count, and each backend's kernels can impose divisibility constraints. On this topology, `PP=3` may be more robust than `TP=3`, but batch-1 pipeline bubbles can limit latency improvement. Benchmark both rather than assuming 1.5x scaling from two to three cards.

Ollama automatically tries to fit a model on one GPU first to avoid PCIe traffic and then distributes it when necessary. This is good capacity behavior but does not guarantee that all three GPUs improve one-token latency.

## Model capability evidence

The following values come from official model cards or papers and are self-reported by model developers. Harness, reasoning effort, tools, sampling, and benchmark version materially affect the result. `NR` means that exact field/variant was not reported in the cited official source.

| Model | Total / active parameters | Context | SWE-bench result | AIME | LiveCodeBench | GPQA Diamond | Agent/coding result |
|---|---:|---:|---:|---:|---:|---:|---:|
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 27B hybrid; full attention every 4 layers, 4 KV heads | 262k native | 61.7, Pro | NR | v6 90.3 | 89.2 | Terminal-Bench 2.0 73.0 |
| [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) | 35B / 3B MoE | 256k | 79.0 | AIME24 90.7 | NR | 93.5 | Terminal-Bench 2.0 67.8 |
| [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | 35B / 3B LM | 262k native, ~1M extended | 73.4, internal scaffold | NR | v6 80.4 | 86.0 | Terminal-Bench 2.0 51.5 |
| [Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) | 80B hybrid | 262k | 70.6, SWE-Agent | AIME25 83.07 | v6 82.2 | NR | Terminal-Bench 2.0 73.9 |
| [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | 304.180B / NR | 1M | NR | NR | NR | NR | Terminal-Bench 2.1 82.7, DeepSeek Harness |
| [DeepSeek-V4-Pro-0813](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813) | 1.650T / NR | 1M | NR | NR | NR | NR | Terminal-Bench 2.1 87.9, DeepSeek Harness |
| [MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5) | NR in extracted official table | NR | 80.2, Claude Code | AIME25 86.3 | NR | 85.2 | Multi-SWE-Bench 51.3 |
| [gpt-oss-120b](https://arxiv.org/abs/2508.10925) | 116.83B / 5.13B | 131k | 62.4, high effort | AIME25 92.5 | NR | 80.1 | Tau-Bench Retail 67.8 |
| [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) | 753B / 40B, per FreeToken paper | 1M | NR | NR | NR | 91.2 | Terminal-Bench 2.1 81.0 |

For the stated primary task, coding, Qwen3.8-27B and Ornith-1.5-35B-A3B are the immediate quality comparison; neither developer score is a substitute for the same local scaffold. Qwen3-Coder-Next is the large coding target now testable with a measured two-GPU hybrid placement and cleanly resident at Q5 on three cards. DeepSeek-V4-Flash and MiniMax-M2.5 remain near-frontier targets only after a RAM and PCIe upgrade, at materially lower local throughput. DeepSeek-V4-Pro is not: Hugging Face reports 1,659.419 GiB used storage for its 1.650T tensors. Even an impossible all-FP4 Pro representation has a 768.6 GiB lower bound before runtime memory and before its FP8/BF16 tensors.

## What should fit and how fast it may run

### Storage and safe RAM tiers

The safe tier includes room for the OS, runtime allocations, conversion, pinned buffers, and KV/state. It is not just checkpoint size rounded upward.

| Model/format | Approx. checkpoint | Approx. routed expert pool | Safe host RAM | Current 141 GiB RAM / 17.66 GiB `memlock` |
|---|---:|---:|---:|---|
| Qwen3.8-27B Q5_K_M GGUF | 19.8 GB publisher decimal | N/A | 64 GiB | Fits one 3090 as a measured text deployment; optional vision/MTP artifacts need separate margin |
| Ornith-1.5-35B-A3B Q4_K_M GGUF | 21.7 GB publisher decimal | N/A | 64 GiB | Fits one 3090 at modest context |
| Qwen3.6-35B-A3B NVFP4 | 21.8 GiB | ~16.4 GB | 64 GiB | Physical RAM fits. The narrow `memlock` margin needs a live `VmLck` check. |
| Qwen3.6-35B-A3B BF16 | ~65.2 GiB | 64.4 GB | 128 GiB | Physical RAM now fits; current hard `memlock` blocks the pinned expert bank. |
| gpt-oss-120b MXFP4 | 60.8 GiB GGUF; FreeToken uses official HF safetensors | Not isolated here | 128 GiB | Physical RAM now permits a TP=2 experiment; current `memlock` and per-rank RSS must be resolved first. |
| Qwen3.8-Flash-Next NVFP4 | ~126 GiB Hub used storage | 47.7 GiB pinned PLE plus unreported routed-expert source-bank total | Measured accounting required | PLE alone exceeds current `memlock`; 141 GiB host RAM makes this conditional and tight, not a safe deployment. |
| Qwen3-Coder-Next Q4_K_M GGUF | 48.4 GB publisher decimal | N/A | Measured `llama.cpp --fit` placement | 141 GiB increases host-placement margin; retain zero-swap requirement. |
| Qwen3-Coder-Next Q5_K_M GGUF | 56.7 GB publisher decimal | N/A | 64 GiB | Preferred clean three-GPU deployment |
| Qwen3-Coder-Next Q6_K GGUF | 65.5 GB publisher decimal | N/A | 64 GiB | Aggressive three-GPU fit test only; validate context and state margin |
| MiniMax-M2.5 NVFP4 | ~130.3 GiB | ~112 GB estimate | 192 GiB minimum, 256 preferred | Does not fit |
| DeepSeek-V4-Flash official hybrid FP4/FP8 | 155.427 GiB Hub used storage | NR | 256 GiB preferred | Does not fit |
| GLM-5.2 NVFP4 | ~432.9 GiB | ~362 GB estimate | 512 GiB or more | Not a practical platform target |
| Generic dense 70B Q4 | ~35-43 GiB | N/A | 64 GiB marginal, 128 preferred | Fits aggregate VRAM/RAM only with careful context sizing |

### Decode estimates

These ranges are deliberately conservative. They are not benchmark results.

| Model and intended runtime | Two 3090s | Three 3090s | Expected decode on current links | After stable intended Gen3 x16/x16/x8 topology | Evidence |
|---|---|---|---:|---:|---|
| Qwen3.8-27B NVFP4, FreeToken 0.1.2 | One GPU; second can host another service | Same; TP>1 is rejected by the loader | **M 18-20 tok/s warm decode** | Same | Local 2026-08-29 run at 8k context and 200 W: 21.8 GiB VRAM, 3.1 GiB PSS. First request paid an 83-second JIT cost; warm requests did not. |
| Qwen3.8-27B Q5, current llama.cpp | One GPU; second can host another service | Same | **U** | **U** | Retain as a direct mature-runtime comparison against the measured FreeToken NVFP4 path. |
| Ornith-1.5-35B-A3B Q4, current llama.cpp | One GPU; second can host another service | Same | **U** | **U** | Official GGUF is available, but no comparable local throughput evidence was identified |
| Qwen3.6-35B-A3B NVFP4, FreeToken offload on one GPU | Extra GPU not used by this loader | Same | **E 20-30 tok/s** | **U** | Paper measured 39.3 tok/s on a PCIe4 x8 RTX 4060 laptop, not this system; do not convert that result into a Gen3 x16 forecast |
| Qwen3.6-35B-A3B BF16, FreeToken with 141 GiB RAM | Extra GPU not used by this loader | Same | **E 9-20 tok/s after `memlock` is raised** | **U** | The paper's ~36.2 tok/s 3090 result used PCIe4 x16 and is not a throughput forecast for the intended Gen3 layout |
| gpt-oss-120b MXFP4, FreeToken TP=2 with 141 GiB RAM | Host-offloaded, source-level TP=2 supported | 60.8 GiB weights nominally fit 72 GiB, but little KV/runtime room | **M 4.6-6.6 tok/s sustained decode** | **U** | Local 2026-08-29 run: 362 completion tokens in 72.5 s end-to-end at 4k context cap, with a measured 26.6% PCIe-fetch hybrid split. This is one warm request, not a concurrency or long-context benchmark. |
| Qwen3.8-Flash-Next NVFP4, FreeToken | No published two-GPU placement | TP status unverified; third card does not remove host-bank need | **U** | **U** | 47.7 GiB pinned PLE table plus expert source bank; test only after `memlock` and full memory accounting. |
| Qwen3-Coder-Next Q4, llama.cpp `--fit` | Conditional CPU/GPU hybrid on two cards | Resident across three cards | **C 24.8-33.2 tok/s** on a community dual-3090 Q4_K_XL report at 131k-32k context | **U** | Reproduce with Q3/Q4, actual installed RAM, Gen3 x8 links, and 200 W cap before planning around it |
| Qwen3-Coder-Next Q5, llama.cpp three-way placement | Not practical resident | Preferred resident deployment | **U** | **U** | Q5 has more useful quality/context margin than a two-GPU hybrid Q4 test; benchmark rather than extrapolate from community Q4 |
| DeepSeek-V4-Flash, FreeToken after 256 GiB RAM | One-GPU offload path | Extra GPUs help only if this loader's TP is validated | **E 2.5-6 tok/s** | **U** | Paper measured 22-25 on RTX 5090; no 3090 result. The official FP4/FP8 checkpoint still cannot be resident in 72 GiB VRAM. |
| MiniMax-M2.5 NVFP4, FreeToken after 256 GiB RAM | One-GPU offload path | TP unverified | **U, plausibly 3-8 tok/s** | **U** | Checkpoint/active path suggest DSV4-like class; no measurement |
| GLM-5.2 NVFP4 | Insufficient host/GPU platform evidence | 72 GiB aggregate still below paper's 96 GiB GPU and host RAM is insufficient | Not recommended | Not recommended | **M 14.9 tok/s only on RTX PRO 6000 96 GiB + 512 GiB host** |
| Generic dense 70B Q4, llama.cpp/Ollama/vLLM | Fully resident across 48 GiB with limited KV margin | More KV margin and PP option | **U 12-25 tok/s** | **U**, topology dependent | No exact-model benchmark; no NVLink currently |

### Local Qwen3.8-27B Q5 native-context validation: 2026-08-30

A current CUDA llama.cpp build at commit `c841aeeb8bb2fe417038dadfa9b007cf1a9ef950`
successfully loaded the 18.41 GiB Unsloth Q5 GGUF on GPU 0 with a 262,144-token Q8 KV cache in
host RAM. The launch used `--no-kv-offload`, so all attention work between KV storage and output
also ran on CPU; this is not a storage-only context tier.

- Initial placement used about 17.8 GiB VRAM and 9.2 GiB host RSS; GPU 1 remained unused.
- Short-context decode measured 7.10 tok/s versus 18-20 tok/s for the one-card FreeToken NVFP4
  profile with GPU-resident context.
- At 8,016 populated tokens, prefill was 596.8 tok/s and decode was 5.39 tok/s.
- At 32,016 populated tokens, prefill was 535.8 tok/s and decode was 3.19 tok/s.
- At 64,016 populated tokens, prefill was 438.6 tok/s and decode was 2.01 tok/s.
- The native 262k allocation is capacity-valid. Extrapolation suggests approximately 0.6-0.8
  decode tok/s near the limit and 15-25 minutes for first prefill; these are estimates, not a full
  262k measurement.

The correct deployment distinction is therefore: keep Qwen on one GPU with GPU-resident context
for latency, or keep Qwen weights on one GPU and context in RAM when preserving the other GPUs is
more important. Splitting Qwen across GPUs is useful only when GPU-resident long context outweighs
the value of reserving those GPUs for Coder-Next.

### Local gpt-oss-120b TP=2 capacity validation: 2026-08-29

The official root safetensors checkpoint was downloaded at revision
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, excluding the repository's unused Metal and
`original/` duplicate artifacts. Its 15 required root shards total 65,248,893,184 bytes (60.8 GiB).

- FreeToken 0.1.2 ran `gpt-oss-120b` from the local checkpoint at TP=2, `--moe-backend hybrid`,
  `--expert-load serial`, `--memory-ratio 0.85`, 4,096 context tokens, and one concurrent request.
- Exact MXFP4 calibration measured 19.4 GB/s CPU-MoE and 6.73 GB/s PCIe gather. The overlapped
  result selected hybrid execution with 26.6% of decode misses fetched over PCIe.
- The ready service consumed 61.6 GiB PSS across its process tree and 40.5 GiB aggregate VRAM
  (20,001 MiB plus 20,688 MiB from `nvidia-smi`), leaving approximately 3.06 GiB free on each
  card after CUDA graph capture. `MemAvailable` remained 73 GiB.
- A warm 94-token prompt produced 362 completion tokens in 72.518 seconds. Scheduler samples
  after warm-up reported 4.64-6.64 decode tok/s, generally about 5.5 tok/s. This validates the
  two-GPU capacity path but is not a high-concurrency, long-context, or task-quality result.
- During sustained decode, sampled board power was about 125-133 W per GPU, below the intentional
  200 W cap; observed GPU temperatures reached 58 C and 63 C. Both cards reported Gen3 x8 after
  the run. The planned under-load `LnkSta` validation remains outstanding.
- Both TP workers reported `VmSwap: 0 kB`; `free` showed 512 KiB aggregate system swap in use, so
  a formal whole-host zero-swap pass still requires identifying and clearing that unrelated usage.
  `VmLck` remained `0 kB` even though the run succeeded: FreeToken pins banks with CUDA
  `cudaHostRegister`, whose accounting is not reflected by Linux `VmLck` here. The service's
  `LimitMEMLOCK=infinity` is verified and should remain the control.

### Prefill transfer bounds on the current links

Using 6-7.5 GB/s sustained as a Gen3 x8 range, effectively dense MoE prefill must move approximately:

| Expert pool | Transfer time per complete pass | Can overlap with compute? |
|---:|---:|---|
| Qwen NVFP4, ~16.4 GB | 2.2-2.7 s | Yes, if two layer buffers fit |
| Qwen BF16, 64.4 GB | 8.6-10.7 s | Yes |
| DeepSeek-V4-Flash, 155.427 GiB | 20.7-25.9 s | Yes |
| MiniMax-M2.5, ~112 GB estimate | 14.9-18.7 s | Yes |
| GLM-5.2, ~362 GB estimate | 48-60 s | Yes, but still impractical here |

These are transfer-stage bounds, not complete TTFT predictions. FreeToken can hide some or all movement behind GPU compute, while semantic prefix reuse can avoid much of the repeated prefill on later agent turns.

### Cold-start bounds on the SATA SSD

At a best-case 0.55 GB/s sequential rate, before parsing and pinning:

| Checkpoint | Best-case raw read |
|---:|---:|
| 20-22 GiB Qwen | ~40 seconds |
| 61-65 GiB gpt-oss/Qwen BF16 | ~2 minutes |
| 155 GiB DeepSeek-V4-Flash | ~5 minutes |
| 433 GiB GLM | ~13-14 minutes |

A 2-4 TB PCIe 4.0 NVMe drive would improve model switching and FTW conversion substantially, but not steady-state FreeToken decode once the expert pool is resident in RAM.

## Recommended deployment tiers

### Tier 0: use the current hardware before buying around assumptions

1. Run the staged privileged installer at `<FREETOKEN_ROOT>/install-system.sh` to install CUDA toolkit 13.2 and activate the dedicated FreeToken system service; keep Ollama/llama.cpp/vLLM environments separate.
2. Raise the hard locked-memory limit before attempting a large pinned expert bank. The current soft and hard value is 18,521,516 KiB (17.66 GiB); this is below Qwen3.6 BF16's 64.4 GB expert pool and Qwen3.8-Flash-Next's 47.7 GiB PLE table. Confirm `VmLck` and zero swap during the run.
3. Verify the top PCIE1 card can reach stable Gen3 x16; the bottom PCIE4 card operating at Gen3 x8 is expected. Inspect motherboard slot wiring, BIOS, and riser quality. Target stable Gen3 x16 + x16 + x8; prefer that over an unreliable Gen4 experiment.
   ```bash
   nvidia-smi --query-gpu=index,name,pci.bus_id,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current --format=csv
   nvidia-smi topo -m
   lspci -vv -s <PCI-BDF>
   ```
   Use `lspci` to inspect each device's `LnkCap` and `LnkSta` under load. If capabilities are access-denied, rerun only that command with `sudo lspci -vv -s <PCI-BDF>`.
4. Run `ft bench bw --dtype nvfp4,bf16` and retain the per-GPU JSON profile.
5. Benchmark Qwen3.8-27B Q5_K_M and Ornith-1.5-35B-A3B Q4_K_M as text-only one-card candidates.
6. Run Qwen3-Coder-Next Q3 then Q4 with `llama.cpp --fit`, explicit `--fit-target`, and explicit `--fit-ctx` on the two cards at 4k, 16k, and 32k. Retain the computed placement, host RSS, swap state, VRAM, PCIe counters, tok/s, and power result.
7. Use Qwen3.6-35B-A3B NVFP4 as the first FreeToken smoke test. After the lock limit is corrected, run its BF16 checkpoint as the primary FreeToken reproduction before gpt-oss or Flash-Next.

This tier can already provide a strong local coding model. Qwen3.8 and Ornith should compete in the same local quality/factuality suite; Qwen3.6-35B-A3B BF16 becomes the primary FreeToken validation after system prerequisites are fixed.

### Tier 1: installed 141 GiB RAM

The host now has 141 GiB total and 137 GiB available, so this capacity tier is installed. It adds:

- Qwen3.6-35B-A3B BF16, allowing a direct reproduction of the paper's 3090 comparison.
- gpt-oss-120b MXFP4 as a cautious two-GPU TP=2 experiment.
- Comfortable dense 70B Q4 serving and more filesystem cache.

The outstanding requirement is system configuration, not DIMM capacity: FreeToken needs a hard `memlock` limit large enough for the measured pinned expert banks. Capacity without channel bandwidth and locked-memory headroom is a poor FreeToken upgrade.

### Tier 2: 256 GiB RAM plus corrected PCIe lane width/topology

Adds the most interesting near-frontier tier:

- DeepSeek-V4-Flash-0731 official hybrid FP4/FP8 checkpoint.
- MiniMax-M2.5 NVFP4.
- Larger expert caches and safer long-running agent sessions.

For FreeToken, this upgrade is still the safe tier for DeepSeek-V4-Flash and MiniMax-M2.5. It makes Flash plausible, not Pro: Pro's official used storage is 1,659.419 GiB and even a hypothetical all-FP4 representation requires at least 768.6 GiB before runtime state. The 141 GiB host does, however, create a separate conditional Qwen3.8-Flash-Next experiment once its total pinned-bank usage is measured.

### Tier 3: third RTX 3090

Use it for:

- a clean, resident Qwen3-Coder-Next Q5_K_M deployment, with Q6_K only after a successful model-specific fit test;
- all-resident 60-70 GiB quantized models with deliberately measured state/KV margin;
- larger KV/context budgets for smaller models;
- pipeline-parallel vLLM deployments when model support permits;
- independent model replicas or separate coder/reasoner/embedding services.

Do not buy or install it on the assumption that current FreeToken Qwen inference becomes 1.5x faster. It will not use that card for the current Qwen3.5/3.6 MoE path.

### NVLink and power

Two RTX 3090s can use a two-card NVLink bridge if board spacing and card design match. There is no three-way 3090 NVLink topology. NVLink can materially help two-way tensor parallelism, but it does not solve FreeToken host offload and does not connect the third card.

Three stock 3090s can request roughly 1,050 W before CPU, drives, pumps/fans, and transients, but that is not this design's target. The local power study establishes 200 W as the selected efficiency and electrical-budget cap: two cards measured 695 W stable / 705 W peak on the UPS; three split-PSU cards at 200 W plus 25% CPU project to 1,059-1,089 W against the selected 1,100 W room-server budget. The same estimate fails at 50% CPU. A PSU synchronizer starts two PSUs together; it does not by itself validate connector loading, grounding, branch-circuit capacity, transient response, or safe separation of each GPU's power leads. Keep the selected cap, do not seek a higher sustained cap, and repeat the documented final three-GPU GPU-plus-representative-CPU validation after installation.

## Benchmark plan for factual decisions

### Serving matrix

Run every candidate with:

- prompt lengths: 1k, 8k, and 32k tokens;
- output lengths: 256 and 2,048 tokens;
- concurrency: 1 and 4;
- cold start and warm cache;
- first turn, repeated-prefix turn, and context-edited tool turn;
- identical model weights/quantization when comparing engines.
- base decode, MTP/speculative decode where available, and each intended KV-cache type as separate configurations; never transfer a base-model result to an MTP or cache-quantized deployment.

### Systems metrics

| Metric | Why it matters |
|---|---|
| Model load/FTW conversion time | Operational cost of switching models |
| Peak and steady host RSS | Determines the real safe RAM tier |
| Per-GPU VRAM and placement | Detects unexpected CPU spill or unused GPUs |
| Measured `B_P` and `B_H` | Predicts FreeToken's `q*` behavior |
| Prefill tok/s | Controls TTFT for new/edited contexts |
| Decode tok/s | Interactive generation rate |
| TTFT p50/p95/p99 | Agent timeout and user experience |
| Inter-token latency p50/p95/p99 | Detects cache-miss stalls hidden by mean tok/s |
| Request throughput at concurrency 4 | Relevant to subagents and multiple pipeline components |
| Expert-cache hit/miss rate | Validates whether the workload has the locality assumed by the paper |
| Prefix/recurrent-state reuse rate | Quantifies semantic checkpoint value |
| PCIe RX/TX and GPU utilization | Distinguishes transfer, CPU, and GPU bottlenecks |
| CPU memory bandwidth and utilization | Determines whether more cores or faster RAM helps |
| Wall watts and joules/token | Required for an honest local-versus-API cost comparison |

### Quality metrics

Token rate is not capability. Evaluate every quantized model on:

- a fixed coding set such as LiveCodeBench or Aider Polyglot;
- a representative subset of SWE-bench Verified with one fixed agent scaffold;
- tool-call validity and recovery rate;
- long-context retrieval at the actual context lengths used;
- deterministic regression prompts for code review, patch generation, and data analysis;
- output disagreement between BF16 and the intended quantization.

The key custom metric should combine **task utility, latency, and cost**, rather than optimizing tok/s in isolation. A slower near-frontier model may be useful as a second-pass verifier while a resident 27B/35B model handles the latency-critical first pass.

## Practical architecture recommendation

### Coding

Use a two-stage local stack:

1. **Fast resident/default model:** Qwen3.8-27B Q5_K_M or Ornith-1.5-35B-A3B Q4_K_M, selected only after the same local coding, grounded-factuality, tool, and energy suite. Use llama.cpp first; compare Qwen3.8 FP8/NVFP4 and Qwen3.6-35B-A3B BF16 through FreeToken only after CUDA 13 and `memlock` are corrected.
2. **FreeToken verifier experiment at 141 GiB:** Qwen3.6-35B-A3B BF16 is the first large host-bank validation. gpt-oss-120b TP=2 follows only if the Qwen run has safe pinned-memory and latency results. Qwen3.8-Flash-Next is a later, tight experiment requiring a measured full host-bank total.
3. **Slow verifier after 256 GiB upgrade:** MiniMax-M2.5 or DeepSeek-V4-Flash through FreeToken for difficult repository tasks, reviews, and final patch verification.

This is more useful than forcing every request through the largest model.

### Engine selection

- Start with **FreeToken** for supported MoE and the paper reproduction.
- Keep **llama.cpp/Ollama** as the easy, mature GGUF baseline and for multi-GPU dense models.
- Use **vLLM** when the selected model can remain in aggregate VRAM and concurrency matters; prefer pipeline parallelism on this no-NVLink topology.
- Evaluate **KTransformers** only after measuring its AMD AVX2 path; do not transfer AMX-based Intel numbers to the 3970X.
- Keep **AirLLM** outside the serving path.

## Answers in one line

- **Is FreeToken new and working?** Yes, as an integrated edge-MoE serving system with a real beta implementation; most primitives have prior art, while `q*`, semantic state placement, and their graph-integrated composition are the main contributions.
- **Is the Ollama result fair?** Fair enough to support a directional claim under the tested Qwen setup, but incomplete for exact reproducibility and not automatically transferable to this Gen3 x8 host.
- **Does CPU plus GPU explain the speed?** Yes. Dynamic expert locality reduces misses, while `q*` uses PCIe and residual host-memory/CPU bandwidth concurrently instead of leaving either idle.
- **Is AirLLM comparable?** Only for "can it emit a token?" capacity. It is orders of magnitude slower and has no dedicated paper or multi-GPU pooling.
- **Do RAM compression and virtual memory still exist?** Yes, but transparent paging is generally too slow. Quantization and model-aware RAM/VRAM caches are the useful modern equivalents.
- **Can two 3090s run near-SOTA locally?** They can run strong 27B/35B models now and test Qwen3-Coder-Next Q3/Q4 through measured `llama.cpp --fit` CPU/GPU placement. Near-frontier 130-150 GiB MoE checkpoints still require a major RAM and PCIe upgrade and will be much slower.
- **Will a third 3090 transform FreeToken?** Not by itself. It is more useful for all-resident multi-GPU engines, KV headroom, or replicas than for FreeToken's current Qwen path.

## Primary sources

1. [FreeToken paper, arXiv:2608.16157v1](https://arxiv.org/abs/2608.16157)
2. [FreeToken repository](https://github.com/FlashML-org/FreeToken)
3. [FreeToken supported models and backends](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md)
4. [FreeToken installation requirements](https://github.com/FlashML-org/FreeToken/blob/main/docs/install.md)
5. [FreeToken CLI and bandwidth benchmark](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md)
6. [AirLLM repository](https://github.com/lyogavin/airllm)
7. [AirLLM dense loading hooks](https://github.com/lyogavin/airllm/blob/main/air_llm/airllm/airllm_base.py)
8. [vLLM parallelism and scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling.html)
9. [Ollama multi-GPU FAQ](https://github.com/ollama/ollama/blob/main/docs/faq.mdx)
10. [FlexGen](https://arxiv.org/abs/2303.06865)
11. [MoE-Infinity](https://arxiv.org/abs/2401.14361)
12. [Fiddler](https://arxiv.org/abs/2402.07033)
13. [KTransformers](https://github.com/kvcache-ai/ktransformers)
14. [EdgeMoE](https://arxiv.org/abs/2308.14352)
15. [Fast MoE inference with offloading](https://arxiv.org/abs/2312.17238)
16. [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)
17. [Qwen3.8-27B official GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF)
18. [Qwen3.8-Flash-Next FP8 model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
19. [Qwen3.8-Flash-Next NVFP4 metadata](https://huggingface.co/api/models/RadixArk/Qwen3.8-Flash-Next-NVFP4)
20. [Ornith-1.5-35B-A3B model card](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
21. [Ornith-1.5-35B-A3B official GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF)
22. [Qwen3.6-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
23. [Qwen3-Coder-Next model card](https://huggingface.co/Qwen/Qwen3-Coder-Next)
24. [Qwen3-Coder-Next technical report](https://arxiv.org/abs/2603.00729)
25. [llama.cpp `--fit` design discussion](https://github.com/ggml-org/llama.cpp/discussions/18049)
26. [llama.cpp fit-params README](https://github.com/ggml-org/llama.cpp/blob/master/tools/fit-params/README.md)
27. [Hardware Corner Coder-Next report, community measurement](https://www.hardware-corner.net/qwen3-coder-next-hardware-requirements/)
28. [vLLM MTP speculative decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/mtp/)
29. [DeepSeek-V4-Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
30. [MiniMax-M2.5 model card](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)
31. [gpt-oss model card/paper](https://arxiv.org/abs/2508.10925)
32. [GLM-5.2 model card](https://huggingface.co/zai-org/GLM-5.2)
33. [CUDA Unified Memory documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html)
34. [Linux zswap documentation](https://docs.kernel.org/admin-guide/mm/zswap.html)
35. [Hugging Face Accelerate big-model inference](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference)
36. [DeepSpeed ZeRO-Inference](https://www.deepspeed.ai/2022/09/09/zero-inference.html)
37. Local three-RTX-3090 power study (kept outside the public repository)
38. [DeepSeek-V4-Pro model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813)
39. [DeepSeek-V4-Flash Hub metadata](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Flash-0731)
40. [DeepSeek-V4-Pro Hub metadata](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Pro-0813)
