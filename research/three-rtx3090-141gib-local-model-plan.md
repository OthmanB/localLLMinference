# Local model plan: 2 then 3 RTX 3090 GPUs, 141 GiB RAM

Refinement of `freetoken-local-moe-inference-assessment.md`  
Snapshot: 2026-08-29

## Design decision

The immediate design is valid: install the third RTX 3090, its riser, and the second PSU with a synchronizer. The host now reports 141 GiB total and 137 GiB available RAM. Further RAM is deliberately out of scope for this phase. AirLLM is excluded because it solves capacity at non-interactive latency.

The 200 W GPU cap is also a hard design input, not a recommendation to relax. The local three-RTX-3090 power study measured 0.250 aggregate GEMM TFLOPS/W at 200 W versus 0.253 at 220 W. The split-PSU projection for three 200 W cards plus 25% CPU is 1,059-1,089 W against the selected 1,100 W room-server budget; with 50% CPU it is 1,157-1,167 W and over budget. The 200 W target therefore needs final validation with the real three-card LLM/CPU mix, but it must not be raised to chase a small synthetic efficiency difference. GEMM is not LLM inference, so final choice requires tokens/s and tokens/J measurements at this selected electrical cap.

The upgraded host capacity broadens the objective to:

1. Run Qwen3.8-27B Q5_K_M and Ornith-1.5-35B-A3B Q4_K_M on one GPU now as current compact coding candidates.
2. Test Qwen3-Coder-Next on the two installed GPUs through `llama.cpp --fit`, which can place selected tensors in host RAM while retaining dense tensors on GPU.
3. Use all three GPUs later for a clean, fully GPU-resident Qwen3-Coder-Next Q5_K_M deployment; test Q6_K only if its runtime and context margin are measured adequate.
4. Model context capacity from the deployed architecture and KV type, not a generic rule derived from conventional full-attention transformers.
5. Use the new host capacity for FreeToken Qwen3.6-35B-A3B BF16 validation and a cautious gpt-oss-120b TP=2 experiment after CUDA 13 and the hard locked-memory limit are corrected.

The third 3090 is therefore useful immediately for **capacity** and parallel services. It is not expected to make FreeToken Qwen3.6 inference 1.5x faster: current FreeToken explicitly rejects tensor parallelism above one GPU for Qwen3.6-27B and Qwen3.6-35B-A3B.

## Hard constraints

| Constraint | Planning assumption | Result |
|---|---|---|
| System RAM | 141 GiB total; 137 GiB available; 8 GiB unused swap | Qwen3.6-35B-A3B BF16 and a gpt-oss-120b TP=2 experiment now have physical host capacity. DeepSeek-V4-Flash, MiniMax-M2.5, GLM-5.2, Kimi K3, and Qwen3-235B remain outside the safe envelope. A hybrid fit must retain zero swap. |
| GPU VRAM now | 2 x 24 GiB = 48 GiB theoretical | A conventional fully resident model should be no more than about 42-44 GiB to leave runtime and KV-cache headroom. `llama.cpp --fit` is a conditional exception: it can calculate a CPU/GPU placement, but must be evaluated against the actual installed host RAM without disk paging and with RSS monitoring. |
| GPU VRAM after build | 3 x 24 GiB = 72 GiB theoretical | About 58-62 GiB remains the conservative ordinary-model weight budget. Qwen3-Coder-Next has unusually low measured hybrid-state growth, making Q5_K_M a better primary target and Q6_K a model-specific experiment, not an automatic fit. |
| Interconnect | No active NVLink; GPUs report max Gen3 / max x16 and current idle Gen1 / x8 | On this board, the bottom PCIE4 x8 link is expected electrically. Investigate the top PCIE1 x8 link under load. The final target is stable Gen3 x16 + x16 + x8, with reliable width preferred over an unreliable Gen4 attempt. |
| GPU power | Deliberate 200 W per-card cap | Retain this target for planning. It is within 1.2% of the best measured GEMM TFLOPS/W point. Three cards at that cap narrowly meet the 1,100 W room-server budget only at 25% CPU and exceed it at 50% CPU, so final LLM/CPU validation is required. Published 350 W 3090 results overstate performance, not the desired operating point. |
| Storage | SATA SSD | Model load time is acceptable for 16-59 GiB GGUF but model switching is slower than NVMe. Do not use disk paging during decode. |
| FreeToken | Qwen3.6 loaders: TP=1 only; gpt-oss: TP=2 only; hard `memlock` is 17.66 GiB | FreeToken is ideal for the one-GPU Qwen MoE path and can now validate its BF16 host-bank path, but the current lock limit blocks the 64.4 GB Qwen BF16 bank and larger candidates. |

### Required distinction: one model versus three services

- A 20 GiB Qwen model on **one 3090** does not become faster merely because two or three cards are installed.
- Two or three cards let the machine run **independent models concurrently**, for example an interactive Qwen coding server plus a separate embedding/reranking/evaluation server.
- The third card enables a clean resident Qwen3-Coder-Next Q5_K_M deployment and more parallel services. It is a quality, context-margin, and concurrency upgrade rather than the threshold at which Coder-Next first becomes usable.
- Aggregate VRAM still requires runtime, display, fragmentation, and state margin. However, hybrid models must use their measured architecture-specific state footprint rather than a blanket conventional-KV reserve.

## Feasible candidate set under the refined constraints

This is not a global model leaderboard. The original inclusion rule was too shallow: it prioritized physical fit, recency, and availability of a usable quantized release, then only lightly checked capability. That is why gpt-oss-120b appeared despite being an older, weaker factuality candidate. The revised set separates physical feasibility from model quality and marks gpt-oss as an optional capacity benchmark, not a recommended default.

### Mandatory selection gates

Apply these gates in order. Passing an earlier gate does not make a model recommended.

1. **Physical fit:** weights plus model-specific runtime, KV/recurrent state, and reserve must fit in one 24 GiB card, a three-card conservative budget of about 58-62 GiB, or a measured `llama.cpp --fit` CPU/GPU hybrid placement within actual installed RAM. No disk paging, AirLLM layer streaming, or optimistic use of all 72 GiB counts as a fit.
2. **Runtime correctness:** the exact checkpoint/quantization must have a maintained backend, supported tokenizer/template, working tool calls, and a known three-way placement where needed.
3. **Capability:** prefer disclosed coding-agent and reasoning evidence, then validate coding tasks using the same agent scaffold and quantization intended for deployment.
4. **Factuality:** require local grounded-claim and abstention results. A developer's generic hallucination number is only comparable when its task, tools, and scoring match.
5. **Operational margin:** validate 4k, 16k, and a model-specific long-context point; record memory stability, p95 latency, and task completion without OOM or degraded tool formatting. Test KV quantization separately from weight quantization.
6. **Power efficiency:** measure task utility, watts, and joules at the selected 200 W cap. A faster result requiring a higher sustained cap fails this design.

| Recommended use | Model and recommended weight | Weight size | Fits now, 2 GPUs | Fits after third GPU | Official coding/reasoning evidence | Main caveat |
|---:|---|---:|---|---|---|---|
| Primary now | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), [Q5_K_M GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | 19.8 GB (publisher decimal) | Yes, one GPU | Yes, one GPU | Terminal-Bench 2.0 73.0; SWE-Bench Pro 61.7; LiveCodeBench v6 90.3; GPQA-Diamond 89.2 | Hybrid architecture with full attention every fourth layer and four KV heads. The optional vision projector requires separate VRAM margin; benchmark the native MTP path separately for its actual memory and latency effect. |
| Primary MoE now | [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), [official Q4_K_M GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) | 21.7 GB (publisher decimal) | Yes, one GPU | Yes, one GPU | Terminal-Bench 2.0 67.8; SWE-Bench Verified 79.0; SWE-Bench Multilingual 79.6; GPQA 93.5 | Start with the official GGUF in llama.cpp. Its model card requires current vLLM/SGLang versions for native serving, and tool use still needs local validation. |
| FreeToken control | [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [Q4_K_M GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) or [NVIDIA NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) | 20.61 GiB GGUF; 23.45 GB NVIDIA checkpoint | Yes, one GPU | Yes, one GPU | SWE-bench Verified 73.4; Terminal-Bench 2.0 51.5 | Keep as FreeToken's paper-specific comparison path, not the default compact-quality recommendation. Qwen TP is currently limited to one GPU; Ampere has no native NVFP4 tensor-core path. |
| Comparison only | [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash), [AWQ 4-bit](https://huggingface.co/cyankiwi/GLM-4.7-Flash-AWQ-4bit) | 18.79 GiB | Yes, one GPU | Yes, one GPU | SWE-bench Verified 59.2 | Weight release is third-party AWQ; official local support targets development vLLM/SGLang paths. Treat as a compatibility experiment. |
| Primary large coding test now | [Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next), [official Q3/Q4 GGUF](https://huggingface.co/Qwen/Qwen3-Coder-Next-GGUF) | Q4_K_M 48.4 GB (publisher decimal) | Conditional with `llama.cpp --fit` | Yes | SWE-bench Verified 70.6 with SWE-Agent | Record `--fit-target` and `--fit-ctx` against actual installed RAM. It must run without disk paging and with measured RSS, PCIe behavior, context headroom, and power at 200 W. |
| Primary 3-GPU coding deployment | [Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next), [official Q5_K_M GGUF](https://huggingface.co/Qwen/Qwen3-Coder-Next-GGUF) | 56.7 GB (publisher decimal) | No practical resident fit | Yes, preferred initial placement | SWE-bench Verified 70.6 with SWE-Agent | Q6_K (65.5 GB) is a model-specific three-GPU fit experiment, not the starting deployment. No official factuality result exists for this exact checkpoint. |
| Optional capacity baseline, not default | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b), [MXFP4_MOE GGUF](https://huggingface.co/bartowski/openai_gpt-oss-120b-GGUF) | 59.03 GiB | No practical fit | Yes, three GPUs, tight | SWE-bench Verified 62.4 at high reasoning effort; AIME25 92.5 | Its official SimpleQA hallucination rate is 78.2% without browsing, lower is better. It is included only because it fits the 3-GPU class, not as a factuality recommendation. |

All reported model scores are developer-reported and use different harnesses, task versions, tool access, and reasoning budgets. They are directionally useful for shortlisting, not a direct leaderboard.

### FreeToken opportunities enabled by 141 GiB RAM

| Candidate | Placement before third GPU | New capability | Mandatory gate |
|---|---|---|---|
| Qwen3.8-27B FP8/NVFP4 | One GPU, dense fused backend | Current FreeToken model support now includes Qwen3.8 dense. This is a runtime comparison with llama.cpp, not a host-RAM use case. | Run the staged CUDA 13.2/system-service installer, then compare the exact deployed quantization. |
| Qwen3.6-35B-A3B BF16 | One GPU plus host routed-expert bank | The approximately 64.4 GB expert pool now fits physical RAM and is the closest paper-reproduction target. | Raise `memlock` above the measured pinned-bank requirement; current 17.66 GiB is insufficient. |
| gpt-oss-120b MXFP4 | Two GPUs, TP=2, host expert banks | FreeToken's gpt-oss loader shards attention and MXFP4 expert intermediates at TP=2. This becomes a capacity experiment before the riser arrives. | Resolve per-rank pinned-bank/RSS accounting and retain zero swap. TP=3 remains invalid. |
| Qwen3.8-Flash-Next NVFP4 | Unverified FreeToken placement | FreeToken supports this new 125B/6B-active architecture; its 47.7 GiB PLE table alone must be pinned in host RAM. | Do not launch with the current lock limit. Measure all source-bank use before accepting the tight 141 GiB physical fit. |

### Why DeepSeek-V4 Flash and Pro are excluded

They are excluded by physical feasibility, not quality.

| Model | Official already-quantized checkpoint | Official capability evidence | Why it fails the 141 GiB / 3x3090 gate |
|---|---:|---|---|
| [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | 304.180B tensors; 155.427 GiB Hub used storage. FP4 experts plus FP8/BF16 elsewhere. | Terminal Bench 2.1 82.7; HLE 37.8 without tools / 51.5 with tools | It exceeds 72 GiB aggregate VRAM by more than 2x and usable model-VRAM budget by about 2.5x, before runtime/KV. The official release is already mixed FP4/FP8, so merely choosing a "quantized version" does not make it fit. |
| [DeepSeek-V4-Pro-0813](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813) | 1.650T tensors; 1,659.419 GiB Hub used storage. FP4 experts plus FP8/BF16 elsewhere. | Terminal Bench 2.1 87.9; HLE 42.7 without tools / 60.0 with tools | It exceeds aggregate VRAM by about 23x before runtime/KV. It is not a local target for this architecture without a radically different, slow offload design. |

No final-checkpoint factuality or text-hallucination benchmark is published for either DeepSeek release in the inspected official sources. The relevant conclusion is not that they are weak: Pro and Flash have strong disclosed Terminal Bench and HLE results, but those harnesses cannot be directly ranked against the resident candidates' SWE-bench and AIME results. The conclusion is that 72 GiB VRAM and 141 GiB RAM still cannot hold Flash's 155.427 GiB official artifact with runtime/state margin, much less Pro. A non-official extreme low-bit conversion might technically exist later, but it would need independent quality and throughput validation before being treated as comparable to the official release.

### Quality and factuality scorecard

No single "hallucination rate" is comparable across all models. Benchmarks differ in text versus visual modality, answerable versus unanswerable questions, retrieval availability, whether abstention is rewarded, scoring direction, prompts, tools, and agent scaffold. `NR` means an exact official result was not published; it means unknown, not zero.

| Model | Coding evidence | Reasoning evidence | Factuality/hallucination evidence | Selection reading |
|---|---:|---:|---:|---|
| Qwen3.8-27B | Terminal-Bench 2.0 73.0; SWE-Bench Pro 61.7; LiveCodeBench v6 90.3 | GPQA-Diamond 89.2 | NR | Strongest immediate compact baseline on current disclosed coding and reasoning evidence; factuality must be measured locally. |
| Ornith-1.5-35B-A3B | Terminal-Bench 2.0 67.8; SWE-Bench Verified 79.0; SWE-Bench Multilingual 79.6 | GPQA 93.5; AIME24 90.7 | NR | Strong one-card MoE challenger. Compare it directly with Qwen3.8 in the same local scaffold before choosing a default. |
| Qwen3.6-35B-A3B | SWE-bench Verified 73.4 | AIME26 92.7 | HallusionBench 69.8, visual hallucination benchmark | FreeToken/runtime control, not the default compact-quality recommendation. The visual score is not a text hallucination rate. |
| GLM-4.7-Flash | SWE-bench Verified 59.2 | AIME25 91.6 | NR | Worth a compatibility comparison, but does not lead the Qwen candidates on disclosed coding evidence. |
| Qwen3-Coder-Next | SWE-bench Verified 70.6 with SWE-Agent | AIME25 83.07 | NR | Best large coding candidate. Test it on two GPUs with measured `--fit` placement now, then make Q5_K_M the first clean three-GPU deployment. |
| gpt-oss-120b | SWE-bench Verified 62.4, high reasoning | AIME25 92.5, high reasoning | SimpleQA hallucination rate 78.2%, lower is better, no browsing | Optional reasoning/capacity baseline only; not recommended as the default factual model. |
| DeepSeek-V4-Flash | Terminal Bench 2.1 82.7 | HLE 37.8 / 51.5 without/with tools | NR | Strong disclosed agentic coding result on a non-comparable benchmark; rejected solely by memory footprint. |
| DeepSeek-V4-Pro | Terminal Bench 2.1 87.9 | HLE 42.7 / 60.0 without/with tools | NR | Strongest reported Terminal Bench and HLE result in this table, but physically far beyond this system. |

Publisher scores are generally for native or BF16 evaluation configurations. Q4, NVFP4, AWQ, and third-party GGUF variants can change task accuracy, output format, tool reliability, long-context behavior, and memory headroom. The local candidate must be evaluated in its exact deployed quantization.

### Required local intelligence and hallucination gate

Before a model enters a coding path, measure its exact quantized deployment on a frozen local suite:

1. Coding: test pass rate, patch acceptance rate, tool-call validity, and regression rate using one fixed agent scaffold.
2. Grounded factuality: measure supported-claim precision, unsupported-claim rate, citation faithfulness, and abstention recall on time-frozen internal and public source packets.
3. Adversarial factuality: include false-premise, missing-context, contradictory-source, and stale-data cases. Reward a correct "insufficient evidence" answer.
4. Efficiency: measure TTFT, decode tok/s, p95 inter-token latency, watts, and joules per completed task at the intentional 200 W cap.

## Performance expectation

`M` = a measured published result. `C` = community measurement, not verified on this host. `E` = engineering estimate, not a benchmark. `U` = insufficient evidence to estimate reliably.

| Model and runtime | GPU use now | GPU use after third card | Decode tok/s at current 200 W / Gen3 x8 topology | Recommended role |
|---|---|---|---:|---|
| Qwen3.8-27B Q5, current `llama.cpp` | 1 GPU | Still 1 GPU | **U** | Default compact coding, analysis, tool calls, and latency-sensitive pipeline work |
| Ornith-1.5-35B-A3B Q4, current `llama.cpp` | 1 GPU | Still 1 GPU | **U** | One-card MoE coding comparison |
| Qwen3.6-35B-A3B Q4/NVFP4, FreeToken | 1 GPU | Still 1 GPU | **E 20-30** | FreeToken control; especially useful for repeated tool-turn contexts |
| GLM-4.7-Flash Q4, current `llama.cpp`/vLLM-compatible build | 1 GPU | Still 1 GPU | **U 12-28** | Compatibility and quality comparison, not the first deployment |
| gpt-oss-120b Q4, current `llama.cpp` three-way split | Not viable fully resident | 3 GPUs together | **E 5-12** | Optional capacity comparison only, not a deployment target |
| Qwen3-Coder-Next Q4, `llama.cpp --fit` | 2 GPUs, CPU/GPU hybrid | 3 GPUs together | **C 24.8-33.2** on a community dual-3090 Q4_K_XL report, across 131k-32k contexts | First large-model coding test; reproduce locally before using this range for planning |
| Qwen3-Coder-Next Q5, current `llama.cpp` three-way placement | Not practical resident | 3 GPUs together | **U** | Preferred clean three-GPU coding deployment |

The estimates assume one warmed request and no disk reads during decode. Context is architecture- and cache-type-specific: Qwen3.8 has full attention every fourth layer, while Qwen3-Coder-Next is a hybrid recurrent-attention model. Do not infer their fit from a conventional all-attention KV-cache rule. Measure `--fit` at 4k, 16k, 32k, and the intended long-context point, including `--cache-type-k` / `--cache-type-v` choices. Until reproduced locally, use the community Coder-Next throughput only to prioritize a test.

### What the two GPUs deliver now

The best development layout is not two-GPU inference for one Qwen model:

| GPU 0 | GPU 1 | Why |
|---|---|---|
| Qwen3.8-27B Q5 through `llama.cpp` | Ornith-1.5-35B-A3B Q4 through `llama.cpp` | Compare the current compact dense/hybrid baseline and MoE challenger concurrently. |
| Qwen3.8-27B as the main server | Embedding, reranking, test/evaluation model, or idle capacity | Gives the primary application predictable latency while leaving a GPU for the rest of the stack. |
| Qwen3.6-35B-A3B through FreeToken | Qwen3.8-27B only when needed | Isolates the FreeToken comparison while keeping the current compact baseline available. |
| Qwen3-Coder-Next Q3/Q4 through `llama.cpp --fit` | Same distributed model | Use both cards for the hybrid-placement test only. Stop if the process approaches host-memory pressure or invokes disk swapping. |

With 141 GiB RAM, start by serving one model. Then test two simultaneous compact servers at 8k or 16k context and monitor RSS, pinned memory, VRAM, and OOM behavior. For the Coder-Next hybrid or large FreeToken host-bank tests, reserve the machine for one model: record host RSS, `VmLck`, swap activity, GPU allocation, and PCIe counters at every context step. Do not co-host a second model during those tests.

### What the third GPU changes

After the riser and synchronized PSU are installed, there are two distinct operating modes:

| Mode | GPU allocation | Best initial model | Value |
|---|---|---|---|
| Three independent services | 1 + 1 + 1 | Qwen3.8-27B, Ornith-1.5-35B-A3B, plus embedding/reranking/eval | Highest throughput for development and agent pipelines; isolates failures. |
| One large model | 3 GPUs together | Qwen3-Coder-Next Q5 first, then Q6_K only if measured fit is adequate | Gives Coder-Next a clean resident weight budget and more context/runtime margin than the two-GPU hybrid experiment. |

The third GPU does not turn the Qwen3.6 FreeToken path into a three-GPU model. It creates a new, fully resident 45-59 GiB class for `llama.cpp` or another backend whose three-way support is verified.

## Runtime choice by model

| Model | First runtime to try | Do not assume | Reason |
|---|---|---|---|
| Qwen3.8-27B | Current `llama.cpp` Q5 GGUF, then FreeToken FP8/NVFP4 | That optional MTP or vision fits alongside the base model | Fully fits one 3090 as a text model. FreeToken now lists Qwen3.8 dense support, but its fused dense path is a runtime comparison rather than an offload gain. llama.cpp auto-detects native MTP from compatible GGUF metadata; benchmark it separately from base decode. |
| Ornith-1.5-35B-A3B | Current `llama.cpp` Q4 GGUF | That stale vLLM/SGLang versions support its native architecture | Fully fits one 3090. Use its official GGUF first, then test native server paths only on the versions requested by its model card. |
| Qwen3.6-35B-A3B | FreeToken | That TP=2/3 is supported | FreeToken is designed for its MoE cache, hybrid CPU/GPU miss path, prefix reuse, and agent APIs; source currently rejects TP > 1 for this family. |
| GLM-4.7-Flash | `llama.cpp` if GGUF exists for the chosen release, otherwise matching vLLM/SGLang build | That generic current vLLM will run the third-party AWQ immediately | Its local support and weight format need direct validation. |
| gpt-oss-120b | FreeToken TP=2 capacity test, then current `llama.cpp` and its Harmony template | That FreeToken can use three GPUs or that 141 GiB makes its host bank automatically safe | FreeToken gpt-oss source supports TP=2 but rejects TP=3 because its 64 query heads must divide TP size. Two 3090s do not have enough VRAM for the full 59 GiB GGUF, while FreeToken requires measured per-rank pinned-bank/RSS margin and a corrected hard `memlock` limit. |
| Qwen3.8-Flash-Next | FreeToken NVFP4 only after Qwen BF16 validation | That its 47.7 GiB PLE table is the complete host requirement, or that TP=2 is currently supported | FreeToken lists this architecture, its PLE table is pinned in host RAM, and the full routed-expert source-bank total must be measured before attempting the tight 141 GiB host fit. |
| Qwen3-Coder-Next | Current `llama.cpp --fit` | That Q4 is fully resident on two cards, or that Q6 automatically fits three cards | `--fit` adjusts unset placement/context values to a recorded `--fit-target` and `--fit-ctx`. Run Q3/Q4 on the existing two cards first; use Q5 as the primary three-GPU deployment. The runtime auto-detects native MTP from compatible GGUF metadata, but measure its power and task-level benefit. |

## Immediate priority order

1. Run `<FREETOKEN_ROOT>/install-system.sh` with `sudo`. FreeToken 0.1.2 and its CUDA-enabled Python environment are already installed, but the CUDA 13.2 `nvcc` toolkit and system service with unlimited `memlock` still require privileged installation. Do not run a large host-bank launch while the interactive limit remains 17.66 GiB.
2. Run `ft bench bw --dtype nvfp4,bf16` on both existing GPUs and retain its per-GPU profiles. Launch Qwen3.6-35B-A3B NVFP4 as the first FreeToken smoke test, recording `VmLck`, RSS, swap, VRAM, and power.
3. After the smoke test, run Qwen3.6-35B-A3B BF16 as the primary 141 GiB FreeToken validation. Do not start gpt-oss or Flash-Next until this run has stable zero-swap results.
4. On the existing cards, build current llama.cpp and run `llama-cli --fit` with explicit `--fit-target` and `--fit-ctx` for Qwen3-Coder-Next Q3 and Q4 at 4k, 16k, and 32k. Record the computed CPU/GPU split, host RSS, swap, VRAM, and sustained 200 W behavior. This establishes whether the community dual-3090 result is reproducible on this Gen3 x8 system.
5. Benchmark Qwen3.8-27B Q5 and Ornith-1.5-35B-A3B Q4 as text-only one-card candidates with the same coding, tool, factuality, context, and power suite.
6. Run gpt-oss-120b MXFP4 at TP=2 as a 4k-context FreeToken capacity experiment only after its per-rank pinned-bank/RSS plan is established. Treat Qwen3.8-Flash-Next as a later, tight experiment requiring complete memory accounting.
7. Install the third 3090, riser, and PSU synchronizer. Before trusting any throughput comparison, verify all three cards under load. The bottom PCIE4 card at Gen3 x8 is expected; verify the top PCIE1 card reaches stable Gen3 x16 and that the new card does not negotiate below Gen3 x8. The target is stable Gen3 x16 + x16 + x8.
   ```bash
   nvidia-smi --query-gpu=index,name,pci.bus_id,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current --format=csv
   nvidia-smi topo -m
   lspci -vv -s <PCI-BDF>
   ```
   Inspect `LnkCap` and `LnkSta` under load. If capability fields are access-denied, rerun only that command with `sudo lspci -vv -s <PCI-BDF>`. Treat Gen4 as an optional later experiment, not the acceptance criterion.
8. Validate the selected 200 W power cap with the three-card installation and a representative CPU load. Begin the electrical test at 150 W as the power study prescribes, measure the final 200 W target, and do not raise the sustained cap to chase synthetic throughput. Do not compare directly with unrestricted 350 W 3090 benchmarks.
9. On three cards, benchmark Qwen3-Coder-Next Q5 before an optional Q6_K fit experiment or gpt-oss-120b GGUF. Q5 is the intended resident deployment with useful margin.
10. Treat gpt-oss-120b as an optional historical capacity baseline at 4k context, not a high-priority high-reasoning deployment. It needs Harmony formatting and its published no-browsing factuality result is poor.

## Minimal factual benchmark matrix

Use identical prompts, context limits, sampling parameters, and quantization for comparisons.

| Test | Models | Success criterion |
|---|---|---|
| Single-GPU baseline | Qwen3.8-27B NVFP4 FreeToken; Q5 llama.cpp and Ornith Q4 remain comparisons | **Qwen FreeToken initial pass:** stable 8k service, 21.8 GiB VRAM, 3.1 GiB PSS, and 18-20 tok/s warm decode. Complete the 30-minute, tool-validity, and factuality gates before promotion. |
| Two-service test | Qwen3.8 on GPU 0 plus Ornith on GPU 1 | Both remain responsive; record host RSS, pinned RAM, VRAM, p95 TTFT, and watts |
| FreeToken smoke test | Qwen3.6-35B-A3B NVFP4 | Stable 8k context; record `ft bench bw` profile, `VmLck`, RSS, swap, cache hit/miss, TTFT, decode tok/s, and watts. |
| FreeToken BF16 validation | Qwen3.6-35B-A3B BF16 | Stable 4k, then 8k and 16k context with zero swap; pinned memory remains below the configured hard limit and the result is compared with the identical task harness. |
| FreeToken TP=2 capacity test | gpt-oss-120b MXFP4 | **Passed initial 4k launch:** TP=2 consumed 40.5 GiB aggregate VRAM and 61.6 GiB PSS. One warm request sustained 4.6-6.6 decode tok/s after the MXFP4 hybrid calibration. Do not promote it without long-context, concurrency, and factuality gates. |
| Two-GPU Coder-Next fit test | Qwen3-Coder-Next Q3 then Q4 with `llama.cpp --fit` | At 4k, 16k, and 32k, retain zero swap/disk paging and stable latency; record computed placement, RSS, GPU allocation, PCIe counters, tok/s, and watts. |
| Three-GPU topology test | `llama.cpp` benchmark with an intentionally distributed model | Confirm all GPUs receive meaningful VRAM allocation and utilization; record PCIe transfer/error counters |
| Three-GPU coding model | Qwen3-Coder-Next Q5, Q6_K only if it passes fit | Stable 4k, then 16k, 32k, and long-context task point; compare code-task score to Qwen3.8/Ornith and retain runtime/state margin |
| Optional three-GPU capacity baseline | gpt-oss-120b Q4 | Stable 4k context and Harmony/tool correctness; do not promote it without beating Qwen on the local factuality gate |
| Engine comparison | Qwen3.6-35B-A3B: FreeToken versus `llama.cpp`/Ollama | Same weights if possible; report decode tok/s, TTFT p50/p95, inter-token p95, and cache-hit behavior |
| KV/MTP comparison | Qwen3.8 and Qwen3-Coder-Next in llama.cpp; Qwen3.8 in vLLM where applicable | Test base versus MTP and each `--cache-type-k` / `--cache-type-v` candidate separately; accept only a measured task/latency/memory improvement. |

Choose the default model with a combined score of task utility, deadline-miss rate, and joules per completed decision. Use the large three-GPU model only as a second-pass verifier if it improves that score in time-ordered offline evaluation.

## Refined bottom line

- **Current two GPUs:** Start with Qwen3.8-27B Q5 and Ornith-1.5-35B-A3B Q4 as one-card quality candidates. The 141 GiB host now makes Qwen3.6-35B-A3B BF16 the first serious FreeToken validation, after CUDA 13 and hard `memlock` are corrected. Run Qwen3-Coder-Next Q3/Q4 through `llama.cpp --fit` as a measured CPU/GPU hybrid experiment, not as a claimed fully resident fit.
- **After the third GPU:** Qwen3-Coder-Next Q5 is the meaningful resident coding target. Q6_K is an aggressive model-specific fit test, while gpt-oss-120b Q4 remains an optional, tight capacity baseline. Do not assume FreeToken TP=3.
- **Three-GPU service layout:** GPU 0 can hold Qwen3.8-27B Q5 while its native 262k Q8 context lives in RAM, preserving GPU 1 + GPU 2 for Coder-Next. Coder-Next Q4 is the two-card resident target; its 48.4 GB file leaves little runtime margin. Q5 is 56.7 GB and therefore requires measured host offload on two cards rather than being described as fully GPU-resident.
- **FreeToken remains valuable:** use it for the single-GPU Qwen MoE route and agent workload behavior, not as the mechanism that pools all three cards today.
- **No further RAM upgrade needed for the current plan:** 141 GiB enables the Qwen BF16 and cautious gpt-oss FreeToken paths. The tradeoff is that DeepSeek-V4-Flash and MiniMax-M2.5 remain outside the safe host-memory tier; Qwen3.8-Flash-Next is conditional on full pinned-bank accounting.
- **No AirLLM:** its capacity advantage does not compensate for the documented minutes-per-token class of performance.

## Sources

1. [FreeToken paper](https://arxiv.org/abs/2608.16157)
2. [FreeToken supported models/backends](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md)
3. [FreeToken CLI](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md)
4. [FreeToken gpt-oss TP loader](https://github.com/FlashML-org/FreeToken/blob/main/python/freetoken/models/gpt_oss/weight.py)
5. [FreeToken Qwen loader](https://github.com/FlashML-org/FreeToken/blob/main/python/freetoken/models/qwen3_5_moe/weight.py)
6. [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
7. [Qwen3.8-27B official GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF)
8. [Qwen3.8-Flash-Next FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
9. [Qwen3.8-Flash-Next NVFP4 metadata](https://huggingface.co/api/models/RadixArk/Qwen3.8-Flash-Next-NVFP4)
10. [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
11. [Ornith-1.5-35B-A3B official GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF)
12. [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
13. [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)
14. [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash)
15. [Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next)
16. [Qwen3-Coder-Next technical report](https://arxiv.org/abs/2603.00729)
17. [llama.cpp `--fit` discussion](https://github.com/ggml-org/llama.cpp/discussions/18049)
18. [llama.cpp fit-params README](https://github.com/ggml-org/llama.cpp/blob/master/tools/fit-params/README.md)
19. [vLLM MTP speculative decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/mtp/)
20. [Hardware Corner Coder-Next report, community measurement](https://www.hardware-corner.net/qwen3-coder-next-hardware-requirements/)
21. Local three-RTX-3090 power study (kept outside the public repository)
22. [DeepSeek-V4-Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
23. [DeepSeek-V4-Pro model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813)
24. [DeepSeek-V4-Flash Hub metadata](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Flash-0731)
25. [DeepSeek-V4-Pro Hub metadata](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Pro-0813)
