# Strata on two GPUs: one shared engine or two replicas?

Date: 2026-10-15. Architecture assessment for **Strata v0.1.44**, not a new
benchmark. The ongoing Qwen-versus-Flash campaign remains single-GPU Flash;
no topology/configuration change or two-GPU test was made for this note.

## Short answer

**One Strata engine can span two GPUs.** It splits contiguous ranges of model
layers between the cards: **pipeline/layer parallelism, not tensor parallelism**.
This differs from running two independent copies of the whole model. It is the
most promising next candidate to measure here because it can increase distinct
expert-cache coverage without allocating a second full host expert arena. It is
not evidence of twice the speed, or even of faster single-stream decoding. [1]

| Layout | Host experts | GPU expert caches | Requests |
|---|---|---|---|
| One engine, one GPU | One host arena | One cache | Baseline solo or shared-model slots |
| One engine, two-GPU layer split | **One arena shared by both stages** | Each card caches experts for **its own layers** | One model; solo or shared-model slots |
| Two independent engines | A separate arena per engine in the resident-arena configuration | Each duplicates its own hot experts across the full model | Independent queues and model state |

Replicas may still share file-backed **OS page cache** when reading the same
assets. That does not make independently allocated/pinned resident arenas one
shared arena, nor eliminate simultaneous CPU memory-bandwidth traffic.

## What crosses PCIe, and what remains a bottleneck?

For the ordinary layer-split path, stages hand activations across **once per
verify window**, through pinned host RAM; the tagged documentation describes a
handoff of a few hundred KB. NVLink and GPU peer-to-peer access are not required.
This is not a tensor-parallel all-reduce at every layer. Each stage's expert
cache covers a disjoint layer range; uncached expert work still uses the shared
CPU pool and/or expert transfers on the individual card's link. [1]

The small activation handoff therefore does **not** remove RAM/PCIe constraints:

- Expert-cache misses, streaming and adaptive expert movement can consume host
  bandwidth and PCIe bandwidth. Two links share the same host memory subsystem.
- The split also needs dense weights, session/KV state, verify/prompt buffers and
  reserves on the cards. VRAM totals are not an expert-only pooled allocation.
- Chunked prefill can overlap stages, but a serial decode dependency can leave a
  stage waiting. More cache coverage and more stage overlap are different gains.
- Replicas can serve independent clients without cross-stage dependencies, but
  duplicate hot caches and contend for CPU/RAM traffic when their experts miss.

Historical observations for this host: three RTX 3090s, one NUMA node, no NVLink;
GPU0 negotiated **PCIe Gen3 x16 under load**, with about **13.1 GB/s** measured
linear host-to-device bandwidth in earlier work. GPU1/GPU2 have x8 links; idle
Gen1 readings can reflect downclocking and must not be treated as loaded-link
bandwidth measurements. These are historical host observations, not measurements
of a two-GPU Strata run. See the [earlier local report](qwen3.8-flash-next-nvfp4-x16-ddr4-3066-2026-09-07.md).

The current IQ3_S host expert arena is about **46.84 GiB**, while the one-GPU
expert cache is roughly **15 GiB**. Two 24 GiB cards still cannot hold the entire
expert arena **plus** dense weights, KV, buffers and reserves. A split can improve
cache coverage substantially, but full residency must not be assumed. Two
resident replicas would nominally allocate about **93.68 GiB of expert arenas
alone**, before other allocations; measure physical/PSS and pinned memory rather
than equating summed RSS or checkpoint size with unique RAM use.

## GPU count and conversation count are separate settings

Illustration only—**not a deployment change or a complete loader config**:

```json
{
  "gpu": [0, 1],
  "layer_split": "auto",
  "parallel": 1
}
```

This means one model split across two cards, serving one active conversation.
It does not mean one replica per card. The tagged installation guide describes
multi-card selection; auto placement and effective card order should be checked
in the engine log. These GPU numbers are illustrative, not a proposal to take a
card from the active campaign. [1][3]

`"parallel": 2` instead enables up to two conversation slots **inside the same
model/engine**, on one GPU or across a layer split. Slots consume extra VRAM and,
with streamed KV, pinned host RAM, reducing room for experts. [2]

For two-GPU **grouped MTP batching**, explicitly compare `--batch-mtp` with
`--batch-groups 1`. The batch-MTP path verifies one proposal per slot per window;
it is not a promise of the solo MTP4 path's throughput. Default pipelined groups
on a split leave batch MTP off; explicit groups above one also disable it.
Inspect the effective engine settings and fallback messages. Serial MTP groups
and pipelined non-MTP groups can trade decode throughput against admission and
prefill latency. Neither guarantees solo speed or a 2x aggregate gain. [2]

## Most useful next comparison

In a separately scheduled window, compare **one engine on two GPUs** against
**two independent one-GPU replicas**, retaining a **one-GPU baseline**. Keep model,
quantization, context/KV policy, prompts, sampler and approximation settings
matched, and label total hardware allocation explicitly.

Measure one and two active clients, plus **a fresh long prefill arriving during
another client's decode**. On the split, compare serial batch-MTP and pipelined
groups separately. Report per-stream and aggregate throughput, TTFT median/p95,
queueing and inter-token stalls, answer quality/caps, physical host/pinned memory,
per-card VRAM/headroom and expert-cache hits, CPU/RAM-bandwidth pressure, loaded
PCIe generation/width and transfer traffic. Repeat with fresh prompts and record
cache reuse. A throughput win that causes unacceptable p95 admission latency or
quality loss is not an unconditional serving upgrade.

## Tagged sources

Architecture and option semantics above are based on the locally available
tagged v0.1.44 documentation, rather than performance extrapolation from unrelated
models, GPUs or contributor rigs:

1. [MULTI_GPU.md — layer split, ownership, handoff and limits](https://github.com/Niko1221/Strata/blob/v0.1.44/docs/MULTI_GPU.md)
2. [BATCHING.md — slots, group scheduling, batch MTP and memory costs](https://github.com/Niko1221/Strata/blob/v0.1.44/docs/BATCHING.md)
3. [INSTALL.md — selecting several cards and configuration](https://github.com/Niko1221/Strata/blob/v0.1.44/docs/INSTALL.md)
