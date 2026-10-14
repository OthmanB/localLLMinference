# Qwen3.8 Flash-Next NVFP4 2,048-Token Profile Validation

Date: 2026-09-08

## Profile

The validated service uses the existing GPU0 PCIe Gen3 x16 link and DDR4-3066
memory configuration:

- 262,144-token QSA KV pool.
- 2,048 NVFP4 MoE LRU expert-cache slots.
- 2,048-token prefill chunks.
- Hybrid CPU/GPU MoE, serial expert loading, disk-backed PLE, and D2D reuse.
- One request at a time with no swap.

## Results

A fresh OpenCode window completed a multi-turn build-agent request and a second
`xhigh` smoke test. The first call was slow before the first response because it
paid the cold system and tool-context prefill cost.

The native-context marker probe used 261,949 prompt tokens and 127 output tokens:

| Metric | Result |
|---|---:|
| Wall time | 656.85 s |
| Steady prefill | ~401 tok/s |
| Settled scheduler decode | ~36.7 tok/s |
| Probe `/v1/stats` maximum decode sample | 35.1 tok/s |
| Peak GPU allocation | 23,450 MiB |
| Physical GPU headroom | ~677 MiB |
| Marker | exact, first |
| Swap / OOM | 0 / 0 |

The preceding 2,344-slot/512-token native-context profile measured 39.61 tok/s
settled scheduler decode. The current profile is approximately 7.4% slower on
decode, while its steady prefill is approximately 401 tok/s. PCIe x16 and
DDR4-3066 are unchanged between these profiles; the intentional changes are the
MoE cache size and prefill chunk length.

Host-level monitoring during validation reported approximately 85 GB RAM used
in `htop`. The service cgroup's file-backed accounting is not a host-RAM limit.
