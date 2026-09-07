# Ornith 1.5 35B-A3B NVFP4 Candidate (Superseded)

Model card: <https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4>

## Reserved Placement

Muse Glimmer 30B supersedes Ornith as the next GPU2 candidate. No Ornith
artifacts or service should be installed unless it is deliberately reconsidered.

## Required Validation

1. Inspect model artifacts and supported runtime architecture before download.
2. Measure model load, GPU2 memory allocation, host RAM use, and swap behavior.
3. Establish a conservative single-request cache geometry and run a smoke probe.
4. Calibrate CPU/GPU transfer behavior if the runtime uses MoE offload.
5. Run a native-context marker retrieval and throughput test before considering a
   permanent GPU2 service.
