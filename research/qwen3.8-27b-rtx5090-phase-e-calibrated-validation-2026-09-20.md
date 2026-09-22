# Qwen3.8-27B RTX 5090 Phase E Calibrated Validation

Date: 2026-09-20

## Verdict

**Do not promote or route this checkpoint.** The calibrated FP8-KV export is
operationally viable with vLLM TP2, but its 128-example GSM8K smoke did not
meet the pre-existing quality gate. The stable llama.cpp Q4 service remains the
only live model.

The smoke result is not a statistical rejection of the checkpoint: 128 examples
are too few to establish a 96.5% floor. It is, however, an explicit failure of
the promotion criteria for this run. No authenticated gateway, systemd canary,
or metrics configuration was installed for it.

## Calibrated Artifact

- Export: `/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-e-kv-fp8-calibration-2026-09-20T052906Z/export`
- Source: Qwen BF16 revision `e13a4f0e35203116364e3b3f3f0c82f6ef1afd3c`.
- Recipe: `operations/hardware/rtx5090/modelopt/qwen3.8-27b-w4a16-nvfp4-fp8-attn-kv-fp8-calibrated.yaml`.
- Calibration: 1,024 CNN/DailyMail examples of 512 tokens, sequential 80%
  GPU mapping, batch size one.
- K/V scales: 32 positive non-default tensors, range `0.02260044775903225` to
  `0.3013392984867096`.
- Export `config.json` SHA-256:
  `731d9ea745aa5c171822e4e64b4a863ae3269620a43ca2bec8a1e42c13a89b68`.

The initial 95%-mapping export completed calibration but OOMed while saving.
The 80%-mapping rerun succeeded. Both runs restored the reference service.

## Runtime Compatibility

The initially selected automatic attention backend chose FlashInfer. On this
pinned vLLM 0.29.0 environment, FlashInfer JIT rejects RTX 5090 SM120 despite
the device being newer than its nominal lower bound. Forcing `TRITON_ATTN`
fixed attention selection, but vLLM then selected FlashInfer top-k/top-p
sampling during warm-up. Setting `VLLM_USE_FLASHINFER_SAMPLER=0` selected the
compatible sampler.

The passing temporary invocation retained TP2, eager mode, FP8 E4M3 KV,
loopback-only bind, `--max-model-len 262144`, `--max-num-seqs 4`, and:

```text
--attention-config {"backend":"TRITON_ATTN","use_trtllm_attention":false}
VLLM_USE_FLASHINFER_SAMPLER=0
```

The candidate server log confirms ModelOpt FP8 checkpoint detection, Triton
attention on both ranks, and the explicit non-FlashInfer sampler. It contains no
default-K/V-scale warning. The remaining q-scale-to-k-scale notice applies only
to FlashInfer or FlashAttention; the validated serving path is Triton.

## Operational Evidence

Run: `/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-e-calibrated-triton-nosampler-vllm-2026-09-20T055707Z/candidate/results.json`

- Normal EOS and forced `lookup_symbol` tool calls passed.
- A 250,016-token request completed in 165.44 seconds.
- Two simultaneous 196,017-token requests completed in 222.54 seconds.
- Prefill/decode interference passed.
- The 300-second sustained test completed all 18 requests and 9,216 output
  tokens.
- Across 719 telemetry samples, peak temperature was 76 C, peak reported power
  was 505.87 W, minimum free device memory was 2,928 MiB, and no thermal stop
  triggered.

Run: `/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-e-cache-restart-vllm-2026-09-20T061821Z/candidate/results.json`

- Two identical 8,208-token prefix requests returned the exact expected value.
- The cached repeat took 0.37 seconds after a 2.57-second first request.
- vLLM exited cleanly, restarted, reloaded the calibrated export, and passed the
  same two-request contract again.

Run: `/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-e-retrieval-vllm-2026-09-20T062127Z/candidate/results.json`

- A 249,041-token buried-fact retrieval request returned
  `phase-d-retrieval-raven-73` exactly in 132.00 seconds.

Every temporary run first tested reference rollback, stopped the reference only
inside the guard, and restored `llama-qwen3.8-q4-native.service` plus
`ai-qwen3.8-q4-native-monitor.service` on exit. Both are active afterward.

## Quality Smoke

The original qualification attestation names GSM8K with a 96.5% minimum accuracy,
100% stop rate, zero request errors, and zero truncations. Its original assets
were not stored locally, but the pinned public `sgl-eval` revision
`6690895609dcbc5df1e7b00dd57c9502b868ec4d` reconstructs the zero-shot boxed
answer prompt and symbolic grader.

An isolated evaluator was installed at `/tmp/phase-e-sgl-eval-venv`. The smoke
used that exact revision with `temperature=1.0`, `top_p=0.95`, seed zero,
thinking enabled, `enable_thinking=true`, four concurrent requests, and a 4,096
token output limit.

Run: `/home/michel/LLMs-tests/rtx5090-qwen38-bench/runs/phase-e-gsm8k-smoke-vllm-2026-09-20T063256Z/candidate/gsm8k/sgl_eval_gsm8k_20260920-153424/metrics.json`

| Metric | Result | Required |
| --- | ---: | ---: |
| Examples | 128 | 1,319 for a full gate |
| Accuracy | 95.3125% (122/128) | >= 96.5% |
| Stop rate | 97.6563% | 100% |
| Truncated rate | 2.3438% (3/128) | 0% |
| Request error rate | 0% | 0% |
| Wall time | 850.24 s | N/A |

This does not establish a quality regression at the full-dataset confidence
level, but it fails the explicit promotion threshold and exposes truncation that
the prior qualification did not have. It also used a reconstructed harness,
because the original command, seed policy, output cap, dataset bytes, and output
artifacts are unavailable locally.

## Required Before Reconsidering Canary

1. Schedule a 1,319-example GSM8K maintenance window. At the smoke rate it is
   approximately 2.4 hours and monopolizes both GPUs, so it cannot coexist with
   the reference service on this host.
2. Resolve the stop/truncation regression at the exact full-run prompt and
   output cap, then meet every recorded quality gate.
3. Run representative coding-agent and multi-turn tool trajectories against the
   calibrated candidate; the one forced-tool contract is not sufficient.
4. Define a latency SLO and add an additive loopback-only vLLM unit, one-replica
   authenticated gateway pool, and vLLM-aware exporter parser. Do not replace
   `qwen3.8-27b-q4-gpukv-native` during a canary.
