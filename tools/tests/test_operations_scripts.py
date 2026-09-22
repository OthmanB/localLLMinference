from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def test_operations_shell_scripts_parse() -> None:
    for script in sorted((ROOT / "operations").rglob("*.sh")):
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{script}: {result.stderr}"


def test_rtx5090_profile_is_explicit_and_loopback_only() -> None:
    profile = ROOT / "operations/hardware/rtx5090"
    installer = (profile / "install-native-q4.sh").read_text(encoding="utf-8")
    service = (profile / "systemd/ai-rtx5090-qwen3.8-q4-native.service").read_text(encoding="utf-8")
    monitor = (profile / "systemd/ai-rtx5090-monitor.service").read_text(encoding="utf-8")

    assert "AI_SERVER_RTX5090_CONFIRM=install" in installer
    assert "systemctl disable" not in installer
    assert "127.0.0.1" in service
    assert "GGML_CUDA_ENABLE_UNIFIED_MEMORY" not in service
    assert "127.0.0.1" in monitor


def test_rtx5090_baseline_removes_managed_memory_override() -> None:
    baseline = (ROOT / "tools/rtx5090_baseline.py").read_text(encoding="utf-8")

    assert 'environment.pop(MANAGED_MEMORY_VARIABLE, None)' in baseline
    assert '"--host",\n        "127.0.0.1"' in baseline
    assert '"--ctx-size",\n        str(args.ctx_size)' in baseline
    assert 'default=85.0' in baseline


def test_rtx5090_phase_d_runner_requires_guarded_rollback() -> None:
    runner = (ROOT / "tools/rtx5090_phase_d.py").read_text(encoding="utf-8")
    wrapper = (ROOT / "operations/run-rtx5090-phase-d-benchmark.sh").read_text(encoding="utf-8")

    assert 'PRODUCTION_SERVICE = "llama-qwen3.8-q4-native.service"' in runner
    assert '"--host",\n                    "127.0.0.1"' in runner
    assert '"--thermal-stop-c"' in runner
    assert 'default=85.0' in runner
    assert "effective_pattern_tokens" in runner
    assert 'payload["ignore_eos"] = True' in runner
    assert "sampler_started = False" in runner
    assert "use_trtllm_attention" in runner
    assert "vllm-attention-config" in runner
    assert "VLLM_USE_FLASHINFER_SAMPLER" in runner
    assert '"--spec-method", "mtp"' in runner
    assert "vllm-spec-tokens" in runner
    assert "vllm-enable-cuda-graph" in runner
    assert "vllm-cudagraph-metrics" in runner
    assert "vllm-compilation-config" in runner
    assert "vllm-gdn-prefill-backend" in runner
    assert "vllm-breakable-cudagraph" in runner
    assert "smoke_only" in runner
    assert "deterministic_output" in runner
    assert "cuda_graph_evidence" in runner
    assert "cuda_graph_runtime" in runner
    assert "full_capture_batch_sizes" in runner
    assert "full_capture_token_counts" in runner
    assert "full_capture_request_counts" in runner
    assert "full_runtime_batch_sizes" in runner
    assert "full_runtime_token_counts" in runner
    assert "full_runtime_request_counts" in runner
    assert "cuda_graph_decode_shape" in runner
    assert "workload_error_lines" in runner
    assert "Traceback (most recent call last)" in runner
    assert "periodic ten-second logger" in runner
    assert "CG Capture: mode=FULL" in runner
    assert "vllm_runtime_info" in runner
    assert "metrics_before_stop" in runner
    assert "KeyboardInterrupt: candidate interrupted" in runner
    assert "Preserve candidate artifacts if an unexpected harness defect occurs" in runner
    assert "stream_capacity" in runner
    assert 'finish_reason == "length"' in runner
    assert "decode_tokens_per_second" in runner
    assert "aggregate_decode_tokens_per_second" in runner
    assert "failed_cache_neutrality" in runner
    assert "accepted_per_drafted_token" in runner
    assert "vllm-max-num-batched-tokens" in runner
    assert "vllm-long-prefill-token-threshold" in runner
    assert "vllm-enable-chunked-prefill" in runner
    assert "time_until_all_users_enter_decode_seconds" in runner
    assert "p95_itl_while_another_user_prefills_seconds" in runner
    assert "stable_three_user_decode" in runner
    assert "interference-only" in runner
    assert "decode_p95_itl_during_new_prefill_seconds" in runner
    assert "cached-continuation-interference-only" in runner
    assert "cached-c3-queued-cold-only" in runner
    assert "cached_c3_queued_cold" in runner
    assert "client_side_cold_admission_after_c0" in runner
    assert "required_consecutive_idle_samples" in runner
    assert "gateway_queue_delay_seconds" in runner
    assert "backend_queue_delay_seconds" in runner
    assert "all_three_active_scheduler_samples" in runner
    assert "vllm_scheduler_samples" in runner
    assert "request_queue_time_seconds_sum" in runner
    assert "preemptions_delta" in runner
    assert "continuation_reused_local_prefix" in runner
    assert "active_request_was_cold" in runner
    assert "active_p95_itl_during_continuation_seconds" in runner
    assert "cached continuation cache/QoS contract failed" in runner
    assert "decode_only" in runner
    assert "if args.decode_only and (args.decode_concurrency" in runner
    assert "context_workloads" in runner
    assert "after-context-warm" in runner
    assert "decode-warm-prefixes" in runner
    assert "cache_correctness" in runner
    assert "--cache-only" in runner
    assert "cache-repro-only" in runner
    assert "vllm-enable-prefix-caching" in runner
    assert "--no-enable-prefix-caching" in runner
    assert "vllm:prefix_cache_hits_total" in runner
    assert "vllm:prompt_tokens_by_source_total" in runner
    assert "--gsm8k-executable" in runner
    assert "enable_thinking=true" in runner
    assert "SGLang exposes the checkpoint path" in runner
    assert "capacity_only" in runner
    assert "phase-d-retrieval-raven-73" in runner
    assert 'Verifying rollback before benchmark' in wrapper
    assert 'systemctl start "${SERVICE}"' in wrapper
    assert 'systemctl start "${MONITOR}"' in wrapper
    assert 'systemctl reset-failed "${SERVICE}"' in wrapper
    assert 'trap restore_service EXIT INT TERM' in wrapper
    assert "stop_candidate" in wrapper
    assert 'setsid runuser -u "${BENCH_USER}"' in wrapper
    assert "AI_SERVER_PHASE_D_MODEL" in wrapper
    assert "AI_SERVER_PHASE_D_RUN_PREFIX" in wrapper
    assert "AI_SERVER_PHASE_D_VLLM" in wrapper

    scheduler_sweep = (ROOT / "operations/run-rtx5090-phase-h-scheduler-sweep.sh").read_text(encoding="utf-8")
    interference = (ROOT / "operations/run-rtx5090-phase-h-decode-under-prefill.sh").read_text(encoding="utf-8")

    assert "2048 4096 8192 16384" in scheduler_sweep
    assert "--vllm-max-num-batched-tokens" in scheduler_sweep
    assert "--vllm-enable-chunked-prefill" in scheduler_sweep
    assert "--interference-only" in interference
    assert "--interference-decode-prompt-tokens 196000" in interference
    assert "--interference-prefill-prompt-tokens 196000" in interference
    assert "AI_SERVER_PHASE_H_MAX_NUM_BATCHED_TOKENS" in interference
    assert "AI_SERVER_PHASE_H_LONG_PREFILL_TOKEN_THRESHOLD" in interference

    cached_continuation = (ROOT / "operations/run-rtx5090-phase-i-cached-continuation.sh").read_text(encoding="utf-8")

    assert "--cached-continuation-interference-only" in cached_continuation
    assert "--vllm-enable-prefix-caching" in cached_continuation
    assert "--cached-continuation-append-tokens" in cached_continuation
    assert "--vllm-long-prefill-token-threshold" in cached_continuation

    phase_j = (ROOT / "operations/run-rtx5090-phase-j-cached-c3-queued-cold.sh").read_text(encoding="utf-8")

    assert "--cached-c3-queued-cold-only" in phase_j
    assert "--max-num-seqs 3" in phase_j
    assert "--vllm-enable-prefix-caching" in phase_j
    assert "--vllm-cudagraph-capture-sizes 1 3" in phase_j
    assert "--vllm-max-num-batched-tokens 4096" in phase_j
    assert "--vllm-long-prefill-token-threshold 256" in phase_j
    assert "--cached-c3-agent-append-tokens 1024 2048 4096" in phase_j
    assert "--cached-c3-agent-output-tokens 2048" in phase_j
    assert "--cached-c3-cold-output-tokens 256" in phase_j
    assert "--cached-c3-max-workload-seconds 900" in phase_j
    assert '"${REQUEST_TIMEOUT}"' in phase_j
    assert "AI_SERVER_PHASE_J_REQUEST_TIMEOUT" in phase_j
    assert "--vllm-max-num-queued-tokens" not in phase_j

    phase_k_native = (ROOT / "operations/run-rtx5090-phase-k-native-admission.sh").read_text(encoding="utf-8")
    phase_k_c2 = (ROOT / "operations/run-rtx5090-phase-k-c2-admission.sh").read_text(encoding="utf-8")
    phase_k_gateway = (ROOT / "operations/run-rtx5090-phase-k-gateway-c2-acceptance.sh").read_text(encoding="utf-8")

    assert "--cached-c3-native-admission-only" in phase_k_native
    assert "--cached-c3-c2-admission-only" in phase_k_c2
    assert "AI_SERVER_PHASE_K_WORKLOAD_MODE" in phase_k_gateway
    assert "--request-port" in phase_k_gateway
    assert "--metrics-port" in phase_k_gateway
    assert "systemctl" not in phase_k_gateway
    for phase_k in (phase_k_native, phase_k_c2):
        assert "--max-num-seqs 3" in phase_k
        assert "--vllm-enable-prefix-caching" in phase_k
        assert "--vllm-cudagraph-capture-sizes 1 3" in phase_k
        assert "--vllm-max-num-batched-tokens 4096" in phase_k
        assert "--vllm-long-prefill-token-threshold 256" in phase_k
        assert "--cached-c3-agent-append-tokens 1024 2048 4096" in phase_k
        assert "--cached-c3-cold-output-tokens 256" in phase_k
        assert "--cached-c3-max-workload-seconds 900" in phase_k

    recovery = (ROOT / "operations/run-rtx5090-phase-e-gsm8k-recovery-256.sh").read_text(encoding="utf-8")
    assert "--recovery-gsm8k-executable" in recovery
    assert "AI_SERVER_PHASE_E_GSM8K_EXAMPLES" in recovery
    assert "--recovery-gsm8k-examples \"${EXAMPLES}\"" in recovery
    assert "--recovery-gsm8k-continuation-max-tokens 4096" in recovery
    assert "--recovery-gsm8k-concurrency 3" in recovery
    assert "AI_SERVER_PHASE_E_GSM8K_RUN_TIMEOUT" in recovery
    assert "--request-port \"${GATEWAY_PORT}\"" in recovery
    assert "--metrics-port \"${BACKEND_PORT}\"" in recovery
    assert "--vllm-enable-cuda-graph" in recovery
    assert "--vllm-cudagraph-capture-sizes 1 3" in recovery
    assert "--vllm-gdn-prefill-backend triton" in recovery
    assert "AI_SERVER_PHASE_D_ALLOWED_GPU0_PID" in recovery


def test_rtx5090_phase_e_calibration_preserves_the_weight_contract() -> None:
    wrapper = (ROOT / "operations/run-rtx5090-phase-e-calibration.sh").read_text(encoding="utf-8")
    recipe = (
        ROOT / "operations/hardware/rtx5090/modelopt/qwen3.8-27b-w4a16-nvfp4-fp8-attn-kv-fp8-calibrated.yaml"
    ).read_text(encoding="utf-8")
    checksums = (ROOT / "operations/hardware/rtx5090/modelopt/qwen3.8-27b-bf16-e13a4f0e.sha256").read_text(
        encoding="utf-8"
    )

    assert "AI_SERVER_PHASE_E_CONFIRM=calibrate" in wrapper
    assert 'trap restore_service EXIT INT TERM' in wrapper
    assert 'setsid runuser -u "${BENCH_USER}"' in wrapper
    assert "--use_seq_device_map" in wrapper
    assert "--gpu_max_mem_percentage 0.80" in wrapper
    assert "--batch_size 1" in wrapper
    assert "--calib_size 1024" in wrapper
    assert "--dataset cnn_dailymail" in wrapper
    assert "THERMAL_STOP_C=85" in wrapper
    assert "sha256sum --check" in wrapper
    assert "kv_fp8:" in recipe
    assert "$import: kv_fp8" in recipe
    assert "$import: kv_fp8_cast" not in recipe
    assert "*mlp*gate_proj*weight_quantizer*" in recipe
    assert "*self_attn*weight_quantizer" in recipe
    assert "config.json" in checksums
    assert "model-00018-of-00018.safetensors" in checksums


def test_rtx5090_live_policy_compatibility_unit_uses_real_policy() -> None:
    compatibility = (ROOT / "operations/systemd/nvidia-qwen3.8-gpu0-policy.service").read_text(
        encoding="utf-8"
    )

    assert "Requires=nvidia-qwen3.8-gpu-policy.service" in compatibility
    assert "After=nvidia-qwen3.8-gpu-policy.service" in compatibility
    assert "ExecStart=/usr/bin/true" in compatibility


def test_qwen_vllm_tp2_cutover_preserves_the_qualified_contract() -> None:
    cutover = (ROOT / "operations/deploy-qwen3.8-vllm-tp2-cutover.sh").read_text(encoding="utf-8")
    backend = (ROOT / "operations/systemd/ai-qwen3.8-vllm-tp2.service").read_text(encoding="utf-8")
    gateway = (ROOT / "operations/systemd/ai-qwen3.8-vllm-tp2-gateway.service").read_text(encoding="utf-8")
    gateway_env = (ROOT / "operations/config/ai-qwen3.8-vllm-tp2-gateway.env").read_text(encoding="utf-8")

    assert 'MODEL_ALIAS=qwen3.8-27b-q4-gpukv-native' in cutover
    assert "--activate" in cutover
    assert "--rollback" in cutover
    assert "restore_legacy_from_state" in cutover
    assert "verify_public_c3_graph" in cutover
    assert "verify_public_long_admission" in cutover
    assert "--tensor-parallel-size 2" in backend
    assert "--max-num-seqs 3" in backend
    assert "--kv-cache-dtype fp8_e4m3" in backend
    assert "--gdn-prefill-backend triton" in backend
    assert "--cudagraph-capture-sizes 1 3" in backend
    assert "--served-model-name qwen3.8-27b-q4-gpukv-native" in backend
    assert "Requires=nvidia-qwen3.8-gpu0-policy.service" in backend
    assert "--host 0.0.0.0 --port 8080" in gateway
    assert '"cold_min_input_tokens":100000' in gateway_env
    assert '"dispatch_running_limit":2' in gateway_env


def test_profile_marker_precedes_dependent_service_restarts() -> None:
    script = (ROOT / "operations/qwen-profile-switch.sh").read_text(encoding="utf-8")
    marker = script.index('install -m 0644 -o root -g root "${marker_tmp}" "${PROFILE_MARKER}"')
    exporter_restart = script.index('systemctl restart "${EXPORTER}"')
    gateway_restart = script.index('systemctl restart "${GATEWAY}"')
    assert marker < exporter_restart < gateway_restart
    assert "restore_profile_enablement" in script
    assert "trap 'on_signal 130' INT" in script
    assert "trap 'on_signal 143' TERM" in script
