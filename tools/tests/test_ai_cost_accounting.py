from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_cost_accounting import (
    CostAccounting,
    api_cost_usd,
    counter_delta,
    host_power_watts,
    rate_from_components,
    trapezoid_energy_joules,
)
from ai_metrics_exporter import (
    bucket_points,
    cached_input_mode,
    gateway_metrics_lines,
    MetricsState,
    nvidia_metrics,
    parse_gateway_metrics,
    parse_freetoken_stats,
    parse_llama_metrics,
    parse_vllm_metrics,
    retain_nonzero_rate,
    validate_backend_gpu_ownership,
)
from ai_cpu_power_profiler import (
    CpuPowerProfiler,
    bucket_index,
    bucket_label,
    cpu_utilization_percent,
    interpolate_power,
    rapl_power_watts,
)


class CostAccountingTest(unittest.TestCase):
    def test_llama_parser_records_prometheus_string_values(self) -> None:
        _, observation = parse_llama_metrics(
            "\n".join(
                [
                    "llamacpp:prompt_tokens_total 131",
                    "llamacpp:prompt_tokens_cached_total 98",
                    "llamacpp:tokens_predicted_total 2739",
                    "llamacpp:requests_processing 1",
                ]
            ),
            {},
            set(),
        )
        self.assertEqual(
            observation,
            {
                "prompt_tokens": 131.0,
                "cached_prompt_tokens": 98.0,
                "completion_tokens": 2739.0,
                "requests_active": 1.0,
            },
        )

    def test_timestamped_trapezoid_energy(self) -> None:
        self.assertEqual(trapezoid_energy_joules(100, 200, 4), 600)
        self.assertEqual(trapezoid_energy_joules(100, 200, 0), 0)

    def test_counter_reset_uses_current_value(self) -> None:
        self.assertEqual(counter_delta(None, 10), 0)
        self.assertEqual(counter_delta(10, 16), 6)
        self.assertEqual(counter_delta(16, 3), 3)

    def test_idle_modes_do_not_double_count_gpu_idle_draw(self) -> None:
        powers = {"gpu-a": 20, "gpu-b": 50}
        self.assertEqual(
            host_power_watts(
                {"mode": "baseline_non_gpu_w", "baseline_non_gpu_w": 240}, powers
            ),
            310,
        )
        self.assertEqual(
            host_power_watts(
                {
                    "mode": "wall_idle_total_w",
                    "wall_idle_total_w": 300,
                    "default_gpu_idle_reference_w": 20,
                },
                powers,
            ),
            330,
        )

    def test_api_pricing_treats_processed_and_cached_tokens_as_disjoint(self) -> None:
        workload, output_only = api_cost_usd(
            1_000_000,
            1_000_000,
            1_000_000,
            {
                "input_usd_per_million_tokens": 1.0,
                "cached_input_usd_per_million_tokens": 0.1,
                "output_usd_per_million_tokens": 2.0,
            },
        )
        self.assertAlmostEqual(workload, 3.1)
        self.assertAlmostEqual(output_only, 2.0)

    def test_tariff_components_are_versioned_inputs(self) -> None:
        self.assertAlmostEqual(
            rate_from_components(
                {
                    "energy": 40.49,
                    "renewable": 4.18,
                    "fuel_adjustment": -10.96,
                }
            ),
            33.71,
        )

    def test_active_energy_host_energy_and_api_cost_are_separate(self) -> None:
        timestamp = datetime(2026, 9, 7, tzinfo=UTC).timestamp()
        config = {
            "host_id": "test-host",
            "max_sample_gap_seconds": 20,
            "baseline": {
                "mode": "wall_idle_total_w",
                "wall_idle_total_w": 300,
                "default_gpu_idle_reference_w": 20,
            },
            "tariffs": [
                {
                    "id": "tariff",
                    "effective_from": "2026-09-01",
                    "components_jpy_per_kwh": {"rate": 36.0},
                }
            ],
        }
        pricing = {
            "fx_rates": [{"id": "fx", "effective_from": "2026-09-01", "jpy_per_usd": 100}],
            "prices": [
                {
                    "id": "price",
                    "provider": "example",
                    "model": "remote",
                    "effective_from": "2026-09-01",
                    "input_usd_per_million_tokens": 1.0,
                    "cached_input_usd_per_million_tokens": 0.1,
                    "output_usd_per_million_tokens": 2.0,
                }
            ],
            "comparisons": [
                {"id": "comparison", "local_model": "local", "price_id": "price", "comparison": "reference"}
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            accounting = CostAccounting(config, pricing, Path(temporary) / "state.json")
            idle_gpu = {"gpu-a": {"power_watts": 20, "model": "local", "active": False}}
            active_gpu = {"gpu-a": {"power_watts": 120, "model": "local", "active": True}}
            no_tokens = {"gpu-a": {"model": "local", "prompt_tokens": 0, "cached_prompt_tokens": 0, "completion_tokens": 0, "cached_input_mode": "observed"}}
            tokens = {"gpu-a": {"model": "local", "prompt_tokens": 100, "cached_prompt_tokens": 10, "completion_tokens": 50, "cached_input_mode": "observed"}}
            later_tokens = {"gpu-a": {"model": "local", "prompt_tokens": 120, "cached_prompt_tokens": 12, "completion_tokens": 60, "cached_input_mode": "observed"}}
            accounting.observe(timestamp, idle_gpu, no_tokens)
            accounting.observe(timestamp + 10, active_gpu, tokens)
            accounting.observe(timestamp + 20, active_gpu, later_tokens)

            self.assertAlmostEqual(accounting.state["gpu_energy_joules"]["gpu-a"], 1_900)
            self.assertAlmostEqual(accounting.state["active_model_energy_joules"]["local\x1fgpu-a"], 1_200)
            self.assertAlmostEqual(accounting.state["host_energy_joules"], 7_500)
            self.assertAlmostEqual(accounting.state["energy_covered_completion_tokens"]["local\x1fgpu-a"], 10)
            self.assertAlmostEqual(accounting.state["api_workload_cost_usd"]["local\x1fgpu-a\x1fprice\x1fcomparison\x1fobserved"], 0.0002412)
            self.assertAlmostEqual(accounting.state["api_output_cost_usd"]["local\x1fgpu-a\x1fprice\x1fcomparison\x1fobserved"], 0.00012)

    def test_missing_power_samples_are_reported_not_integrated(self) -> None:
        timestamp = datetime(2026, 9, 7, tzinfo=UTC).timestamp()
        config = {
            "host_id": "test-host",
            "max_sample_gap_seconds": 5,
            "baseline": {"mode": "baseline_non_gpu_w", "baseline_non_gpu_w": 100},
            "tariffs": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            accounting = CostAccounting(config, {}, Path(temporary) / "state.json")
            gpu = {"gpu-a": {"power_watts": 100, "model": None, "active": False}}
            accounting.observe(timestamp, gpu, {})
            accounting.observe(timestamp + 10, {}, {})
            accounting.observe(timestamp + 20, {}, {})
            self.assertEqual(accounting.state["host_energy_joules"], 0)
            self.assertEqual(accounting.state["integration_gap_seconds_total"], 20)

    def test_missing_gpu_excludes_only_host_estimate(self) -> None:
        timestamp = datetime(2026, 9, 7, tzinfo=UTC).timestamp()
        config = {
            "host_id": "test-host",
            "max_sample_gap_seconds": 20,
            "baseline": {"mode": "baseline_non_gpu_w", "baseline_non_gpu_w": 100},
            "tariffs": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            accounting = CostAccounting(config, {}, Path(temporary) / "state.json")
            two_gpus = {
                "gpu-a": {"power_watts": 100, "model": None, "active": False},
                "gpu-b": {"power_watts": 200, "model": None, "active": False},
            }
            accounting.observe(timestamp, two_gpus, {})
            accounting.observe(timestamp + 10, {"gpu-a": two_gpus["gpu-a"]}, {})
            self.assertEqual(accounting.state["gpu_energy_joules"]["gpu-a"], 1_000)
            self.assertEqual(accounting.state["host_energy_joules"], 0)
            self.assertEqual(accounting.state["integration_gap_seconds_total"], 10)

    def test_state_persists_across_exporter_restart(self) -> None:
        timestamp = datetime(2026, 9, 7, tzinfo=UTC).timestamp()
        config = {
            "host_id": "test-host",
            "max_sample_gap_seconds": 20,
            "baseline": {"mode": "baseline_non_gpu_w", "baseline_non_gpu_w": 100},
            "tariffs": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "state.json"
            gpu = {"gpu-a": {"power_watts": 100, "model": None, "active": False}}
            accounting = CostAccounting(config, {}, state_path)
            accounting.observe(timestamp, gpu, {})
            accounting.observe(timestamp + 10, gpu, {})
            restarted = CostAccounting(config, {}, state_path)
            self.assertEqual(restarted.state["gpu_energy_joules"]["gpu-a"], 1_000)
            self.assertEqual(restarted.state["host_energy_joules"], 2_000)

    def test_confirmed_reboot_is_not_a_sampling_gap(self) -> None:
        timestamp = datetime(2026, 9, 7, tzinfo=UTC).timestamp()
        config = {
            "host_id": "test-host",
            "max_sample_gap_seconds": 20,
            "baseline": {"mode": "baseline_non_gpu_w", "baseline_non_gpu_w": 100},
            "tariffs": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "state.json"
            gpu = {"gpu-a": {"power_watts": 100, "model": None, "active": False}}
            before_reboot = CostAccounting(config, {}, state_path, boot_id="boot-a", boot_time_seconds=1)
            before_reboot.observe(timestamp, gpu, {})
            after_reboot = CostAccounting(config, {}, state_path, boot_id="boot-b", boot_time_seconds=2)
            after_reboot.observe(timestamp + 3_600, gpu, {})
            self.assertEqual(after_reboot.state["host_energy_joules"], 0)
            self.assertEqual(after_reboot.state["integration_gap_seconds_total"], 0)
            self.assertEqual(after_reboot.state["reboot_downtime_seconds_total"], 3_600)
            metrics = {name: value for name, _labels, value in after_reboot.metrics()}
            self.assertEqual(metrics["ai_host_reboot_downtime_seconds_total"], 3_600)
            self.assertEqual(metrics["ai_host_boot_time_seconds"], 2)

    def test_configured_model_counters_start_at_zero(self) -> None:
        timestamp = datetime(2026, 9, 7, tzinfo=UTC).timestamp()
        config = {
            "host_id": "test-host",
            "baseline": {"mode": "baseline_non_gpu_w", "baseline_non_gpu_w": 100},
            "tariffs": [
                {
                    "id": "tariff",
                    "effective_from": "2026-09-01",
                    "components_jpy_per_kwh": {"rate": 36},
                }
            ],
        }
        pricing = {
            "fx_rates": [{"id": "fx", "effective_from": "2026-09-01", "jpy_per_usd": 100}],
            "prices": [
                {
                    "id": "price",
                    "provider": "example",
                    "model": "remote",
                    "effective_from": "2026-09-01",
                    "input_usd_per_million_tokens": 1,
                    "output_usd_per_million_tokens": 2,
                }
            ],
            "comparisons": [{"id": "comparison", "local_model": "local", "price_id": "price"}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            accounting = CostAccounting(config, pricing, Path(temporary) / "state.json")
            accounting.observe(
                timestamp,
                {"gpu-a": {"power_watts": 100, "model": "local", "active": False}},
                {
                    "gpu-a": {
                        "model": "local",
                        "prompt_tokens": 0,
                        "cached_prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cached_input_mode": "observed",
                    }
                },
            )
            metrics = {name: value for name, _labels, value in accounting.metrics()}
            self.assertEqual(metrics["ai_model_active_gpu_energy_joules_total"], 0)
            self.assertEqual(metrics["ai_model_energy_covered_completion_tokens_total"], 0)
            self.assertEqual(metrics["ai_model_api_workload_cost_jpy_total"], 0)

    def test_terminal_completion_tokens_match_prior_active_energy(self) -> None:
        timestamp = datetime(2026, 9, 7, tzinfo=UTC).timestamp()
        config = {
            "host_id": "test-host",
            "max_sample_gap_seconds": 20,
            "baseline": {"mode": "baseline_non_gpu_w", "baseline_non_gpu_w": 100},
            "tariffs": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            accounting = CostAccounting(config, {}, Path(temporary) / "state.json")
            inactive = {"gpu-a": {"power_watts": 20, "model": "local", "active": False}}
            active = {"gpu-a": {"power_watts": 120, "model": "local", "active": True}}
            no_tokens = {"gpu-a": {"model": "local", "completion_tokens": 0}}
            completed = {"gpu-a": {"model": "local", "completion_tokens": 10}}
            accounting.observe(timestamp, inactive, no_tokens)
            accounting.observe(timestamp + 10, active, no_tokens)
            accounting.observe(timestamp + 20, active, no_tokens)
            accounting.observe(timestamp + 30, inactive, completed)
            self.assertEqual(
                accounting.state["energy_covered_completion_tokens"]["local\x1fgpu-a"], 10
            )

    def test_dual_gpu_model_energy_does_not_duplicate_token_costs(self) -> None:
        timestamp = datetime(2026, 9, 7, tzinfo=UTC).timestamp()
        config = {
            "host_id": "test-host",
            "max_sample_gap_seconds": 20,
            "baseline": {"mode": "baseline_non_gpu_w", "baseline_non_gpu_w": 100},
            "tariffs": [],
        }
        pricing = {
            "prices": [
                {
                    "id": "price",
                    "provider": "example",
                    "model": "remote",
                    "effective_from": "2026-09-01",
                    "input_usd_per_million_tokens": 1,
                    "output_usd_per_million_tokens": 2,
                }
            ],
            "comparisons": [{"id": "comparison", "local_model": "qwen", "price_id": "price"}],
            "fx_rates": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            accounting = CostAccounting(config, pricing, Path(temporary) / "state.json")
            inactive = {
                "gpu-1": {"power_watts": 20, "model": "qwen", "active": False},
                "gpu-2": {"power_watts": 20, "model": "qwen", "active": False},
            }
            active = {
                "gpu-1": {"power_watts": 120, "model": "qwen", "active": True},
                "gpu-2": {"power_watts": 120, "model": "qwen", "active": True},
            }
            initial = {
                "gpu-1": {"model": "qwen", "completion_tokens": 0, "account_tokens": True},
                "gpu-2": {"model": "qwen", "completion_tokens": 0, "account_tokens": False},
            }
            completed = {
                "gpu-1": {
                    "model": "qwen",
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "account_tokens": True,
                },
                "gpu-2": {
                    "model": "qwen",
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "account_tokens": False,
                },
            }
            later = {
                "gpu-1": completed["gpu-1"] | {"prompt_tokens": 120, "completion_tokens": 60},
                "gpu-2": completed["gpu-2"] | {"prompt_tokens": 120, "completion_tokens": 60},
            }
            accounting.observe(timestamp, inactive, initial)
            accounting.observe(timestamp + 10, active, completed)
            accounting.observe(timestamp + 20, active, later)

            self.assertEqual(accounting.state["active_model_energy_joules"]["qwen\x1fgpu-1"], 1200)
            self.assertEqual(accounting.state["active_model_energy_joules"]["qwen\x1fgpu-2"], 1200)
            api_keys = set(accounting.state["api_workload_cost_usd"])
            self.assertEqual(api_keys, {"qwen\x1fgpu-1\x1fprice\x1fcomparison\x1funobserved_assumed_uncached"})

    def test_metrics_filter_retired_models_from_persisted_state(self) -> None:
        timestamp = datetime(2026, 9, 7, tzinfo=UTC).timestamp()
        config = {
            "host_id": "test-host",
            "baseline": {"mode": "baseline_non_gpu_w", "baseline_non_gpu_w": 100},
            "tariffs": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            accounting = CostAccounting(config, {}, Path(temporary) / "state.json")
            accounting.observe(
                timestamp,
                {"gpu-a": {"power_watts": 20, "model": "retired", "active": False}},
                {"gpu-a": {"model": "retired", "completion_tokens": 0}},
            )
            accounting.observe(
                timestamp + 10,
                {"gpu-a": {"power_watts": 20, "model": "current", "active": False}},
                {"gpu-a": {"model": "current", "completion_tokens": 0}},
            )

            metrics = accounting.metrics({"current"})
            model_labels = [labels["model"] for _name, labels, _value in metrics if "model" in labels]
            self.assertIn("current", model_labels)
            self.assertNotIn("retired", model_labels)

            accounting.state["active_model_energy_joules"]["current\x1fgpu-old"] = 1.0
            metrics = accounting.metrics({"current"}, {"current": {"gpu-a"}})
            current_uuids = {
                labels["gpu_uuid"]
                for _name, labels, _value in metrics
                if labels.get("model") == "current"
            }
            self.assertEqual(current_uuids, {"gpu-a"})


class MultiGpuExporterTest(unittest.TestCase):
    def test_vllm_parser_normalizes_counters_and_rates(self) -> None:
        labels = {"host_id": "test", "model": "qwen", "gpu": "0", "gpu_uuid": "gpu-a"}
        metrics = "\n".join(
            [
                'vllm:prompt_tokens_total{model_name="qwen"} 100',
                'vllm:prompt_tokens_cached_total{model_name="qwen"} 80',
                'vllm:generation_tokens_total{model_name="qwen"} 40',
                'vllm:num_requests_running{model_name="qwen"} 2',
                'vllm:request_success_total{model_name="qwen",finished_reason="stop"} 3',
                'vllm:request_success_total{model_name="qwen",finished_reason="length"} 1',
                'vllm:time_to_first_token_seconds_count{model_name="qwen"} 4',
                'vllm:time_to_first_token_seconds_sum{model_name="qwen"} 8',
            ]
        )
        counters: dict[str, tuple[float, float]] = {}
        rates: dict[str, float] = {}
        first, observation = parse_vllm_metrics(metrics, labels, set(), rates, counters, 10)
        second, _ = parse_vllm_metrics(
            metrics.replace(" 100", " 120", 1).replace(" 40", " 50", 1),
            labels,
            set(),
            rates,
            counters,
            20,
        )
        self.assertEqual(observation["requests_active"], 2)
        self.assertIn('ai_model_requests_completed_total{host_id="test",model="qwen",gpu="0",gpu_uuid="gpu-a"} 4.0', first)
        self.assertIn('ai_model_ttft_seconds{host_id="test",model="qwen",gpu="0",gpu_uuid="gpu-a"} 2.0', first)
        self.assertIn('ai_model_prefill_tokens_per_second{host_id="test",model="qwen",gpu="0",gpu_uuid="gpu-a"} 2.0', second)
        self.assertIn('ai_model_decode_tokens_per_second{host_id="test",model="qwen",gpu="0",gpu_uuid="gpu-a"} 1.0', second)
    def test_backend_gpu_ownership_is_unique_across_backends(self) -> None:
        with self.assertRaisesRegex(ValueError, "multiple metrics backends"):
            validate_backend_gpu_ownership(
                {
                    "qwen_gpu1": {"gpus": ["1"]},
                    "qwen_duplicate": {"gpus": ["1"]},
                }
            )

    def test_two_independent_qwen_backends_keep_replica_and_cache_status(self) -> None:
        backends = {
            "qwen_atx_gpu1": {
                "url": "http://127.0.0.1:8080/metrics",
                "model": "qwen3.8-27b-atx-iq4xs-m-262144-gpu1",
                "gpus": ["1"],
            },
            "qwen_atx_gpu2": {
                "url": "http://127.0.0.1:8081/metrics",
                "model": "qwen3.8-27b-atx-iq4xs-m-262144-gpu2",
                "gpus": ["2"],
            },
            "muse": {
                "url": "http://127.0.0.1:8082/metrics",
                "model": "muse",
                "gpus": ["0"],
            },
        }
        gpu_fields = {
            "name": "3090",
            "power_limit_watts": 300,
            "power_watts": 20,
            "utilization_percent": 0,
            "temperature_celsius": 40,
            "memory_used_bytes": 1,
            "memory_total_bytes": 2,
        }
        gpus = [
            {**gpu_fields, "gpu": str(index), "gpu_uuid": f"gpu-{index}"}
            for index in range(3)
        ]
        llama_metrics = "\n".join(
            [
                "llamacpp:requests_processing 0",
                "llamacpp:prompt_tokens_total 10",
                "llamacpp:prompt_tokens_cached_total 7",
                "llamacpp:tokens_predicted_total 20",
            ]
        )
        with patch.dict("os.environ", {"AI_SERVING_PROFILE": "atx-dual"}), patch(
            "ai_metrics_exporter.read_nvidia_gpus", return_value=gpus
        ), patch(
            "ai_metrics_exporter.fetch_metrics",
            side_effect=[llama_metrics, llama_metrics, llama_metrics],
        ), patch("ai_metrics_exporter.read_cpu_profiler", return_value=None), patch(
            "ai_metrics_exporter.read_proc_stat_cpu", return_value=None
        ):
            state = MetricsState(backends, CostAccounting.disabled())

        body = state.get()
        self.assertIn('ai_serving_profile_info{profile="atx-dual"} 1', body)
        self.assertIn('ai_backend_up{backend="qwen_atx_gpu1",model="qwen3.8-27b-atx-iq4xs-m-262144-gpu1"} 1', body)
        self.assertIn('ai_backend_up{backend="qwen_atx_gpu2",model="qwen3.8-27b-atx-iq4xs-m-262144-gpu2"} 1', body)
        self.assertIn('replica="qwen_atx_gpu1"', body)
        self.assertIn('replica="qwen_atx_gpu2"', body)
        self.assertIn('model="qwen3.8-27b-atx-iq4xs-m-262144-gpu1",gpu="1",gpu_uuid="gpu-1",mode="unobserved"', body)
        self.assertIn('model="qwen3.8-27b-atx-iq4xs-m-262144-gpu2",gpu="2",gpu_uuid="gpu-2",mode="unobserved"', body)
        self.assertIn('model="muse",gpu="0",gpu_uuid="gpu-0",mode="observed"', body)
        self.assertNotIn(
            'ai_model_cached_prompt_tokens_total{host_id="unconfigured",model="qwen3.8-27b-atx',
            body,
        )

    def test_gateway_observation_is_bounded_and_safe_when_unavailable(self) -> None:
        samples = parse_gateway_metrics(
            "\n".join(
                [
                    'ai_gateway_pool_route_decisions_total{pool="qwen",replica="gpu1",reason="affinity",session_id="secret"} 3',
                    'ai_gateway_pool_affinity_hits_total{pool="qwen",replica="gpu1"} 2',
                    'ai_gateway_pool_affinity_misses_total{pool="qwen",replica="gpu2"} 1',
                    'ai_gateway_pool_saturation_rejections_total{pool="qwen",replica="gpu1"} 4',
                    'ai_gateway_pool_cold_failover_total{pool="qwen",replica="gpu2"} 1',
                ]
            )
        )
        self.assertEqual(samples["route"][("qwen", "gpu1", "affinity")], 3)
        self.assertNotIn("secret", repr(samples))
        lines = gateway_metrics_lines(set(), None)
        self.assertIn("ai_gateway_metrics_up 0.0", lines)
        self.assertIn('ai_gateway_metrics_observation_info{status="unknown"} 1', lines)
        self.assertIn(
            'ai_gateway_pool_saturation_rejections_total{pool="unknown",replica="unknown"} 0.0',
            lines,
        )

    def test_llamampere_cache_mode_does_not_normalize_cached_tokens(self) -> None:
        output, observation = parse_llama_metrics(
            "llamacpp:prompt_tokens_total 10\nllamacpp:prompt_tokens_cached_total 7",
            {"model": "qwen", "gpu_uuid": "gpu-1"},
            set(),
            cached_input_mode="unobserved",
        )
        self.assertEqual(observation["cached_prompt_tokens"], 0.0)
        self.assertFalse(any(line.startswith("ai_model_cached_prompt_tokens_total") for line in output))
        self.assertEqual(cached_input_mode({"runtime": "llamAmpere"}), "unobserved")

    def test_nvidia_metrics_labels_all_physical_gpus(self) -> None:
        comments: set[str] = set()
        lines = nvidia_metrics(
            comments,
            [
                {"gpu": "0", "gpu_uuid": "gpu-0", "name": "3090", "power_limit_watts": 300, "power_watts": 20, "utilization_percent": 0, "temperature_celsius": 40, "memory_used_bytes": 1, "memory_total_bytes": 2},
                {"gpu": "1", "gpu_uuid": "gpu-1", "name": "3090", "power_limit_watts": 300, "power_watts": 20, "utilization_percent": 0, "temperature_celsius": 40, "memory_used_bytes": 1, "memory_total_bytes": 2},
                {"gpu": "2", "gpu_uuid": "gpu-2", "name": "3090", "power_limit_watts": 300, "power_watts": 20, "utilization_percent": 0, "temperature_celsius": 40, "memory_used_bytes": 1, "memory_total_bytes": 2},
            ],
            {"0": "muse", "1": "qwen", "2": "qwen"},
            "test-host",
        )
        model_lines = [line for line in lines if line.startswith("ai_gpu_power_draw_watts{")]
        self.assertEqual(len(model_lines), 3)
        self.assertTrue(any('gpu="0"' in line and 'model="muse"' in line for line in model_lines))
        self.assertEqual(sum('model="qwen"' in line for line in model_lines), 2)

    def test_metrics_state_scrapes_dual_gpu_backend_once(self) -> None:
        backends = {
            "qwen": {
                "url": "http://127.0.0.1:8080/metrics",
                "model": "qwen",
                "gpus": ["1", "2"],
            },
            "muse": {
                "url": "http://127.0.0.1:8082/metrics",
                "model": "muse",
                "gpus": ["0"],
            },
        }
        gpu_fields = {
            "gpu": "0",
            "gpu_uuid": "gpu-0",
            "name": "3090",
            "power_limit_watts": 300,
            "power_watts": 20,
            "utilization_percent": 0,
            "temperature_celsius": 40,
            "memory_used_bytes": 1,
            "memory_total_bytes": 2,
        }
        gpus = [{**gpu_fields, "gpu": str(index), "gpu_uuid": f"gpu-{index}"} for index in range(3)]
        llama_metrics = "\n".join(
            [
                "llamacpp:requests_processing 0",
                "llamacpp:prompt_tokens_total 10",
                "llamacpp:prompt_tokens_cached_total 0",
                "llamacpp:tokens_predicted_total 20",
            ]
        )
        with patch.dict("os.environ", {"AI_SERVING_PROFILE": ""}), patch(
            "ai_metrics_exporter.read_nvidia_gpus", return_value=gpus
        ), patch(
            "ai_metrics_exporter.fetch_metrics", side_effect=[llama_metrics, llama_metrics]
        ), patch("ai_metrics_exporter.read_cpu_profiler", return_value=None), patch(
            "ai_metrics_exporter.read_proc_stat_cpu", return_value=None
        ):
            state = MetricsState(backends, CostAccounting.disabled())

        body = state.get()
        self.assertEqual(body.count('ai_model_up{host_id="unconfigured",model="qwen",gpu="1"'), 1)
        self.assertEqual(body.count('ai_model_up{host_id="unconfigured",model="qwen",gpu="2"'), 1)
        self.assertEqual(body.count('ai_model_up{host_id="unconfigured",model="muse",gpu="0"'), 1)
        self.assertIn('ai_serving_profile_info{profile="unknown"} 1', body)
        self.assertNotIn("flash_next_262k", body)
        self.assertNotIn("qwen3.8-27b-q4-gpukv192", body)


class ThroughputRetentionTest(unittest.TestCase):
    def labels(self) -> dict[str, str]:
        return {"host_id": "ai-server", "model": "flash", "gpu": "0", "gpu_uuid": "gpu-a"}

    def values_for(self, lines: list[str], metric: str) -> list[str]:
        return [line for line in lines if line.startswith(metric + "{")]

    def test_freetoken_zero_decode_holds_plateau(self) -> None:
        state: dict[str, float] = {}
        labels = self.labels()
        first, _ = parse_freetoken_stats(
            {"throughput": {"decode_tps": 30.5, "prefill_tps": 400.0}, "requests": {}},
            labels,
            set(),
            state,
        )
        self.assertTrue(all(line.endswith(" 30.5") for line in self.values_for(first, "ai_model_decode_tokens_per_second")))
        second, _ = parse_freetoken_stats(
            {"throughput": {"decode_tps": 0.0, "prefill_tps": 0.0}, "requests": {}},
            labels,
            set(),
            state,
        )
        self.assertTrue(all(line.endswith(" 30.5") for line in self.values_for(second, "ai_model_decode_tokens_per_second")))
        self.assertTrue(all(line.endswith(" 400.0") for line in self.values_for(second, "ai_model_prefill_tokens_per_second")))
        third, _ = parse_freetoken_stats(
            {"throughput": {"decode_tps": 12.0, "prefill_tps": 0.0}, "requests": {}},
            labels,
            set(),
            state,
        )
        self.assertTrue(all(line.endswith(" 12.0") for line in self.values_for(third, "ai_model_decode_tokens_per_second")))
        self.assertTrue(all(line.endswith(" 400.0") for line in self.values_for(third, "ai_model_prefill_tokens_per_second")))

    def test_llama_zero_decode_holds_plateau(self) -> None:
        state: dict[str, float] = {}
        labels = self.labels()
        _, _ = parse_llama_metrics("llamacpp:predicted_tokens_seconds 25", labels, set(), state)
        output, _ = parse_llama_metrics("llamacpp:predicted_tokens_seconds 0", labels, set(), state)
        decode_lines = self.values_for(output, "ai_model_decode_tokens_per_second")
        self.assertTrue(all(line.endswith(" 25.0") for line in decode_lines))
        raw_lines = [line for line in output if line.startswith("llamacpp:predicted_tokens_seconds")]
        self.assertTrue(all(line.endswith(" 0") for line in raw_lines))

    def test_retention_is_scoped_per_model(self) -> None:
        state: dict[str, float] = {}
        retain_nonzero_rate(state, "a", "ai_model_decode_tokens_per_second", 20)
        value = retain_nonzero_rate(state, "b", "ai_model_decode_tokens_per_second", 0)
        self.assertEqual(value, 0)


class HostCostPeriodTest(unittest.TestCase):
    config = {
        "host_id": "test-host",
        "timezone": "Asia/Tokyo",
        "max_sample_gap_seconds": 20_000,
        "baseline": {"mode": "baseline_non_gpu_w", "baseline_non_gpu_w": 120},
        "tariffs": [
            {"id": "tariff", "effective_from": "2026-01-01", "components_jpy_per_kwh": {"rate": 36.0}}
        ],
    }

    def samples_with_labels(self) -> list[tuple[str, dict[str, str], float]]:
        return self.accounting.metrics()

    def test_day_period_resets_at_jst_midnight(self) -> None:
        tz = ZoneInfo("Asia/Tokyo")
        first_jst_day = datetime(2026, 9, 7, 23, 0, tzinfo=tz)
        second_jst_day = first_jst_day + timedelta(hours=3)
        gpu = {"gpu-a": {"power_watts": 20, "model": None, "active": False}}
        with tempfile.TemporaryDirectory() as temporary:
            self.accounting = CostAccounting(self.config, {}, Path(temporary) / "state.json")
            self.accounting.observe(first_jst_day.timestamp(), gpu, {})
            self.accounting.observe(second_jst_day.timestamp(), gpu, {})
            self.accounting.observe(second_jst_day.timestamp() + 3600, gpu, {})

            periods = {
                name: (labels, value)
                for name, labels, value in self.samples_with_labels()
                if name.startswith("ai_host_electricity_cost_jpy_") and name != "ai_host_electricity_cost_jpy_total"
            }
            today_labels, today_value = periods["ai_host_electricity_cost_jpy_today"]
            # 140 W wall power for one hour: 504000 J = 0.14 kWh at 36 JPY/kWh.
            self.assertAlmostEqual(today_value, 0.14 * 36, places=6)
            self.assertEqual(today_labels["period"], second_jst_day.date().isoformat())

            week_labels, week_value = periods["ai_host_electricity_cost_jpy_week_to_date"]
            self.assertEqual(
                week_labels["period"],
                (
                    second_jst_day.date()
                    - timedelta(days=second_jst_day.weekday())
                ).isoformat(),
            )
            self.assertGreater(week_value, today_value)

            month_labels, _month_value = periods["ai_host_electricity_cost_jpy_month_to_date"]
            self.assertEqual(month_labels["period"], first_jst_day.strftime("%Y-%m"))

    def test_period_rebases_after_counter_decrease(self) -> None:
        timestamp = datetime(2026, 9, 7, 12, 0, tzinfo=UTC).timestamp()
        gpu = {"gpu-a": {"power_watts": 20, "model": None, "active": False}}
        with tempfile.TemporaryDirectory() as temporary:
            self.accounting = CostAccounting(self.config, {}, Path(temporary) / "state.json")
            self.accounting.observe(timestamp, gpu, {})
            self.accounting.observe(timestamp + 600, gpu, {})
            self.accounting.state["host_cost_periods"]["tariff\x1fday"]["baseline"] = 5.0
            self.accounting.observe(timestamp + 1200, gpu, {})
            today = [
                value
                for name, _labels, value in self.samples_with_labels()
                if name == "ai_host_electricity_cost_jpy_today"
            ]
            self.assertEqual(today, [0.0])


class CpuProfilerTest(unittest.TestCase):
    def test_rapl_power_delta_and_wrap(self) -> None:
        watts = rapl_power_watts(1_000_000.0, 6_000_000.0, 10_000_000.0, 5.0)
        self.assertIsNotNone(watts)
        self.assertAlmostEqual(watts if watts is not None else -1.0, 1.0, places=9)
        wrapped = rapl_power_watts(9_000_000.0, 2_000_000.0, 10_000_000.0, 2.0)
        self.assertIsNotNone(wrapped)
        self.assertAlmostEqual(wrapped if wrapped is not None else -1.0, 1.5, places=9)
        self.assertIsNone(rapl_power_watts(None, 2_000_000.0, 10_000_000.0, 2.0))

    def test_bucket_index_and_label(self) -> None:
        self.assertEqual(bucket_index(24.99, 5.0), 4)
        self.assertEqual(bucket_label(4, 5.0), "20-25")
        self.assertEqual(bucket_label(bucket_index(100.0, 5.0), 5.0), "95-100")
        self.assertEqual(bucket_index(-1.0, 5.0), 0)

    def test_bucket_window_drops_oldest(self) -> None:
        profiler = CpuPowerProfiler(
            {"host_id": "test", "bucket_width_percent": 5.0, "bucket_window_samples": 3},
            None,
        )
        for watts in (10.0, 20.0, 30.0, 40.0, 50.0):
            profiler.update_bucket(2.5, watts)
        self.assertEqual(profiler.state["buckets"]["00-05"], [30.0, 40.0, 50.0])

    def test_cpu_utilization_percent(self) -> None:
        util = cpu_utilization_percent((100.0, 10.0), (200.0, 60.0))
        self.assertAlmostEqual(util if util is not None else -1.0, 50.0)
        self.assertIsNone(cpu_utilization_percent(None, (200.0, 60.0)))

    def test_interpolate_power(self) -> None:
        points = [(2.5, 10.0), (7.5, 50.0), (12.5, 90.0)]
        at_mid = interpolate_power(points, 5.0)
        self.assertAlmostEqual(at_mid if at_mid is not None else -1.0, 30.0)
        self.assertEqual(interpolate_power(points, 0.0), 10.0)
        self.assertEqual(interpolate_power(points, 20.0), 90.0)
        self.assertEqual(interpolate_power([(7.5, 50.0)], 1.0), 50.0)
        self.assertIsNone(interpolate_power([], 1.0))

    def test_bucket_points_from_labels(self) -> None:
        self.assertEqual(bucket_points({'bucket="20-25"': 40.0}), [(22.5, 40.0)])


class WallIdlePlusCpuTest(unittest.TestCase):
    baseline = {
        "mode": "wall_idle_plus_cpu_w",
        "base_idle_total_w": 120.0,
        "default_gpu_idle_reference_w": 20.0,
    }
    pricing = {"prices": [], "comparisons": [], "fx_rates": []}
    tariff = [
        {"id": "t", "effective_from": "2026-01-01", "components_jpy_per_kwh": {"rate": 36.0}}
    ]

    def make(self, baseline: dict) -> CostAccounting:
        return CostAccounting(
            {"host_id": "test", "tariffs": self.tariff, "baseline": baseline},
            self.pricing,
            None,
        )

    def values(self, accounting: CostAccounting) -> dict[str, float]:
        output: dict[str, float] = {}
        for name, _labels, value in accounting.metrics():
            output.setdefault(name, value)
        return output

    def test_host_power_includes_cpu_only_when_measured(self) -> None:
        gpus = {"gpu-a": 40.0, "gpu-b": 20.0}
        self.assertAlmostEqual(host_power_watts(self.baseline, gpus), 140.0)
        self.assertAlmostEqual(host_power_watts(self.baseline, gpus, 80.0), 220.0)

    def test_observe_integrates_cpu_and_host_energy(self) -> None:
        accounting = self.make(self.baseline)
        gpus = {"gpu-a": {"power_watts": 20.0, "model": None, "active": False}}
        accounting.observe(100.0, gpus, {}, cpu_power_watts=100.0)
        accounting.observe(110.0, gpus, {}, cpu_power_watts=200.0)
        values = self.values(accounting)
        self.assertAlmostEqual(values["ai_host_cpu_energy_joules_total"], 1500.0, places=6)
        # Host: trapezoid of (120+100) and (120+200) over 10 s = 2700 J.
        self.assertAlmostEqual(values["ai_host_estimated_energy_joules_total"], 2700.0, places=6)

    def test_flat_baseline_mode_still_records_cpu_energy(self) -> None:
        accounting = self.make(
            {
                "mode": "wall_idle_total_w",
                "wall_idle_total_w": 300.0,
                "default_gpu_idle_reference_w": 20.0,
            }
        )
        gpus = {"gpu-a": {"power_watts": 20.0, "model": None, "active": False}}
        accounting.observe(100.0, gpus, {}, cpu_power_watts=100.0)
        accounting.observe(110.0, gpus, {}, cpu_power_watts=200.0)
        values = self.values(accounting)
        self.assertAlmostEqual(values["ai_host_cpu_energy_joules_total"], 1500.0, places=6)
        # Flat 300 W baseline must ignore CPU power.
        self.assertAlmostEqual(values["ai_host_estimated_energy_joules_total"], 3000.0, places=6)


if __name__ == "__main__":
    unittest.main()
