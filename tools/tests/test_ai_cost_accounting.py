from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_cost_accounting import (
    CostAccounting,
    api_cost_usd,
    counter_delta,
    host_power_watts,
    rate_from_components,
    trapezoid_energy_joules,
)
from ai_metrics_exporter import parse_llama_metrics


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

    def test_api_pricing_separates_cached_and_output_cost(self) -> None:
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
        self.assertAlmostEqual(workload, 2.1)
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
            self.assertAlmostEqual(accounting.state["api_workload_cost_usd"]["local\x1fgpu-a\x1fprice\x1fcomparison\x1fobserved"], 0.0002292)
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


if __name__ == "__main__":
    unittest.main()
