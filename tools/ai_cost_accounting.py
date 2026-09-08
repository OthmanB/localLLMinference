"""Durable energy and API-price accounting for the AI metrics exporter."""

from __future__ import annotations

from datetime import UTC, datetime, date, timedelta
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
from zoneinfo import ZoneInfo


JOULES_PER_KWH = 3_600_000.0
TOKENS_PER_MILLION = 1_000_000.0
STATE_VERSION = 1
KEY_SEPARATOR = "\x1f"


def metric_key(*parts: str) -> str:
    return KEY_SEPARATOR.join(parts)


def split_metric_key(value: str) -> tuple[str, ...]:
    return tuple(value.split(KEY_SEPARATOR))


def counter_delta(previous: float | None, current: float) -> float:
    """Return a monotonic-counter increment, treating a decrease as a reset."""
    if previous is None:
        return 0.0
    return current - previous if current >= previous else current


def trapezoid_energy_joules(previous_w: float, current_w: float, elapsed_seconds: float) -> float:
    """Integrate two timestamped power samples without assuming a scrape interval."""
    if elapsed_seconds <= 0:
        return 0.0
    return (previous_w + current_w) * 0.5 * elapsed_seconds


def active_on_date(entry: Mapping[str, Any], today: date) -> bool:
    start = entry.get("effective_from")
    end = entry.get("effective_to")
    if start and today < date.fromisoformat(str(start)):
        return False
    if end and today > date.fromisoformat(str(end)):
        return False
    return True


def rate_from_components(components: Mapping[str, Any]) -> float:
    return sum(float(value) for value in components.values() if value is not None)


def gpu_excess_power_watts(baseline: Mapping[str, Any], gpu_powers: Mapping[str, float]) -> float:
    """Sum GPU draw above the configured per-GPU idle references."""
    references = baseline.get("gpu_idle_reference_w_by_uuid", {})
    default_reference = float(baseline.get("default_gpu_idle_reference_w", 0.0))
    return sum(
        max(power - float(references.get(uuid, default_reference)), 0.0)
        for uuid, power in gpu_powers.items()
    )


def host_power_watts(
    baseline: Mapping[str, Any],
    gpu_powers: Mapping[str, float],
    cpu_power_w: float | None = None,
) -> float:
    """Calculate wall power with exactly one GPU-idle accounting convention."""
    mode = baseline.get("mode")
    if mode == "baseline_non_gpu_w":
        return float(baseline["baseline_non_gpu_w"]) + sum(gpu_powers.values())
    if mode == "wall_idle_total_w":
        return float(baseline["wall_idle_total_w"]) + gpu_excess_power_watts(baseline, gpu_powers)
    if mode == "wall_idle_plus_cpu_w":
        cpu_power = float(cpu_power_w) if cpu_power_w is not None else 0.0
        return (
            float(baseline["base_idle_total_w"])
            + cpu_power
            + gpu_excess_power_watts(baseline, gpu_powers)
        )
    raise ValueError(f"unsupported baseline mode: {mode!r}")


def api_cost_usd(
    input_tokens: float,
    cached_input_tokens: float,
    output_tokens: float,
    price: Mapping[str, Any],
) -> tuple[float, float]:
    """Return complete-workload and output-only API costs in USD."""
    input_rate = float(price["input_usd_per_million_tokens"])
    cached_rate = price.get("cached_input_usd_per_million_tokens")
    cached_rate = input_rate if cached_rate is None else float(cached_rate)
    output_rate = float(price["output_usd_per_million_tokens"])
    # Runtime cached-prompt counters are a subset of total prompt tokens.
    uncached_input_tokens = max(input_tokens - cached_input_tokens, 0.0)
    workload = (
        uncached_input_tokens * input_rate
        + cached_input_tokens * cached_rate
        + output_tokens * output_rate
    ) / TOKENS_PER_MILLION
    return workload, output_tokens * output_rate / TOKENS_PER_MILLION


def read_host_boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError:
        return "unavailable"


def read_host_boot_time_seconds() -> float:
    try:
        with open("/proc/stat", encoding="ascii") as proc_stat:
            for line in proc_stat:
                if line.startswith("btime "):
                    return float(line.split()[1])
    except OSError:
        pass
    return 0.0


class CostAccounting:
    """Integrate physical energy and account token deltas into configured price scenarios."""

    def __init__(
        self,
        config: Mapping[str, Any],
        pricing: Mapping[str, Any],
        state_path: Path | None,
        boot_id: str | None = None,
        boot_time_seconds: float | None = None,
    ) -> None:
        self.config = dict(config)
        self.pricing = dict(pricing)
        self.state_path = state_path
        self.host_id = str(self.config["host_id"])
        self.max_gap_seconds = float(self.config.get("max_sample_gap_seconds", 20.0))
        self.baseline = self.config["baseline"]
        self.timezone = ZoneInfo(str(self.config.get("timezone", "Asia/Tokyo")))
        self.boot_id = read_host_boot_id() if boot_id is None else boot_id
        self.boot_time_seconds = (
            read_host_boot_time_seconds() if boot_time_seconds is None else boot_time_seconds
        )
        self.state = self._load_state()

    @classmethod
    def disabled(cls) -> "CostAccounting":
        return cls(
            {
                "host_id": "unconfigured",
                "baseline": {"mode": "baseline_non_gpu_w", "baseline_non_gpu_w": 0},
            },
            {"prices": [], "comparisons": [], "fx_rates": []},
            None,
        )

    def _empty_state(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "last_sample": None,
            "boot_id": None,
            "last_tokens": {},
            "gpu_energy_joules": {},
            "active_model_energy_joules": {},
            "energy_covered_completion_tokens": {},
            "models_with_unmatched_active_energy": {},
            "host_energy_joules": 0.0,
            "cpu_energy_joules": 0.0,
            "host_energy_cost_jpy": {},
            "host_cost_periods": {},
            "active_model_energy_cost_jpy": {},
            "api_workload_cost_usd": {},
            "api_output_cost_usd": {},
            "api_workload_cost_jpy": {},
            "api_output_cost_jpy": {},
            "integration_gap_seconds_total": 0.0,
            "reboot_downtime_seconds_total": 0.0,
        }

    def _load_state(self) -> dict[str, Any]:
        if self.state_path is None or not self.state_path.exists():
            return self._empty_state()
        try:
            loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return self._empty_state()
        if not isinstance(loaded, dict) or loaded.get("version") != STATE_VERSION:
            return self._empty_state()
        state = self._empty_state()
        state.update(loaded)
        return state

    def _save_state(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.state_path.parent,
            prefix=f".{self.state_path.name}.",
            text=True,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(self.state, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.state_path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    @staticmethod
    def _increment(values: dict[str, float], key: str, amount: float) -> None:
        if amount:
            values[key] = float(values.get(key, 0.0)) + amount

    def _tariffs(self, today: date) -> list[Mapping[str, Any]]:
        return [
            tariff
            for tariff in self.config.get("tariffs", [])
            if active_on_date(tariff, today)
        ]

    def _fx_rates(self, today: date) -> list[Mapping[str, Any]]:
        return [rate for rate in self.pricing.get("fx_rates", []) if active_on_date(rate, today)]

    def _prices_for_model(self, model: str, today: date) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
        prices = {
            str(price["id"]): price
            for price in self.pricing.get("prices", [])
            if active_on_date(price, today)
        }
        selected: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        for comparison in self.pricing.get("comparisons", []):
            if comparison.get("local_model") not in {"*", model}:
                continue
            price = prices.get(str(comparison.get("price_id")))
            if price is not None:
                selected.append((price, comparison))
        return selected

    def _price_labels(self, price_id: str, comparison_id: str) -> dict[str, str]:
        price = next(
            (entry for entry in self.pricing.get("prices", []) if str(entry.get("id")) == price_id),
            {},
        )
        comparison = next(
            (entry for entry in self.pricing.get("comparisons", []) if str(entry.get("id")) == comparison_id),
            {},
        )
        return {
            "provider": str(price.get("provider", "unknown")),
            "remote_model": str(price.get("model", price_id)),
            "price_id": price_id,
            "scenario": str(price.get("scenario", "standard")),
            "comparison": str(comparison.get("comparison", "unclassified")),
        }

    def observe(
        self,
        timestamp_seconds: float,
        gpus: Mapping[str, Mapping[str, Any]],
        models: Mapping[str, Mapping[str, Any]],
        cpu_power_watts: float | None = None,
    ) -> None:
        """Observe one exporter sample keyed by GPU UUID."""
        today = datetime.fromtimestamp(timestamp_seconds, UTC).date()
        previous = self.state.get("last_sample")
        if isinstance(previous, Mapping):
            elapsed = timestamp_seconds - float(previous.get("timestamp_seconds", timestamp_seconds))
            previous_gpus = previous.get("gpus", {})
            rebooted = bool(self.state.get("boot_id")) and self.state["boot_id"] != self.boot_id
            if rebooted:
                self.state["reboot_downtime_seconds_total"] = float(
                    self.state.get("reboot_downtime_seconds_total", 0.0)
                ) + max(elapsed, 0.0)
                covered_models = set()
            elif 0 < elapsed <= self.max_gap_seconds and isinstance(previous_gpus, Mapping):
                covered_models = self._integrate_energy(
                    elapsed,
                    previous_gpus,
                    gpus,
                    today,
                    previous.get("cpu_power_watts"),
                    cpu_power_watts,
                )
            elif elapsed > self.max_gap_seconds:
                self.state["integration_gap_seconds_total"] = float(
                    self.state.get("integration_gap_seconds_total", 0.0)
                ) + elapsed
                covered_models = set()
            else:
                covered_models = set()
        else:
            covered_models = set()

        self._initialize_model_counters(models, today)
        self._account_tokens(models, today, covered_models)
        self._update_cost_periods(datetime.fromtimestamp(timestamp_seconds, self.timezone).date())
        self.state["boot_id"] = self.boot_id
        self.state["last_sample"] = {
            "timestamp_seconds": timestamp_seconds,
            "cpu_power_watts": cpu_power_watts,
            "gpus": {
                uuid: {
                    "power_watts": float(sample["power_watts"]),
                    "model": sample.get("model"),
                    "active": bool(sample.get("active")),
                }
                for uuid, sample in gpus.items()
            },
        }
        self._save_state()

    def _initialize_model_counters(
        self, models: Mapping[str, Mapping[str, Any]], today: date
    ) -> None:
        """Expose zero-valued configured counters before the first request."""
        for uuid, sample in models.items():
            model = str(sample["model"])
            model_key = metric_key(model, uuid)
            self.state["active_model_energy_joules"].setdefault(model_key, 0.0)
            self.state["energy_covered_completion_tokens"].setdefault(model_key, 0.0)
            for tariff in self._tariffs(today):
                cost_key = metric_key(model, uuid, str(tariff["id"]))
                self.state["active_model_energy_cost_jpy"].setdefault(cost_key, 0.0)
            cached_mode = str(sample.get("cached_input_mode", "unobserved_assumed_uncached"))
            for price, comparison in self._prices_for_model(model, today):
                price_id = str(price["id"])
                comparison_id = str(comparison["id"])
                cost_key = metric_key(model, uuid, price_id, comparison_id, cached_mode)
                self.state["api_workload_cost_usd"].setdefault(cost_key, 0.0)
                self.state["api_output_cost_usd"].setdefault(cost_key, 0.0)
                for fx in self._fx_rates(today):
                    fx_key = metric_key(
                        model,
                        uuid,
                        price_id,
                        comparison_id,
                        cached_mode,
                        str(fx["id"]),
                    )
                    self.state["api_workload_cost_jpy"].setdefault(fx_key, 0.0)
                    self.state["api_output_cost_jpy"].setdefault(fx_key, 0.0)

    def _integrate_energy(
        self,
        elapsed: float,
        previous_gpus: Mapping[str, Any],
        current_gpus: Mapping[str, Mapping[str, Any]],
        today: date,
        previous_cpu_watts: float | None = None,
        current_cpu_watts: float | None = None,
    ) -> set[str]:
        previous_power: dict[str, float] = {}
        current_power: dict[str, float] = {}
        covered_models: set[str] = set()
        for uuid, current in current_gpus.items():
            before = previous_gpus.get(uuid)
            if not isinstance(before, Mapping):
                continue
            before_watts = float(before.get("power_watts", 0.0))
            current_watts = float(current["power_watts"])
            energy = trapezoid_energy_joules(before_watts, current_watts, elapsed)
            self._increment(self.state["gpu_energy_joules"], uuid, energy)
            previous_power[uuid] = before_watts
            current_power[uuid] = current_watts

            model = current.get("model")
            if model and model == before.get("model") and current.get("active") and before.get("active"):
                model_key = metric_key(str(model), uuid)
                covered_models.add(model_key)
                self.state["models_with_unmatched_active_energy"][model_key] = True
                self._increment(self.state["active_model_energy_joules"], model_key, energy)
                for tariff in self._tariffs(today):
                    tariff_id = str(tariff["id"])
                    cost_key = metric_key(str(model), uuid, tariff_id)
                    self._increment(
                        self.state["active_model_energy_cost_jpy"],
                        cost_key,
                        energy / JOULES_PER_KWH * rate_from_components(tariff["components_jpy_per_kwh"]),
                    )

        cpu_energy = 0.0
        if previous_cpu_watts is not None and current_cpu_watts is not None:
            cpu_energy = trapezoid_energy_joules(
                float(previous_cpu_watts), float(current_cpu_watts), elapsed
            )
        self._increment(self.state, "cpu_energy_joules", cpu_energy)

        # A wall-power estimate needs a complete, stable GPU set. Individual GPU
        # energy can still be integrated when one device's samples are available.
        complete_gpu_set = bool(previous_gpus) and set(previous_gpus) == set(current_gpus)
        if complete_gpu_set:
            host_energy = trapezoid_energy_joules(
                host_power_watts(self.baseline, previous_power, previous_cpu_watts),
                host_power_watts(self.baseline, current_power, current_cpu_watts),
                elapsed,
            )
            self.state["host_energy_joules"] = float(self.state["host_energy_joules"]) + host_energy
            for tariff in self._tariffs(today):
                self._increment(
                    self.state["host_energy_cost_jpy"],
                    str(tariff["id"]),
                    host_energy / JOULES_PER_KWH * rate_from_components(tariff["components_jpy_per_kwh"]),
                )
        else:
            self.state["integration_gap_seconds_total"] = float(
                self.state.get("integration_gap_seconds_total", 0.0)
            ) + elapsed
        return covered_models

    def _account_tokens(
        self,
        models: Mapping[str, Mapping[str, Any]],
        today: date,
        covered_models: set[str],
    ) -> None:
        for uuid, sample in models.items():
            model = str(sample["model"])
            model_key = metric_key(model, uuid)
            current = {
                "prompt": float(sample.get("prompt_tokens", 0.0)),
                "cached": float(sample.get("cached_prompt_tokens", 0.0)),
                "completion": float(sample.get("completion_tokens", 0.0)),
            }
            previous = self.state["last_tokens"].get(model_key)
            self.state["last_tokens"][model_key] = current
            if not isinstance(previous, Mapping):
                continue
            prompt_delta = counter_delta(float(previous.get("prompt", 0.0)), current["prompt"])
            cached_delta = counter_delta(float(previous.get("cached", 0.0)), current["cached"])
            completion_delta = counter_delta(float(previous.get("completion", 0.0)), current["completion"])
            if model_key in covered_models or self.state["models_with_unmatched_active_energy"].get(model_key):
                self._increment(
                    self.state["energy_covered_completion_tokens"], model_key, completion_delta
                )
                if completion_delta:
                    self.state["models_with_unmatched_active_energy"].pop(model_key, None)
            for price, comparison in self._prices_for_model(model, today):
                price_id = str(price["id"])
                comparison_id = str(comparison["id"])
                cached_mode = str(sample.get("cached_input_mode", "unobserved_assumed_uncached"))
                cost_key = metric_key(model, uuid, price_id, comparison_id, cached_mode)
                workload_usd, output_usd = api_cost_usd(
                    prompt_delta,
                    cached_delta,
                    completion_delta,
                    price,
                )
                self._increment(self.state["api_workload_cost_usd"], cost_key, workload_usd)
                self._increment(self.state["api_output_cost_usd"], cost_key, output_usd)
                for fx in self._fx_rates(today):
                    fx_key = metric_key(
                        model,
                        uuid,
                        price_id,
                        comparison_id,
                        cached_mode,
                        str(fx["id"]),
                    )
                    jpy_per_usd = float(fx["jpy_per_usd"])
                    self._increment(self.state["api_workload_cost_jpy"], fx_key, workload_usd * jpy_per_usd)
                    self._increment(self.state["api_output_cost_jpy"], fx_key, output_usd * jpy_per_usd)

    def _update_cost_periods(self, local_day: date) -> None:
        """Reset host cost period baselines at calendar day/week/month boundaries."""
        monday = local_day - timedelta(days=local_day.weekday())
        period_keys = {
            "day": local_day.isoformat(),
            "week": monday.isoformat(),
            "month": local_day.strftime("%Y-%m"),
        }
        for tariff in self._tariffs(local_day):
            self.state["host_energy_cost_jpy"].setdefault(str(tariff["id"]), 0.0)
        periods = self.state["host_cost_periods"]
        for tariff_id, cumulative in self.state["host_energy_cost_jpy"].items():
            for kind, period_key in period_keys.items():
                key = metric_key(tariff_id, kind)
                entry = periods.get(key)
                if (
                    not isinstance(entry, Mapping)
                    or entry.get("period") != period_key
                    or float(entry.get("baseline", 0.0)) > cumulative
                ):
                    periods[key] = {"period": period_key, "baseline": float(cumulative)}

    def metrics(self) -> list[tuple[str, dict[str, str], float]]:
        """Return accounting counter samples for Prometheus exposition."""
        output: list[tuple[str, dict[str, str], float]] = []
        for uuid, joules in self.state["gpu_energy_joules"].items():
            output.append(("ai_gpu_energy_joules_total", {"host_id": self.host_id, "gpu_uuid": uuid}, float(joules)))
        for key, joules in self.state["active_model_energy_joules"].items():
            model, uuid = split_metric_key(key)
            output.append(("ai_model_active_gpu_energy_joules_total", {"host_id": self.host_id, "model": model, "gpu_uuid": uuid}, float(joules)))
        for key, tokens in self.state["energy_covered_completion_tokens"].items():
            model, uuid = split_metric_key(key)
            output.append(("ai_model_energy_covered_completion_tokens_total", {"host_id": self.host_id, "model": model, "gpu_uuid": uuid}, float(tokens)))
        output.append(("ai_host_estimated_energy_joules_total", {"host_id": self.host_id}, float(self.state["host_energy_joules"])))
        output.append(("ai_host_cpu_energy_joules_total", {"host_id": self.host_id}, float(self.state.get("cpu_energy_joules", 0.0))))
        output.append(("ai_energy_integration_gap_seconds_total", {"host_id": self.host_id}, float(self.state["integration_gap_seconds_total"])))
        output.append(("ai_host_reboot_downtime_seconds_total", {"host_id": self.host_id}, float(self.state["reboot_downtime_seconds_total"])))
        output.append(("ai_host_boot_time_seconds", {"host_id": self.host_id}, self.boot_time_seconds))
        for tariff_id, value in self.state["host_energy_cost_jpy"].items():
            output.append(("ai_host_electricity_cost_jpy_total", {"host_id": self.host_id, "tariff_id": tariff_id}, float(value)))
        period_metrics = {
            "day": "ai_host_electricity_cost_jpy_today",
            "week": "ai_host_electricity_cost_jpy_week_to_date",
            "month": "ai_host_electricity_cost_jpy_month_to_date",
        }
        for key, entry in self.state["host_cost_periods"].items():
            tariff_id, kind = split_metric_key(key)
            cumulative = float(self.state["host_energy_cost_jpy"].get(tariff_id, 0.0))
            to_date = max(cumulative - float(entry.get("baseline", 0.0)), 0.0)
            output.append(
                (
                    period_metrics[kind],
                    {"host_id": self.host_id, "tariff_id": tariff_id, "period": str(entry["period"])},
                    to_date,
                )
            )
        for key, value in self.state["active_model_energy_cost_jpy"].items():
            model, uuid, tariff_id = split_metric_key(key)
            output.append(("ai_model_active_gpu_electricity_cost_jpy_total", {"host_id": self.host_id, "model": model, "gpu_uuid": uuid, "tariff_id": tariff_id}, float(value)))
        for metric, values in (
            ("ai_model_api_workload_cost_usd_total", self.state["api_workload_cost_usd"]),
            ("ai_model_api_output_only_cost_usd_total", self.state["api_output_cost_usd"]),
        ):
            for key, value in values.items():
                model, uuid, price_id, comparison_id, cached_mode = split_metric_key(key)
                labels = {
                    "host_id": self.host_id,
                    "model": model,
                    "gpu_uuid": uuid,
                    "cached_input_mode": cached_mode,
                }
                labels.update(self._price_labels(price_id, comparison_id))
                output.append((metric, labels, float(value)))
        for metric, values in (
            ("ai_model_api_workload_cost_jpy_total", self.state["api_workload_cost_jpy"]),
            ("ai_model_api_output_only_cost_jpy_total", self.state["api_output_cost_jpy"]),
        ):
            for key, value in values.items():
                model, uuid, price_id, comparison_id, cached_mode, fx_id = split_metric_key(key)
                labels = {
                    "host_id": self.host_id,
                    "model": model,
                    "gpu_uuid": uuid,
                    "cached_input_mode": cached_mode,
                    "fx_id": fx_id,
                }
                labels.update(self._price_labels(price_id, comparison_id))
                output.append((metric, labels, float(value)))
        return output

    def configuration_metrics(self, timestamp_seconds: float) -> list[tuple[str, dict[str, str], float]]:
        today = datetime.fromtimestamp(timestamp_seconds, UTC).date()
        baseline_mode = str(self.baseline["mode"])
        baseline_value = float(
            self.baseline.get(
                "baseline_non_gpu_w",
                self.baseline.get("wall_idle_total_w", self.baseline.get("base_idle_total_w", 0.0)),
            )
        )
        output = [
            ("ai_host_power_accounting_configured", {"host_id": self.host_id, "baseline_mode": baseline_mode}, 1.0),
            ("ai_host_baseline_power_watts", {"host_id": self.host_id, "baseline_mode": baseline_mode}, baseline_value),
        ]
        for tariff in self._tariffs(today):
            output.append(
                (
                    "ai_electricity_tariff_jpy_per_kwh",
                    {"host_id": self.host_id, "tariff_id": str(tariff["id"]), "scenario": str(tariff.get("scenario", "actual"))},
                    rate_from_components(tariff["components_jpy_per_kwh"]),
                )
            )
        for price in self.pricing.get("prices", []):
            if not active_on_date(price, today):
                continue
            labels = {
                "price_id": str(price["id"]),
                "provider": str(price["provider"]),
                "remote_model": str(price["model"]),
                "scenario": str(price.get("scenario", "standard")),
            }
            for kind, field in (
                ("input", "input_usd_per_million_tokens"),
                ("cached_input", "cached_input_usd_per_million_tokens"),
                ("output", "output_usd_per_million_tokens"),
            ):
                if price.get(field) is not None:
                    output.append(("ai_api_price_usd_per_million_tokens", labels | {"kind": kind}, float(price[field])))
        for fx in self._fx_rates(today):
            output.append(("ai_fx_jpy_per_usd", {"fx_id": str(fx["id"])}, float(fx["jpy_per_usd"])))
        return output
