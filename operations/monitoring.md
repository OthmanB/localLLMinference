# Monitoring

Prometheus and Grafana run on the monitoring host. The AI server exporter listens
on port 9108 and combines FreeToken, llama.cpp, `nvidia-smi`, host-memory, energy,
electricity-cost, and API-equivalent-cost metrics.

Expected model labels:

- `model="qwen3.8-flash-next-nvfp4-262k"`
- `gpu="0"`
- `model="qwen3.8-27b-q4-gpukv192"`
- `gpu="1"`
- `model="muse-glimmer-30b-kquant17"`
- `gpu="2"`

All exporter-owned GPU and accounting series also include immutable `host_id` and
`gpu_uuid` labels. Do not identify a GPU solely by the global GPU index.

The dashboard should show request processing, prefill tokens/s, decode tokens/s,
cumulative prompt/generated tokens, GPU utilization, measured power draw and limit,
temperature, VRAM, host memory, energy, electricity cost, API-equivalent costs,
and exporter/service health.

Throughput plateau semantics: runtimes report zero decode/prefill throughput
whenever their scheduler is idle. The exporter therefore retains the last nonzero
decode and prefill sample per model and keeps exporting it until the next nonzero
measurement replaces it. Panels render both series with `stepAfter` so the graph
shows a plateau between measurements rather than a zero baseline with one-second
spikes. After an exporter restart, the retained value starts at zero again until
the model completes its first new batch.

Host memory metrics are exposed as:

- `ai_host_memory_total_bytes`
- `ai_host_memory_available_bytes`
- `ai_host_memory_used_bytes`

The Prometheus scrape job is in `config/prometheus-ai-server.yml`. Grafana
should use the existing Prometheus data source and import
`config/grafana-ai-server-dashboard.json` into dashboard UID
`ai-server-qwen-q4`. The authenticated API procedure is documented in
`deployment-record.md`; verify the resulting dashboard at
`/d/ai-server-qwen-q4`.

After changing the exporter or dashboard source files, restart the exporter and
re-import the dashboard with overwrite enabled:

```bash
sudo systemctl restart ai-metrics-exporter.service
```

Run `sudo operations/promote-freetoken-flash-next-262k.sh` on the AI server to
install the permanent service and local gateway/exporter configuration. Then
import the dashboard source on the monitoring host with overwrite enabled.

## CPU Power Attribution

`tools/ai_cpu_power_profiler.py` runs as the always-on
`ai-cpu-power-profiler.service` and listens on port 9109. Every
`sampling_interval_seconds` (default 10 s) it reads the AMD/Intel RAPL
`package-0` energy counter delta and the `/proc/stat` utilization delta. Each
`(utilization, watts)` pair is stored in one of twenty 5-point utilization
buckets, and each bucket keeps a moving window of the last
`bucket_window_samples` (default 20) samples, dropping the oldest. Median,
standard deviation, maximum, and count per bucket are exposed as
`ai_cpu_power_bucket_*` metrics, which lets the power-versus-utilization curve
be learned passively during normal operation.

RAPL energy counters are root-only (sysfs mode `0440`) and the powercap domain
can boot disabled. udev cannot change sysfs attribute permissions, so the
profiler unit fixes both at start with root-privileged `ExecStartPre=+` lines:
it writes `1` to the domain `enabled` files and then `chmod o+r` the
`energy_uj` counters, while the service itself stays unprivileged. Run
`sudo operations/install-cpu-power-profiler.sh` once to install the unit and
the config from `config/cpu-power-profiler.json.example`.
`ai_cpu_profiler_rapl_available` is 1 when the RAPL `package-0` energy counter
is readable and exposes a nonzero `max_energy_range_uj`; it deliberately does
not require the powercap `enabled` flag to be nonzero. Some kernels report
`enabled=0` while the counter still accumulates (observed on this host), so
gating counter reads on the flag would silently disable working RAPL. The flag
is still exported for diagnostics as `ai_cpu_profiler_rapl_enabled` (1 when it
reads nonzero). If the counter is unreadable the metric is 0 and it re-arms on
the next sample once the counter becomes readable.

The main exporter resolves CPU package power per scrape in this order
(`ai_host_cpu_power_source_info` reports the selection):

1. `rapl`: live package power from the profiler endpoint.
2. `interpolated`: piecewise-linear interpolation across the learned bucket
   medians at the current utilization.
3. `linear`: `min(max_watts, max_watts x utilization / saturate_utilization_percent)`
   from the `cpu_power.linear` block of the cost config, a rough bootstrap model.
4. `unavailable`: the CPU term is treated as 0 W for that interval, so the
   whole-host estimate is a lower bound (only when RAPL is down, no learned
   bucket data exists, and no `cpu_power.linear` fallback is configured).

With the baseline mode `wall_idle_plus_cpu_w` (`base_idle_total_w` instead of
`wall_idle_total_w`), whole-host power becomes `base_idle_total_w +
cpu_power_w + sum(max(gpu_draw - idle_reference, 0))`, so idle periods are no
longer charged the old flat 300 W and GPU-resident models add no CPU term. The
selected split is exposed as `ai_host_power_attribution_watts{source=...}` and
CPU energy accumulates in `ai_host_cpu_energy_joules_total`. Older flat modes
keep their original semantics and ignore CPU power for host estimates while
still recording `ai_host_cpu_energy_joules_total`.

After switching modes, re-measure the base: with the host otherwise idle
(inference clients routed away from this machine), set `base_idle_total_w` to
`UPS_watts - cpu_package_watts - sum(max(gpu_draw - idle_reference, 0))` and record
`measured_at`. Because GPU power enters the estimate as *excess* above the idle
reference, the idle GPUs stay inside the base; at rest they sit at their
references, so the last term is ~0 and the base is just `UPS_watts -
cpu_package_watts` (e.g. 200 - 73 = 127 W). The validation load sequence runs
the load probes only, because the assistant's own model would otherwise pollute
the CPU measurement.

## Energy and Cost Accounting

The exporter samples actual `nvidia-smi` `power.draw` every five seconds and uses
timestamped trapezoidal integration. `power.limit` is exposed for operations but
is never treated as consumption. State is atomically retained at
`/var/lib/ai-metrics-exporter/cost-accounting-state.json`; runtime counter resets
are interpreted as a new counter value rather than a negative token delta.

`/etc/ai-server/ai-cost-accounting.json` is installed from
`config/ai-cost-accounting.json.example`. Its optional `timezone` key selects the
calendar used by the day/week/month cost views; the current value is
`Asia/Tokyo`. Its baseline mode is `wall_idle_plus_cpu_w` with `base_idle_total_w`
of 127 W (measured 2026-09-08): the 200 W rest wall draw minus the ~73 W CPU
package floor, so the base covers the platform (chassis fans, motherboard, RAM,
disks) plus the three idle GPUs. Whole-host power is therefore `127 +
cpu_power_w + sum(max(actual GPU draw - 20, 0))` W. This is a host-level
allocation and must not be presented as a per-model inference cost.

The electricity tariff and API prices are explicit, dated configuration inputs.
The installed files are readable by the exporter service account but writable only
by root. The current TEPCO scenario excludes the temporary subsidy and the fixed
monthly basic charge. The JPY API figures use the configurable planning FX rate,
not a billed exchange rate. Update the source JSON files, review the price source
and effective date, then explicitly install the reviewed file into `/etc/ai-server`.
The installer and promotion scripts create missing config files but preserve
existing local calibration and pricing inputs.

Key counters:

- `ai_gpu_energy_joules_total`: actual energy for each UUID with consecutive valid samples.
- `ai_model_active_gpu_energy_joules_total`: direct GPU energy only when the named model was active in both endpoint samples.
- `ai_host_estimated_energy_joules_total`: full host estimate, integrated only when the entire GPU set is available.
- `ai_energy_integration_gap_seconds_total`: elapsed time intentionally excluded from the whole-host estimate because samples were missing, late, or incomplete.
- `ai_host_reboot_downtime_seconds_total`: elapsed time between samples across a confirmed changed Linux boot ID; it is not charged to the host-energy estimate.
- `ai_host_boot_time_seconds`: Unix time of the current Linux host boot. Grafana's `dateTimeAsIso` unit expects milliseconds, so the dashboard multiplies this metric by 1000.
- `ai_host_electricity_cost_jpy_today`, `_week_to_date`, and `_month_to_date`: host electricity cost accumulated since the current calendar day, Monday, and month start in the configured timezone, each with a `period` label and a reset at the next boundary. Because the whole-host series contains the idle baseline, these curves ramp linearly while idle and steepen while GPUs are active. They exclude the fixed monthly basic charge.
- `ai_model_energy_covered_completion_tokens_total`: completion-token denominator paired with observed active-energy intervals. This includes terminal token deltas reported when a request finishes after validated active samples.
- `ai_host_electricity_cost_jpy_total` and `ai_model_active_gpu_electricity_cost_jpy_total`: host and direct-GPU electricity views respectively.
- `ai_model_api_workload_cost_{usd,jpy}_total`: token-priced remote API comparison including input, cached input, and output.
- `ai_model_api_output_only_cost_{usd,jpy}_total`: output-only remote API comparison.

The FreeToken endpoint does not expose a cached-input token counter. Its
API-comparison metrics carry `cached_input_mode="unobserved_assumed_uncached"` and
must not be interpreted as a measured cache discount. For runtimes that report
both counters, cached prompt tokens are treated as a subset of total prompt tokens
and replace the corresponding regular-input rate. Prices not represented in
`ai-api-pricing.json`, including unverified Qwen Model Studio and RunInfra Flash
Next rates, are intentionally absent rather than estimated.

After each exporter restart, successfully scraped configured models emit zero-valued
active-energy and API-comparison counters before their first request. This keeps
cost panels visible without claiming that direct inference energy was measured.
The direct-GPU JPY-per-1K panel intentionally remains unavailable until a
completion has matching active-energy coverage.

When the AI server is off, its exporter cannot emit a local zero sample. The
monitoring host therefore provides `up{job="ai-server"}`: `0` means the target
could not be scraped, whether from power-off, network loss, or exporter failure.
On its next successful sample, a changed Linux boot ID records the interval in
`ai_host_reboot_downtime_seconds_total` instead of
`ai_energy_integration_gap_seconds_total`; neither counter adds the 300 W host
baseline. A changed boot ID confirms a reboot, not whether it was deliberate or
caused by power loss.

The dashboard's selected-period counters use `last_over_time(...) -
first_over_time(...)` rather than `increase(...)`. This reports the exact observed
counter delta in the selected window and avoids Prometheus boundary extrapolation
when a new exporter series starts partway through that window.

Because that view depends on the selected dashboard time range, the
`Whole-Host Electricity Cost` stat is not a daily or billing total. At the idle
300 W baseline and the current TEPCO scenario (33.71 JPY/kWh), the host costs
roughly 10 JPY per hour. For calendar totals use the `Host Cost Today`,
`Host Cost This Week`, and `Host Cost This Month` panels and the `Host
Electricity Cost To Date` time series, which reset at their own JST boundaries.

Useful PromQL range queries:

```promql
# Measured active direct-GPU electricity cost for one model and tariff.
last_over_time(ai_model_active_gpu_electricity_cost_jpy_total{host_id="ai-server",model="muse-glimmer-30b-kquant17"}[$__range])
- first_over_time(ai_model_active_gpu_electricity_cost_jpy_total{host_id="ai-server",model="muse-glimmer-30b-kquant17"}[$__range])

# Full host electricity cost. Do not attribute this number to one model.
last_over_time(ai_host_electricity_cost_jpy_total{host_id="ai-server"}[$__range])
- first_over_time(ai_host_electricity_cost_jpy_total{host_id="ai-server"}[$__range])

# Equivalent remote API workload cost under each configured comparison and FX scenario.
last_over_time(ai_model_api_workload_cost_jpy_total{host_id="ai-server",model="muse-glimmer-30b-kquant17"}[$__range])
- first_over_time(ai_model_api_workload_cost_jpy_total{host_id="ai-server",model="muse-glimmer-30b-kquant17"}[$__range])

# Direct-GPU JPY per 1,000 completions with matching energy coverage.
1000 * sum by (model, tariff_id) (
  last_over_time(ai_model_active_gpu_electricity_cost_jpy_total{host_id="ai-server"}[$__range])
  - first_over_time(ai_model_active_gpu_electricity_cost_jpy_total{host_id="ai-server"}[$__range])
)
/ on (model) group_left sum by (model) (
  last_over_time(ai_model_energy_covered_completion_tokens_total{host_id="ai-server"}[$__range])
  - first_over_time(ai_model_energy_covered_completion_tokens_total{host_id="ai-server"}[$__range])
)
```
