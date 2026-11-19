# Flash 512K + Qwen single-GPU production promotion

## Selected allocation

| Physical GPU | UUID | Service | Context | Backend |
|---|---|---|---:|---|
| 0 and 1 | `GPU-8c1e5d04-43a0-dba7-ebf6-c623bfa2f325`, `GPU-bd704e22-5ba1-e0c3-0f66-309a6104b367` | `strata-flash-512k.service` | 524288, one active session | `127.0.0.1:8082` |
| 2 | `GPU-ae2d59f3-4764-179b-5e78-e5e9c320e744` | `llamampere-qwen3.8-atx-iq4xs-m-gpu2.service` | 262144, one active session | `127.0.0.1:8081` |

This preserves the validated two-GPU Flash allocation. At preparation time the
driver reports negotiated widths x16 / x8 / x8 for GPUs0 / 1 / 2, not two x16
links. The units pin UUIDs and the Flash preflight checks index/UUID identity.

## Promotion

Prepare/check without privileges:

```sh
python3 /home/obenomar/localLLMinference/operations/promote_flash_512k.py --check
```

Apply the system-level boot replacement:

```sh
sudo python3 /home/obenomar/localLLMinference/operations/promote_flash_512k.py
```

The installer:

- Backs up existing private configuration and service states under
  `/var/backups/ai-server/flash512-<UTC timestamp>/` (root-only).
- Copies the tested Strata runtime into `/opt/ai-server/strata/ffa80d7/`, records
  file hashes, and verifies it at Flash startup. Model weights are reused.
- Disables/stops Muse and Qwen GPU1/old tensor service, removing boot conflicts.
- Installs the Flash unit and pins the remaining Qwen unit to GPU2's UUID.
- Preserves the gateway credential and unrelated environment settings while
  replacing the backend list with Flash plus the remaining Qwen replica. The
  generic Qwen pool remains available with one member.
- Updates monitoring to attribute Flash to GPUs0/1 and Qwen to GPU2. Host power,
  tariff and pricing calibration files remain their existing deployment values.
- Enables and starts both model services, verifies context/slots, gateway model
  discovery/readiness and small completion requests, and checks actual GPU
  process placement. No large-context benchmarks are run.
- Restores prior files/unit states if promotion fails. Original backups remain.

Models published through the existing authenticated gateway on port8088:

- `qwen3.8-flash-next-iq3_s`
- `qwen3.8-27b-atx-iq4xs-m-262144-gpu2`
- `qwen3.8-27b-atx-iq4xs-m-262144` (single-member pool)

Completion record: `/var/lib/ai-server/flash-deployment.json`.
Logs: `/var/log/ai-server/flash-512k-{server,engine}.log`.

## Flash thinking modes

The Flash template implements `low`, `medium`, and `xhigh`. Strata accepts
`high` and maps it to `xhigh`; these are not two different model levels.
`none` disables thinking. API requests may select any of those names.

The persisted server configuration uses `reasoning_effort: high`, because
Strata 0.1.56's configuration validator accepts `high` but rejects `xhigh` in
that particular field. At request time both spellings map to xhigh. With no
hardcoded `sampling` block, Strata selects the model-card defaults appropriate
to thinking or non-thinking requests.

For an already-installed service, apply the corrected default using:

```sh
sudo python3 /home/obenomar/localLLMinference/operations/set_flash_thinking.py
```

This backs up the root-owned config, restarts Flash only, and checks that a
default request returns thinking. It leaves Qwen and GPU allocation unchanged.
The OMP and OpenCode clients explicitly select xhigh; their requests override
the previous server default even before the persisted server file is updated.

## Promotion status

The system-level replacement has been applied. Gateway `/healthz` reports
exactly `flash_iq3s_512k` and `qwen_atx_gpu2` (verified 2026-11-19), and the
generic Qwen pool exposes the single `qwen_atx_gpu2` member. Muse, the GPU1
llamAmpere replica, and the stock tensor-split service are disabled at boot and
guarded by the Flash unit's `Conflicts=` list. The completion record is
`/var/lib/ai-server/flash-deployment.json` on the server; pre-promotion private
state is under `/var/backups/ai-server/flash512-<UTC timestamp>/`. The research
supervisors and overnight timer were disabled in the user scope.
