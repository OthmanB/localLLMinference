#!/usr/bin/python3
"""Install and verify Flash(0,1) + Qwen(2), preserving private gateway credentials.

Run --check without privileges to inspect preparation. Actual system deployment
requires root. On an unsuccessful promotion, restore prior files/unit states.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import pwd
import shlex
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any

from flash_preflight import EXPECTED, digest, gpu_map

OPS = Path(__file__).resolve().parent
ROOT = OPS.parent
SOURCE = ROOT / 'research/flash-reliability-2026-11-05/strata-candidate'
RELEASE = Path('/opt/ai-server/strata/ffa80d7')
FLASH = 'strata-flash-512k.service'
QWEN = 'llamampere-qwen3.8-atx-iq4xs-m-gpu2.service'
OLD = ['llama-muse-glimmer-30b-131k.service', 'llamampere-qwen3.8-atx-iq4xs-m-gpu1.service',
       'llama-qwen3.8-q4-tensor-262k.service']
PROFILE = 'flash512-qwen-single'
FLASH_MODEL = 'qwen3.8-flash-next-iq3_s'
QWEN_MODEL = 'qwen3.8-27b-atx-iq4xs-m-262144-gpu2'


def run(*args, check=True):
    return subprocess.run(list(args), check=check, text=True, capture_output=True, timeout=180)


def patch_env(text, updates):
    """Preserve credentials and unrelated settings byte-for-byte; replace only public keys."""
    lines, found = [], set()
    for line in text.splitlines(keepends=True):
        key = line.split('=', 1)[0] if '=' in line and not line.lstrip().startswith('#') else None
        if key is not None and key in updates:
            if key not in found:
                lines.append(key + '=' + updates[key] + '\n')
                found.add(key)
        else:
            lines.append(line)
    if lines and not lines[-1].endswith('\n'):
        lines[-1] += '\n'
    lines.extend(key + '=' + value + '\n' for key, value in updates.items() if key not in found)
    return ''.join(lines)


def profile_values():
    backends = [dict(name='flash_iq3s_512k', base_url='http://127.0.0.1:8082',
                     models=[FLASH_MODEL], health_path='/health'),
                dict(name='qwen_atx_gpu2', base_url='http://127.0.0.1:8081',
                     models=[QWEN_MODEL], health_path='/health')]
    pools = [dict(name='qwen3.8-27b-atx-iq4xs-m-262144-pool', model='qwen3.8-27b-atx-iq4xs-m-262144',
                  replicas=['qwen_atx_gpu2'], session_header='X-Inference-Session', max_inflight=1)]
    metrics = dict(flash_iq3s_512k=dict(url='http://127.0.0.1:8082/metrics/prometheus',
        model=FLASH_MODEL, gpus=['0', '1'], runtime='Strata', cached_input_mode='observed'),
        qwen_atx_gpu2=dict(url='http://127.0.0.1:8081/metrics', model=QWEN_MODEL,
                          gpus=['2'], runtime='llamAmpere', cached_input_mode='unobserved'))
    compact = lambda value: json.dumps(value, separators=(',', ':'))
    return ({'LAN_INFERENCE_BACKENDS': compact(backends), 'LAN_INFERENCE_POOLS': compact(pools)},
            {'AI_SERVING_PROFILE': PROFILE, 'AI_METRICS_BACKENDS': compact(metrics)})


def atomic_write(path, content, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
    try:
        with os.fdopen(fd, 'w') as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(name, mode)
        if mode == 0o640:
            os.chown(name, 0, pwd.getpwnam('obenomar').pw_gid)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def install_runtime():
    if RELEASE.exists():
        manifest = json.loads((RELEASE / 'release-manifest.json').read_text())
        if manifest['source_binary_sha256'] != digest(SOURCE / 'build-campaign/strata'):
            raise RuntimeError('Different runtime already occupies the pinned release path')
        for name, expected in manifest['files'].items():
            if digest(RELEASE / name) != expected:
                raise RuntimeError(f'Existing pinned release changed: {name}')
        return
    RELEASE.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=RELEASE.parent, prefix='.ffa80d7-'))
    try:
        for name in ('serve', 'strata_control', 'strata_cli', 'tools'):
            shutil.copytree(SOURCE / name, staging / name,
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'), symlinks=False)
        for path in SOURCE.iterdir():
            if path.is_file() and path.suffix in ('.py', '.json', '.toml', '.txt'):
                shutil.copy2(path, staging / path.name)
        (staging / 'build-campaign').mkdir()
        for name in ('strata', 'libstrata_kq.so'):
            shutil.copy2(SOURCE / 'build-campaign' / name, staging / 'build-campaign' / name)
        files = {str(p.relative_to(staging)): digest(p) for p in staging.rglob('*') if p.is_file()}
        for path in staging.rglob('*'):
            os.chmod(path, 0o755 if path.is_dir() or path.name == 'strata' else 0o644)
        os.chmod(staging, 0o755)
        (staging / 'release-manifest.json').write_text(json.dumps(dict(
            commit='ffa80d73133ea5c548aefe57889a16e8085a938f',
            source_binary_sha256=digest(SOURCE / 'build-campaign/strata'), files=files), indent=2) + '\n')
        staging.rename(RELEASE)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def read_env(path):
    values = {}
    for line in path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            parsed = shlex.split(value)
            values[key] = parsed[0] if len(parsed) == 1 else value
    return values


def http(url, token=None, body=None, *, timeout: float = 60):
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    request = urllib.request.Request(url, headers=headers,
        data=json.dumps(body).encode() if body is not None else None)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_http(url, token=None, *, seconds: float = 120, ready=None, poll_seconds: float = 1):
    """A started Type=simple process need not have bound its HTTP socket yet."""
    until = time.monotonic() + seconds
    last_error = 'endpoint not ready'
    while True:
        left = until - time.monotonic()
        if left <= 0:
            raise TimeoutError(f'{url} did not become ready within {seconds}s: {last_error}')
        try:
            value = http(url, token, timeout=min(5, left))
            if ready is None or ready(value):
                return value
            last_error = 'endpoint responded but readiness condition is not met'
        except urllib.error.HTTPError as exc:
            # Retry transient startup/unavailable responses, not invalid tokens
            # or permanently incorrect routes/configuration.
            if exc.code not in (429, 500, 502, 503, 504):
                raise
            last_error = f'HTTP {exc.code}'
        except (OSError, ValueError) as exc:
            last_error = str(exc)
        time.sleep(min(poll_seconds, max(0, until - time.monotonic())))


def wait_health(port, capacity=None):
    return wait_http(f'http://127.0.0.1:{port}/health', seconds=1200, poll_seconds=3,
        ready=lambda value: value.get('status') == 'ok' and value.get('loaded', True)
        and (capacity is None or value.get('max_context') == capacity))


def check_inputs():
    cfg = json.loads((OPS / 'config/flash-512k.json').read_text())
    for flag in ('--native', '--ple-gguf', '--expert-profile', '--pack', '--mtp'):
        path = Path(cfg['args'][cfg['args'].index(flag) + 1])
        if not path.exists():
            raise FileNotFoundError(path)
    for path in (SOURCE / 'build-campaign/strata', SOURCE / 'build-campaign/libstrata_kq.so',
                 Path('/home/obenomar/.local/share/strata-flash-next-20261014/.venv/bin/python')):
        if not path.exists():
            raise FileNotFoundError(path)
    if gpu_map() != EXPECTED:
        raise RuntimeError('GPU index/UUID allocation differs from the validated mapping')
    if cfg['gpu'] != [0, 1] or cfg['parallel'] != 1 or cfg['env']['CUDA_VISIBLE_DEVICES'].split(',') != [EXPECTED['0'], EXPECTED['1']]:
        raise ValueError('Flash GPU or slot configuration drift')
    print('Prepared: Flash GPUs 0,1 / 524288 tokens / one session; Qwen GPU 2 / 262144 tokens.', flush=True)


def promote():
    check_inputs()
    etc = Path('/etc/ai-server')
    for name in ('lan-inference-gateway.env', 'ai-metrics-exporter.env'):
        if not (etc / name).exists():
            raise FileNotFoundError(etc / name)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = Path('/var/backups/ai-server') / ('flash512-' + stamp)
    backup.mkdir(parents=True, mode=0o700)
    paths = [etc / 'flash-512k.json', etc / 'lan-inference-gateway.env', etc / 'ai-metrics-exporter.env',
        etc / 'qwen-serving-profile', Path('/etc/systemd/system') / FLASH,
        Path('/usr/local/sbin/ai-qwen-profile-preflight'), Path('/usr/local/sbin/ai-flash-preflight'),
        Path('/etc/systemd/system') / (QWEN + '.d/20-physical-gpu.conf'),
        Path('/etc/systemd/system/lan-inference-gateway.service.d/20-flash.conf'),
        Path('/etc/systemd/system/ai-metrics-exporter.service.d/20-flash.conf')]
    saved = []
    for n, path in enumerate(paths):
        record: dict[str, Any] = dict(path=str(path), existed=path.exists())
        if path.exists():
            record.update(mode=path.stat().st_mode & 0o777, uid=path.stat().st_uid, gid=path.stat().st_gid)
            shutil.copy2(path, backup / str(n))
            os.chmod(backup / str(n), 0o600)
        saved.append(record)
    states = {unit: dict(enabled=run('systemctl', 'is-enabled', unit, check=False).stdout.strip(),
                         active=run('systemctl', 'is-active', unit, check=False).stdout.strip())
              for unit in [FLASH, QWEN, *OLD]}
    (backup / 'manifest.json').write_text(json.dumps(dict(files=saved, units=states), indent=2) + '\n')
    print('Backup: ' + str(backup), flush=True)
    try:
        install_runtime()
        for unit in OLD:
            run('systemctl', 'disable', '--now', unit)
        # Reapplying the selected profile may replace an already-running Flash.
        run('systemctl', 'stop', FLASH, check=False)
        used = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                        '--format=csv,noheader,nounits'], text=True)
        for line in used.splitlines():
            index, mib = line.split(',')
            if index.strip() in ('0', '1') and int(mib.strip()) > 256:
                raise RuntimeError('Flash GPUs still occupied after stopping old units')
        atomic_write(etc / 'flash-512k.json', (OPS / 'config/flash-512k.json').read_text(), 0o640)
        atomic_write(Path('/etc/systemd/system') / FLASH, (OPS / 'systemd' / FLASH).read_text(), 0o644)
        atomic_write(Path('/usr/local/sbin/ai-flash-preflight'), (OPS / 'flash_preflight.py').read_text(), 0o755)
        atomic_write(Path('/usr/local/sbin/ai-qwen-profile-preflight'), (OPS / 'qwen-profile-preflight.sh').read_text(), 0o755)
        atomic_write(Path('/etc/systemd/system') / (QWEN + '.d/20-physical-gpu.conf'),
            '[Service]\nEnvironment=CUDA_VISIBLE_DEVICES=' + EXPECTED['2'] + '\n', 0o644)
        for name in ('lan-inference-gateway', 'ai-metrics-exporter'):
            atomic_write(Path('/etc/systemd/system') / f'{name}.service.d/20-flash.conf',
                         f'[Unit]\nAfter={FLASH} {QWEN}\n', 0o644)
        gateway, metrics = profile_values()
        atomic_write(etc / 'lan-inference-gateway.env', patch_env((etc / 'lan-inference-gateway.env').read_text(), gateway), 0o600)
        atomic_write(etc / 'ai-metrics-exporter.env', patch_env((etc / 'ai-metrics-exporter.env').read_text(), metrics), 0o640)
        atomic_write(etc / 'qwen-serving-profile', PROFILE + '\n', 0o644)
        run('systemd-analyze', 'verify', '/etc/systemd/system/' + FLASH)
        run('systemctl', 'daemon-reload')
        run('systemctl', 'enable', FLASH, QWEN)
        print('Starting Qwen on GPU 2...', flush=True)
        run('systemctl', 'start', QWEN)
        wait_health(8081)
        print('Starting Flash on GPUs 0,1 (model loading may take a few minutes)...', flush=True)
        run('systemctl', 'start', FLASH)
        flash_health = wait_health(8082, 524288)
        qwen_slots = http('http://127.0.0.1:8081/slots')
        if not isinstance(qwen_slots, list) or len(qwen_slots) != 1 or qwen_slots[0].get('n_ctx') != 262144:
            raise RuntimeError('Qwen single-slot 262144 capacity verification failed')
        run('systemctl', 'restart', 'lan-inference-gateway.service', 'ai-metrics-exporter.service')
        env = read_env(etc / 'lan-inference-gateway.env')
        token = env[env.get('LAN_INFERENCE_CLIENT_API_KEY_ENV', 'LAN_GATEWAY_CLIENT_TOKEN')]
        print('Waiting for gateway HTTP startup and backend readiness...', flush=True)
        wait_http('http://127.0.0.1:8088/healthz', ready=lambda value: value.get('status') == 'ok')
        wait_http('http://127.0.0.1:8088/readyz', token, ready=lambda value: value.get('status') == 'ready')
        models = http('http://127.0.0.1:8088/v1/models', token)
        ids = {x['id'] for x in models['data']}
        if not {FLASH_MODEL, QWEN_MODEL, 'qwen3.8-27b-atx-iq4xs-m-262144'} <= ids:
            raise RuntimeError('Gateway does not publish both intended services')
        for model in (FLASH_MODEL, QWEN_MODEL):
            reply = http('http://127.0.0.1:8088/v1/chat/completions', token, dict(
                model=model, messages=[dict(role='user', content='Reply with exactly READY.')],
                max_tokens=32, temperature=0, reasoning_effort='none', chat_template_kwargs={'enable_thinking': False}))
            if 'READY' not in (reply['choices'][0]['message'].get('content') or ''):
                raise RuntimeError('Gateway completion check failed for ' + model)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,process_name',
                                        '--format=csv,noheader'], text=True)
        flash_gpus, qwen_gpus = set(), set()
        for line in apps.splitlines():
            gpu, process = (x.strip() for x in line.split(',', 1))
            if process == str(RELEASE / 'build-campaign/strata'):
                flash_gpus.add(gpu)
            if process.startswith('/opt/ai-server/llamampere/') and process.endswith('/llama-server'):
                qwen_gpus.add(gpu)
        if flash_gpus != {EXPECTED['0'], EXPECTED['1']} or qwen_gpus != {EXPECTED['2']}:
            raise RuntimeError('Actual model process GPU placement differs from requested allocation')
        for unit in (FLASH, QWEN):
            if run('systemctl', 'is-enabled', unit).stdout.strip() != 'enabled':
                raise RuntimeError('Boot persistence verification failed: ' + unit)
        record = dict(profile=PROFILE, flash_health=flash_health, qwen_slots=qwen_slots,
                      models=sorted(ids), flash_gpu_uuids=sorted(flash_gpus), qwen_gpu_uuids=sorted(qwen_gpus),
                      backup=str(backup), verified_at=datetime.now(timezone.utc).isoformat())
        atomic_write(Path('/var/lib/ai-server/flash-deployment.json'), json.dumps(record, indent=2) + '\n', 0o644)
        print('SUCCESS: Flash 512K on GPUs 0,1; Qwen 262K on GPU 2; both verified through gateway and enabled at boot.', flush=True)
    except Exception:
        print('Promotion failed; restoring previous files and boot states. Evidence/backup: ' + str(backup), flush=True)
        run('systemctl', 'stop', FLASH, QWEN, check=False)
        for n, record in enumerate(saved):
            path = Path(record['path'])
            if record['existed']:
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup / str(n), path)
                os.chmod(path, record['mode'])
                os.chown(path, record['uid'], record['gid'])
            elif path.exists():
                path.unlink()
        run('systemctl', 'daemon-reload')
        for unit, state in states.items():
            run('systemctl', 'enable' if state['enabled'] == 'enabled' else 'disable', unit, check=False)
            if state['active'] == 'active':
                run('systemctl', 'start', unit, check=False)
        run('systemctl', 'restart', 'lan-inference-gateway.service', 'ai-metrics-exporter.service', check=False)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        check_inputs()
        print('Check only: applying boot services and private gateway settings requires root.')
        return
    if os.geteuid() != 0:
        raise SystemExit('Run with sudo: python3 ' + str(Path(__file__).resolve()))
    with Path('/run/ai-server-install.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        promote()


if __name__ == '__main__':
    main()
