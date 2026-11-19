#!/usr/bin/python3
"""Persist Flash's highest supported thinking default without changing its GPUs.

Run with sudo. Backs up the deployed config, applies Strata's `high` server
setting (the Flash template maps it to xhigh), and verifies the restarted server.
"""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import urllib.request

CONFIG = Path('/etc/ai-server/flash-512k.json')
SHARED = CONFIG.with_suffix('').with_suffix('.shared-settings.json')
SERVICE = 'strata-flash-512k.service'
OLD_SAMPLING = {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'presence_penalty': 1.5}


def desired_config(config):
    updated = dict(config)
    if updated.get('sampling') not in (None, {}, OLD_SAMPLING):
        raise ValueError('Custom sampling settings detected; refusing to replace them automatically')
    updated['reasoning_effort'] = 'high'  # Config validator accepts high; template maps it to xhigh.
    updated.pop('sampling', None)          # Strata selects thinking/non-thinking model-card defaults.
    return updated


def atomic_json(path, value, stat):
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as target:
            json.dump(value, target, indent=2)
            target.write('\n')
            target.flush()
            os.fsync(target.fileno())
        os.chmod(name, stat.st_mode & 0o777)
        os.chown(name, stat.st_uid, stat.st_gid)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def request(path, body=None):
    req = urllib.request.Request('http://127.0.0.1:8082' + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def wait_ready():
    end = time.monotonic() + 1200
    while time.monotonic() < end:
        try:
            value = request('/health')
            if value.get('status') == 'ok' and value.get('loaded') and value.get('max_context') == 524288:
                return
        except (OSError, ValueError):
            pass
        time.sleep(3)
    raise TimeoutError('Flash did not return healthy at 524288 context')


def main():
    if os.geteuid() != 0:
        raise SystemExit('Run: sudo python3 /home/obenomar/localLLMinference/operations/set_flash_thinking.py')
    with Path('/run/ai-server-install.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = json.loads(CONFIG.read_text())
        new = desired_config(old)
        config_stat = CONFIG.stat()
        shared_exists = SHARED.exists()
        shared_old = json.loads(SHARED.read_text()) if shared_exists else None
        shared_stat = SHARED.stat() if shared_exists else None
        backup = Path('/var/backups/ai-server') / ('flash-thinking-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
        backup.mkdir(parents=True, mode=0o700)
        shutil.copy2(CONFIG, backup / CONFIG.name)
        if shared_exists:
            shutil.copy2(SHARED, backup / SHARED.name)
        print('Backup:', backup, flush=True)
        try:
            atomic_json(CONFIG, new, config_stat)
            if shared_exists:
                # Preserve unrelated user web-chat settings while lifting a stale off override.
                assert shared_old is not None and shared_stat is not None
                shared_new = dict(shared_old)
                shared_new['reasoning_effort'] = 'high'
                atomic_json(SHARED, shared_new, shared_stat)
            subprocess.run(['systemctl', 'restart', SERVICE], check=True, timeout=180)
            print('Waiting for Flash to load...', flush=True)
            wait_ready()
            settings = request('/settings')
            if settings.get('defaults', {}).get('reasoning_effort') not in (None, 'high'):
                raise RuntimeError('A shared setting still overrides the requested default')
            reply = request('/v1/chat/completions', dict(model='qwen3.8-flash-next-iq3_s',
                messages=[{'role': 'user', 'content': 'Compute 123 times 456 carefully, then answer.'}], max_tokens=1024))
            message = reply['choices'][0]['message']
            if not message.get('reasoning_content'):
                raise RuntimeError('Default request returned no thinking; inspect the response/server logs')
            print('SUCCESS: default is high (Flash xhigh); thinking response verified; model-card sampling is active.', flush=True)
        except Exception:
            print('Update failed; restoring the previous config and restarting Flash.', flush=True)
            atomic_json(CONFIG, old, config_stat)
            if shared_exists:
                assert shared_stat is not None
                atomic_json(SHARED, shared_old, shared_stat)
            subprocess.run(['systemctl', 'restart', SERVICE], check=False, timeout=180)
            raise


if __name__ == '__main__':
    main()
