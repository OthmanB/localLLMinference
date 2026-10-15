#!/usr/bin/env python3
"""Run the sealed additional cohort after primary scoring, within original deadlines."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.request

import run_campaign as runtime

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def status(root, stage, **kw):
    row = dict(stage=stage, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), **kw)
    runtime.atomic(root/'status.json', row)
    with (root/'progress.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--strata-root',type=Path,default=Path(os.getenv('STRATA_ROOT','~/.local/share/strata-flash-next-20261014')).expanduser())
    p.add_argument('--swe-python',default=os.getenv('SWEBENCH_PYTHON',str(Path('~/.local/share/swebench-venv/bin/python').expanduser())))
    a = p.parse_args()
    root = HERE/'extra-campaign'
    root.mkdir(mode=0o700,exist_ok=True)
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM,lambda *_:runtime.STOP.set())
        signal.signal(signal.SIGINT,lambda *_:runtime.STOP.set())
        primary = HERE/'runs/primary-single-instance'
        limits = read(primary/'ledger.json')
        generation_deadline = limits['generation_deadline_epoch']
        envelope_deadline = limits['envelope_deadline_epoch']
        status(root,'waiting',generation_deadline_epoch=generation_deadline,envelope_deadline_epoch=envelope_deadline)
        while True:
            if runtime.STOP.is_set() or time.time() >= generation_deadline:
                raise RuntimeError('Stopped or inherited generation deadline reached before expansion')
            primary_phase = read(primary/'ledger.json').get('phase')
            main_state = subprocess.run(['systemctl','--user','show','qwen27-flash-comparison-20261015.service',
                                         '-p','ActiveState','--value'],text=True,capture_output=True).stdout.strip()
            prep = read(HERE/'extra-cohort/preparation-status.json')
            ready_images = prep.get('cached_count',0) == 40
            if primary_phase == 'complete' and main_state not in ('active','activating','deactivating') and ready_images:
                break
            if main_state == 'failed':
                raise RuntimeError('Primary campaign failed; resolve original run before adding requests')
            status(root,'waiting',primary_phase=primary_phase,primary_unit=main_state,
                   cached_images=prep.get('cached_count',0),total_images=40)
            runtime.STOP.wait(60)
        if runtime.unit('llama-muse-glimmer-30b-131k.service','ActiveState') != 'inactive':
            raise RuntimeError('Muse holds GPU0; no automatic service handoff')
        runtime.RUN_FOR_NOTES = root
        pids = {}
        for index,name in enumerate(runtime.QWEN_UNITS,1):
            current = subprocess.check_output(['systemctl','cat',name],text=True)
            if current != (HERE/f'campaign/qwen-gpu{index}.service.txt').read_text():
                raise RuntimeError('Qwen serving settings differ from primary cohort')
            runtime.SERVICE_TEXTS[name] = current
            pids[name] = runtime.unit(name,'MainPID')
        runtime.protect(pids)
        cfg = read(HERE/'campaign/flash-native.json')
        cfg['log'] = str(root/'flash.engine.log')
        cfg_path = root/'flash-native.json'
        runtime.atomic(cfg_path,cfg)
        env = dict(os.environ, **cfg['env'], PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
                   HF_DATASETS_OFFLINE='1',HF_HUB_OFFLINE='1')
        py = str(a.strata_root/'.venv/bin/python')
        server = None
        stop_monitor = threading.Event()
        threading.Thread(target=runtime.monitor,args=(root,stop_monitor),daemon=True).start()
        extra = HERE/'runs/additional40-single-instance'
        try:
            runtime.command(root,'image-verification',[a.swe_python,str(HERE/'extra_prepare.py'),'check-images'],
                            env,envelope_deadline,pids)
            if not (extra/'ledger.json').exists():
                runtime.command(root,'initialize',[a.swe_python,str(HERE/'extra_swe.py'),'init','--run-dir',str(extra)],
                                env,envelope_deadline,pids)
            status(root,'loading-flash')
            with (root/'flash.server.log').open('a') as log:
                server = subprocess.Popen([py,'-m','serve.server','--engine','strata','--config',str(cfg_path),
                                           '--port','18144'],cwd=a.strata_root,env=env,stdout=log,
                                          stderr=subprocess.STDOUT,start_new_session=True)
            for _ in range(360):
                if server.poll() is not None:
                    raise RuntimeError('Additional-cohort Flash startup failed')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18144/health',timeout=5) as r:
                        if json.load(r).get('loaded'):
                            break
                except OSError:
                    pass
                if runtime.STOP.wait(5):
                    raise RuntimeError('Stopped during Flash load')
            else:
                raise RuntimeError('Flash startup timeout')
            cmd = [a.swe_python,str(HERE/'extra_swe.py'),'run','--run-dir',str(extra),'--swe-phase']
            if read(extra/'ledger.json')['phase'] != 'initialized':
                cmd.append('--resume')
            status(root,'additional-generation-and-scoring',tasks_per_model=40,combined_tasks_per_model=80)
            runtime.command(root,'swe',cmd,env,envelope_deadline,pids)
            subprocess.run([py,str(HERE/'combined_swe_report.py')],check=True,cwd=HERE,env=env)
            status(root,'complete',message='Additional cohort scored; combined report generated')
        except Exception as exc:
            status(root,'failed',error=str(exc))
            raise
        finally:
            runtime.stop_process(server)
            stop_monitor.set()


if __name__ == '__main__':
    main()
