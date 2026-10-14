#!/usr/bin/env python3
"""Durable 30h campaign: paired supplements then paired SWE, existing Qwen unchanged."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import psutil

HERE = Path(__file__).resolve().parent
STOP = threading.Event()
QWEN_UNITS = ['llamampere-qwen3.8-atx-iq4xs-m-gpu1.service',
              'llamampere-qwen3.8-atx-iq4xs-m-gpu2.service']


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)


def unit(unit_name, prop):
    return subprocess.check_output(['systemctl', 'show', unit_name, '-p', prop, '--value'], text=True).strip()


def note(run, phase, **details):
    row = dict(phase=phase, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **details)
    atomic(run / 'campaign-status.json', row)
    with (run / 'campaign-progress.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def protect(pids):
    for name, pid in pids.items():
        if unit(name, 'MainPID') != pid or unit(name, 'ActiveState') != 'active':
            raise RuntimeError('Deployed Qwen service changed: ' + name)


def stop_process(process):
    if process and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=150)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def monitor(run, stop):
    with (run / 'resources.jsonl').open('a') as out:
        while not stop.is_set():
            try:
                gpu = subprocess.check_output(['nvidia-smi',
                    '--query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu',
                    '--format=csv,noheader,nounits'], text=True).strip()
                memory = psutil.virtual_memory()
                out.write(json.dumps(dict(epoch=time.time(), gpu=gpu,
                                           available_ram_bytes=memory.available,
                                           host_swap_bytes=psutil.swap_memory().used)) + '\n')
                out.flush()
                temperatures = [float(line.split(',')[5]) for line in gpu.splitlines()]
                if max(temperatures) >= 88 or memory.available < 8 * 1024**3:
                    note(run,'resource-stop',temperatures=temperatures,available_ram_bytes=memory.available)
                    STOP.set()
                    return
            except subprocess.SubprocessError:
                pass
            stop.wait(5)


def command(run, phase, cmd, env, deadline, pids, allow_incomplete=False):
    note(run, phase, remaining_seconds=round(deadline-time.time()))
    with (run / f'{phase}.log').open('a') as log:
        child = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            while child.poll() is None:
                protect(pids)
                if STOP.is_set() or time.time() >= deadline:
                    stop_process(child)
                    raise TimeoutError(f'{phase} interrupted/deadline reached')
                time.sleep(10)
            if allow_incomplete and child.returncode == 2:
                note(run,phase+'-incomplete',message='Partial coverage retained; proceeding to reserved SWE phase.')
            elif child.returncode:
                raise RuntimeError(f'{phase} exited {child.returncode}; see {phase}.log')
            return child.returncode
        finally:
            stop_process(child)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, default=HERE / 'campaign')
    p.add_argument('--strata-root', type=Path, default=Path(os.getenv('STRATA_ROOT', '~/.local/share/strata-flash-next-20261014')).expanduser())
    p.add_argument('--evalplus-python', default=os.getenv('EVALPLUS_PYTHON', str(Path('~/.local/share/evalplus-venv/bin/python').expanduser())))
    p.add_argument('--swe-python', default=os.getenv('SWEBENCH_PYTHON', str(Path('~/.local/share/swebench-venv/bin/python').expanduser())))
    a = p.parse_args()
    run = a.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    os.chmod(run, 0o700)
    with (run / 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, lambda *_: STOP.set())
        signal.signal(signal.SIGINT, lambda *_: STOP.set())
        if unit('llama-muse-glimmer-30b-131k.service', 'ActiveState') != 'inactive':
            raise RuntimeError('Muse must remain stopped for Flash GPU0')
        pids = {name: unit(name, 'MainPID') for name in QWEN_UNITS}
        protect(pids)
        origin = run / 'campaign-origin.json'
        if not origin.exists():
            atomic(origin, dict(start_epoch=time.time(), generation_deadline_epoch=time.time()+24*3600,
                                deadline_epoch=time.time()+30*3600, qwen_pids=pids))
        limits = json.loads(origin.read_text())
        deadline = limits['deadline_epoch']
        cfg = json.loads((a.strata_root / 'strata-iq3_s.json').read_text())
        cfg.update(gpu=0, parallel=1, host='127.0.0.1', port=18144, reasoning_effort='none',
                   log=str(run / 'flash.engine.log'))
        cfg.pop('layer_split', None)
        args = cfg['args']
        for option, value in [('--max-context','262144'), ('--kv','int8'), ('--kv-resident','32768'),
                              ('--vram-reserve-mib','1536'), ('--rope-scaling','none'), ('--rope-scale','1')]:
            if option in args:
                args[args.index(option)+1] = value
            else:
                args.extend([option, value])
        if '--layer-split' in args or '--gpus' in args:
            raise RuntimeError('Refusing multi-GPU Flash configuration')
        cfg['env'] = dict(CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                          STRATA_ROUTE_TAIL_SKIP='0', STRATA_ADAPT_OVERLAP='0',
                          STRATA_HC_Q8='0', STRATA_MMQ_KQ='0')
        config_path = run / 'flash-native.json'
        atomic(config_path, cfg)
        env = dict(os.environ, **cfg['env'], STRATA_ROOT=str(a.strata_root),
                   STRATA_TOKENIZER=cfg['tokenizer'], EVALPLUS_PYTHON=a.evalplus_python,
                   PYTHONUNBUFFERED='1', HF_DATASETS_OFFLINE='1', HF_HUB_OFFLINE='1',
                   PYTHONDONTWRITEBYTECODE='1')
        strata_python = str(a.strata_root / '.venv/bin/python')
        stop = threading.Event()
        threading.Thread(target=monitor, args=(run, stop), daemon=True).start()
        server = None
        try:
            note(run, 'loading-flash', deadline_epoch=deadline)
            with (run / 'flash.server.log').open('a') as log:
                server = subprocess.Popen([strata_python, '-m', 'serve.server', '--engine','strata',
                                           '--config',str(config_path),'--port','18144'],
                                          cwd=a.strata_root, env=env, stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
            for _ in range(360):
                if server.poll() is not None:
                    raise RuntimeError('Flash startup failed')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18144/health', timeout=5) as r:
                        health = json.load(r)
                    if health.get('loaded'):
                        atomic(run / 'flash-health.json', health)
                        break
                except OSError:
                    pass
                if STOP.wait(5):
                    raise RuntimeError('Stopped during load')
            else:
                raise RuntimeError('Flash startup timeout')
            # Preserve exact active deployments before any agent traffic.
            for index, name in enumerate(QWEN_UNITS, 1):
                (run / f'qwen-gpu{index}.service.txt').write_text(subprocess.check_output(['systemctl','cat',name],text=True))
            for model, base in [('qwen','http://127.0.0.1:8080'),('qwen2','http://127.0.0.1:8081'),('flash','http://127.0.0.1:18144')]:
                with urllib.request.urlopen(base+'/v1/models',timeout=10) as r:
                    atomic(run / f'{model}-models.json', json.load(r))
            swe_run = HERE / 'runs/primary-single-instance'
            if not (swe_run/'ledger.json').exists():
                command(run,'swe-init',[a.swe_python,str(HERE/'paired_swe.py'),'init','--run-dir',str(swe_run)],env,deadline,pids)
                ledger = json.loads((swe_run/'ledger.json').read_text())
                ledger['generation_deadline_epoch'] = limits['generation_deadline_epoch']
                ledger['envelope_deadline_epoch'] = deadline
                atomic(swe_run/'ledger.json',ledger)
            if not json.loads((swe_run/'ledger.json').read_text())['smoke_gate']:
                cmd = [a.swe_python,str(HERE/'paired_swe.py'),'run','--run-dir',str(swe_run),
                       '--swe-phase','--smoke-only']
                if json.loads((swe_run/'ledger.json').read_text())['phase'] != 'initialized':
                    cmd.append('--resume')
                command(run,'swe-smoke',cmd,env,deadline,pids)
            phases = [('speed',strata_python,3*3600), ('capacity',strata_python,30*60),
                      ('evalplus',a.evalplus_python,5*3600)]
            for phase, python, allocation in phases:
                if (run / f'{phase}.complete').exists():
                    continue
                available = min(allocation, max(1,int(limits['generation_deadline_epoch']-time.time()-14*3600)))
                cmd = [python,str(HERE/'supplementary.py'),phase,'--run-dir',str(run/'supplementary'),
                       '--overall-seconds',str(available)]
                if phase == 'speed':
                    cmd.append('--qwen-cache-prompt-false')
                if phase == 'evalplus':
                    cmd.extend(['--request-seconds','300','--evaluation-seconds','1800'])
                code = command(run,phase,cmd,env,min(deadline,time.time()+available+300),pids,allow_incomplete=True)
                (run / f'{phase}.{"complete" if code == 0 else "incomplete"}').write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            resume = json.loads((swe_run/'ledger.json').read_text())['phase'] != 'initialized'
            cmd = [a.swe_python,str(HERE/'paired_swe.py'),'run','--run-dir',str(swe_run),'--swe-phase']
            if resume:
                cmd.append('--resume')
            command(run,'swe',cmd,env,deadline,pids)
            note(run,'complete',message='Paired generation and official evaluation complete; review scores.',
                 deadline_epoch=deadline)
        except Exception as exc:
            note(run,'blocked-or-failed',error=str(exc),deadline_epoch=deadline)
            raise
        finally:
            stop_process(server)
            stop.set()
            protect(pids)


if __name__ == '__main__':
    main()
