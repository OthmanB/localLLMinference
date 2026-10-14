#!/usr/bin/env python3
"""Durable serial trial runner. Waits for model prep and manual Muse handoff."""
import json
import os
import pathlib
import subprocess
import threading
import time
import urllib.request

import psutil

HERE = pathlib.Path(__file__).resolve().parent
RUNTIME = pathlib.Path((os.environ['RUNTIME_ROOT'] + '/strata-flash-next-20261014'))
DATA = pathlib.Path((os.environ['MODEL_STORE'] + '/strata-flash-next-20261014'))
PYTHON = str(RUNTIME / '.venv/bin/python')
MUSE = 'llama-muse-glimmer-30b-131k.service'
PROTECTED = ['llamampere-qwen3.8-atx-iq4xs-m-gpu1.service',
             'llamampere-qwen3.8-atx-iq4xs-m-gpu2.service']
ENV = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='0',
           PATH='/usr/local/cuda-13.2/bin:' + os.environ['PATH'],
           XDG_CONFIG_HOME=str(DATA / 'config-home'), PYTHONUNBUFFERED='1',
           STRATA_ROUTE_TAIL_SKIP='0', STRATA_ADAPT_OVERLAP='0',
           STRATA_HC_Q8='1', STRATA_MMQ_KQ='0')
# Scheduling heuristic only, not a user-specified pass/fail threshold.
# At 500 tok/s, a 262k prompt still takes about 8.7 minutes to prefill.
COMPACT_PRIORITY_PREFILL_TPS = 500


def performance(result):
    timings = result.get('timings') or {}
    usage = result.get('usage') or {}
    details = usage.get('prompt_tokens_details') or {}
    cached = timings.get('cache_n', details.get('cached_tokens'))
    prefill = timings.get('prompt_per_second')
    decode = timings.get('predicted_per_second') or result.get('client_decode_tps')
    # Keep missing/unknown timings explicit; never use SSE event counts as tokens.
    return dict(prefill_tps=prefill, decode_tps=decode, ttft_s=result.get('ttft_s'),
                prompt_tokens=usage.get('prompt_tokens'), cached_prompt_tokens=cached,
                fully_uncached=(cached == 0),
                prefill_speedup_vs_historical_ft102=(prefill / 102 if prefill else None),
                projected_262k_prefill_s=(262144 / prefill if prefill else None),
                projected_1000_token_generation_s=(1000 / decode if decode else None),
                projection_note='Arithmetic estimates only; populated 262k measurement is authoritative.')


def value(unit, prop, user=False):
    cmd = ['systemctl'] + (['--user'] if user else [])
    return subprocess.check_output(cmd + ['show', unit, '-p', prop, '--value'], text=True).strip()


def note(stage, **details):
    state = dict(stage=stage, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **details)
    (HERE / 'state.json').write_text(json.dumps(state, indent=2))
    with (HERE / 'progress.jsonl').open('a') as f:
        f.write(json.dumps(state) + '\n')
    print(json.dumps(state), flush=True)


def check_protected(pids):
    for unit, pid in pids.items():
        if value(unit, 'ActiveState') != 'active' or value(unit, 'MainPID') != pid:
            raise RuntimeError(f'Protected service changed: {unit}; aborting trial')


def wait_prepared():
    while value('strata-flash-prepare-20261014.service', 'ActiveState', True) in ('active', 'activating'):
        files = {p.name: p.stat().st_size for p in (DATA / 'models').rglob('*') if p.is_file()}
        note('preparing', message='Model transfer/packing still running; no inference launched.',
             muse_state=value(MUSE, 'ActiveState'), artifact_bytes=files)
        time.sleep(120)
    if value('strata-flash-prepare-20261014.service', 'ExecMainStatus', True) != '0':
        raise RuntimeError('Model preparation failed; inspect prepare.log')
    if not (RUNTIME / 'strata-unsloth-ud-q4_k_xl.json').exists():
        raise RuntimeError('Preparation exited without the Q4 run configuration')


def config_for(tag, context):
    cfg = json.loads((RUNTIME / f'strata-{tag}.json').read_text())
    cfg['gpu'] = 0
    cfg['host'], cfg['port'] = '127.0.0.1', 18144
    cfg['parallel'] = 1
    cfg['reasoning_effort'] = 'none'
    cfg['log'] = str(HERE / f'{tag}-{context}.engine.log')
    env = cfg.setdefault('env', {})
    env.update({k: ENV[k] for k in ('CUDA_DEVICE_ORDER', 'CUDA_VISIBLE_DEVICES',
                                  'STRATA_ROUTE_TAIL_SKIP', 'STRATA_ADAPT_OVERLAP',
                                  'STRATA_MMQ_KQ')})
    # Q8 HC reads are the Unsloth release default; do not override the base
    # GSQ model's precision path with that Unsloth-specific choice.
    if tag.startswith('unsloth-'):
        env['STRATA_HC_Q8'] = '1'
    else:
        env['STRATA_HC_Q8'] = '0'
    args = cfg['args']
    if '--layer-split' in args or '--gpus' in args:
        raise RuntimeError('Refusing multi-GPU configuration')
    if '--gpu' in args:
        args[args.index('--gpu') + 1] = '0'
    args[args.index('--max-context') + 1] = str(context)
    if '--kv' not in args:
        args.extend(['--kv', 'int8'])
    if context >= 65536 and '--kv-resident' not in args:
        args.extend(['--kv-resident', '32768'])
    cfg.pop('layer_split', None)
    path = HERE / f'{tag}-{context}.json'
    path.write_text(json.dumps(cfg, indent=2))
    return path, cfg


def samples(stop, tag, context, server):
    with (HERE / f'{tag}-{context}.resources.jsonl').open('a') as f:
        while not stop.is_set():
            try:
                gpu = subprocess.check_output([
                    'nvidia-smi', '--query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw,pcie.link.gen.current,pcie.link.width.current',
                    '--format=csv,noheader,nounits'], text=True).strip()
                processes = [psutil.Process(server.pid)] + psutil.Process(server.pid).children(recursive=True)
                row = dict(epoch=time.time(), gpu=gpu,
                           available_ram=psutil.virtual_memory().available,
                           swap_used=psutil.swap_memory().used,
                           rss=sum(p.memory_info().rss for p in processes),
                           process_storage_read_bytes=sum(p.io_counters().read_bytes for p in processes),
                           process_storage_write_bytes=sum(p.io_counters().write_bytes for p in processes))
                # Cgroup accounting includes the trial's anonymous allocations
                # and file cache; summed process RSS alone can double-count shared
                # mappings and miss file cache outside the process working set.
                group = pathlib.Path('/sys/fs/cgroup') / next(
                    line.split(':', 2)[2].lstrip('/') for line in
                    pathlib.Path('/proc/self/cgroup').read_text().splitlines()
                    if line.startswith('0::'))
                row['cgroup_memory_bytes'] = int((group / 'memory.current').read_text())
                row['cgroup_swap_bytes'] = int((group / 'memory.swap.current').read_text())
                stats = dict(line.split() for line in (group / 'memory.stat').read_text().splitlines())
                row['cgroup_anon_bytes'] = int(stats.get('anon', 0))
                row['cgroup_file_bytes'] = int(stats.get('file', 0))
                f.write(json.dumps(row) + '\n')
                f.flush()
            except (psutil.Error, subprocess.SubprocessError, OSError, ValueError, StopIteration):
                pass
            stop.wait(2)


def run_context(tag, context, pids, quality=False):
    check_protected(pids)
    if value(MUSE, 'ActiveState') != 'inactive':
        raise RuntimeError('Muse must be stopped before experimental inference')
    path, cfg = config_for(tag, context)
    note('testing', quant=tag, context=context)
    with (HERE / f'{tag}-{context}.server.log').open('a') as log:
        server = subprocess.Popen([PYTHON, '-m', 'serve.server', '--engine', 'strata',
                                   '--config', str(path), '--port', '18144'],
                                  cwd=RUNTIME, env=ENV, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        stop = threading.Event()
        monitor = threading.Thread(target=samples, args=(stop, tag, context, server), daemon=True)
        monitor.start()
        try:
            deadline = time.monotonic() + 1800
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'Experimental server exited {server.returncode}')
                if time.monotonic() > deadline:
                    raise RuntimeError('Experimental load timeout')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18144/health', timeout=5) as r:
                        health = json.load(r)
                    if health.get('loaded') is True:
                        (HERE / f'{tag}-{context}.health.json').write_text(json.dumps(health, indent=2))
                        break
                except OSError:
                    pass
                time.sleep(5)
            label = f'{tag}-{context}'
            subprocess.run([PYTHON, str(HERE / 'context_probe.py'), '--base', 'http://127.0.0.1:18144',
                            '--model', cfg['model_name'], '--label', label, '--context', str(context)],
                           env=ENV, check=True, timeout=7200)
            if quality:
                subprocess.run([PYTHON, str(HERE / 'probe.py'), '--base', 'http://127.0.0.1:18144',
                                '--model', cfg['model_name'], '--label', label],
                               env=ENV, check=True, timeout=3600)
            result = json.loads((HERE / 'results' / label / f'context-{context}.json').read_text())
            result['performance'] = performance(result)
            (HERE / 'results' / label / f'context-{context}.json').write_text(json.dumps(result, indent=2))
            check_protected(pids)
            return result
        finally:
            # Stop just our process group, including the GPU engine, gracefully.
            import signal
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
            stop.set()
            monitor.join(timeout=10)


def series(tag, pids):
    results = []
    for context in (4096, 32768, 131072, 262144):
        result = run_context(tag, context, pids, quality=context == 32768)
        results.append(result)
        metrics = performance(result)
        rate = metrics['decode_tps']
        note('context-complete', quant=tag, context=context, **metrics,
             markers=result.get('markers_present'), finish=result.get('finish_reason'),
             usage=result.get('usage'), timings=result.get('timings'))
        # A severe short-context speed failure makes spending hours at native
        # context unhelpful. A narrower miss still earns a long-context attempt.
        if context == 32768 and (not rate or rate < 20 or not all(result['markers_present'])):
            note('screen-failed', quant=tag, decode_tps=rate)
            break
    return results


def main():
    pids = {u: value(u, 'MainPID') for u in PROTECTED}
    try:
        wait_prepared()
        check_protected(pids)
        # User elected manual system-service control. Never try to kill Muse
        # directly (Restart=on-failure would immediately reload it).
        while value(MUSE, 'ActiveState') != 'inactive':
            note('waiting-for-muse-stop', command='sudo systemctl stop ' + MUSE,
                 message='Artifacts ready. User-controlled GPU0 handoff required.')
            time.sleep(60)
        q4 = series('unsloth-ud-q4_k_xl', pids)
        short = next(x for x in q4 if x['context_limit'] == 32768)
        measured = performance(short)
        rate = measured['decode_tps'] or 0
        prefill = measured['prefill_tps']
        # Prefill is central to the revised objective. Good decode alone must
        # not defer the compact candidate when fresh-prompt ingestion is slow.
        prioritize_compact = rate < 40 or prefill is None or prefill < COMPACT_PRIORITY_PREFILL_TPS
        quant = 'IQ2_XS' if prioritize_compact else 'IQ3_S'
        note('preparing-comparison', quant=quant, q4_screen=measured,
             reason='Prefer compact IQ2_XS when decode is below the preferred 40 tok/s or '
                    'prefill is unknown/below the provisional 500 tok/s scheduling threshold; '
                    'otherwise compare IQ3_S. This is not a replacement pass/fail threshold.')
        # Use the root SSD for the comparison so Q4 + comparison + MTP do not
        # crowd the model-store SSD. All shared draft assets stay in DATA.
        models = RUNTIME / 'comparison-models'
        models.mkdir(exist_ok=True)
        if psutil.disk_usage(str(RUNTIME)).free < 95 * 1024**3:
            raise RuntimeError('Insufficient free root disk for compact comparison; preserving Q4 results')
        with (HERE / 'comparison-prepare.log').open('a') as log:
            subprocess.run([PYTHON, str(RUNTIME / 'setup.py'), '--setup', '--family', 'qwen',
                            '--model', quant, '--context', '32768', '--kv', 'int8', '--vision', 'no',
                            '--gpu', '0', '--parallel', '1', '--experimental-speed-projection', 'off',
                            '--backend', 'cuda', '--yes', '--no-start', '--no-browser',
                            '--port', '18144', '--host', '127.0.0.1',
                            '--data-dir', str(DATA), '--models-dir', str(models)],
                           cwd=RUNTIME, env=ENV, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=86400)
        series(quant.lower(), pids)
        check_protected(pids)
        note('trials-complete', message='Experimental engine stopped. Review raw results before promotion.',
             restore_command='sudo systemctl start ' + MUSE)
    except Exception as exc:
        note('blocked-or-failed', error=str(exc), restore_command='sudo systemctl start ' + MUSE)
        raise


if __name__ == '__main__':
    main()
