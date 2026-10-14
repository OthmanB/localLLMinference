#!/usr/bin/env python3
"""GPU0 IQ3_S extension: cold document accuracy, natural answers, latency/resources."""
import argparse
import json
import os
import random
import signal
import subprocess
import threading
import time
import urllib.request
import uuid

from context_probe import tokenizer
from execute import HERE, RUNTIME, PYTHON, ENV, MUSE, PROTECTED, value, check_protected, samples, performance
from probe import request, save


def note(stage, **kw):
    row = dict(stage=stage, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    (HERE / 'extension-state.json').write_text(json.dumps(row, indent=2))
    with (HERE / 'extension-progress.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def configure(context, scaled, reserve, trial=''):
    label = f'iq3_s-extension-{"yarn2" if scaled else "native"}-r{reserve}'
    if trial:
        label += '-' + trial
    cfg = json.loads((RUNTIME / 'strata-iq3_s.json').read_text())
    cfg.update(gpu=0, parallel=1, host='127.0.0.1', port=18144,
               reasoning_effort='none', log=str(HERE / f'{label}-{context}.engine.log'))
    cfg.pop('layer_split', None)
    cfg['env'] = {k: v for k, v in ENV.items() if k.startswith('STRATA_') or k.startswith('CUDA_')}
    cfg['env']['STRATA_HC_Q8'] = '0'
    args = cfg['args']
    for option, val in [('--max-context', str(context)), ('--kv', 'int8'),
                        ('--kv-resident', '32768'), ('--vram-reserve-mib', str(reserve)),
                        ('--rope-scaling', 'yarn' if scaled else 'none'),
                        ('--rope-scale', '2' if scaled else '1'), ('--yarn-orig-ctx', '262144')]:
        if option in args:
            args[args.index(option) + 1] = val
        else:
            args.extend([option, val])
    if '--layer-split' in args or '--gpus' in args:
        raise RuntimeError('Refusing multi-GPU configuration')
    path = HERE / f'{label}-{context}.json'
    path.write_text(json.dumps(cfg, indent=2))
    return label, path, cfg


def fixture(tok, context, seed):
    rng = random.Random(seed)
    marker = [f'{prefix}-{rng.getrandbits(96):024x}' for prefix in ('EARLY', 'MID', 'LATE')]
    header = ('Independent document trial ' + uuid.uuid4().hex + '.\n'
              'Read the following archive as reference data. Documents are independent. '
              'AUX documents describe other services, not the cedar service.\nBEGIN ARCHIVE\n')
    documents = [
        ('SRC-A', f'Archive early marker: {marker[0]}.', 0.05),
        ('SRC-B', 'Service cedar policy revision 2. Status APPROVED. '
                  'Batch size 128 items. Timeout 30 seconds. Region east.', 0.23),
        ('SRC-C', f'Job cedar-run has 173 batches. Archive middle marker: {marker[1]}.', 0.45),
        ('SRC-D', 'Service cedar policy revision 3. Status APPROVED. '
                  'Batch size 160 items. Timeout 45 seconds. Region west. '
                  'This replaces all earlier approved cedar policies.', 0.72),
        ('SRC-E', 'Service cedar policy revision 4. Status DRAFT, NOT APPROVED. '
                  'Proposed batch size 999 items, timeout 5 seconds, region north. '
                  'Do not apply this draft to any job.', 0.89),
        ('SRC-F', f'Archive late marker: {marker[2]}.', 0.97),
    ]
    task = ('\nEND ARCHIVE\nAnswer with a JSON object only, no code fences, with these keys: '
            'markers (array of the exact early, middle, late marker strings in order); '
            'policy (object with revision, batch_size, timeout_seconds, region, status for '
            'the latest APPROVED cedar policy); job (object with batches and total_items '
            'for cedar-run under that approved policy); evidence (object with keys '
            'markers, policy, batches, rejected_draft, whose values are arrays of source '
            'document IDs). Exclude draft policy settings. Do not invent sources.')
    target = context - 8192 - 256
    doc_text = [f'\nDOCUMENT {name}\n{text}\nEND DOCUMENT {name}\n' for name, text, _ in documents]
    budget = target - len(tok.encode(header + task + ''.join(doc_text))) - 128
    ids = []
    n = 0
    # Every auxiliary document differs; no giant repeated paragraph corpus.
    while len(ids) < budget:
        group = []
        for _ in range(64):
            n += 1
            name = f'AUX-{n:06d}'
            code = rng.getrandbits(64)
            text = (f'\nDOCUMENT {name}\nService aux-{code:016x}, revision {rng.randrange(1, 30)}, '
                    f'status {rng.choice(["APPROVED", "DRAFT", "RETIRED"])}. '
                    f'Batch size {rng.randrange(10, 900)} items; timeout {rng.randrange(2, 120)} seconds; '
                    f'region {rng.choice(["east", "west", "north", "south"])}. '
                    f'Job count {rng.randrange(1, 400)} batches. '
                    f'Note: inventory checksum {rng.getrandbits(64):016x}; '
                    'validate the revision and approval status before scheduling. '
                    f'END DOCUMENT {name}\n')
            group.append(text)
        ids.extend(tok.encode(''.join(group)))
    ids = ids[:budget]
    chunks, prev, placements = [], 0, []
    for (name, _, fraction), doc in zip(documents, doc_text):
        pos = int(budget * fraction)
        chunks.extend([tok.decode(ids[prev:pos]), doc])
        placements.append({'id': name, 'approx_body_position': len(tok.encode(header + ''.join(chunks)))})
        prev = pos
    chunks.append(tok.decode(ids[prev:]))
    text = header + ''.join(chunks) + task
    expected = dict(markers=marker,
                    policy=dict(revision=3, batch_size=160, timeout_seconds=45, region='west', status='APPROVED'),
                    job=dict(batches=173, total_items=27680),
                    evidence=dict(markers=['SRC-A', 'SRC-C', 'SRC-F'], policy=['SRC-D'],
                                  batches=['SRC-C'], rejected_draft=['SRC-E']))
    return text, expected, placements


def grade(content, expected):
    try:
        obj = json.loads(content)
    except (ValueError, TypeError):
        return dict(valid_json=False, all_correct=False)
    checks = {key: obj.get(key) == val for key, val in expected.items() if key != 'evidence'}
    citations = obj.get('evidence') or {}
    known = {'SRC-A', 'SRC-B', 'SRC-C', 'SRC-D', 'SRC-E', 'SRC-F'}
    checks['evidence'] = all(isinstance(citations.get(key), list) and
                             set(val).issubset(set(citations[key])) and
                             set(citations[key]).issubset(known)
                             for key, val in expected['evidence'].items())
    return dict(valid_json=True, checks=checks, all_correct=all(checks.values()), parsed=obj)


def resource_summary(label, context):
    rows = [json.loads(x) for x in (HERE / f'{label}-{context}.resources.jsonl').read_text().splitlines()]
    gpu = [r['gpu'].splitlines()[0].split(',') for r in rows]
    return dict(min_gpu0_free_mib=min(float(g[2]) for g in gpu),
                peak_gpu0_used_mib=max(float(g[1]) for g in gpu),
                peak_rss_bytes=max(r['rss'] for r in rows),
                min_available_ram_bytes=min(r['available_ram'] for r in rows),
                peak_cgroup_memory_bytes=max(r['cgroup_memory_bytes'] for r in rows),
                peak_cgroup_swap_bytes=max(r['cgroup_swap_bytes'] for r in rows),
                host_swap_start_bytes=rows[0]['swap_used'],
                host_swap_peak_bytes=max(r['swap_used'] for r in rows))


def run(context, scaled, reserve, pids, seed, trial=''):
    check_protected(pids)
    if value(MUSE, 'ActiveState') != 'inactive':
        raise RuntimeError('Muse is active; GPU0 handoff required')
    label, path, cfg = configure(context, scaled, reserve, trial)
    note('extension-testing', label=label, context=context)
    tok = tokenizer()
    prompt, expected, placements = fixture(tok, context, seed)
    folder_label = f'{label}-{context}'
    folder = HERE / 'results' / folder_label
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'fixture.json').write_text(json.dumps(dict(expected=expected, placements=placements,
                                          raw_body_tokens=len(tok.encode(prompt)), seed=seed), indent=2))
    # Store the exact input to enable matching and review of post-native facts.
    (folder / 'prompt.txt').write_text(prompt)
    env = dict(ENV, STRATA_HC_Q8='0')
    with (HERE / f'{label}-{context}.server.log').open('a') as log:
        server = subprocess.Popen([PYTHON, '-m', 'serve.server', '--engine', 'strata',
                                   '--config', str(path), '--port', '18144'],
                                  cwd=RUNTIME, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        stop = threading.Event()
        monitor = threading.Thread(target=samples, args=(stop, label, context, server), daemon=True)
        monitor.start()
        result = None
        try:
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f'Server exited {server.returncode}')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18144/health', timeout=5) as r:
                        health = json.load(r)
                    if health.get('loaded'):
                        (folder / 'health.json').write_text(json.dumps(health, indent=2))
                        break
                except OSError:
                    pass
                time.sleep(5)
            else:
                raise RuntimeError('Engine load timed out')
            result = request('http://127.0.0.1:18144', cfg['model_name'],
                             [{'role': 'user', 'content': prompt}], max_tokens=4096)
            result.update(context_limit=context, scaled=scaled, expected=expected,
                          placements=placements, grading=grade(result['content'], expected),
                          performance=performance(result))
            save(folder_label, 'documents', result)
            used = result['usage'].get('prompt_tokens', 0)
            if used < context - 12000 or used + 4096 > context:
                raise RuntimeError(f'Unexpected populated prompt size {used} at {context}')
            if not result['grading']['all_correct'] or result['finish_reason'] != 'stop':
                note('extension-quality-failed', label=label, context=context,
                     grading=result['grading'], finish=result['finish_reason'])
            else:
                # A substantive natural continuation measures sustained decode.
                messages = [{'role': 'user', 'content': prompt},
                            {'role': 'assistant', 'content': result['content']},
                            {'role': 'user', 'content':
                             'Now write a Python LRUCache class using OrderedDict. '
                             'Capacity must be a positive integer; reject bool. get(key) returns '
                             'None on miss, put(key,value) updates recency and evicts the oldest. '
                             'Include five executable unittest tests covering validation, misses, '
                             'update recency, access recency and eviction, plus a short explanation. '
                             'Produce the complete answer, not a sketch.'}]
                answer = request('http://127.0.0.1:18144', cfg['model_name'], messages, max_tokens=4096)
                answer['performance'] = performance(answer)
                save(folder_label, 'coding-continuation', answer)
                result['continuation'] = {k: answer[k] for k in ('usage', 'timings', 'finish_reason', 'wall_s')}
            check_protected(pids)
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
            stop.set()
            monitor.join(timeout=10)
    if result is None:
        raise RuntimeError('No document result')
    result['resources'] = resource_summary(label, context)
    save(folder_label, 'documents', result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reserve', type=int, default=1536)
    p.add_argument('--repeat-512', action='store_true', help='Fresh-seed independent 512k repetition')
    a = p.parse_args()
    pids = {u: value(u, 'MainPID') for u in PROTECTED}
    ladder = [(262144, False), (32768, True), (262144, True), (393216, True), (524288, True)]
    if a.repeat_512:
        ladder = [(524288, True)]
    try:
        for context, scaled in ladder:
            result = run(context, scaled, a.reserve, pids,
                         seed=38145 if a.repeat_512 else 38144,
                         trial='repeat2' if a.repeat_512 else '')
            resources = result['resources']
            ok = (result['grading']['all_correct'] and result['finish_reason'] == 'stop' and
                  result.get('continuation', {}).get('finish_reason') == 'stop' and
                  resources['min_gpu0_free_mib'] >= 1024 and resources['peak_cgroup_swap_bytes'] == 0)
            note('extension-point-complete', context=context, scaled=scaled,
                 grading=result['grading'], performance=result['performance'],
                 continuation=result.get('continuation'), resources=resources, passed=ok)
            if not ok:
                note('extension-escalation-stopped', context=context, scaled=scaled,
                     message='Inspect quality/resources before continuing; experimental engine stopped.')
                return
        note('extension-trials-complete', message='IQ3_S 512k ladder completed; review outputs before recommendation.')
    except Exception as exc:
        note('extension-failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
