#!/usr/bin/env python3
"""Render only aggregate results; optionally wait, commit exact reports and push."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def aggregate(study):
    campaign = study / 'campaign'
    status = load(campaign / 'campaign-status.json')
    origin = load(campaign / 'campaign-origin.json')
    ledger = load(study / 'runs/primary-single-instance/ledger.json')
    swe = {}
    for model, tasks in ledger.get('tasks', {}).items():
        attempts = [attempt for task in tasks.values() for attempt in task.get('attempts', [])]
        resolved_tasks = [t for t in tasks.values() if t.get('evaluation', {}).get('resolved') is True]
        total_wall = sum(a.get('wall_seconds', 0) for a in attempts)
        solved_wall = sum(t['attempts'][-1].get('wall_seconds', 0) for t in resolved_tasks if t.get('attempts'))
        swe[model] = dict(denominator=len(tasks), attempted=sum(bool(t.get('attempts')) for t in tasks.values()),
                         submitted=sum(t['state'] == 'submitted' for t in tasks.values()),
                         resolved=len(resolved_tasks),
                         officially_evaluated=sum(isinstance(t.get('evaluation', {}).get('resolved'), bool) for t in tasks.values()),
                         states=dict(Counter(t['state'] for t in tasks.values())),
                         terminal_errors=dict(Counter(t.get('attempts', [{}])[-1].get('reason') or 'none'
                                            for t in tasks.values() if t.get('attempts'))),
                         total_attempt_wall_s=total_wall,
                         mean_resolved_task_wall_s=solved_wall/len(resolved_tasks) if resolved_tasks else None,
                         total_attempt_seconds_per_resolved=total_wall/len(resolved_tasks) if resolved_tasks else None,
                         completion_tokens=sum(a.get('metrics', {}).get('completion_tokens', 0) for a in attempts),
                         model_calls=sum(a.get('metrics', {}).get('backend_requests', 0) for a in attempts))
    # Strict allowlists: no commands, paths, model answers, trajectories or secrets.
    evalplus = {}
    for dataset, values in load(campaign/'supplementary/evalplus/summary.json').items():
        evalplus[dataset] = {}
        for model, d in values.items():
            official = d.get('official', {})
            evalplus[dataset][model] = {key: d.get(key) for key in ('status','denominator','attempted','outcomes','sanitizer_failures')}
            evalplus[dataset][model]['official'] = {key: official.get(key) for key in ('status','denominator','base_pass_at_1','plus_pass_at_1','wall_s')}
    speed = load(campaign/'supplementary/speed/summary.json')
    capacity = load(campaign/'supplementary/capacity/summary.json')
    return dict(updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                phase=status.get('phase','not-started'),
                start_epoch=origin.get('start_epoch'), deadline_epoch=origin.get('deadline_epoch'),
                models=dict(qwen='Qwen3.8-27B ATX IQ4_XS-M / llamAmpere MTP3 / GPU1',
                            flash='Qwen3.8-Flash-Next IQ3_S / Strata 0.1.44 MTP4 / GPU0'),
                swe_phase=ledger.get('phase','not-started'), swe=swe, evalplus=evalplus,
                speed={k:speed.get(k) for k in ('planned_requests','attempted_requests','groups','paired_delta_distributions','denominator_note')},
                capacity=dict(aggregate_rates=capacity.get('aggregate_rates'),
                              note='2 Qwen GPU replicas versus 1 Flash GPU; includes queue/prefill, not single-GPU speed.'))


def fmt(value, percent=False):
    if value is None:
        return 'pending'
    return f'{value*100:.1f}%' if percent else f'{value:.1f}'


def render(data):
    lines = ['# Qwen3.8-27B versus Flash-Next IQ3_S — paired campaign', '',
             '<!-- generated-by: publish_results.py -->', '',
             f'Updated: {data["updated_utc"]}. Campaign phase: **{data["phase"]}**.', '',
             '## Configurations', '', '| | Qwen3.8-27B | Flash-Next |', '|---|---|---|',
             '| Quant | ATX IQ4_XS-M | full-expert IQ3_S |',
             '| Runtime | deployed llamAmpere, MTP3 | Strata 0.1.44, MTP4 |',
             '| Primary allocation | one RTX 3090, GPU1 | one RTX 3090, GPU0 + host RAM |',
             '| Matched context | 262144 native | 262144 native, no scaling |', '',
             'Identical task text and generation policy per benchmark. This is a comparison',
             'of serving configurations, not an isolated quantization experiment.', '',
             '## SWE-bench Verified (frozen 40-task subset)', '',
             f'Phase: **{data["swe_phase"]}**. Patch submission is not resolution.', '',
             '| Metric | Qwen | Flash |', '|---|---:|---:|']
    for title, key in [('Attempted tasks','attempted'),('Submitted patches','submitted'),
                       ('Officially evaluated submissions','officially_evaluated'),('Officially resolved / 40','resolved'),
                       ('Total model calls','model_calls'),('Total output tokens','completion_tokens'),
                       ('Mean resolved-task wall seconds','mean_resolved_task_wall_s'),
                       ('All attempt seconds per resolved task','total_attempt_seconds_per_resolved')]:
        vals = [data['swe'].get(m, {}).get(key) for m in ('qwen','flash')]
        lines.append(f'| {title} | {fmt(vals[0])} | {fmt(vals[1])} |')
    lines += ['', 'Denominator stays 40 for each model. Missing official reports and unattempted',
              'tasks remain explicit; incomplete coverage is not a completed accuracy score.', '',
              '## HumanEval+ and MBPP+', '', '| Dataset | Qwen base / plus pass@1 | Flash base / plus pass@1 | Coverage/status |',
              '|---|---:|---:|---|']
    for name in ('humaneval','mbpp'):
        values = data['evalplus'].get(name, {})
        scores, coverage = [], []
        for model in ('qwen','flash'):
            d = values.get(model, {})
            o = d.get('official', {})
            scores.append(fmt(o.get('base_pass_at_1'),True)+' / '+fmt(o.get('plus_pass_at_1'),True))
            coverage.append(f'{model}: {d.get("attempted",0)}/{d.get("denominator","?")} {d.get("status","pending")}')
        lines.append(f'| {name} | {scores[0]} | {scores[1]} | {"; ".join(coverage)} |')
    lines += ['', '## Matched natural coding workload: median engine rates and TTFT', '',
              '| Body target | Qwen prefill / decode tok/s | Flash prefill / decode tok/s | Qwen / Flash TTFT s |',
              '|---:|---:|---:|---:|']
    groups = data['speed'].get('groups') or {}
    for size in (4096,32768,131072,245760):
        values = []
        for model in ('qwen','flash'):
            g = groups.get(f'code/{size}/{model}', {})
            metrics = g.get('all_responses', {})
            values.append({k:metrics.get(k,{}).get('median') for k in ('engine_prefill_tps','engine_decode_tps','ttft_s')})
        q,f = values
        lines.append(f'| {size} | {fmt(q["engine_prefill_tps"])} / {fmt(q["engine_decode_tps"])} | '
                     f'{fmt(f["engine_prefill_tps"])} / {fmt(f["engine_decode_tps"])} | {fmt(q["ttft_s"])} / {fmt(f["ttft_s"])} |')
    lines += ['', 'Three fresh repeats are planned per point. These medians include valid responses;',
              'the aggregate JSON separately exposes caps, cache reuse/unknowns, failures, actual',
              'token counts, ranges and zero-reported-cache distributions. Do not interpret',
              'different natural output lengths as equal-token work.', '',
              '## Deployment capacity', '',
              'Measured separately: two Qwen replicas on GPU1+GPU2 versus two queued requests',
              'to one Flash replica on GPU0. This is a hardware-allocation comparison.', '',
              '## Interpretation and reproduction', '',
              'Use PLAN.md, README.md and SUPPLEMENTARY.md for fixed budgets, sampler settings,',
              'held-out evaluation, failure accounting and reproduction. Flash separately passed',
              '512k YaRN document/code trials; that capability is outside this native matched test.',
              'Raw agent/model outputs, held-out fixtures, credentials and local telemetry are',
              'not published. results-summary.json contains filtered aggregate measurements.', '']
    return '\n'.join(lines)


def git(repo, *args):
    proc = subprocess.run(['git',*args],cwd=repo,text=True,capture_output=True,check=True)
    print(proc.stdout, end='')
    return proc.stdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study-root',type=Path,default=HERE)
    p.add_argument('--public-repo',type=Path)
    p.add_argument('--wait-unit')
    p.add_argument('--commit-and-push',action='store_true')
    a = p.parse_args()
    if a.wait_unit:
        while subprocess.check_output(['systemctl','--user','show',a.wait_unit,'-p','ActiveState','--value'],text=True).strip() in ('active','activating','deactivating'):
            time.sleep(60)
    if a.commit_and_push:
        if a.public_repo is None:
            p.error('--public-repo required for publication')
        if git(a.public_repo,'status','--porcelain').strip():
            raise RuntimeError('Public tree has other changes; preserving them. Generate/review reports manually.')
        git(a.public_repo,'fetch','origin')
        git(a.public_repo,'merge','--ff-only','origin/main')
    data = aggregate(a.study_root)
    destination = a.public_repo/'research'/HERE.name if a.public_repo else a.study_root
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'RESULTS.md').write_text(render(data))
    (destination/'results-summary.json').write_text(json.dumps(data,indent=2)+'\n')
    manifest_path = destination/'PUBLICATION.json'
    if manifest_path.exists():
        manifest = load(manifest_path)
        manifest.update(aggregate_snapshot_utc=data['updated_utc'], aggregate_campaign_phase=data['phase'],
                        aggregate_swe_phase=data['swe_phase'],
                        publication_status='Aggregate snapshot; consult phase and official coverage before interpreting scores')
        for entry in manifest.get('included', []):
            entry['sha256'] = hashlib.sha256((destination/entry['path']).read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('Rendered aggregate reports; phase:',data['phase'])
    if a.commit_and_push:
        paths = [f'research/{HERE.name}/RESULTS.md',f'research/{HERE.name}/results-summary.json']
        if manifest_path.exists():
            paths.append(f'research/{HERE.name}/PUBLICATION.json')
        git(a.public_repo,'status','--short')
        git(a.public_repo,'diff','--check')
        git(a.public_repo,'diff','--',*paths)
        git(a.public_repo,'log','--oneline','-10')
        git(a.public_repo,'add','--',*paths)
        if git(a.public_repo,'diff','--cached','--name-only').splitlines() != sorted(paths):
            raise RuntimeError('Unexpected staged publication files; no commit made')
        git(a.public_repo,'commit','-m','Report paired Qwen and Flash benchmark results')
        git(a.public_repo,'push','origin','main')


if __name__ == '__main__':
    main()
