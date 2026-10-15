#!/usr/bin/env python3
"""Snapshot failed task IDs, request termination signals and paired outcomes."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, default=Path(__file__).resolve().parent/'runs/primary-single-instance')
    p.add_argument('--include-snippets', action='store_true', help='Private diagnostic output only')
    a = p.parse_args()
    ledger = json.loads((a.run_dir/'ledger.json').read_text())
    failed = {model:{key for key,t in tasks.items() if t['state'] == 'failed'}
              for model,tasks in ledger['tasks'].items()}
    results = []
    for task_id in sorted(set().union(*failed.values())):
        for model, tasks in ledger['tasks'].items():
            task = tasks[task_id]
            row = {'task_id':task_id, 'model':model, 'state':task['state'],
                   'official_resolved':task.get('evaluation',{}).get('resolved')}
            if task.get('attempts'):
                attempt = task['attempts'][-1]
                row.update(reason=attempt.get('reason'),wall_seconds=attempt.get('wall_seconds'),
                           metrics={k:attempt.get('metrics',{}).get(k) for k in
                                    ('exit_status','agent_calls','backend_requests','completion_tokens',
                                     'tool_calls','max_context_tokens','request_errors')})
                folder = a.run_dir/attempt['directory']
                requests = folder/'requests.jsonl'
                if requests.exists():
                    records = [json.loads(line) for line in requests.read_text().splitlines() if line.strip()]
                    row['finish_reasons'] = dict(Counter(reason for record in records
                                                        for reason in record.get('finish_reasons',[])))
                    row['http_errors'] = [record['error'] for record in records if record.get('error')]
                trajectory = folder/'trajectory.json'
                if trajectory.exists():
                    traj = json.loads(trajectory.read_text())
                    info = traj.get('info') or {}
                    row['exit_message'] = info.get('exit_message')
                    assistants = [m for m in traj.get('messages',[]) if m.get('role') == 'assistant']
                    row['assistant_messages'] = len(assistants)
                    row['assistant_messages_with_reasoning'] = sum(bool(m.get('reasoning_content')) for m in assistants)
                    if a.include_snippets:
                        row['last_assistants'] = [{k:m.get(k) for k in ('content','tool_calls')}
                                                  for m in assistants[-3:]]
            results.append(row)
    report = {'failed_sets':{model:sorted(ids) for model,ids in failed.items()},
              'intersection':sorted(set.intersection(*failed.values())), 'paired_outcomes':results}
    if not a.include_snippets:
        (a.run_dir/'failure-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
