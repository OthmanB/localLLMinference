#!/usr/bin/env python3
"""Symmetric post-hoc audit of an ambiguous marker schema; originals untouched."""
import argparse
from collections import Counter
import json
from pathlib import Path


def semantic_grade(content, expected):
    try:
        obj = json.loads(content)
    except ValueError:
        return {'all_correct': False, 'checks': {'valid_json': False}}
    if not isinstance(obj, dict):
        return {'all_correct': False, 'checks': {'valid_json': False}}
    markers = obj.get('markers')
    marker_schema = 'array'
    if isinstance(markers, dict) and set(markers) == {'early', 'middle', 'late'}:
        markers = [markers[k] for k in ('early', 'middle', 'late')]
        marker_schema = 'keyed_object'
    checks = {'markers': markers == expected['markers'],
              'policy': obj.get('policy') == expected['policy'],
              'job': obj.get('job') == expected['job']}
    evidence = obj.get('evidence')
    checks['evidence'] = isinstance(evidence, dict) and set(evidence) == set(expected['evidence']) and all(
        isinstance(evidence.get(k), list) and sorted(evidence[k]) == sorted(v)
        for k, v in expected['evidence'].items())
    return {'all_correct': all(checks.values()), 'checks': checks, 'marker_schema': marker_schema}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study-root', type=Path, default=Path(__file__).resolve().parent)
    a = p.parse_args()
    speed = a.study_root / 'campaign/supplementary/speed'
    rows = []
    for path in sorted((speed/'private/records').glob('documents-*.json')):
        record = json.loads(path.read_text())
        fixture = json.loads((speed/'private/fixtures'/f'{record["pair_id"]}.json').read_text())
        rows.append({'key':record['key'], 'model':record['backend'],
                     'target':record['target_body_tokens'],
                     'original_strict_correct':record.get('grading',{}).get('all_correct',False),
                     'semantic_audit':semantic_grade(record['content'], fixture['expected'])})
    summary = {}
    for model in ('qwen','flash'):
        selected = [row for row in rows if row['model'] == model]
        summary[model] = {'denominator':len(selected),
                          'original_strict_correct':sum(row['original_strict_correct'] for row in selected),
                          'semantic_correct':sum(row['semantic_audit']['all_correct'] for row in selected),
                          'marker_schemas':dict(Counter(row['semantic_audit'].get('marker_schema','invalid') for row in selected))}
    result = {'method':'post-hoc semantic audit, applied to every model/point; original grades unchanged',
              'reason':'Prompt did not specify markers must be an array. Accept exact keyed early/middle/late objects too; citation order is not semantic.',
              'models':summary,'records':rows}
    (a.study_root/'campaign/document-grade-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
