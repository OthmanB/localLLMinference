#!/usr/bin/env python3
"""Compact recorded extended-context table, independent of the live runner."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = []
for path in (ROOT / 'results').glob('iq3_s-extension-*/documents.json'):
    d = json.loads(path.read_text())
    if 'resources' not in d:
        continue
    coding = json.loads((path.parent / 'coding-continuation.json').read_text()) if (path.parent / 'coding-continuation.json').exists() else {}
    grade = json.loads((path.parent / 'independent-code-grade.json').read_text()) if (path.parent / 'independent-code-grade.json').exists() else {}
    row = dict(label=path.parent.name, context=d['context_limit'], scaled=d['scaled'],
               actual_input=d['usage']['prompt_tokens'], cache_n=d['timings']['cache_n'],
               prefill_tps=d['timings']['prompt_per_second'], ttft_s=d['ttft_s'],
               document_decode_tps=d['timings']['predicted_per_second'],
               document_output=d['usage']['completion_tokens'], document_finish=d['finish_reason'],
               document_grade=d['grading']['all_correct'],
               code_decode_tps=coding.get('timings', {}).get('predicted_per_second'),
               code_output=coding.get('usage', {}).get('completion_tokens'),
               code_finish=coding.get('finish_reason'), code_grade=grade.get('passed'),
               resources=d['resources'], placements=d['placements'])
    rows.append(row)
rows.sort(key=lambda x: (x['scaled'], x['context']))
(ROOT / 'extension-summary.json').write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
