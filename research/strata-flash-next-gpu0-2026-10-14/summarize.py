#!/usr/bin/env python3
"""Summarize saved trial timings/resources without making inference requests."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = []
for folder in sorted((ROOT / 'results').glob('unsloth-ud-q4_k_xl-*')):
    for path in folder.glob('context-*.json'):
        result = json.loads(path.read_text())
        samples_path = ROOT / f'{folder.name}.resources.jsonl'
        samples = [json.loads(line) for line in samples_path.read_text().splitlines()] if samples_path.exists() else []
        gpu0 = [s['gpu'].splitlines()[0].split(',') for s in samples]
        row = dict(label=folder.name, context=result['context_limit'],
                   prompt_tokens=result['usage']['prompt_tokens'],
                   completion_tokens=result['usage']['completion_tokens'],
                   prefill_tps=result['timings']['prompt_per_second'],
                   decode_tps=result['timings']['predicted_per_second'],
                   ttft_s=result['ttft_s'], wall_s=result['wall_s'],
                   cached_tokens=result['timings']['cache_n'],
                   markers=result['markers_present'], finish=result['finish_reason'],
                   gpu0_peak_used_mib=max((float(g[1]) for g in gpu0), default=None),
                   gpu0_min_free_mib=min((float(g[2]) for g in gpu0), default=None))
        for key in ('rss', 'cgroup_memory_bytes', 'cgroup_anon_bytes', 'cgroup_file_bytes',
                    'swap_used', 'cgroup_swap_bytes'):
            row['peak_' + key] = max((s[key] for s in samples if key in s), default=None)
        row['min_available_ram_bytes'] = min((s['available_ram'] for s in samples), default=None)
        rows.append(row)
rows.sort(key=lambda row: row['context'])
(ROOT / 'q4-initial-summary.json').write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
