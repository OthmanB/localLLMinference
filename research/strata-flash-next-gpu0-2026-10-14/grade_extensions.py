#!/usr/bin/env python3
"""Independent behavioral grading of the saved LRU continuations."""
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
DRIVER = '''
from collections import OrderedDict
import random
for invalid in (True, False, 0, -1, 1.5, "2", None):
    try:
        LRUCache(invalid)
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError("invalid capacity accepted: " + repr(invalid))
c = LRUCache(2)
assert c.get("missing") is None
c.put("a", 1); c.put("b", 2); assert c.get("a") == 1
c.put("c", 3); assert c.get("b") is None
c.put("a", 9); c.put("d", 4); assert c.get("c") is None and c.get("a") == 9
rng = random.Random(38144)
for capacity in (1, 2, 7):
    c, ref = LRUCache(capacity), OrderedDict()
    for _ in range(500):
        key = rng.randrange(12)
        if rng.random() < 0.6:
            value = rng.randrange(1000)
            c.put(key, value)
            ref[key] = value; ref.move_to_end(key)
            if len(ref) > capacity: ref.popitem(last=False)
        else:
            expected = ref.get(key)
            if key in ref: ref.move_to_end(key)
            assert c.get(key) == expected, (capacity, key, expected)
print("Independent validation + recency + eviction + 1500 randomized operations: PASS")
'''


def main():
    rows = []
    for path in sorted((ROOT / 'results').glob('iq3_s-extension-*/coding-continuation.json')):
        result = json.loads(path.read_text())
        row: dict[str, Any] = dict(label=path.parent.name, finish=result['finish_reason'],
                   output_tokens=result['usage'].get('completion_tokens'))
        blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', result['content'], re.DOTALL)
        try:
            tree = ast.parse('\n\n'.join(blocks))
            # Remove the generated unittest entry point so independent checks
            # run too, instead of exiting inside unittest.main().
            tree.body = [node for node in tree.body if not (isinstance(node, ast.If) and
                         '__name__' in ast.unparse(node.test))]
            source = ast.unparse(tree)
            if 'class LRUCache' not in source:
                raise ValueError('Requested LRUCache class absent from complete code fences')
            proc = subprocess.run([sys.executable, '-c', source + '\n' + DRIVER],
                                  env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1'),
                                  text=True, capture_output=True, timeout=10)
            row.update(passed=proc.returncode == 0, stdout=proc.stdout, stderr=proc.stderr)
        except Exception as exc:
            row.update(passed=False, error=str(exc))
        (path.parent / 'independent-code-grade.json').write_text(json.dumps(row, indent=2))
        rows.append(row)
    (ROOT / 'extension-code-grades.json').write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
