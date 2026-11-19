#!/usr/bin/python3
"""Check pinned Flash runtime and physical GPU identities before serving."""
import hashlib
import json
from pathlib import Path
import subprocess

EXPECTED = {
    '0': 'GPU-8c1e5d04-43a0-dba7-ebf6-c623bfa2f325',
    '1': 'GPU-bd704e22-5ba1-e0c3-0f66-309a6104b367',
    '2': 'GPU-ae2d59f3-4764-179b-5e78-e5e9c320e744',
}


def gpu_map():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
    result = {}
    for line in output.splitlines():
        index, uuid = line.split(',', 1)
        result[index.strip()] = uuid.strip()
    return result


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if gpu_map() != EXPECTED:
        raise SystemExit('GPU index/UUID mapping changed; refusing a different physical allocation.')
    root = Path('/opt/ai-server/strata/ffa80d7')
    manifest = json.loads((root / 'release-manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if digest(root / name) != expected:
            raise SystemExit(f'Pinned runtime changed: {name}')
    print('Pinned Flash runtime verified; Flash owns GPUs 0,1 and Qwen owns GPU 2.')


if __name__ == '__main__':
    main()
