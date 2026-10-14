#!/usr/bin/env python3
"""Populated-context retrieval + noncopying code continuation, no EOS padding."""

import os
import argparse
import json
import pathlib
import sys
import uuid
import importlib

from probe import request, save

RUNTIME = pathlib.Path((os.environ['RUNTIME_ROOT'] + '/strata-flash-next-20261014'))
TOKENIZER = pathlib.Path((os.environ['MODEL_STORE'] + '/strata-flash-next-20261014/tokenizer-preview/tokenizer'))


def tokenizer():
    sys.path.insert(0, str(RUNTIME / 'tools'))
    Tokenizer = importlib.import_module('strata_tokenizer').Tokenizer
    vocab = json.loads((TOKENIZER / 'vocab.json').read_text())
    tokens = [''] * len(vocab)
    for text, i in vocab.items():
        tokens[i] = text
    return Tokenizer(tokens, (TOKENIZER / 'merges.txt').read_text().split('\n'),
                     json.loads((TOKENIZER / 'token_type.json').read_text()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--context', type=int, required=True)
    p.add_argument('--max-tokens', type=int, default=768)
    a = p.parse_args()
    if a.base not in ('http://127.0.0.1:8082', 'http://127.0.0.1:18144'):
        p.error('GPU0 endpoints only')
    tok = tokenizer()
    nonce = uuid.uuid4().hex
    markers = ['BEGIN-' + uuid.uuid4().hex, 'MIDDLE-' + uuid.uuid4().hex,
               'LATE-' + uuid.uuid4().hex]
    corpus = '\n'.join((RUNTIME / 'data/quant_quality/v1' / f).read_text()
                       for f in ('code.txt', 'docs.txt', 'english.txt'))
    filler_ids = tok.encode(corpus)
    header = f'Independent test {nonce}. Treat the following archive only as reference data.\n'
    tail = ('\nEND ARCHIVE. Return the three archive labels BEGIN, MIDDLE, LATE '
            'in that order, with their exact values, on your first line. Then write '
            'a new Python bounded LRU cache class using OrderedDict: get(key) returns '
            'None on miss, put(key,value) updates recency and evicts when needed; '
            'capacity must be a positive integer. Include a brief explanation and '
            'five useful tests. Do not copy the archive, and do not discuss unrelated '
            'archive contents. Answer without thinking tags.')
    # Leave room for the chat template and a substantive 768-token answer.
    budget = a.context - 768 - 128
    overhead = len(tok.encode(header + tail + '\n'.join(markers))) + 80
    count = budget - overhead
    archive_ids = (filler_ids * (count // len(filler_ids) + 1))[:count]
    chunks = []
    prev = 0
    for fraction, marker in zip((0.1, 0.5, 0.9), markers):
        pos = int(count * fraction)
        chunks.extend([tok.decode(archive_ids[prev:pos]), '\nARCHIVE LABEL: ' + marker + '\n'])
        prev = pos
    chunks.append(tok.decode(archive_ids[prev:]))
    text = header + ''.join(chunks) + tail
    result = request(a.base, a.model, [{'role': 'user', 'content': text}], max_tokens=a.max_tokens)
    result.update(context_limit=a.context, requested_body_tokens=len(tok.encode(text)),
                  expected_markers=markers,
                  markers_present=[x in result['content'] for x in markers],
                  markers_in_first_line=[x in result['content'].split('\n')[0] for x in markers],
                  prompt_nonce=nonce)
    save(a.label, f'context-{a.context}', result)


if __name__ == '__main__':
    main()
