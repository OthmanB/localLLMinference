"""Read-only local tokenizer and deterministic substantive document fixtures."""
import importlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any


CODE_TASK = ("Write a complete Python LRUCache using OrderedDict. Capacity must be a positive "
             "integer, rejecting bool. get(key) returns None on miss; put updates recency and "
             "evicts the oldest. Include five executable unittest tests for validation, misses, "
             "update recency, access recency, and eviction, and a concise explanation. "
             "Finish naturally; do not pad or repeat the archive.")


def local_tokenizer():
    root = Path(os.environ["STRATA_ROOT"])
    directory = Path(os.environ["STRATA_TOKENIZER"])
    sys.path.insert(0, str(root / "tools"))
    cls = importlib.import_module("strata_tokenizer").Tokenizer
    vocab = json.loads((directory / "vocab.json").read_text())
    if len(vocab) != 248320 or set(vocab.values()) != set(range(248320)):
        raise ValueError("Expected contiguous shared n_vocab=248320")
    tokens = [""] * len(vocab)
    for text, i in vocab.items():
        tokens[i] = text
    return cls(tokens, (directory / "merges.txt").read_text().split("\n"),
               json.loads((directory / "token_type.json").read_text()))


def fixture(tok, target, seed, tag, kind="documents") -> dict[str, Any]:
    rng = random.Random(seed)
    markers = [f"{p}-{rng.getrandbits(96):024x}" for p in ("EARLY", "MID", "LATE")]
    docs = [
        ("SRC-A", f"Early archive marker: {markers[0]}.", .05),
        ("SRC-B", "Cedar policy revision 2 APPROVED: batch_size 128, timeout_seconds 30, region east.", .23),
        ("SRC-C", f"Job cedar-run: 173 batches. Middle archive marker: {markers[1]}.", .45),
        ("SRC-D", "Cedar policy revision 3 APPROVED: batch_size 160, timeout_seconds 45, region west. Supersedes earlier approved cedar policies.", .72),
        ("SRC-E", "Cedar policy revision 4 DRAFT NOT APPROVED: batch_size 999, timeout_seconds 5, region north. Never apply this draft.", .89),
        ("SRC-F", f"Late archive marker: {markers[2]}.", .97),
    ]
    expected = dict(markers=markers,
                    policy=dict(revision=3, batch_size=160, timeout_seconds=45, region="west", status="APPROVED"),
                    job=dict(batches=173, total_items=27680),
                    evidence=dict(markers=["SRC-A", "SRC-C", "SRC-F"], policy=["SRC-D"],
                                  batches=["SRC-C"], rejected_draft=["SRC-E"]))
    header = f"{tag}\nReference archive. AUX services are not cedar.\nBEGIN ARCHIVE\n"
    task = ("\nEND ARCHIVE\nReturn only one JSON object with markers (early, middle, late strings), "
            "policy (revision,batch_size,timeout_seconds,region,status of latest APPROVED cedar policy), "
            "job (batches,total_items for cedar-run), evidence (markers,policy,batches,rejected_draft; "
            "each an array of exact source IDs). Cite only documents proving each answer, with no extra IDs. "
            "Compute total_items from the approved batch size and job batches.")
    if kind == "code":
        task = "\nEND ARCHIVE\nUsing this populated context as background, answer this coding task:\n" + CODE_TASK
    rendered = [f"\nDOCUMENT {name}\n{text}\nEND DOCUMENT {name}\n" for name, text, _ in docs]
    budget = target - len(tok.encode(header + "".join(rendered) + task))
    if budget < 100:
        raise ValueError("body token target too small")
    auxiliary = []
    count = 0
    n = 0
    # Whole unique documents, no EOS padding, no truncated evidence documents.
    def auxiliary_doc(n):
        name = f"AUX-{n:06d}"
        doc = (f"\nDOCUMENT {name}\nService aux-{rng.getrandbits(64):016x}, revision {rng.randrange(1,30)}, "
               f"status {rng.choice(['APPROVED','DRAFT','RETIRED'])}; batch size {rng.randrange(10,900)}; "
               f"timeout {rng.randrange(2,120)} seconds; region {rng.choice(['east','west','north','south'])}; "
               f"job count {rng.randrange(1,400)} batches; inventory checksum {rng.getrandbits(64):016x}. "
               f"Validate approval before scheduling.\nEND DOCUMENT {name}\n")
        return doc
    while count < budget:
        n += 1
        doc = auxiliary_doc(n)
        size = len(tok.encode(doc))
        if count + size > budget:
            break
        auxiliary.append(doc)
        count += size
    def assemble():
        pieces = []
        prev = 0
        for (_, _, fraction), doc in zip(docs, rendered):
            pos = int(len(auxiliary) * fraction)
            pieces.extend(auxiliary[prev:pos])
            pieces.append(doc)
            prev = pos
        pieces.extend(auxiliary[prev:])
        return header + "".join(pieces) + task
    # BPE boundaries differ between standalone documents and concatenated text.
    # Correct using actual full-body encodings, keeping every document whole.
    for _ in range(12):
        text = assemble()
        actual = len(tok.encode(text))
        if actual > target:
            auxiliary.pop()
        elif target - actual > 150:
            average = max(1, (actual - len(tok.encode(header + task + "".join(rendered)))) / max(1, len(auxiliary)))
            for _ in range(max(1, int((target - actual) / average))):
                n += 1
                auxiliary.append(auxiliary_doc(n))
        else:
            break
    text = assemble()
    actual = len(tok.encode(text))
    if abs(actual - target) > max(256, target * .01):
        raise ValueError(f"fixture target drift: {actual} vs {target}")
    return dict(text=text, expected=expected, body_tokens=actual, target_body_tokens=target,
                seed=seed, tag=tag, kind=kind,
                placements=[dict(id=name, body_token_offset=len(tok.encode(text[:text.index('DOCUMENT ' + name)])))
                            for name, _, _ in docs])


def grade(content, expected):
    try:
        obj = json.loads(content)
    except (ValueError, TypeError):
        return dict(outcome="json_format", all_correct=False, valid_json=False)
    if not isinstance(obj, dict):
        return dict(outcome="json_format", all_correct=False, valid_json=False)
    checks = {k: obj.get(k) == v for k, v in expected.items()}
    return dict(valid_json=True, checks=checks, all_correct=all(checks.values()),
                outcome="correct" if all(checks.values()) else "answer_incorrect")
