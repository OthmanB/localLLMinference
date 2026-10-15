"""Offline extra-cohort tests: no inference, image pulls or service operations."""
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import extra_prepare as prep
import extra_swe


class ExtraTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=prep.HERE, prefix="extra-unit-")
        self.root = Path(self.temp.name)
        self.original = {"tasks": [{"instance_id": f"old-{i}"} for i in range(40)],
                         "selection": {"quotas": {"repo/a": 20, "repo/b": 19, "repo/exhausted": 1}}}
        self.rows = [{"instance_id": f"new-{repo}-{i}", "repo": f"repo/{repo}", "problem_statement": "x" * (i + 1),
                      "base_commit": "a" * 40, "version": "1"} for repo in ("a", "b") for i in range(80)]
        self.baseline = {name: prep.sha(prep.HERE / name) for name in prep.PROTECTED}

    def tearDown(self):
        self.assertEqual(self.baseline, {name: prep.sha(prep.HERE / name) for name in prep.PROTECTED})
        self.temp.cleanup()

    def fixture(self):
        selected, quotas, availability, smoke = prep.selection(self.rows, self.original)
        manifest = {"selection": {"smoke_ids": smoke}, "tasks": [
            {"instance_id": r["instance_id"], "repo": r["repo"]} for r in selected]}
        manifest["manifest_sha256"] = prep.body_hash(manifest)
        policy = {"manifest_sha256": manifest["manifest_sha256"], "smoke_ids": smoke, "count": 40,
                  "source_hashes": {n: prep.sha(prep.HERE / n) for n in (*prep.PROTECTED, "extra_prepare.py", "extra_swe.py")}}
        prep.write(self.root / "manifest.json", manifest)
        prep.write(self.root / "policy.json", policy)
        prep.write(self.root / "original-ids.json", [r["instance_id"] for r in self.original["tasks"]])
        prep.write(self.root / "seal.HOST-ONLY.json", {"files": {n: prep.sha(self.root / n) for n in ("manifest.json", "policy.json", "original-ids.json")}})
        return manifest

    def test_exact_disjoint_deterministic_metadata_selection(self):
        augmented = self.rows + [{"instance_id": "old-0", "repo": "repo/a", "problem_statement": "old"}]
        a = prep.selection(augmented, self.original)
        b = prep.selection(list(reversed(augmented)), self.original)
        self.assertEqual(a, b)
        self.assertEqual(len(a[0]), 40)
        self.assertEqual(len({r["instance_id"] for r in a[0]}), 40)
        self.assertFalse({r["instance_id"] for r in a[0]} & {f"old-{i}" for i in range(40)})
        self.assertEqual(a[1], {"repo/a": 21, "repo/b": 19, "repo/exhausted": 0})

    def test_each_input_length_stratum_represented_and_no_outcome_influence(self):
        chosen, quotas, _, _ = prep.selection(self.rows, self.original)
        for repo, count in quotas.items():
            if not count:
                continue
            population = sorted([r for r in self.rows if r["repo"] == repo], key=lambda r: len(r["problem_statement"]))
            picked = {r["instance_id"] for r in chosen if r["repo"] == repo}
            for i in range(count):
                stratum = population[i * len(population) // count:(i + 1) * len(population) // count]
                self.assertEqual(len(picked & {r["instance_id"] for r in stratum}), 1)
        poisoned = [{**r, "patch": "gold-poison", "resolved": True, "image_cached": False} for r in self.rows]
        self.assertEqual([r["instance_id"] for r in chosen], [r["instance_id"] for r in prep.selection(poisoned, self.original)[0]])

    def test_wrapper_uses_new_different_repo_smokes_and_leaves_source_untouched(self):
        manifest = self.fixture()
        import paired_swe
        old_smoke = paired_swe.SMOKE
        try:
            swe, _, _ = extra_swe.controller(self.root)
            self.assertEqual(swe.SMOKE, manifest["selection"]["smoke_ids"])
            self.assertFalse(set(swe.SMOKE) & set(old_smoke))
            repos = {r["instance_id"]: r["repo"] for r in manifest["tasks"]}
            self.assertNotEqual(repos[swe.SMOKE[0]], repos[swe.SMOKE[1]])
        finally:
            paired_swe.SMOKE = old_smoke

    def test_source_hash_guard_fails_before_import_or_work(self):
        self.fixture()
        real_sha = prep.sha

        def changed(path):
            return "changed" if Path(path) == prep.HERE / "extra_swe.py" else real_sha(path)

        with patch.object(prep, "sha", side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, "source hash guard failed"):
                extra_swe.controller(self.root)

    def test_missing_image_prevents_initialization(self):
        inherited = {"generation_deadline_epoch": 10**12, "envelope_deadline_epoch": 10**12 + 1}
        with patch.object(extra_swe, "controller", return_value=(None, {}, {"original_manifest_file_sha256": "x", "primary_deadlines": inherited})), \
             patch.object(extra_swe, "deadlines", return_value={"generation_deadline_epoch": 10**12, "envelope_deadline_epoch": 10**12 + 1}), \
             patch.object(prep, "sha", return_value="x"), \
             patch.object(prep, "images_cached", side_effect=RuntimeError("missing images")):
            from argparse import Namespace
            target = self.root / "new-run"
            with self.assertRaisesRegex(RuntimeError, "missing images"):
                extra_swe.init(Namespace(cohort=self.root, run_dir=target, primary_run=self.root / "primary"))
            self.assertFalse(target.exists())

    def test_exact_deadlines_read_without_mutation(self):
        primary = self.root / "primary"
        primary.mkdir()
        prep.write(primary / "ledger.json", {"generation_deadline_epoch": 1000, "envelope_deadline_epoch": 2000})
        before = prep.sha(primary / "ledger.json")
        self.assertEqual(extra_swe.deadlines(primary), {"generation_deadline_epoch": 1000, "envelope_deadline_epoch": 2000})
        self.assertEqual(prep.sha(primary / "ledger.json"), before)


if __name__ == "__main__":
    unittest.main()
