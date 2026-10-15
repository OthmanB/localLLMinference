"""Small offline statistical and ledger-contract tests; no runner imports."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import combined_swe_report as report


def task(state, resolved=None, **extra):
    result = {"state": state, "attempts": [], **extra}
    if resolved is not None:
        result["evaluation"] = {"resolved": resolved}
    return report.task_view(result)


def row(i, q, f, repo="org/repo", cohort="original40"):
    return {"instance_id": f"org__repo-{i}", "repo": repo, "cohort": cohort,
            "models": {"qwen": q, "flash": f}}


class StatisticsTests(unittest.TestCase):
    def test_submission_does_not_mean_resolved(self):
        result = report.summarize([row(1, task("submitted"), task("submitted", True))], 100)
        self.assertEqual(result["pairs"]["unknown_pairs"], 1)
        self.assertIsNone(result["paired_difference95"])
        self.assertIsNone(result["mcnemar_exact_two_sided_p"])
        q = result["models"]["qwen"]
        self.assertEqual(q["submitted_missing_official_bool"], 1)
        self.assertIsNone(q["wilson95"])
        self.assertIsNone(q["complete_success_score"])
        self.assertEqual(q["success_fraction_upper_bound"], 1)

    def test_unknown_pairs_and_all_four_determinate_cells(self):
        yes, no = task("submitted", True), task("submitted", False)
        rows = [row(1, yes, yes), row(2, yes, no), row(3, no, yes), row(4, no, no),
                row(5, task("deadline"), no), row(6, task("infra_failed"), yes)]
        result = report.summarize(rows, 100)
        self.assertEqual(result["pairs"], {"both_resolved": 1, "qwen_only": 1, "flash_only": 1,
                                           "both_unresolved": 1, "unknown_pairs": 2})
        self.assertEqual(result["models"]["qwen"]["infra_failed"], 1)
        self.assertIsNone(result["paired_difference_qwen_minus_flash"])

    def test_terminal_failure_protocol_not_official_score(self):
        failed = task("failed")
        self.assertIs(failed["outcome"], False)
        self.assertFalse(failed["official_scored"])
        self.assertEqual(failed["category"], "generation_no_patch_failure")
        self.assertIsNone(task("failed", patch="a patch exists")["outcome"])
        for state in ("queued", "running", "deadline", "interrupted", "infra_failed"):
            self.assertIsNone(task(state, False)["outcome"])
        self.assertIsNone(task("submitted", "false")["outcome"])

    def test_disjoint_and_duplicate_ids_rejected(self):
        one = [{"instance_id": "org__repo-1"}]
        report.enforce_disjoint([one, [{"instance_id": "org__repo-2"}]])
        for cohorts in ([one, one], [one + one]):
            with self.assertRaises(ValueError):
                report.enforce_disjoint(cohorts)

    def test_exact_mcnemar_known_values(self):
        self.assertEqual(report.mcnemar(0, 0), 1)
        self.assertEqual(report.mcnemar(5, 0), .0625)
        self.assertEqual(report.mcnemar(0, 5), .0625)
        self.assertEqual(report.mcnemar(3, 1), .625)
        self.assertEqual(report.mcnemar(4, 4), 1)

    def test_wilson_edges_and_known_value(self):
        low, high = report.wilson(0, 40)
        self.assertAlmostEqual(low, 0)
        self.assertAlmostEqual(high, .0876216011972866)
        low2, high2 = report.wilson(40, 40)
        self.assertAlmostEqual(low2, 1-high)
        self.assertAlmostEqual(high2, 1)
        self.assertAlmostEqual(report.wilson(20, 40)[0], .35199526933465375)

    def test_bootstrap_seed_and_fixed_strata(self):
        rows = [{"cohort": "a", "repo": "r", "qwen": True, "flash": False},
                {"cohort": "a", "repo": "r", "qwen": False, "flash": True},
                {"cohort": "b", "repo": "s", "qwen": True, "flash": False}]
        first = report.paired_bootstrap(rows, 1000)
        self.assertEqual(first, report.paired_bootstrap(rows, 1000))
        self.assertEqual(first, [-1/3, 1])
        self.assertIsNone(report.paired_bootstrap([rows[0], rows[2]], 100))
        with self.assertRaises(ValueError):
            report.paired_bootstrap([{**rows[0], "qwen": None}], 10)

    def test_attempt_metrics_include_retries_but_not_missing_as_zero(self):
        view = task("submitted", True, attempts=[
            {"outcome": "infra_retry", "wall_seconds": 4, "metrics": {"completion_tokens": 5, "backend_requests": 2, "agent_calls": 1}},
            {"outcome": "submitted", "wall_seconds": 8, "metrics": {"agent_calls": 3}},
        ])
        result = report.model_summary([view])
        self.assertEqual(result["infra_retry_attempts"], 1)
        self.assertEqual(result["metrics"]["backend_calls"]["total_observed"], 5)
        self.assertEqual(result["metrics"]["output_tokens"]["attempts_measured"], 1)
        self.assertIsNone(result["metrics"]["output_tokens"]["mean_task_total_if_all_attempts_measured"])
        self.assertEqual(result["metrics"]["wall_seconds"]["mean_task_total_if_all_attempts_measured"], 12)

    def test_manifest_and_ledger_contract_and_output_filter(self):
        # All fixture writes remain inside the authorized study directory.
        with tempfile.TemporaryDirectory(prefix="combined-report-tests-", dir=report.HERE) as temp:
            root = Path(temp)
            original_hash = None
            ledger = {}
            ledger_path = root/"runs/primary-single-instance/ledger.json"
            for cohort, run, source, offset in (
                ("original40", "primary-single-instance", "runs/primary-single-instance/frozen/source-manifest.json", 0),
                ("additional40", "additional40-single-instance", "extra-cohort/manifest.json", 40),
            ):
                data = {"selection": {"seed": 20260830 if offset == 0 else 20261016},
                        "tasks": [{"instance_id": f"org__repo-{i}", "repo": "org/repo"} for i in range(offset, offset+40)]}
                if offset:
                    data["original_manifest_sha256"] = original_hash
                data["manifest_sha256"] = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                original_hash = data["manifest_sha256"] if not offset else original_hash
                target = root/source
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(data))
                if not offset:
                    ledger = {"phase": "complete", "tasks": {m: {
                        t["instance_id"]: {"state": "submitted", "patch": "SECRET_PATCH", "evaluation": {"resolved": True, "report": "/machine/secret"},
                                           "attempts": [{"error": "SECRET_RESPONSE", "directory": "/machine/secret", "metrics": {"completion_tokens": 7}}]}
                        for t in data["tasks"]} for m in report.MODELS}}
                    ledger_path = root/"runs"/run/"ledger.json"
                    ledger_path.write_text(json.dumps(ledger))
            result = report.build_report(root, 100)
            self.assertEqual(result["cohorts"]["combined80"]["tasks"], 80)
            self.assertEqual(result["cohorts"]["additional40"]["pairs"]["unknown_pairs"], 40)
            self.assertEqual(result["cohorts"]["original40"]["models"]["qwen"]["complete_success_score"], 1)
            self.assertFalse(result["integration_gate"]["ready_for_complete_comparison"])
            for output in (json.dumps(result), report.markdown(result)):
                self.assertNotIn("SECRET", output)
                self.assertNotIn("/machine/secret", output)
                self.assertNotIn(str(root), output)
            del ledger["tasks"]["flash"]["org__repo-0"]
            ledger_path.write_text(json.dumps(ledger))
            with self.assertRaises(ValueError):
                report.build_report(root, 100)

    def test_json_duplicate_task_key_rejected(self):
        with tempfile.TemporaryDirectory(prefix="combined-report-tests-", dir=report.HERE) as temp:
            path = Path(temp)/"ledger.json"
            path.write_text('{"tasks":{"qwen":{"org__repo-1":{},"org__repo-1":{}}}}')
            with self.assertRaises(ValueError):
                report.load(path)


if __name__ == "__main__":
    unittest.main()
