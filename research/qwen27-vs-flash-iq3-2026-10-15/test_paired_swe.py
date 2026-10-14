"""Offline controller contract tests. No HTTP requests or Docker launches."""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import paired_swe as swe
import official_eval


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=swe.HERE, prefix="unit-")
        self.root = Path(self.temp.name)
        swe.STOP.clear()

    def tearDown(self):
        self.temp.cleanup()
        swe.STOP.clear()

    def test_failure_denominator_never_uses_submitted_as_resolved(self):
        tasks = {str(i): {"state": "failed", "attempts": []} for i in range(40)}
        tasks["0"]["state"] = "submitted"
        tasks["1"].update(state="submitted", evaluation={"resolved": True})
        tasks["2"].update(state="submitted", evaluation={"resolved": False})
        tasks["3"]["state"] = "deadline"
        summary = swe.summarize({"phase": "evaluation", "tasks": {"qwen": tasks}})["models"]["qwen"]
        self.assertEqual(summary["denominator"], 40)
        self.assertEqual(summary["official_resolved"], 1)
        self.assertEqual(summary["resolved_fraction_all_tasks"], 0.025)
        self.assertEqual(summary["not_officially_evaluated"], 38)

    def test_only_infrastructure_retries(self):
        self.assertEqual(swe.classify("APIConnectionError", "connection refused")[0], "infra_retry")
        for status in ["LimitsExceeded", "RepeatedFormatError", "Submitted", "BadRequestError"]:
            self.assertEqual(swe.classify(status, "old observation: connection refused")[0], "failed")
        self.assertEqual(swe.classify("ContextWindowExceededError", "connection refused")[0], "failed")
        self.assertEqual(swe.classify("", "connection refused", timed_out=True), ("failed", "task_timeout"))
        self.assertEqual(swe.classify("", "", interrupted=True)[0], "interrupted")

    def test_metrics_count_format_errors_and_expose_usage_missing(self):
        path = self.root / "trajectory.json"
        response = {"usage": {"prompt_tokens": 100, "completion_tokens": 7}}
        swe.atomic(path, {"info": {"exit_status": "LimitsExceeded"}, "messages": [
            {"role": "assistant", "extra": {"actions": [{"command": "true"}], "response": response}},
            {"role": "user", "extra": {"interrupt_type": "FormatError", "response": response}},
            {"role": "tool", "extra": {"raw_output": "hello"}},
        ]})
        result = swe.metrics(path)
        self.assertEqual(result["agent_calls"], 2)
        self.assertEqual(result["completion_tokens"], 14)
        self.assertEqual(result["max_context_tokens"], 107)
        self.assertEqual(result["tool_calls"], 1)
        (self.root / "requests.jsonl").write_text(json.dumps({"latency_seconds": 3, "error": "disconnect"}) + "\n")
        result = swe.metrics(path)
        self.assertEqual(result["usage_missing_calls"], 1)
        self.assertEqual(result["request_latency_seconds"], [3])

    def test_official_wrapper_is_offline_and_bounded_without_image_pull(self):
        client, spec = Mock(), Mock()
        spec.image, spec.instance_id = "cached:latest", "flask__flask-1"
        official_eval.cached_container(spec, client, "fresh", Mock())
        client.images.get.assert_called_once_with("cached:latest")
        client.images.pull.assert_not_called()
        kwargs = client.containers.create.call_args.kwargs
        self.assertEqual(kwargs["network_mode"], "none")
        self.assertEqual(kwargs["nano_cpus"], 2_000_000_000)
        self.assertEqual(kwargs["mem_limit"], "6g")

    def test_predictions_include_every_failure_as_empty(self):
        ledger = {"tasks": {"qwen": {
            "a": {"state": "submitted", "patch": "diff --git a/x b/x"},
            "b": {"state": "failed", "patch": "must not leak stale patch"},
            "c": {"state": "deadline"},
        }}}
        path = self.root / "preds.jsonl"
        swe.predictions(self.root, ledger, "qwen", ["a", "b", "c"], path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["model_patch"] for row in rows[1:]], ["", ""])

    def test_official_report_is_required_and_false_is_terminal(self):
        (self.root / "qwen").mkdir()
        ledger = {"phase": "smoke_evaluation", "envelope_deadline_epoch": time.time() + 60,
                  "tasks": {"qwen": {"a": {"state": "submitted", "patch": "patch", "attempts": []}}},
                  "evaluations": []}
        config = {"models": {"qwen": {}}, "python": "not-executed"}

        def fake_supervise(command, folder, log, deadline, heartbeat, cleanup, record):
            path = folder / "logs/run_evaluation" / record["run_id"] / "qwen/a/report.json"
            path.parent.mkdir(parents=True)
            swe.atomic(path, {"a": {"resolved": False}})
            return 0, 1, False

        with patch.object(swe, "supervise", side_effect=fake_supervise):
            self.assertTrue(swe.evaluate(self.root, config, ledger, ["a"], "smoke"))
        self.assertFalse(ledger["tasks"]["qwen"]["a"]["evaluation"]["resolved"])
        with patch.object(swe, "supervise") as supervisor:
            self.assertTrue(swe.evaluate(self.root, config, ledger, ["a"], "full"))
            supervisor.assert_not_called()
        del ledger["tasks"]["qwen"]["a"]["evaluation"]
        with patch.object(swe, "supervise", return_value=(0, 1, False)):
            self.assertFalse(swe.evaluate(self.root, config, ledger, ["a"], "full"))
        self.assertNotEqual(ledger["evaluations"][0]["run_id"], ledger["evaluations"][1]["run_id"])

    def test_immutable_validation_detects_mutation(self):
        (self.root / "frozen").mkdir()
        for name in ("paired_swe.py", "instrumented_model.py", "official_eval.py"):
            (self.root / "frozen" / name).write_bytes((swe.HERE / name).read_bytes())
        source = self.root / "source.py"
        source.write_text("original")
        swe.atomic(self.root / "immutable-sha256.json", {"source.py": swe.digest(source)})
        swe.atomic(self.root / "config.json", {"versions": {}, "installed_source_hashes": {}})
        swe.verify(self.root)
        source.write_text("modified")
        with self.assertRaisesRegex(RuntimeError, "immutable artifact changed"):
            swe.verify(self.root)

    def test_process_recovery_never_kills_reused_pid(self):
        with patch.object(swe, "process_identity", return_value="new-start"), patch.object(os, "killpg") as kill:
            swe.recover_process({"process_pid": 1234, "process_identity": "old-start"})
            kill.assert_not_called()

    def test_worker_retries_infrastructure_once_and_preserves_attempts(self):
        (self.root / "qwen").mkdir()
        ledger = {"phase": "generation", "generation_deadline_epoch": time.time() + 60,
                  "tasks": {"qwen": {"a": {"state": "queued", "attempts": []}}}}
        config = {"order": ["a"], "mini_extra": "not-executed", "task_seconds": 1800,
                  "infra_retries": 1, "dataset": {"name": "offline", "split": "test"}}
        calls = []

        def fake(command, folder, log, deadline, heartbeat, cleanup, record):
            calls.append(record["id"])
            if len(calls) == 1:
                log.write_text("APIConnectionError: connection refused")
                return 1, 1, False
            log.write_text("submitted")
            swe.atomic(log.parent / "trajectory.json", {"info": {"exit_status": "Submitted", "submission": "diff --git a/x b/x"}})
            return 0, 1, False

        with patch.object(swe, "load_yaml", return_value={"environment": {"run_args": []}}), \
             patch.object(swe, "supervise", side_effect=fake):
            swe.worker(self.root, config, ledger, "qwen", ["a"])
        task = ledger["tasks"]["qwen"]["a"]
        self.assertEqual(task["state"], "submitted")
        self.assertEqual(len(set(calls)), 2)
        self.assertEqual([a["outcome"] for a in task["attempts"]], ["infra_retry", "submitted"])
        # A successful prediction must never be sampled again on resume.
        with patch.object(swe, "supervise") as supervisor:
            swe.worker(self.root, config, ledger, "qwen", ["a"])
            supervisor.assert_not_called()

    def test_worker_marks_unattempted_deadline_without_request(self):
        (self.root / "qwen").mkdir()
        ledger = {"phase": "generation", "generation_deadline_epoch": time.time() - 1,
                  "tasks": {"qwen": {"a": {"state": "queued", "attempts": []}}}}
        with patch.object(swe, "supervise") as supervisor:
            swe.worker(self.root, {}, ledger, "qwen", ["a"])
            supervisor.assert_not_called()
        self.assertEqual(ledger["tasks"]["qwen"]["a"]["state"], "deadline")

    def test_atomic_write_leaves_valid_full_document(self):
        path = self.root / "ledger.json"
        swe.atomic(path, {"states": ["running"] * 100})
        swe.atomic(path, {"states": ["failed"] * 40})
        self.assertEqual(swe.load(path), {"states": ["failed"] * 40})
        self.assertEqual(list(self.root.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
