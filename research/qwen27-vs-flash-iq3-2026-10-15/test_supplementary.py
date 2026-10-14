"""Offline contract tests: no network connections or generated-code execution."""
import io
import json
import importlib.util
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from typing import Any

import supplementary as bench
from supplementary_fixtures import fixture, grade
from supplementary_http import StreamState, payload, sse_events, stream_chat

HERE = Path(__file__).resolve().parent


class CharacterTokenizer:
    def encode(self, text):
        return list(text)


class Contracts(unittest.TestCase):
    def test_sse_framing(self):
        raw = b': comment\r\nevent: message\r\ndata: {"choices": [],\r\ndata: "usage": {"completion_tokens": 17}}\r\n\r\ndata: [DONE]'
        events = list(sse_events(io.BytesIO(raw)))
        self.assertEqual(json.loads(events[0])["usage"]["completion_tokens"], 17)
        self.assertEqual(events[1], "[DONE]")

    def test_usage_not_event_count_and_timing_denominators(self):
        state = StreamState()
        state.feed(json.dumps(dict(choices=[dict(delta=dict(role="assistant"))])), .1)
        self.assertIsNone(state.first)
        state.feed(json.dumps(dict(choices=[dict(delta=dict(content="many tokens batched"))])), 2)
        state.feed(json.dumps(dict(choices=[dict(delta=dict(content=" second chunk"), finish_reason="stop")])), 4)
        state.feed(json.dumps(dict(choices=[], usage=dict(prompt_tokens=100, completion_tokens=27,
                                                        prompt_tokens_details=dict(cached_tokens=11)),
                                  timings=dict(prompt_per_second=50, predicted_per_second=10))), 4.1)
        state.feed("[DONE]", 4.2)
        result = state.result(5)
        self.assertEqual(result["ttft_s"], 2)
        self.assertEqual(result["output_tokens_per_wall_s"], 27 / 5)
        self.assertEqual(result["engine_decode_tps"], 10)
        self.assertIsNone(result["client_decode_tps"])
        self.assertEqual(result["cache_observation"], "reuse_reported")

    def test_missing_cache_counts_are_unknown(self):
        self.assertEqual(StreamState().result(1)["cache_observation"], "unknown")
        state = StreamState()
        state.timings = dict(cache_n=0)
        self.assertEqual(state.result(1)["cache_observation"], "zero_reported")

    def test_payload_same_controls(self):
        body = payload("model", bench.paired_messages("nonce", "task"), 4096)
        self.assertEqual(body["reasoning_effort"], "none")
        self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])
        self.assertTrue(body["messages"][0]["content"].startswith("nonce"))
        self.assertNotIn("cache_prompt", body)
        self.assertFalse(payload("m", [], 1, False)["cache_prompt"])

    def test_fixture_revisions_arithmetic_exact_citations(self):
        data = fixture(CharacterTokenizer(), 16000, 1, "fresh-tag")
        self.assertLess(abs(data["body_tokens"] - 16000), 256)
        self.assertEqual(data, fixture(CharacterTokenizer(), 16000, 1, "fresh-tag"))
        self.assertEqual(data["expected"]["job"]["total_items"], 173 * 160)
        for placement in data["placements"]:
            self.assertIn("END DOCUMENT " + placement["id"], data["text"])
        self.assertTrue(grade(json.dumps(data["expected"]), data["expected"])["all_correct"])
        wrong = json.loads(json.dumps(data["expected"]))
        wrong["evidence"]["policy"].append("SRC-E")
        self.assertFalse(grade(json.dumps(wrong), data["expected"])["all_correct"])
        self.assertEqual(grade("```json\n{}\n```", data["expected"])["outcome"], "json_format")
        self.assertEqual(grade("[]", data["expected"])["outcome"], "json_format")

    def test_capacity_denominators_include_queue(self):
        rows: list[dict[str, Any]] = [dict(usage=dict(completion_tokens=100), started_monotonic=10,
                     ended_monotonic=12, wall_s=2, error=None),
                dict(usage=dict(completion_tokens=50), started_monotonic=10,
                     ended_monotonic=15, wall_s=5, error=None)]
        result = bench.capacity_group(rows)
        self.assertEqual(result["aggregate_output_tokens_per_wall_s"], 150 / 5)
        self.assertEqual(result["per_request_wall_s"]["median"], 3.5)
        rows[1]["error"] = "transport failed"
        self.assertIsNone(bench.capacity_group(rows)["aggregate_output_tokens_per_wall_s"])

    def test_official_denominator_never_drops_failures(self):
        results = dict(eval={"a": [dict(base_status="pass", plus_status="pass")],
                             "b": [dict(base_status="fail", plus_status="fail")],
                             "c": [dict(base_status="pass", plus_status="fail")]})
        score = bench.official_score(results, ["a", "b", "c"])
        self.assertEqual(score["base_pass_at_1"], 2 / 3)
        self.assertEqual(score["plus_pass_at_1"], 1 / 3)
        with self.assertRaises(ValueError):
            bench.official_score(results, ["a", "b", "c", "missing"])

    def test_http_mock_stream_and_incomplete(self):
        class Sock:
            def settimeout(self, value):
                pass
            def shutdown(self, value):
                pass
        class Response(io.BytesIO):
            status = 200
            def getheader(self, name, default=None):
                return "text/event-stream"
        class Connection:
            sock = Sock()
            raw = b'data: {"choices":[{"delta":{"content":"abc"},"finish_reason":"length"}]}\n\ndata: {"usage":{"completion_tokens":12}}\n\ndata: [DONE]\n\n'
            def __init__(self, *args, **kwargs):
                pass
            def connect(self):
                pass
            def request(self, *args, **kwargs):
                pass
            def getresponse(self):
                return Response(self.raw)
            def close(self):
                pass
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path = Path(directory) / "raw.sse"
            with patch("supplementary_http.http.client.HTTPConnection", Connection):
                result = stream_chat("http://unused/v1", {}, time.monotonic() + 5, path)
                self.assertEqual(result["outcome"], "output_cap")
                self.assertEqual(result["usage"]["completion_tokens"], 12)
                Connection.raw = Connection.raw.replace(b'data: [DONE]\n\n', b'')
                result = stream_chat("http://unused/v1", {}, time.monotonic() + 5, path)
                self.assertEqual(result["outcome"], "infrastructure")
                self.assertEqual(result["content"], "abc")

    def test_resume_error_records_no_regeneration(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            args = bench.parser().parse_args(["speed", "--run-dir", directory])
            run = bench.Run(args)
            row = StreamState().result(1)
            row.update(error="HTTP 503", error_category="infrastructure", outcome="infrastructure")
            with patch("supplementary.stream_chat", return_value=row) as client:
                first = run.request("task", "qwen", bench.paired_messages("fresh", "text"))
                again = run.request("task", "qwen", bench.paired_messages("fresh", "text"))
                self.assertEqual(first, again)
                self.assertEqual(client.call_count, 1)
                with self.assertRaises(ValueError):
                    run.request("task", "qwen", bench.paired_messages("changed", "text"))
            run.lock.close()

    def test_official_command_and_score_artifact_without_execution(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            args = bench.parser().parse_args(["evalplus", "--run-dir", directory])
            run = bench.Run(args)
            folder = run.private / "evalplus" / "humaneval" / "qwen"
            folder.mkdir(parents=True)
            samples = folder / "samples.jsonl"
            samples.write_text('{"task_id":"a","solution":""}\n')
            bench.save(folder / "samples_eval_results.json", dict(hash="fixture-hash", eval={
                "a": [dict(base_status="fail", plus_status="fail")]}))
            with patch("supplementary.subprocess.Popen") as process:
                process.return_value.wait.return_value = 0
                info = bench.evaluate_samples(run, "humaneval", "qwen", samples, ["a"], "fixture-hash")
                self.assertEqual(process.call_args.args[0][-2:], ["--parallel", "2"])
                self.assertEqual(process.call_args.kwargs["cwd"], folder / "work")
                self.assertTrue(process.call_args.kwargs["start_new_session"])
                self.assertEqual(info["base_pass_at_1"], 0)
                self.assertEqual(info["denominator"], 1)
                self.assertEqual(info["status"], "complete")
                bench.evaluate_samples(run, "humaneval", "qwen", samples, ["a"], "fixture-hash")
                self.assertEqual(process.call_count, 1)
            run.lock.close()

    @unittest.skipUnless(os.environ.get("STRATA_ROOT") and os.environ.get("STRATA_TOKENIZER"), "operator tokenizer env not set")
    def test_real_readonly_tokenizer_targets(self):
        from supplementary_fixtures import local_tokenizer
        tok = local_tokenizer()
        for target in (4096, 32768, 131072, 245760):
            for kind in ("documents", "code"):
                data = fixture(tok, target, 2715, "offline-validation-tag", kind)
                self.assertLessEqual(abs(data["body_tokens"] - target), max(256, target * .01))
                self.assertLess(data["body_tokens"] + 8192 + 1024, 262144)
                self.assertEqual(len(data["placements"]), 6)

    @unittest.skipUnless(importlib.util.find_spec("evalplus"), "EvalPlus interpreter not selected")
    def test_cached_official_datasets_and_sanitizer(self):
        import importlib
        data = importlib.import_module("evalplus.data")
        sanitize = importlib.import_module("evalplus.sanitize").sanitize
        def no_download(*args, **kwargs):
            raise AssertionError("offline test must not download")
        with patch("wget.download", no_download):
            for getter, hasher, count in ((data.get_human_eval_plus, data.get_human_eval_plus_hash, 164),
                                           (data.get_mbpp_plus, data.get_mbpp_plus_hash, 378)):
                tasks = getter()
                self.assertEqual(len(tasks), count)
                self.assertTrue(hasher())
                task = next(iter(tasks.values()))
                self.assertIn(task["prompt"].strip(), bench.evalplus_prompt(task))
                entry = task["entry_point"]
                source = f"```python\ndef {entry}(*args):\n    return None\n```"
                self.assertIn(f"def {entry}(", sanitize(source, entrypoint=entry))


if __name__ == "__main__":
    unittest.main()
