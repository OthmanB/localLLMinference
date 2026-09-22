from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gsm8k_recovery_eval import (  # noqa: E402
    Example,
    ParsedResponse,
    PROMPT_PREFIX,
    RECOVERY_FOLLOWUP,
    TRANSCRIPT_RECONSTRUCTION,
    answers_equal,
    build_prompt,
    build_continuation_messages,
    build_summary,
    extract_boxed_answer,
    evaluate,
    load_dataset,
    parse_numeric_answer,
    parse_response,
    recovery_eligible,
    score_visible_content,
)


def test_last_balanced_boxed_answer_and_safe_numeric_scoring() -> None:
    assert build_prompt("problem") == PROMPT_PREFIX + "\n\nproblem"
    content = r"Work \boxed{1 + 1}, then \boxed{\frac{3}{2}} and an unmatched \boxed{9"
    assert extract_boxed_answer(content) == r"\frac{3}{2}"
    assert answers_equal(r"\frac{3}{2}\text{ minutes}", "1.5")
    assert answers_equal(r"-\frac{1}{2}", "-0.5")
    assert answers_equal(r"\$1,000", "1000")
    assert parse_numeric_answer("2 + 2") is None
    assert score_visible_content(r"The answer is \boxed{160\text{ minutes}}", "160")["status"] == "correct"
    assert score_visible_content(r"\boxed{7}", "8")["status"] == "incorrect"
    assert score_visible_content("No box", "8")["status"] == "no_answer"


def test_dataset_validates_row_count_and_selected_uniqueness(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    rows = [
        {"problem": "one", "expected_answer": 1},
        {"problem": "two", "expected_answer": 2},
        {"problem": "three", "expected_answer": 3},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    examples, digest, row_count = load_dataset(path, count=2, expected_rows=3)
    assert [example.example_id for example in examples] == ["gsm8k-0", "gsm8k-1"]
    assert len({example.question_sha256 for example in examples}) == 2
    assert len(digest) == 64
    assert row_count == 3

    duplicate = rows.copy()
    duplicate[1] = duplicate[0].copy()
    path.write_text("\n".join(json.dumps(row) for row in duplicate) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not unique"):
        load_dataset(path, count=2, expected_rows=3)


def test_response_parsing_preserves_visible_and_reasoning_fields() -> None:
    parsed = parse_response(
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "role": "assistant",
                        "content": "visible",
                        "reasoning": "private reasoning",
                        "reasoning_content": "alternate",
                    },
                }
            ],
            "usage": {"completion_tokens": 4},
        }
    )
    assert parsed.visible_content == "visible"
    assert parsed.reasoning == "private reasoning"
    assert parsed.finish_reason == "length"
    assert parsed.usage == {"completion_tokens": 4}


def test_recovery_eligibility_is_strict() -> None:
    assert recovery_eligible("unfinished", "length")
    assert recovery_eligible(r"\boxed{wrong}", "length")
    assert not recovery_eligible(r"\boxed{7}", "length")
    assert not recovery_eligible("unfinished", "stop")
    assert not recovery_eligible("", "stop")
    assert recovery_eligible("", "stop", recover_empty_stop=True)


def test_continuation_reconstructs_transcript_without_moving_reasoning_to_content() -> None:
    response = ParsedResponse(
        raw={},
        message={"content": "visible", "reasoning": "reasoning"},
        visible_content="visible",
        reasoning="reasoning",
        finish_reason="length",
        usage=None,
    )
    messages = build_continuation_messages("problem prompt", response)
    assert messages == [
        {"role": "user", "content": "problem prompt"},
        {"role": "assistant", "content": "visible", "reasoning_content": "reasoning"},
        {"role": "user", "content": RECOVERY_FOLLOWUP},
    ]
    assert "reasoning" not in messages[1]["content"]


def test_summary_separates_strict_and_recovered_accounting() -> None:
    examples = [
        Example("gsm8k-0", 0, "one", "1", "a" * 64),
        Example("gsm8k-1", 1, "two", "2", "b" * 64),
        Example("gsm8k-2", 2, "three", "3", "c" * 64),
    ]
    records = [
        {
            "example_id": "gsm8k-0",
            "phase": "first_pass",
            "status": "correct",
            "finish_reason": "stop",
            "continuation_eligible": False,
        },
        {
            "example_id": "gsm8k-1",
            "phase": "first_pass",
            "status": "no_answer",
            "finish_reason": "length",
            "continuation_eligible": True,
        },
        {
            "example_id": "gsm8k-1",
            "phase": "recovery",
            "status": "correct",
            "finish_reason": "stop",
            "continuation_eligible": False,
        },
        {
            "example_id": "gsm8k-2",
            "phase": "first_pass",
            "status": "incorrect",
            "finish_reason": "stop",
            "continuation_eligible": False,
        },
    ]
    summary = build_summary(
        examples,
        records,
        {"dataset_sha256": "d" * 64, "transcript_reconstruction": TRANSCRIPT_RECONSTRUCTION},
    )
    assert summary["complete"] is True
    assert summary["counts"]["continuation_eligible"] == 1
    assert summary["counts"]["continuation_attempts"] == 1
    assert summary["counts"]["continuation_success"] == 1
    assert summary["accuracy"]["strict_first_pass"] == pytest.approx(1 / 3)
    assert summary["accuracy"]["recovered"] == pytest.approx(2 / 3)
    assert summary["provenance"]["dataset_sha256"] == "d" * 64
    assert TRANSCRIPT_RECONSTRUCTION in summary["provenance"].values()


def test_summary_marks_transport_errors_incomplete() -> None:
    examples = [Example("gsm8k-0", 0, "one", "1", "a" * 64)]
    summary = build_summary(
        examples,
        [
            {
                "example_id": "gsm8k-0",
                "phase": "first_pass",
                "status": "error",
                "finish_reason": None,
                "continuation_eligible": False,
            }
        ],
        {},
    )
    assert summary["complete"] is False
    assert summary["counts"]["errors"] == 1


def test_evaluate_persists_each_request_and_recovery_transcript(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = tmp_path / "test.jsonl"
    rows = [{"problem": "one", "expected_answer": 1}, {"problem": "two", "expected_answer": 2}]
    rows.extend({"problem": f"problem-{index}", "expected_answer": index} for index in range(2, 1319))
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_post(url: str, payload: dict[str, object], api_key: str, timeout: float) -> dict[str, object]:
        calls.append(payload)
        if len(calls) == 1:
            return {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "unfinished", "reasoning": "thought"},
                    }
                ]
            }
        if len(calls) == 2:
            return {"choices": [{"finish_reason": "stop", "message": {"content": r"\boxed{1}"}}]}
        return {"choices": [{"finish_reason": "stop", "message": {"content": r"\boxed{3}"}}]}

    monkeypatch.setattr("gsm8k_recovery_eval.post_json", fake_post)
    args = SimpleNamespace(
        base_url="http://127.0.0.1:1/v1",
        model="candidate",
        dataset=dataset,
        count=2,
        output_dir=tmp_path / "output",
        max_tokens=4096,
        continuation_max_tokens=512,
        temperature=1.0,
        top_p=0.95,
        seed=0,
        concurrency=1,
        timeout=5.0,
        api_key="secret-key",
        recover_empty_stop=False,
        allow_fallback_scorer=True,
    )
    summary = evaluate(args)
    predictions = [json.loads(line) for line in (args.output_dir / "predictions.jsonl").read_text().splitlines()]
    audits = [json.loads(line) for line in (args.output_dir / "request-response-audit.jsonl").read_text().splitlines()]
    assert len(calls) == len(predictions) == len(audits) == 3
    assert summary["counts"]["continuation_success"] == 1
    assert calls[1]["messages"][1] == {
        "role": "assistant",
        "content": "unfinished",
        "reasoning_content": "thought",
    }
    assert calls[1]["chat_template_kwargs"] == {"enable_thinking": True, "preserve_thinking": True}
    assert all("secret-key" not in json.dumps(audit) for audit in audits)


def test_evaluate_runs_bounded_concurrent_examples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = tmp_path / "test.jsonl"
    rows = [{"problem": str(index), "expected_answer": index} for index in range(1319)]
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    lock = threading.Lock()
    started = threading.Event()
    active = 0
    peak_active = 0

    def fake_post(url: str, payload: dict[str, object], api_key: str, timeout: float) -> dict[str, object]:
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
            if active == 3:
                started.set()
        assert started.wait(timeout=1)
        with lock:
            active -= 1
        answer = payload["messages"][0]["content"].rsplit("\n\n", 1)[1]
        return {"choices": [{"finish_reason": "stop", "message": {"content": rf"\boxed{{{answer}}}"}}]}

    monkeypatch.setattr("gsm8k_recovery_eval.post_json", fake_post)
    args = SimpleNamespace(
        base_url="http://127.0.0.1:1/v1",
        model="candidate",
        dataset=dataset,
        count=3,
        output_dir=tmp_path / "output",
        max_tokens=4096,
        continuation_max_tokens=512,
        temperature=1.0,
        top_p=0.95,
        seed=0,
        concurrency=3,
        timeout=5.0,
        api_key="",
        recover_empty_stop=False,
        allow_fallback_scorer=True,
    )

    summary = evaluate(args)

    assert peak_active == 3
    assert summary["counts"]["final_after_recovery"]["correct"] == 3
    assert len((args.output_dir / "predictions.jsonl").read_text().splitlines()) == 3
    assert len((args.output_dir / "request-response-audit.jsonl").read_text().splitlines()) == 3


def test_evaluate_scores_on_the_main_thread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = tmp_path / "test.jsonl"
    rows = [{"problem": str(index), "expected_answer": index} for index in range(1319)]
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    score_threads: list[int] = []

    def fake_post(url: str, payload: dict[str, object], api_key: str, timeout: float) -> dict[str, object]:
        answer = payload["messages"][0]["content"].rsplit("\n\n", 1)[1]
        return {"choices": [{"finish_reason": "stop", "message": {"content": rf"\boxed{{{answer}}}"}}]}

    def fake_score(content: str, expected_answer: str) -> dict[str, object]:
        score_threads.append(threading.get_ident())
        return {"status": "correct", "boxed_answer": expected_answer, "expected_answer": expected_answer}

    monkeypatch.setattr("gsm8k_recovery_eval.post_json", fake_post)
    monkeypatch.setattr("gsm8k_recovery_eval.score_visible_content", fake_score)
    args = SimpleNamespace(
        base_url="http://127.0.0.1:1/v1",
        model="candidate",
        dataset=dataset,
        count=3,
        output_dir=tmp_path / "output",
        max_tokens=4096,
        continuation_max_tokens=512,
        temperature=1.0,
        top_p=0.95,
        seed=0,
        concurrency=3,
        timeout=5.0,
        api_key="",
        recover_empty_stop=False,
        allow_fallback_scorer=True,
    )

    evaluate(args)

    assert score_threads == [threading.get_ident()] * 3
