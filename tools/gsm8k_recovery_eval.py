#!/usr/bin/env python3
"""Run a standalone, recovery-aware GSM8K evaluation over an OpenAI API."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

try:
    from sgl_eval._vendored.nemo_skills.math_grader import (
        extract_answer as _pinned_extract_answer,
        math_equal as _pinned_math_equal,
    )
except ImportError:
    _pinned_extract_answer = None
    _pinned_math_equal = None


DEFAULT_DATASET = Path("/home/michel/.cache/sgl_eval/gsm8k/test.jsonl")
EXPECTED_DATASET_ROWS = 1319
DEFAULT_COUNT = 256
PROMPT_PREFIX = r"Solve the following math problem. Make sure to put the answer (and only answer) inside \boxed{}."
RECOVERY_FOLLOWUP = r"Continue and finish with the answer (and only answer) inside \boxed{}."
TRANSCRIPT_RECONSTRUCTION = "transcript reconstruction, not KV/token continuation"
NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
UNIT_SUFFIX_RE = re.compile(r"(?:[A-Za-z]+|%)(?:\s+(?:[A-Za-z]+|%))*")


class DatasetError(ValueError):
    """The pinned dataset does not satisfy the evaluator contract."""


class ResponseParseError(ValueError):
    """An HTTP response was not a usable chat completion."""


class ScoringError(ValueError):
    """The configured GSM8K scorer could not score an answer."""


@dataclass(frozen=True)
class Example:
    example_id: str
    row_index: int
    problem: str
    expected_answer: str
    question_sha256: str


@dataclass(frozen=True)
class ParsedResponse:
    raw: dict[str, Any]
    message: dict[str, Any]
    visible_content: str
    reasoning: str
    finish_reason: str | None
    usage: Any


@dataclass(frozen=True)
class CompletionAttempt:
    example: Example
    phase: str
    payload: dict[str, Any]
    response: dict[str, Any] | None
    parsed: ParsedResponse | None
    error: Exception | None
    elapsed_seconds: float


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_dataset(
    path: Path,
    count: int = DEFAULT_COUNT,
    expected_rows: int = EXPECTED_DATASET_ROWS,
) -> tuple[list[Example], str, int]:
    """Load and validate the JSONL dataset, selecting its first unique rows."""
    if count < 1:
        raise DatasetError("count must be positive")
    raw = path.read_bytes()
    dataset_sha256 = sha256_bytes(raw)
    lines = raw.splitlines()
    if len(lines) != expected_rows:
        raise DatasetError(f"expected {expected_rows} dataset rows, found {len(lines)}")
    if count > len(lines):
        raise DatasetError(f"count {count} exceeds dataset rows {len(lines)}")

    examples: list[Example] = []
    all_ids: set[str] = set()
    all_hashes: set[str] = set()
    for row_index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise DatasetError(f"invalid JSON on row {row_index + 1}: {error.msg}") from error
        if not isinstance(row, dict):
            raise DatasetError(f"row {row_index + 1} is not an object")
        problem = row.get("problem")
        if not isinstance(problem, str) or not problem:
            raise DatasetError(f"row {row_index + 1} has no non-empty problem")
        if "expected_answer" not in row:
            raise DatasetError(f"row {row_index + 1} has no expected_answer")
        expected_answer = str(row["expected_answer"])
        if parse_numeric_answer(expected_answer) is None:
            raise DatasetError(f"row {row_index + 1} has a non-numeric expected_answer")
        example_id = str(row.get("id") or f"gsm8k-{row_index}")
        question_sha256 = sha256_bytes(problem.encode("utf-8"))
        if example_id in all_ids or question_sha256 in all_hashes:
            raise DatasetError("dataset rows are not unique")
        all_ids.add(example_id)
        all_hashes.add(question_sha256)
        if row_index < count:
            examples.append(Example(example_id, row_index, problem, expected_answer, question_sha256))

    if len(examples) != count:
        raise DatasetError(f"selected {len(examples)} rows, expected {count}")
    return examples, dataset_sha256, len(lines)


def build_prompt(problem: str) -> str:
    return f"{PROMPT_PREFIX}\n\n{problem}"


def _balanced_group_end(text: str, opening_index: int) -> int | None:
    if opening_index >= len(text) or text[opening_index] != "{":
        return None
    depth = 0
    escaped = False
    for index in range(opening_index, len(text)):
        character = text[index]
        if character == "{" and not escaped:
            depth += 1
        elif character == "}" and not escaped:
            depth -= 1
            if depth == 0:
                return index
        escaped = character == "\\" and not escaped
        if character != "\\":
            escaped = False
    return None


def extract_boxed_answer(content: str) -> str | None:
    """Return the content of the last balanced ``\\boxed{...}`` occurrence."""
    candidates: list[str] = []
    for match in re.finditer(r"\\boxed\s*\{", content):
        opening_index = match.end() - 1
        closing_index = _balanced_group_end(content, opening_index)
        if closing_index is not None:
            candidates.append(content[opening_index + 1 : closing_index])
    return candidates[-1] if candidates else None


def _remove_known_tex_groups(text: str) -> str | None:
    """Remove unit-formatting groups without interpreting expressions."""
    pattern = re.compile(r"\\(?:text|mathrm|textrm|mbox)\s*\{")
    while True:
        match = pattern.search(text)
        if match is None:
            return text
        opening_index = match.end() - 1
        closing_index = _balanced_group_end(text, opening_index)
        if closing_index is None:
            return None
        text = text[: match.start()] + text[closing_index + 1 :]


def _allowed_suffix(value: str) -> bool:
    return not value or UNIT_SUFFIX_RE.fullmatch(value.strip()) is not None


def _parse_numeric(value: str, depth: int = 0) -> Fraction | None:
    if depth > 4:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    cleaned = cleaned.replace("\\left", "").replace("\\right", "")
    cleaned = re.sub(r"\\(?:!|,|;|:|>| )", "", cleaned)
    cleaned = cleaned.replace("\\%", "%").replace("\\$", "$").replace("$", "")
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", cleaned).strip()
    without_units = _remove_known_tex_groups(cleaned)
    if without_units is None:
        return None
    cleaned = without_units.strip()

    if cleaned[:1] in {"+", "-"} and cleaned[1:].lstrip().startswith((r"\frac", r"\dfrac", r"\tfrac")):
        value = _parse_numeric(cleaned[1:].lstrip(), depth + 1)
        return -value if value is not None and cleaned[0] == "-" else value

    for command in (r"\frac", r"\dfrac", r"\tfrac"):
        if not cleaned.startswith(command):
            continue
        position = len(command)
        while position < len(cleaned) and cleaned[position].isspace():
            position += 1
        if position >= len(cleaned) or cleaned[position] != "{":
            return None
        numerator_end = _balanced_group_end(cleaned, position)
        if numerator_end is None:
            return None
        position = numerator_end + 1
        while position < len(cleaned) and cleaned[position].isspace():
            position += 1
        if position >= len(cleaned) or cleaned[position] != "{":
            return None
        denominator_end = _balanced_group_end(cleaned, position)
        if denominator_end is None or not _allowed_suffix(cleaned[denominator_end + 1 :]):
            return None
        numerator = _parse_numeric(cleaned[cleaned.find("{") + 1 : numerator_end], depth + 1)
        denominator = _parse_numeric(cleaned[position + 1 : denominator_end], depth + 1)
        if numerator is None or denominator in (None, 0):
            return None
        return numerator / denominator

    if cleaned.count("/") == 1:
        numerator_text, denominator_text = cleaned.split("/", 1)
        numerator = _parse_numeric(numerator_text, depth + 1)
        denominator = _parse_numeric(denominator_text, depth + 1)
        if numerator is not None and denominator not in (None, 0):
            return numerator / denominator
        return None

    match = NUMBER_RE.fullmatch(cleaned)
    if match is not None:
        try:
            return Fraction(Decimal(match.group(0)))
        except (InvalidOperation, ValueError, ZeroDivisionError):
            return None
    number_match = NUMBER_RE.match(cleaned)
    if number_match is None or not _allowed_suffix(cleaned[number_match.end() :]):
        return None
    try:
        return Fraction(Decimal(number_match.group(0)))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def parse_numeric_answer(value: str) -> Fraction | None:
    """Parse numeric GSM8K answers without evaluating arbitrary expressions."""
    return _parse_numeric(value)


def answers_equal(candidate: str, expected: str) -> bool:
    candidate_number = parse_numeric_answer(candidate)
    expected_number = parse_numeric_answer(expected)
    return candidate_number is not None and expected_number is not None and candidate_number == expected_number


def score_visible_content(content: str, expected_answer: str) -> dict[str, object]:
    if _pinned_extract_answer is not None and _pinned_math_equal is not None:
        boxed_answer = _pinned_extract_answer(content, extract_from_boxed=True, relaxed=False)
        if boxed_answer is None:
            return {"status": "no_answer", "boxed_answer": None, "expected_answer": expected_answer}
        try:
            correct = bool(
                _pinned_math_equal(
                    expected_answer,
                    boxed_answer,
                    numeric_precision=15,
                    timeout_seconds=10,
                )
            )
        except Exception as error:
            raise ScoringError(f"pinned GSM8K scorer failed: {error}") from error
        return {
            "status": "correct" if correct else "incorrect",
            "boxed_answer": boxed_answer,
            "expected_answer": expected_answer,
            "scorer": "sgl_eval-nemo_skills-pinned",
        }

    boxed_answer = extract_boxed_answer(content)
    if boxed_answer is None or parse_numeric_answer(boxed_answer) is None:
        return {"status": "no_answer", "boxed_answer": boxed_answer, "expected_answer": expected_answer}
    return {
        "status": "correct" if answers_equal(boxed_answer, expected_answer) else "incorrect",
        "boxed_answer": boxed_answer,
        "expected_answer": expected_answer,
        "scorer": "fallback-numeric",
    }


def parse_response(response: dict[str, Any]) -> ParsedResponse:
    if not isinstance(response, dict):
        raise ResponseParseError("completion response is not a JSON object")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ResponseParseError("completion response has no choices")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ResponseParseError("completion choice has no message")
    content = message.get("content")
    if content is None:
        visible_content = ""
    elif isinstance(content, str):
        visible_content = content
    else:
        raise ResponseParseError("completion message content is not a string")
    reasoning = message.get("reasoning")
    if not isinstance(reasoning, str):
        reasoning = message.get("reasoning_content")
    if not isinstance(reasoning, str):
        reasoning = ""
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        finish_reason = str(finish_reason)
    return ParsedResponse(response, message, visible_content, reasoning, finish_reason, response.get("usage"))


def recovery_eligible(
    visible_content: str,
    finish_reason: str | None,
    recover_empty_stop: bool = False,
) -> bool:
    """Allow only one recovery for an unboxed length stop, plus the opt-in empty stop."""
    boxed_answer = extract_boxed_answer(visible_content)
    if boxed_answer is not None and parse_numeric_answer(boxed_answer) is not None:
        return False
    if finish_reason == "length":
        return True
    return recover_empty_stop and finish_reason == "stop" and not visible_content


def build_continuation_messages(prompt: str, response: ParsedResponse) -> list[dict[str, str]]:
    """Reconstruct a chat transcript; this does not resume server-side KV state."""
    return [
        {"role": "user", "content": prompt},
        {
            "role": "assistant",
            "content": response.visible_content,
            "reasoning_content": response.reasoning,
        },
        {"role": "user", "content": RECOVERY_FOLLOWUP},
    ]


def completion_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def safe_url(url: str) -> str:
    parts = urlsplit(url)
    hostname = parts.hostname or ""
    if parts.port is not None:
        hostname = f"{hostname}:{parts.port}"
    return urlunsplit((parts.scheme, hostname, parts.path, "", ""))


def post_json(url: str, payload: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ResponseParseError("completion response is not a JSON object")
    return value


def request_payload(
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "n": 1,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
    }


def _error_message(error: Exception, api_key: str) -> str:
    message = str(error).replace(api_key, "[redacted]") if api_key else str(error)
    return message[:500]


def _status_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(record.get("status", "error") for record in records)
    return {status: counts.get(status, 0) for status in ("correct", "incorrect", "no_answer", "error")}


def _reason_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        record.get("finish_reason") if record.get("finish_reason") is not None else "none"
        for record in records
    )
    return dict(sorted(counts.items()))


def build_summary(
    examples: list[Example],
    prediction_records: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    first = {record["example_id"]: record for record in prediction_records if record.get("phase") == "first_pass"}
    recovery = {
        record["example_id"]: record for record in prediction_records if record.get("phase") == "recovery"
    }
    expected_ids = {example.example_id for example in examples}
    missing_first = sorted(expected_ids - set(first))
    missing_recovery = sorted(
        example_id
        for example_id, record in first.items()
        if record.get("continuation_eligible") and example_id not in recovery
    )
    first_records = [first[example.example_id] for example in examples if example.example_id in first]
    recovery_records = [recovery[example.example_id] for example in examples if example.example_id in recovery]
    final_statuses: list[str] = []
    for example in examples:
        first_record = first.get(example.example_id)
        if first_record is None:
            continue
        final_statuses.append(recovery.get(example.example_id, first_record).get("status", "error"))
    first_statuses = _status_counts(first_records)
    recovery_statuses = _status_counts(recovery_records)
    final_counts = _status_counts([{"status": status} for status in final_statuses])
    example_count = len(examples)
    error_count = first_statuses["error"] + recovery_statuses["error"]
    complete = (
        not missing_first
        and not missing_recovery
        and len(first_records) == example_count
        and error_count == 0
    )
    recovery_correct = recovery_statuses["correct"]
    strict_accuracy = first_statuses["correct"] / example_count if example_count else 0.0
    recovered_accuracy = final_counts["correct"] / example_count if example_count else 0.0
    return {
        "status": "complete" if complete else "incomplete",
        "complete": complete,
        "counts": {
            "examples": example_count,
            "first_pass": first_statuses,
            "final_after_recovery": final_counts,
            "continuation_eligible": sum(
                1 for record in first_records if record.get("continuation_eligible")
            ),
            "continuation_attempts": len(recovery_records),
            "continuation_success": recovery_correct,
            "recovery": recovery_statuses,
            "errors": error_count,
        },
        "strict_first_pass_accuracy": strict_accuracy,
        "recovered_accuracy": recovered_accuracy,
        "accuracy": {
            "strict_first_pass": strict_accuracy,
            "recovered": recovered_accuracy,
            "recovery_only": recovery_correct / len(recovery_records) if recovery_records else None,
        },
        "finish_reasons": {
            "first_pass": _reason_counts(first_records),
            "recovery": _reason_counts(recovery_records),
        },
        "error_types": {
            "first_pass": dict(
                sorted(Counter(record.get("error_type", "unknown") for record in first_records if record.get("status") == "error").items())
            ),
            "recovery": dict(
                sorted(Counter(record.get("error_type", "unknown") for record in recovery_records if record.get("status") == "error").items())
            ),
        },
        "incomplete_examples": {"first_pass": missing_first, "recovery": missing_recovery},
        "provenance": provenance,
    }


def _audit_record(
    example: Example,
    phase: str,
    url: str,
    payload: dict[str, Any],
    response: dict[str, Any] | None,
    parsed: ParsedResponse | None,
    error: Exception | None,
    elapsed_seconds: float,
    api_key: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "example_id": example.example_id,
        "row_index": example.row_index,
        "question_sha256": example.question_sha256,
        "phase": phase,
        "timestamp": time.time(),
        "elapsed_seconds": elapsed_seconds,
        "url": safe_url(url),
        "request": payload,
        "transcript_mode": TRANSCRIPT_RECONSTRUCTION if phase == "recovery" else "initial chat request",
        "raw_response": response,
    }
    if parsed is not None:
        result["response_message"] = {
            "content": parsed.visible_content,
            "reasoning": parsed.message.get("reasoning"),
            "reasoning_content": parsed.message.get("reasoning_content"),
        }
        result["finish_reason"] = parsed.finish_reason
    if error is not None:
        result["error"] = {"type": type(error).__name__, "message": _error_message(error, api_key)}
    return result


def _prediction_record(
    example: Example,
    phase: str,
    parsed: ParsedResponse | None,
    score: dict[str, object],
    continuation_eligible: bool,
    error: Exception | None,
    api_key: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "example_id": example.example_id,
        "row_index": example.row_index,
        "question_sha256": example.question_sha256,
        "phase": phase,
        "expected_answer": example.expected_answer,
        "status": score["status"],
        "boxed_answer": score.get("boxed_answer"),
        "visible_content": parsed.visible_content if parsed is not None else "",
        "finish_reason": parsed.finish_reason if parsed is not None else None,
        "usage": parsed.usage if parsed is not None else None,
        "reasoning_present": bool(parsed.reasoning) if parsed is not None else False,
        "continuation_eligible": continuation_eligible,
    }
    if error is not None:
        record["status"] = "error"
        record["error_type"] = type(error).__name__
        record["error_message"] = _error_message(error, api_key)
    return record


def _write_jsonl(stream: Any, value: dict[str, Any]) -> None:
    json.dump(value, stream, ensure_ascii=True, separators=(",", ":"))
    stream.write("\n")
    stream.flush()


def _fetch_completion(
    example: Example,
    phase: str,
    url: str,
    payload: dict[str, Any],
    args: argparse.Namespace,
) -> CompletionAttempt:
    started = time.monotonic()
    response: dict[str, Any] | None = None
    parsed: ParsedResponse | None = None
    error: Exception | None = None
    try:
        response = post_json(url, payload, args.api_key, args.timeout)
        parsed = parse_response(response)
    except Exception as caught:
        error = caught
    return CompletionAttempt(example, phase, payload, response, parsed, error, time.monotonic() - started)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    if (
        _pinned_extract_answer is None or _pinned_math_equal is None
    ) and not getattr(args, "allow_fallback_scorer", False):
        raise ValueError(
            "pinned sgl-eval GSM8K scorer is unavailable; use its environment or explicitly allow fallback"
        )
    examples, dataset_sha256, dataset_rows = load_dataset(args.dataset, args.count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    audit_path = args.output_dir / "request-response-audit.jsonl"
    summary_path = args.output_dir / "summary.json"
    url = completion_url(args.base_url)
    provenance = {
        "evaluator": "gsm8k_recovery_eval.py",
        "dataset_path": str(args.dataset),
        "dataset_sha256": dataset_sha256,
        "dataset_rows": dataset_rows,
        "selected_count": len(examples),
        "selected_ids": [example.example_id for example in examples],
        "selected_question_hashes": {
            example.example_id: example.question_sha256 for example in examples
        },
        "selected": [
            {"id": example.example_id, "question_sha256": example.question_sha256} for example in examples
        ],
        "prompt_prefix": PROMPT_PREFIX,
        "model": args.model,
        "base_url": safe_url(args.base_url),
        "max_tokens": args.max_tokens,
        "continuation_max_tokens": args.continuation_max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "thinking": True,
        "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
        "concurrency": args.concurrency,
        "timeout": args.timeout,
        "recover_empty_stop": args.recover_empty_stop,
        "transcript_reconstruction": TRANSCRIPT_RECONSTRUCTION,
        "scorer": "sgl_eval-nemo_skills-pinned"
        if _pinned_extract_answer is not None and _pinned_math_equal is not None
        else "fallback-numeric",
    }
    prediction_records: list[dict[str, Any]] = []
    with predictions_path.open("w", encoding="utf-8") as predictions, audit_path.open(
        "w", encoding="utf-8"
    ) as audit:
        def persist(audit_record: dict[str, Any], prediction_record: dict[str, Any]) -> None:
            _write_jsonl(audit, audit_record)
            _write_jsonl(predictions, prediction_record)

        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            remaining_examples = iter(examples)
            futures: dict[Any, tuple[str, ParsedResponse | None]] = {}

            def submit_first() -> bool:
                try:
                    example = next(remaining_examples)
                except StopIteration:
                    return False
                payload = request_payload(
                    args.model,
                    [{"role": "user", "content": build_prompt(example.problem)}],
                    args.max_tokens,
                    args.temperature,
                    args.top_p,
                    args.seed,
                )
                futures[executor.submit(_fetch_completion, example, "first_pass", url, payload, args)] = (
                    "first_pass",
                    None,
                )
                return True

            for _ in range(min(args.concurrency, len(examples))):
                submit_first()

            while futures:
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    phase, first_parsed = futures.pop(future)
                    attempt = future.result()
                    if phase == "first_pass":
                        record_error = attempt.error
                        if attempt.error is None and attempt.parsed is not None:
                            try:
                                score = score_visible_content(attempt.parsed.visible_content, attempt.example.expected_answer)
                            except Exception as caught:
                                record_error = caught
                                score = {
                                    "status": "error",
                                    "boxed_answer": None,
                                    "expected_answer": attempt.example.expected_answer,
                                }
                        else:
                            score = {
                                "status": "error",
                                "boxed_answer": None,
                                "expected_answer": attempt.example.expected_answer,
                            }
                        continuation_eligible = (
                            record_error is None
                            and attempt.parsed is not None
                            and recovery_eligible(
                                attempt.parsed.visible_content,
                                attempt.parsed.finish_reason,
                                args.recover_empty_stop,
                            )
                        )
                        prediction_record = _prediction_record(
                            attempt.example,
                            "first_pass",
                            attempt.parsed,
                            score,
                            continuation_eligible,
                            record_error,
                            args.api_key,
                        )
                        persist(
                            _audit_record(
                                attempt.example,
                                "first_pass",
                                url,
                                attempt.payload,
                                attempt.response,
                                attempt.parsed,
                                record_error,
                                attempt.elapsed_seconds,
                                args.api_key,
                            ),
                            prediction_record,
                        )
                        prediction_records.append(prediction_record)
                        if continuation_eligible and attempt.parsed is not None:
                            payload = request_payload(
                                args.model,
                                build_continuation_messages(build_prompt(attempt.example.problem), attempt.parsed),
                                args.continuation_max_tokens,
                                args.temperature,
                                args.top_p,
                                args.seed,
                            )
                            futures[
                                executor.submit(
                                    _fetch_completion,
                                    attempt.example,
                                    "recovery",
                                    url,
                                    payload,
                                    args,
                                )
                            ] = ("recovery", attempt.parsed)
                        else:
                            submit_first()
                        continue

                    record_error = attempt.error
                    if attempt.error is None and attempt.parsed is not None and first_parsed is not None:
                        try:
                            score = score_visible_content(
                                first_parsed.visible_content + attempt.parsed.visible_content,
                                attempt.example.expected_answer,
                            )
                        except Exception as caught:
                            record_error = caught
                            score = {
                                "status": "error",
                                "boxed_answer": None,
                                "expected_answer": attempt.example.expected_answer,
                            }
                    else:
                        score = {
                            "status": "error",
                            "boxed_answer": None,
                            "expected_answer": attempt.example.expected_answer,
                        }
                    prediction_record = _prediction_record(
                        attempt.example,
                        "recovery",
                        attempt.parsed,
                        score,
                        False,
                        record_error,
                        args.api_key,
                    )
                    prediction_record["transcript_reconstruction"] = TRANSCRIPT_RECONSTRUCTION
                    prediction_record["recovered_visible_content"] = (
                        first_parsed.visible_content + attempt.parsed.visible_content
                        if first_parsed is not None and attempt.parsed is not None
                        else first_parsed.visible_content
                        if first_parsed is not None
                        else ""
                    )
                    persist(
                        _audit_record(
                            attempt.example,
                            "recovery",
                            url,
                            attempt.payload,
                            attempt.response,
                            attempt.parsed,
                            record_error,
                            attempt.elapsed_seconds,
                            args.api_key,
                        ),
                        prediction_record,
                    )
                    prediction_records.append(prediction_record)
                    submit_first()

    summary = build_summary(examples, prediction_records, provenance)
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=True)
        stream.write("\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible server base URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gsm8k-recovery"))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--continuation-max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument(
        "--allow-fallback-scorer",
        action="store_true",
        help="Allow the restricted numeric scorer when the pinned sgl-eval scorer is unavailable",
    )
    parser.add_argument(
        "--recover-empty-stop",
        action="store_true",
        help="Also recover an empty visible response whose finish_reason is stop",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.count < 1 or args.max_tokens < 1 or args.continuation_max_tokens < 1 or args.concurrency < 1 or args.timeout <= 0:
        parser.error("count, token limits, concurrency, and timeout must be positive")
    try:
        summary = evaluate(args)
    except (DatasetError, OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
