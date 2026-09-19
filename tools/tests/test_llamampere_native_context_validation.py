from __future__ import annotations

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llamampere_native_context_validation import (  # noqa: E402
    CONTEXT_TOKENS,
    SENTINEL_BEGIN,
    SENTINEL_END,
    TARGET_TOKENS,
    RunResult,
    validate_result,
)


def test_validate_result_requires_exact_prompt_token_count() -> None:
    result = RunResult(
        gpu=1,
        port=18121,
        run=1,
        response={
            "truncated": False,
            "tokens_evaluated": TARGET_TOKENS,
            "_prompt_tokens_expected": TARGET_TOKENS + 1,
            "content": f"{SENTINEL_BEGIN}\n{SENTINEL_END}",
        },
        error=None,
        metrics=[{"gpu": 1}],
        host_metrics=[{"timestamp": 1}],
        log_path="server.log",
        cleanup_ok=True,
    )

    failures = validate_result(result)

    assert any("expected=" in failure for failure in failures)


def test_validate_result_accepts_an_exact_full_context_run() -> None:
    result = RunResult(
        gpu=2,
        port=18122,
        run=1,
        response={
            "truncated": False,
            "tokens_evaluated": TARGET_TOKENS,
            "_prompt_tokens_expected": TARGET_TOKENS,
            "content": f"{SENTINEL_BEGIN}\n{SENTINEL_END}",
        },
        error=None,
        metrics=[{"gpu": 2}],
        host_metrics=[{"timestamp": 1}],
        log_path="server.log",
        cleanup_ok=True,
    )

    assert validate_result(result) == []
    assert CONTEXT_TOKENS > TARGET_TOKENS
