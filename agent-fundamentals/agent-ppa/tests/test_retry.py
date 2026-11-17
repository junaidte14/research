
import pytest

from tools.retry import (
    TransientToolError,
    execute_with_retry
)


def test_retry_succeeds_after_temporary_failures():
    attempts = {"count": 0}

    def unreliable_tool():
        attempts["count"] += 1

        if attempts["count"] < 3:
            raise TransientToolError("Temporary service failure.")

        return "Success"

    result = execute_with_retry(
        unreliable_tool,
        {},
        max_attempts=3,
        base_delay_seconds=0
    )

    assert result == "Success"
    assert attempts["count"] == 3


def test_retry_stops_after_maximum_attempts():
    attempts = {"count": 0}

    def unreliable_tool():
        attempts["count"] += 1
        raise TransientToolError("Service unavailable.")

    with pytest.raises(TransientToolError):
        execute_with_retry(
            unreliable_tool,
            {},
            max_attempts=3,
            base_delay_seconds=0
        )

    assert attempts["count"] == 3


def test_permanent_error_is_not_retried():
    attempts = {"count": 0}

    def failing_tool():
        attempts["count"] += 1
        raise ValueError("Invalid input.")

    with pytest.raises(ValueError):
        execute_with_retry(
            failing_tool,
            {},
            max_attempts=3,
            base_delay_seconds=0
        )

    assert attempts["count"] == 1


def test_retry_rejects_invalid_attempt_count():
    with pytest.raises(ValueError):
        execute_with_retry(
            lambda: "Success",
            {},
            max_attempts=0
        )
