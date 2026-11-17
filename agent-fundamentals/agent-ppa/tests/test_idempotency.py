
from tools.idempotency import IdempotencyStore, idempotency_store


def test_new_operation_can_be_claimed(tmp_path):
    store = IdempotencyStore(
        tmp_path / "test.db"
    )

    result = store.claim(
        "call_001",
        "send_message",
        {
            "recipient": "test@example.com",
            "message": "Hello"
        }
    )

    assert result["status"] == "new"


def test_duplicate_operation_is_detected(tmp_path):
    store = IdempotencyStore(
        tmp_path / "test.db"
    )

    arguments = {
        "recipient": "test@example.com",
        "message": "Hello"
    }

    store.claim("call_002", "send_message", arguments)
    store.finish("call_002", "Message approved (simulation only).")

    result = store.claim(
        "call_002",
        "send_message",
        arguments
    )

    assert result["status"] == "completed"
    assert result["result"] == "Message approved (simulation only)."


def test_same_key_with_different_arguments_is_rejected(tmp_path):
    store = IdempotencyStore(
        tmp_path / "test.db"
    )

    store.claim(
        "call_003",
        "send_message",
        {
            "recipient": "first@example.com",
            "message": "Hello"
        }
    )

    result = store.claim(
        "call_003",
        "send_message",
        {
            "recipient": "second@example.com",
            "message": "Hello"
        }
    )

    assert result["status"] == "conflict"


def test_in_progress_operation_is_detected(tmp_path):
    store = IdempotencyStore(
        tmp_path / "test.db"
    )

    arguments = {"expression": "2 + 2"}

    store.claim(
        "call_004",
        "calculator",
        arguments
    )

    result = store.claim(
        "call_004",
        "calculator",
        arguments
    )

    assert result["status"] == "in_progress"


def test_failed_operation_is_not_reexecuted_automatically(tmp_path):
    store = IdempotencyStore(
        tmp_path / "test.db"
    )

    arguments = {"expression": "2 / 0"}

    store.claim(
        "call_005",
        "calculator",
        arguments
    )

    store.finish(
        "call_005",
        "Tool execution error: Division by zero.",
        status="failed"
    )

    result = store.claim(
        "call_005",
        "calculator",
        arguments
    )

    assert result["status"] == "failed"
    assert "Division by zero" in result["result"]


def test_lookup_returns_none_for_new_operation(tmp_path):
    store = IdempotencyStore(
        tmp_path / "test.db"
    )
    result = store.lookup("new-operation")

    assert result is None


def test_lookup_returns_completed_operation(tmp_path):
    store = IdempotencyStore(
        tmp_path / "test.db"
    )
    store.claim(
        "operation-123",
        "send_message",
        {"recipient": "test@example.com", "message": "Hello"}
    )

    store.finish(
        "operation-123",
        "Message approved (simulation only).",
        status="completed"
    )

    result = store.lookup("operation-123")

    assert result == {
        "status": "completed",
        "result": "Message approved (simulation only)."
    }
