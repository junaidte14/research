
from tools.idempotency import IdempotencyStore
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

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

    claim = store.claim(
        "call_002",
        "send_message",
        arguments
    )

    store.finish(
        idempotency_key="call_002",
        lease_token=claim["lease_token"],
        result="Message approved (simulation only)."
    )

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

    claim = store.claim(
        "call_005",
        "calculator",
        arguments
    )

    store.finish(
        idempotency_key="call_005",
        lease_token=claim["lease_token"],
        result="Tool execution error: Division by zero.",
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
    claim = store.claim(
        "operation-123",
        "send_message",
        {
            "recipient": "test@example.com",
            "message": "Hello"
        }
    )

    store.finish(
        idempotency_key="operation-123",
        lease_token=claim["lease_token"],
        result="Message approved (simulation only).",
        status="completed"
    )

    result = store.lookup("operation-123")

    assert result == {
        "status": "completed",
        "result": "Message approved (simulation only)."
    }

def test_concurrent_claim_allows_only_one_new_operation(tmp_path):
    store = IdempotencyStore(
        tmp_path / "concurrent_test.db"
    )

    def claim_operation():
        return store.claim(
            idempotency_key="concurrent_operation_001",
            tool_name="send_message",
            arguments={
                "recipient": "test@example.com",
                "message": "Hello"
            }
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda _: claim_operation(),
                range(8)
            )
        )

    statuses = [result["status"] for result in results]

    assert statuses.count("new") == 1
    assert statuses.count("in_progress") == 7

def test_new_operation_receives_lease(tmp_path):

    store = IdempotencyStore(
        tmp_path / "lease_test.db"
    )

    result = store.claim(
        idempotency_key="lease_test_001",
        tool_name="calculator",
        arguments={"expression": "10 + 5"}
    )

    assert result["status"] == "new"
    assert result["lease_token"] is not None

    with store._connect() as connection:

        operation = connection.execute(
            """
            SELECT *
            FROM operations
            WHERE idempotency_key = ?
            """,
            ("lease_test_001",)
        ).fetchone()

    assert operation["lease_token"] == result["lease_token"]
    assert operation["attempt_count"] == 1
    assert operation["lease_expires_at"] is not None

def test_operation_can_be_finished_with_valid_lease(tmp_path):

    store = IdempotencyStore(
        tmp_path / "finish_test.db"
    )

    claim = store.claim(
        idempotency_key="finish_001",
        tool_name="calculator",
        arguments={"expression": "10 + 5"}
    )

    result = store.finish(
        idempotency_key="finish_001",
        lease_token=claim["lease_token"],
        result="15"
    )

    assert result is True

    operation = store.lookup("finish_001")

    assert operation["status"] == "completed"
    assert operation["result"] == "15"

def test_operation_cannot_be_finished_with_invalid_lease(tmp_path):

    store = IdempotencyStore(
        tmp_path / "invalid_lease_test.db"
    )

    store.claim(
        idempotency_key="finish_002",
        tool_name="calculator",
        arguments={"expression": "10 + 5"}
    )

    result = store.finish(
        idempotency_key="finish_002",
        lease_token="incorrect-token",
        result="15"
    )

    assert result is False

    operation = store.lookup("finish_002")

    assert operation["status"] == "in_progress"
    assert operation["result"] is None

def test_operation_has_future_lease_expiration(tmp_path):

    store = IdempotencyStore(
        tmp_path / "future_lease.db"
    )

    result = store.claim(
        idempotency_key="future_lease_001",
        tool_name="calculator",
        arguments={"expression": "10 + 5"},
        lease_duration_seconds=60
    )

    with store._connect() as connection:
        operation = connection.execute(
            """
            SELECT lease_expires_at
            FROM operations
            WHERE idempotency_key = ?
            """,
            ("future_lease_001",)
        ).fetchone()

    expiration = datetime.fromisoformat(
        operation["lease_expires_at"]
    )

    assert expiration > datetime.now(timezone.utc)

def test_expired_operation_requires_recovery(tmp_path):

    store = IdempotencyStore(
        tmp_path / "expired_test.db"
    )

    claim = store.claim(
        idempotency_key="expired_001",
        tool_name="calculator",
        arguments={"expression": "10 + 5"}
    )

    with store._connect() as connection:
        connection.execute(
            """
            UPDATE operations
            SET lease_expires_at = ?
            WHERE idempotency_key = ?
            """,
            (
                "2000-01-01T00:00:00+00:00",
                "expired_001"
            )
        )

    recovered_count = store.recover_expired_operations()

    assert recovered_count == 1

    operation = store.lookup("expired_001")

    assert operation["status"] == "recovery_required"
    assert operation["result"] is None

def test_active_operation_is_not_recovered(tmp_path):

    store = IdempotencyStore(
        tmp_path / "active_test.db"
    )

    store.claim(
        idempotency_key="active_001",
        tool_name="calculator",
        arguments={"expression": "10 + 5"},
        lease_duration_seconds=300
    )

    recovered_count = store.recover_expired_operations()

    assert recovered_count == 0

    operation = store.lookup("active_001")

    assert operation["status"] == "in_progress"

def test_old_worker_cannot_finish_after_recovery(tmp_path):

    store = IdempotencyStore(
        tmp_path / "stale_worker_test.db"
    )

    claim = store.claim(
        idempotency_key="stale_001",
        tool_name="calculator",
        arguments={"expression": "10 + 5"}
    )

    with store._connect() as connection:
        connection.execute(
            """
            UPDATE operations
            SET lease_expires_at = ?
            WHERE idempotency_key = ?
            """,
            (
                "2000-01-01T00:00:00+00:00",
                "stale_001"
            )
        )

    store.recover_expired_operations()

    result = store.finish(
        idempotency_key="stale_001",
        lease_token=claim["lease_token"],
        result="15"
    )

    assert result is False

    operation = store.lookup("stale_001")

    assert operation["status"] == "recovery_required"

def test_valid_owner_can_renew_lease(tmp_path):
    store = IdempotencyStore(tmp_path / "renew_valid.db")

    claim = store.claim(
        "renew_001",
        "calculator",
        {"expression": "5 + 5"},
        lease_duration_seconds=1
    )

    renewed = store.renew_lease(
        idempotency_key="renew_001",
        lease_token=claim["lease_token"],
        lease_duration_seconds=300
    )

    assert renewed is True

    with store._connect() as connection:
        operation = connection.execute(
            """
            SELECT lease_expires_at
            FROM operations
            WHERE idempotency_key = ?
            """,
            ("renew_001",)
        ).fetchone()

    expiry = datetime.fromisoformat(operation["lease_expires_at"])
    assert expiry > datetime.now(timezone.utc)

def test_invalid_owner_cannot_renew_lease(tmp_path):
    store = IdempotencyStore(tmp_path / "renew_invalid.db")

    store.claim(
        "renew_002",
        "calculator",
        {"expression": "5 + 5"}
    )

    renewed = store.renew_lease(
        idempotency_key="renew_002",
        lease_token="incorrect-token",
        lease_duration_seconds=300
    )

    assert renewed is False

def test_expired_lease_cannot_be_renewed(tmp_path):
    store = IdempotencyStore(tmp_path / "renew_expired.db")

    claim = store.claim(
        "renew_003",
        "calculator",
        {"expression": "5 + 5"}
    )

    with store._connect() as connection:
        connection.execute(
            """
            UPDATE operations
            SET lease_expires_at = ?
            WHERE idempotency_key = ?
            """,
            ("2000-01-01T00:00:00+00:00", "renew_003")
        )

    renewed = store.renew_lease(
        idempotency_key="renew_003",
        lease_token=claim["lease_token"],
        lease_duration_seconds=300
    )

    assert renewed is False

def test_completed_operation_cannot_renew_lease(tmp_path):
    store = IdempotencyStore(tmp_path / "renew_completed.db")

    claim = store.claim(
        "renew_004",
        "calculator",
        {"expression": "5 + 5"}
    )

    finished = store.finish(
        idempotency_key="renew_004",
        lease_token=claim["lease_token"],
        result="10"
    )

    assert finished is True

    renewed = store.renew_lease(
        idempotency_key="renew_004",
        lease_token=claim["lease_token"],
        lease_duration_seconds=300
    )

    assert renewed is False

def test_recovery_required_operations_can_be_listed(tmp_path):
    store = IdempotencyStore(tmp_path / "recovery_list.db")

    claim = store.claim(
        "recovery_list_001",
        "calculator",
        {"expression": "5 + 5"}
    )

    with store._connect() as connection:
        connection.execute(
            """
            UPDATE operations
            SET lease_expires_at = ?
            WHERE idempotency_key = ?
            """,
            ("2000-01-01T00:00:00+00:00", "recovery_list_001")
        )

    store.recover_expired_operations()

    operations = store.get_recovery_required_operations()

    assert len(operations) == 1
    assert operations[0]["idempotency_key"] == "recovery_list_001"
    assert operations[0]["tool_name"] == "calculator"
    assert operations[0]["status"] == "recovery_required"
    assert operations[0]["attempt_count"] == 1

def test_recovery_list_excludes_other_statuses(tmp_path):
    store = IdempotencyStore(tmp_path / "recovery_filter.db")

    store.claim(
        "active_001",
        "calculator",
        {"expression": "5 + 5"}
    )

    completed_claim = store.claim(
        "completed_001",
        "calculator",
        {"expression": "10 + 5"}
    )

    store.finish(
        idempotency_key="completed_001",
        lease_token=completed_claim["lease_token"],
        result="15"
    )

    operations = store.get_recovery_required_operations()

    assert operations == []

def test_recovery_list_is_empty_when_no_operations_need_recovery(tmp_path):
    store = IdempotencyStore(tmp_path / "recovery_empty.db")

    operations = store.get_recovery_required_operations()

    assert operations == []