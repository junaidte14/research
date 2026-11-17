import pytest
import json
from tools.policy import ToolPolicy
from tools.idempotency import IdempotencyStore

from tools.registry import (
    TOOL_DEFINITIONS,
    TOOL_REGISTRY,
    execute_tool
)

import tools.registry as registry

def test_all_tools_are_registered():
    assert "calculator" in TOOL_REGISTRY
    assert "get_current_datetime" in TOOL_REGISTRY
    assert "analyze_text" in TOOL_REGISTRY


def test_tool_definitions_are_generated():
    names = [
        item["function"]["name"]
        for item in TOOL_DEFINITIONS
    ]

    assert "calculator" in names
    assert "get_current_datetime" in names
    assert "analyze_text" in names


def test_calculator_execution():
    result = execute_tool(
        "calculator",
        json.dumps({"expression": "10 + 5"})
    )

    assert result == "15"


def test_invalid_calculator_arguments():
    result = execute_tool(
        "calculator",
        json.dumps({"wrong_field": "10 + 5"})
    )

    assert "Invalid arguments" in result


def test_unexpected_argument_is_rejected():
    result = execute_tool(
        "calculator",
        json.dumps({
            "expression": "10 + 5",
            "admin": True
        })
    )

    assert "Invalid arguments" in result


def test_unknown_tool():
    result = execute_tool(
        "unknown_tool",
        "{}"
    )

    assert "Unknown tool" in result

def test_authorized_tool_executes():
    policy = ToolPolicy(
        allowed_permissions=frozenset({"math:calculate"})
    )

    result = execute_tool(
        "calculator",
        '{"expression": "10 + 5"}',
        policy=policy
    )

    assert result == "15"


def test_unauthorized_tool_is_blocked():
    policy = ToolPolicy(
        allowed_permissions=frozenset()
    )

    result = execute_tool(
        "calculator",
        '{"expression": "10 + 5"}',
        policy=policy
    )

    assert "Tool authorization error" in result
    assert "math:calculate" in result


def test_authorization_precedes_argument_validation():
    policy = ToolPolicy(
        allowed_permissions=frozenset()
    )

    result = execute_tool(
        "calculator",
        '{"unexpected": true}',
        policy=policy
    )

    assert "Tool authorization error" in result
    assert "Invalid arguments" not in result


def test_approved_action_executes():
    policy = ToolPolicy(
        allowed_permissions=frozenset({"message:send"})
    )

    def approve_action(tool_name, arguments):
        assert tool_name == "send_message"
        assert arguments["recipient"] == "test@example.com"
        return True

    result = execute_tool(
        "send_message",
        '{"recipient":"test@example.com","message":"Hello"}',
        policy=policy,
        approval_callback=approve_action
    )

    assert "SIMULATION ONLY" in result


def test_rejected_action_does_not_execute():
    policy = ToolPolicy(
        allowed_permissions=frozenset({"message:send"})
    )

    def reject_action(tool_name, arguments):
        return False

    result = execute_tool(
        "send_message",
        '{"recipient":"test@example.com","message":"Hello"}',
        policy=policy,
        approval_callback=reject_action
    )

    assert "cancelled by user" in result


def test_approval_required_but_no_handler():
    policy = ToolPolicy(
        allowed_permissions=frozenset({"message:send"})
    )

    result = execute_tool(
        "send_message",
        '{"recipient":"test@example.com","message":"Hello"}',
        policy=policy
    )

    assert "No approval handler" in result
    assert "blocked" in result


def test_unauthorized_action_never_requests_approval():
    policy = ToolPolicy(
        allowed_permissions=frozenset()
    )

    approval_called = False

    def approve_action(tool_name, arguments):
        nonlocal approval_called
        approval_called = True
        return True

    result = execute_tool(
        "send_message",
        '{"recipient":"test@example.com","message":"Hello"}',
        policy=policy,
        approval_callback=approve_action
    )

    assert "Tool authorization error" in result
    assert approval_called is False




@pytest.fixture
def isolated_idempotency_store(tmp_path, monkeypatch):
    store = IdempotencyStore(
        tmp_path / "test_idempotency.db"
    )
    monkeypatch.setattr(
        registry,
        "idempotency_store",
        store
    )
    return store


def test_idempotent_operation_executes_only_once(
    isolated_idempotency_store,
    monkeypatch
):
    tool = registry.TOOL_REGISTRY["calculator"]
    execution_count = 0

    def fake_calculator(expression):
        nonlocal execution_count
        execution_count += 1
        return f"result:{expression}"

    monkeypatch.setattr(tool, "function", fake_calculator)

    first_result = registry.execute_tool(
        "calculator",
        '{"expression": "10 + 5"}',
        idempotency_key="test-operation-001"
    )

    second_result = registry.execute_tool(
        "calculator",
        '{"expression": "10 + 5"}',
        idempotency_key="test-operation-001"
    )

    assert first_result == "result:10 + 5"
    assert second_result == first_result
    assert execution_count == 1


def test_idempotency_key_rejects_different_arguments(
    isolated_idempotency_store,
    monkeypatch
):
    tool = registry.TOOL_REGISTRY["calculator"]
    execution_count = 0

    def fake_calculator(expression):
        nonlocal execution_count
        execution_count += 1
        return f"result:{expression}"

    monkeypatch.setattr(tool, "function", fake_calculator)

    registry.execute_tool(
        "calculator",
        '{"expression": "10 + 5"}',
        idempotency_key="test-operation-002"
    )

    result = registry.execute_tool(
        "calculator",
        '{"expression": "20 + 5"}',
        idempotency_key="test-operation-002"
    )

    assert "Idempotency conflict" in result
    assert execution_count == 1


def test_in_progress_operation_is_not_executed_again(
    isolated_idempotency_store,
    monkeypatch
):
    tool = registry.TOOL_REGISTRY["calculator"]
    execution_count = 0

    def fake_calculator(expression):
        nonlocal execution_count
        execution_count += 1
        return f"result:{expression}"

    monkeypatch.setattr(tool, "function", fake_calculator)

    isolated_idempotency_store.claim(
        "test-operation-003",
        "calculator",
        {"expression": "10 + 5"}
    )

    result = registry.execute_tool(
        "calculator",
        '{"expression": "10 + 5"}',
        idempotency_key="test-operation-003"
    )

    assert "already in progress" in result
    assert execution_count == 0


def test_failed_operation_is_not_executed_again(
    isolated_idempotency_store,
    monkeypatch
):
    tool = registry.TOOL_REGISTRY["calculator"]
    execution_count = 0

    def fake_calculator(expression):
        nonlocal execution_count
        execution_count += 1
        return f"result:{expression}"

    monkeypatch.setattr(tool, "function", fake_calculator)

    claim = isolated_idempotency_store.claim(
        "test-operation-004",
        "calculator",
        {"expression": "10 + 5"}
    )

    isolated_idempotency_store.finish(
        idempotency_key = "test-operation-004",
        result = "Previous simulated failure",
        lease_token = claim["lease_token"],
        status="failed"
    )

    result = registry.execute_tool(
        "calculator",
        '{"expression": "10 + 5"}',
        idempotency_key="test-operation-004"
    )

    assert "Automatic re-execution is blocked" in result
    assert execution_count == 0

def test_recovery_required_operation_is_not_executed(
    isolated_idempotency_store,
    monkeypatch
):
    import tools.registry as registry

    execution_count = 0

    def fake_calculator(expression):
        nonlocal execution_count
        execution_count += 1
        return "10"

    monkeypatch.setattr(
        registry,
        "idempotency_store",
        isolated_idempotency_store
    )

    monkeypatch.setattr(
        registry.TOOL_REGISTRY["calculator"],
        "function",
        fake_calculator
    )

    claim = isolated_idempotency_store.claim(
        "recovery_registry_001",
        "calculator",
        {"expression": "5 + 5"}
    )

    with isolated_idempotency_store._connect() as connection:
        connection.execute(
            """
            UPDATE operations
            SET lease_expires_at = ?
            WHERE idempotency_key = ?
            """,
            (
                "2000-01-01T00:00:00+00:00",
                "recovery_registry_001"
            )
        )

    isolated_idempotency_store.recover_expired_operations()

    result = registry.execute_tool(
        tool_name="calculator",
        arguments='{"expression": "5 + 5"}',
        idempotency_key="recovery_registry_001"
    )

    assert "Recovery required" in result
    assert execution_count == 0

def test_lost_lease_during_execution_does_not_report_success(
    isolated_idempotency_store,
    monkeypatch
):
    import tools.registry as registry

    monkeypatch.setattr(
        registry,
        "idempotency_store",
        isolated_idempotency_store
    )

    def fake_calculator(expression):
        # Simulate the lease expiring while the tool is executing.
        with isolated_idempotency_store._connect() as connection:
            connection.execute(
                """
                UPDATE operations
                SET lease_expires_at = ?
                WHERE idempotency_key = ?
                """,
                (
                    "2000-01-01T00:00:00+00:00",
                    "lost_lease_001"
                )
            )

        # Simulate another process detecting the expired lease.
        isolated_idempotency_store.recover_expired_operations()

        # The original worker still returns a result.
        return "10"

    monkeypatch.setattr(
        registry.TOOL_REGISTRY["calculator"],
        "function",
        fake_calculator
    )

    result = registry.execute_tool(
        tool_name="calculator",
        arguments='{"expression": "5 + 5"}',
        idempotency_key="lost_lease_001"
    )

    assert "lease was lost" in result.lower()

    operation = isolated_idempotency_store.lookup(
        "lost_lease_001"
    )

    assert operation["status"] == "recovery_required"
    assert operation["result"] is None


def test_duplicate_request_does_not_trigger_second_approval(
    isolated_idempotency_store,
    monkeypatch
):
    tool = registry.TOOL_REGISTRY["send_message"]

    approval_count = 0
    execution_count = 0
    duplicate_result = None

    def fake_send_message(recipient, message):
        nonlocal execution_count
        execution_count += 1
        return "SIMULATION ONLY: Message processed."

    monkeypatch.setattr(
        tool,
        "function",
        fake_send_message
    )

    policy = ToolPolicy(
        allowed_permissions=frozenset({"message:send"})
    )

    def approve_action(tool_name, arguments):
        nonlocal approval_count, duplicate_result

        approval_count += 1

        # The first request must already own the operation.
        operation = isolated_idempotency_store.lookup(
            "duplicate_approval_001"
        )
        assert operation["status"] == "in_progress"

        # Simulate a duplicate request arriving while approval
        # for the first request is still pending.
        duplicate_result = registry.execute_tool(
            tool_name="send_message",
            arguments=(
                '{"recipient":"test@example.com",'
                '"message":"Hello"}'
            ),
            policy=policy,
            approval_callback=approve_action,
            idempotency_key="duplicate_approval_001"
        )

        return True

    first_result = registry.execute_tool(
        tool_name="send_message",
        arguments=(
            '{"recipient":"test@example.com",'
            '"message":"Hello"}'
        ),
        policy=policy,
        approval_callback=approve_action,
        idempotency_key="duplicate_approval_001"
    )

    assert "SIMULATION ONLY" in first_result
    assert "already in progress" in duplicate_result
    assert approval_count == 1
    assert execution_count == 1


def test_rejected_approval_cancels_idempotent_operation(
    isolated_idempotency_store,
    monkeypatch
):
    tool = registry.TOOL_REGISTRY["send_message"]

    execution_count = 0
    approval_count = 0

    def fake_send_message(recipient, message):
        nonlocal execution_count
        execution_count += 1
        return "SIMULATION ONLY: Message processed."

    monkeypatch.setattr(
        tool,
        "function",
        fake_send_message
    )

    policy = ToolPolicy(
        allowed_permissions=frozenset({"message:send"})
    )

    def reject_action(tool_name, arguments):
        nonlocal approval_count
        approval_count += 1
        return False

    result = registry.execute_tool(
        tool_name="send_message",
        arguments=(
            '{"recipient":"test@example.com",'
            '"message":"Hello"}'
        ),
        policy=policy,
        approval_callback=reject_action,
        idempotency_key="rejected_approval_001"
    )

    assert "cancelled by user" in result.lower()

    operation = isolated_idempotency_store.lookup(
        "rejected_approval_001"
    )

    assert operation["status"] == "cancelled"
    assert operation["result"] == "Tool execution cancelled by user."

    # A repeated request using the same key must not prompt again.
    repeated_result = registry.execute_tool(
        tool_name="send_message",
        arguments=(
            '{"recipient":"test@example.com",'
            '"message":"Hello"}'
        ),
        policy=policy,
        approval_callback=reject_action,
        idempotency_key="rejected_approval_001"
    )

    assert "Previous execution was cancelled" in repeated_result
    assert approval_count == 1
    assert execution_count == 0


def test_expired_lease_during_approval_prevents_execution(
    isolated_idempotency_store,
    monkeypatch
):
    tool = registry.TOOL_REGISTRY["send_message"]

    execution_count = 0

    def fake_send_message(recipient, message):
        nonlocal execution_count
        execution_count += 1
        return "SIMULATION ONLY: Message processed."

    monkeypatch.setattr(
        tool,
        "function",
        fake_send_message
    )

    policy = ToolPolicy(
        allowed_permissions=frozenset({"message:send"})
    )

    def approve_after_lease_expiry(tool_name, arguments):
        # Simulate the human taking longer than the lease duration.
        with isolated_idempotency_store._connect() as connection:
            connection.execute(
                """
                UPDATE operations
                SET lease_expires_at = ?
                WHERE idempotency_key = ?
                """,
                (
                    "2000-01-01T00:00:00+00:00",
                    "approval_lease_expired_001"
                )
            )

        return True

    result = registry.execute_tool(
        tool_name="send_message",
        arguments=(
            '{"recipient":"test@example.com",'
            '"message":"Hello"}'
        ),
        policy=policy,
        approval_callback=approve_after_lease_expiry,
        idempotency_key="approval_lease_expired_001"
    )

    assert "Recovery required" in result
    assert execution_count == 0

    operation = isolated_idempotency_store.lookup(
        "approval_lease_expired_001"
    )

    assert operation["status"] == "recovery_required"
    assert operation["result"] is None
