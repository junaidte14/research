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

    isolated_idempotency_store.claim(
        "test-operation-004",
        "calculator",
        {"expression": "10 + 5"}
    )

    isolated_idempotency_store.finish(
        "test-operation-004",
        "Previous simulated failure",
        status="failed"
    )

    result = registry.execute_tool(
        "calculator",
        '{"expression": "10 + 5"}',
        idempotency_key="test-operation-004"
    )

    assert "Automatic re-execution is blocked" in result
    assert execution_count == 0
