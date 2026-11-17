
import json

import agent
import tools.registry as registry

from tools.idempotency import IdempotencyStore


class FakeLLM:
    def __init__(self, tool_call_id):
        self.tool_call_id = tool_call_id
        self.calls = 0

    def complete_with_tools(self, messages, tools):
        self.calls += 1

        if self.calls == 1:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": self.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": json.dumps({
                                "expression": "10 + 5"
                            })
                        }
                    }
                ]
            }

        return {
            "role": "assistant",
            "content": "The result is 15.",
            "tool_calls": []
        }


def test_agent_passes_tool_call_id_as_idempotency_key(
    tmp_path,
    monkeypatch
):
    store = IdempotencyStore(
        tmp_path / "agent_test.db"
    )

    monkeypatch.setattr(
        registry,
        "idempotency_store",
        store
    )

    calculator_tool = registry.TOOL_REGISTRY["calculator"]
    execution_count = 0

    def fake_calculator(expression):
        nonlocal execution_count
        execution_count += 1
        return "15"

    monkeypatch.setattr(
        calculator_tool,
        "function",
        fake_calculator
    )

    original_execute_tool = registry.execute_tool

    def tracked_execute_tool(*args, **kwargs):
        return original_execute_tool(*args, **kwargs)

    monkeypatch.setattr(
        agent,
        "execute_tool",
        tracked_execute_tool
    )

    llm = FakeLLM(tool_call_id="call_test_001")

    first_messages = [
        {"role": "system", "content": "Test agent"}
    ]

    second_messages = [
        {"role": "system", "content": "Test agent"}
    ]

    first_answer = agent.run_agent_turn(
        llm,
        first_messages
    )

    second_answer = agent.run_agent_turn(
        llm,
        second_messages
    )

    assert first_answer == "The result is 15."
    assert second_answer == "The result is 15."
    assert execution_count == 1

    stored_operation = store.lookup(
        "call_test_001",
        "calculator",
        {"expression": "10 + 5"}
    )

    assert stored_operation["status"] == "completed"
