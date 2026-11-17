
from types import SimpleNamespace

import pytest

import litellm_adapter 
from litellm_adapter import LiteLLMAdapter


def test_complete_passes_timeout(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Hello from the model"
                    )
                )
            ]
        )

    monkeypatch.setattr(
        litellm_adapter,
        "completion",
        fake_completion
    )

    adapter = LiteLLMAdapter(model="test-model")

    result = adapter.complete([
        {"role": "user", "content": "Hello"}
    ])

    assert result == "Hello from the model"
    assert captured["timeout"] == 30
    assert captured["model"] == "test-model"


def test_complete_with_tools_passes_timeout(monkeypatch):
    captured = {}

    fake_tool_call = SimpleNamespace(
        id="call_123",
        function=SimpleNamespace(
            name="calculator",
            arguments='{"expression":"2 + 2"}'
        )
    )

    def fake_completion(**kwargs):
        captured.update(kwargs)

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[fake_tool_call]
                    )
                )
            ]
        )

    monkeypatch.setattr(
        litellm_adapter,
        "completion",
        fake_completion
    )

    adapter = LiteLLMAdapter(model="test-model")

    result = adapter.complete_with_tools(
        messages=[
            {"role": "user", "content": "Calculate 2 + 2"}
        ],
        tools=[
            {"type": "function", "function": {"name": "calculator"}}
        ]
    )

    assert captured["timeout"] == 30
    assert captured["tool_choice"] == "auto"

    assert result["tool_calls"][0]["function"]["name"] == "calculator"
    assert result["tool_calls"][0]["function"]["arguments"] == (
        '{"expression":"2 + 2"}'
    )


def test_complete_wraps_timeout_exception(monkeypatch):
    def fake_completion(**kwargs):
        raise TimeoutError("Simulated API timeout")

    monkeypatch.setattr(
        litellm_adapter,
        "completion",
        fake_completion
    )

    adapter = LiteLLMAdapter(model="test-model")

    with pytest.raises(RuntimeError) as error:
        adapter.complete([
            {"role": "user", "content": "Hello"}
        ])

    assert "LLM request failed" in str(error.value)
    assert isinstance(error.value.__cause__, TimeoutError)


def test_complete_with_tools_wraps_timeout_exception(monkeypatch):
    def fake_completion(**kwargs):
        raise TimeoutError("Simulated API timeout")

    monkeypatch.setattr(
        litellm_adapter,
        "completion",
        fake_completion
    )

    adapter = LiteLLMAdapter(model="test-model")

    with pytest.raises(RuntimeError) as error:
        adapter.complete_with_tools(
            messages=[
                {"role": "user", "content": "Calculate 2 + 2"}
            ],
            tools=[]
        )

    assert "LLM tool request failed" in str(error.value)
    assert isinstance(error.value.__cause__, TimeoutError)
