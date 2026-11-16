import json
from datetime import datetime

from tools.date_time import get_current_datetime
from tools.text_analyzer import analyze_text
from tools.registry import execute_tool

def test_current_datetime():
    result = json.loads(get_current_datetime())

    assert "datetime" in result
    assert "timezone" in result
    assert "weekday" in result

    # Verify that the returned datetime is valid ISO format.
    datetime.fromisoformat(result["datetime"])

def test_text_analyzer():
    text = "Agentic AI is interesting. Agentic AI is powerful."

    result = json.loads(analyze_text(text=text))

    assert result["word_count"] == 8
    assert result["line_count"] == 1
    assert result["unique_word_count"] == 5

def test_calculator_through_registry():
    result = execute_tool("calculator", '{"expression": "10 + 20"}')
    assert result == "30"

def test_datetime_through_registry():
    result = json.loads(
        execute_tool("get_current_datetime", '{}')
    )
    assert "datetime" in result

def test_text_analyzer_through_registry():
    result = json.loads(
        execute_tool("analyze_text", '{"text": "Hello world"}')
    )
    assert result["word_count"] == 2

def test_unknown_tool():
    result = execute_tool("unknown_tool", "{}")
    assert "Unknown tool" in result

def test_invalid_json():
    result = execute_tool("calculator", "{invalid json}")
    assert "Invalid JSON" in result

def test_missing_argument():
    result = execute_tool("calculator", "{}")
    assert "Invalid arguments" in result