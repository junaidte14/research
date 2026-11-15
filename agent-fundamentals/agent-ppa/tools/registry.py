import json
from tools.calculator import calculator
from tools.date_time import get_current_datetime
from tools.text_analyzer import analyze_text

# 1. Define all available tools for the LLM.
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate a mathematical expression accurately. "
                "Use this tool whenever the user asks for an "
                "arithmetic calculation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "A mathematical expression, for example "
                            "(125 * 8) + 450"
                        )
                    }
                },
                "required": ["expression"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": (
                "Get the current local date, time, "
                "timezone and weekday."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_text",
            "description": (
                "Analyze text and calculate its character count, "
                "word count, line count and unique word count."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to analyze."
                    }
                },
                "required": ["text"],
                "additionalProperties": False
            }
        }
    }
]

# 2. Map tool names to actual Python functions.
AVAILABLE_TOOLS = {
    "calculator": calculator,
    "get_current_datetime": get_current_datetime,
    "analyze_text": analyze_text
}

# 3. Define the expected argument types for validation.
TOOL_ARGUMENT_TYPES = {
    "calculator": {
        "expression": str
    },
    "get_current_datetime": {},
    "analyze_text": {
        "text": str
    }
}


def execute_tool(tool_name: str, arguments: str) -> str:
    """
    Validate and execute a registered tool.
    """
    function = AVAILABLE_TOOLS.get(tool_name)

    if function is None:
        return f"Tool Error: Unknown tool '{tool_name}'"

    try:

        parsed_arguments = json.loads(arguments)

        if not isinstance(parsed_arguments, dict):
            raise ValueError("Tool arguments must be a json object")

        expected_arguments = TOOL_ARGUMENT_TYPES[tool_name]

        if set(parsed_arguments) != set(expected_arguments):
            raise ValueError(
                f"Invalid tool arguments. "
                f"Expected: {list(expected_arguments.keys())}"
            )

        for argument_name, expected_type in expected_arguments.items():
            if type(parsed_arguments[argument_name]) != expected_type:
                raise ValueError(
                    f"Argument '{argument_name}' "
                    f"must be {expected_type.__name__}."
                )

        return function(**parsed_arguments)

    except json.JSONDecodeError:
        return "Tool execution error: Invalid JSON arguments."
    
    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as err:
        return f"Tool execution error: {err}"

