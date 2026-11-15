import json
from tools.calculator import calculator

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
                "required": ["expression"]
            }
        }
    }
]

AVAILABLE_TOOLS = {
    "calculator": calculator
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

        if set(parsed_arguments) != {"expression"}:
            raise ValueError("Invalid calculator arguments")

        expression = parsed_arguments["expression"]

        if not isinstance(expression, str):
            raise ValueError("Expression must be a string")

        return function(expression=expression)

    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as err:
        return f"Tool execution error: {err}"

