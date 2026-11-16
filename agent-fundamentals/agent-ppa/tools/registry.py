from tools.base import Tool
from tools.policy import DEFAULT_TOOL_POLICY, ToolPolicy

from tools.calculator import calculator
from tools.date_time import get_current_datetime
from tools.text_analyzer import analyze_text

from tools.schemas import (
    CalculatorInput,
    DateTimeInput,
    TextAnalyzerInput
)

# single source of truth for registered tools

TOOLS = [
    Tool(
        name="calculator",
        description=(
            "Evaluate a mathematical expression accurately. "
            "Use this tool whenever the user asks for an "
            "arithmetic calculation."
        ),
        args_schema=CalculatorInput,
        function=calculator,
        required_permission="math:calculate"
    ),
    Tool(
        name="get_current_datetime",
        description=(
            "Get the current local date, time, "
            "timezone and weekday."
        ),
        args_schema=DateTimeInput,
        function=get_current_datetime,
        required_permission="system:read_time"
    ),
    Tool(
        name="analyze_text",
        description=(
            "Analyze text and calculate its character count, "
            "word count, line count and unique word count."
        ),
        args_schema=TextAnalyzerInput,
        function=analyze_text,
        required_permission="text:analyze"
    )
]

# Build lookup structures from the tool definitions
TOOL_REGISTRY = {
    tool.name: tool for tool in TOOLS
}

# Automatically generate definitions for the LLM
TOOL_DEFINITIONS = [
    tool.to_llm_definition() for tool in TOOLS
]

def execute_tool(
        tool_name: str, 
        arguments: str,
        policy: ToolPolicy = DEFAULT_TOOL_POLICY
    ) -> str:
    tool = TOOL_REGISTRY.get(tool_name)

    if tool is None:
        return f"Tool Error: Unknown tool '{tool_name}'"

    # Authorization check happens before execution
    if not policy.is_allowed(tool.required_permission):
        return (
            "Tool authorization error: "
            f"Permission '{tool.required_permission}' "
            "is not allowed."
        )
    
    return tool.execute(arguments)