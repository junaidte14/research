from typing import Any, Callable
from tools.base import Tool
from tools.policy import DEFAULT_TOOL_POLICY, ToolPolicy
from tools.audit import write_audit_event

from tools.calculator import calculator
from tools.date_time import get_current_datetime
from tools.text_analyzer import analyze_text
from tools.messaging import send_message

from tools.schemas import (
    CalculatorInput,
    DateTimeInput,
    TextAnalyzerInput,
    MessageInput
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
    ),
    Tool(
        name="send_message",
        description=(
            "Simulate sending a message to the specified recipient "
            "with the supplied message content."
        ),
        args_schema=MessageInput,
        function=send_message,
        required_permission="message:send",
        requires_approval=True
    ),
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
        policy: ToolPolicy = DEFAULT_TOOL_POLICY,
        approval_callback: Callable[[str, dict[str, Any]], bool] | None = None
    ) -> str:
    tool = TOOL_REGISTRY.get(tool_name)

    if tool is None:
        write_audit_event(
            "tool_request",
            tool=tool_name,
            outcome="unknown_tool"
        )
        return f"Tool Error: Unknown tool '{tool_name}'"

    write_audit_event(
        "tool_request",
        tool=tool_name,
        permission=tool.required_permission,
        argument_fields=list(
            tool.args_schema.model_fields.keys()
        )
    )

    # Authorization check happens before execution
    if not policy.is_allowed(tool.required_permission):
        write_audit_event(
            "authorization",
            tool=tool.name,
            permission=tool.required_permission,
            outcome="denied"
        )
        return (
            "Tool authorization error: "
            f"Permission '{tool.required_permission}' "
            "is not allowed."
        )

    write_audit_event(
        "authorization",
        tool=tool.name,
        permission=tool.required_permission,
        outcome="allowed"
    )

    result = tool.execute(
        arguments,
        approval_callback=approval_callback
    )

    if result == "Tool execution cancelled by user.":
        outcome = "cancelled"
    elif result.startswith("Tool approval error:"):
        outcome = "approval_error"
    elif result.startswith("Tool execution error:"):
        outcome = "execution_error"
    else:
        outcome = "completed"

    write_audit_event(
        "tool_execution",
        tool=tool.name,
        permission=tool.required_permission,
        outcome=outcome
    )

    return result