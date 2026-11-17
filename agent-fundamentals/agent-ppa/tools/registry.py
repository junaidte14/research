from typing import Any, Callable
from tools.base import Tool
from tools.policy import DEFAULT_TOOL_POLICY, ToolPolicy
from tools.audit import write_audit_event
from tools.idempotency import idempotency_store

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
        approval_callback: Callable[[str, dict[str, Any]], bool] | None = None,
        idempotency_key: str | None = None
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

    
    # Without an idempotency key, preserve the original behavior.
    if idempotency_key is None:
        result = tool.execute(
            arguments,
            approval_callback=approval_callback
        )

    else:
        try:
            # 1. Validate arguments before approval.
            validated_arguments = tool.validate_arguments(arguments)

            # 2. Check whether this operation was already processed.
            existing = idempotency_store.lookup(
                idempotency_key,
                tool_name,
                validated_arguments
            )

            if existing is not None:
                status = existing["status"]

                if status == "completed":
                    result = existing["result"]

                elif status == "in_progress":
                    result = (
                        "Idempotency notice: This operation "
                        "is already in progress."
                    )

                elif status == "failed":
                    result = (
                        "Previous execution failed. "
                        "Automatic re-execution is blocked. "
                        f"Previous result: {existing['result']}"
                    )

                else:
                    result = (
                        "Idempotency conflict: This key was "
                        "already used for a different operation."
                    )

            else:
                # 3. Ask for approval only for a new operation.
                if tool.requires_approval and approval_callback is None:
                    result = (
                        "Tool approval error: No approval handler "
                        "configured. Action is blocked."
                    )

                elif not tool.request_approval(
                    validated_arguments,
                    approval_callback
                ):
                    result = "Tool execution cancelled by user."

                else:
                    # 4. Atomically claim the operation.
                    claim = idempotency_store.claim(
                        idempotency_key,
                        tool_name,
                        validated_arguments
                    )

                    if claim["status"] == "new":
                        # 5. Execute only after a successful claim.
                        try:
                            result = tool.execute_validated(
                                validated_arguments
                            )
                            idempotency_store.finish(
                                idempotency_key,
                                result,
                                status="completed"
                            )

                        except Exception as error:
                            result = (
                                "Tool execution error: "
                                f"{error}"
                            )
                            idempotency_store.finish(
                                idempotency_key,
                                result,
                                status="failed"
                            )

                    elif claim["status"] == "completed":
                        result = claim["result"]

                    elif claim["status"] == "in_progress":
                        result = (
                            "Idempotency notice: This operation "
                            "is already in progress."
                        )

                    elif claim["status"] == "failed":
                        result = (
                            "Previous execution failed. "
                            "Automatic re-execution is blocked. "
                            f"Previous result: {claim['result']}"
                        )

                    else:
                        result = (
                            "Idempotency conflict: This key was "
                            "already used for a different operation."
                        )

        except Exception as error:
            result = f"Tool execution error: {error}"


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