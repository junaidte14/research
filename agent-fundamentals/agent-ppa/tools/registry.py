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

# Single source of truth for registered tools.
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

# Build lookup structures from the tool definitions.
TOOL_REGISTRY = {
    tool.name: tool for tool in TOOLS
}

TOOL_DEFINITIONS = [
    tool.to_llm_definition() for tool in TOOLS
]

def _existing_operation_result(
    status: str,
    existing: dict[str, Any]
) -> str:
    """Translate an existing idempotency state into a safe result."""

    if status == "completed":
        return existing["result"]

    if status == "in_progress":
        return (
            "Idempotency notice: This operation "
            "is already in progress."
        )

    if status == "failed":
        return (
            "Previous execution failed. "
            "Automatic re-execution is blocked. "
            f"Previous result: {existing['result']}"
        )

    if status == "recovery_required":
        return (
            "Recovery required: The previous operation "
            "has an uncertain execution outcome. "
            "Manual inspection is required. "
            "Automatic re-execution is blocked."
        )

    if status == "cancelled":
        return (
            "Previous execution was cancelled. "
            "This idempotency key cannot be reused."
        )

    return (
        "Idempotency conflict: This key was "
        "already used for a different operation."
    )


def execute_tool(
    tool_name: str,
    arguments: str,
    policy: ToolPolicy = DEFAULT_TOOL_POLICY,
    approval_callback: Callable[
        [str, dict[str, Any]], bool
    ] | None = None,
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

    # 1. Application-level authorization.
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

    # 2. Preserve the original behavior when no idempotency key is provided.
    if idempotency_key is None:
        result = tool.execute(
            arguments,
            approval_callback=approval_callback
        )

    else:
        try:
            # 3. Validate arguments before claiming or approval.
            validated_arguments = tool.validate_arguments(
                arguments
            )

            # 4. Check whether this operation already exists.
            existing = idempotency_store.lookup(
                idempotency_key,
                tool_name,
                validated_arguments
            )

            if existing is not None:
                result = _existing_operation_result(
                    existing["status"],
                    existing
                )

            # 5. Do not create a claim if approval cannot be requested at all.
            elif tool.requires_approval and approval_callback is None:
                result = (
                    "Tool approval error: No approval handler "
                    "configured. Action is blocked."
                )

            else:
                # 6. Atomically claim BEFORE asking for approval.
                claim = idempotency_store.claim(
                    idempotency_key,
                    tool_name,
                    validated_arguments
                )

                if claim["status"] != "new":
                    result = _existing_operation_result(
                        claim["status"],
                        claim
                    )

                else:
                    lease_token = claim["lease_token"]

                    # 7. Only the successful claimant reaches the human approval step.
                    try:
                        approved = tool.request_approval(
                            validated_arguments,
                            approval_callback
                        )
                    except Exception as error:
                        approval_error = (
                            f"Tool approval error: {error}"
                        )

                        cancelled = idempotency_store.cancel(
                            idempotency_key=idempotency_key,
                            lease_token=lease_token,
                            result=approval_error
                        )

                        if cancelled:
                            result = approval_error
                        else:
                            idempotency_store.recover_expired_operations()
                            result = (
                                "Recovery required: Approval handling "
                                "failed and the operation lease was lost. "
                                "Manual inspection is required."
                            )

                    else:
                        if not approved:
                            # 8. Close the claimed operation safely when the human rejects it.
                            cancelled = idempotency_store.cancel(
                                idempotency_key=idempotency_key,
                                lease_token=lease_token
                            )

                            if cancelled:
                                result = (
                                    "Tool execution cancelled by user."
                                )
                            else:
                                idempotency_store.recover_expired_operations()
                                result = (
                                    "Recovery required: The operation "
                                    "could not be cancelled because its "
                                    "lease was lost. Manual inspection "
                                    "is required."
                                )

                        else:
                            # 9. Approval may have taken time. Renew the lease before execution.
                            renewed = idempotency_store.renew_lease(
                                idempotency_key=idempotency_key,
                                lease_token=lease_token
                            )

                            if not renewed:
                                idempotency_store.recover_expired_operations()
                                result = (
                                    "Recovery required: The operation "
                                    "lease expired or was lost while "
                                    "waiting for approval. The tool "
                                    "was not executed."
                                )

                            else:
                                # 10. Execute only after claim, approval and lease renewal.
                                try:
                                    execution_result = (
                                        tool.execute_validated(
                                            validated_arguments
                                        )
                                    )

                                    # 11. Finalize using the same lease ownership token.
                                    finalized = idempotency_store.finish(
                                        idempotency_key=idempotency_key,
                                        lease_token=lease_token,
                                        result=execution_result,
                                        status="completed"
                                    )

                                    if finalized:
                                        result = execution_result
                                    else:
                                        result = (
                                            "Operation execution returned a result, but its lease was lost before finalization. "
                                            "The execution outcome may be uncertain. Manual recovery is required."
                                        )

                                except Exception as error:
                                    execution_error = (
                                        f"Tool execution error: {error}"
                                    )

                                    finalized = idempotency_store.finish(
                                        idempotency_key=idempotency_key,
                                        lease_token=lease_token,
                                        result=execution_error,
                                        status="failed"
                                    )

                                    if finalized:
                                        result = execution_error
                                    else:
                                        result = (
                                            "Tool execution failed, but "
                                            "its lease was lost before "
                                            "the failure could be recorded. "
                                            "Manual recovery is required."
                                        )

        except Exception as error:
            result = f"Tool execution error: {error}"

    # 12. Record the final outcome.
    if result == "Tool execution cancelled by user.":
        outcome = "cancelled"

    elif result.startswith("Tool approval error:"):
        outcome = "approval_error"

    elif result.startswith("Tool execution error:"):
        outcome = "execution_error"

    elif result.startswith("Recovery required:"):
        outcome = "recovery_required"

    elif result.startswith("Tool execution failed,"):
        outcome = "execution_uncertain"

    elif result.startswith("Operation execution returned a result,"):
        outcome = "execution_uncertain"

    elif result.startswith("Idempotency notice:"):
        outcome = "already_in_progress"

    elif result.startswith("Previous execution failed."):
        outcome = "previously_failed"

    elif result.startswith("Previous execution was cancelled."):
        outcome = "cancelled"

    elif result.startswith("Idempotency conflict:"):
        outcome = "idempotency_conflict"

    else:
        outcome = "completed"

    write_audit_event(
        "tool_execution",
        tool=tool.name,
        permission=tool.required_permission,
        outcome=outcome
    )

    return result