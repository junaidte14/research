import json
from typing import Any
from llm_interface import LLMClient
from tools.registry import (TOOL_DEFINITIONS, execute_tool)

MAX_TOOL_ROUNDS = 5


def request_tool_approval(
    tool_name: str,
    arguments: dict[str, Any]
) -> bool:
    """
    Ask the human to approve a validated tool action.
    """

    print("\n" + "=" * 50)
    print("HUMAN APPROVAL REQUIRED")
    print("=" * 50)

    print(f"Tool: {tool_name}")
    print("Validated arguments:")

    print(
        json.dumps(
            arguments,
            indent=4,
            ensure_ascii=False
        )
    )

    print("=" * 50)

    decision = input(
        "Approve this action? (yes/no): "
    ).strip().lower()

    return decision in {"yes", "y"}

def run_agent_turn(
    llm: LLMClient,
    messages: list[dict[str, Any]]
) -> str:
    """
    Run one user turn, allowing the LLM to call tools.
    """

    for round_number in range(MAX_TOOL_ROUNDS):

        #step 1: ask the llm what to do
        assistant_message = llm.complete_with_tools(
            messages=messages, 
            tools=TOOL_DEFINITIONS
        )

        tool_calls = assistant_message.get("tool_calls", [])

        #step 2: No tool requested: return the normal answer

        if not tool_calls:
            content = assistant_message.get("content")

            if not isinstance(content, str):
                raise RuntimeError("LLM returned no final answer")

            messages.append(assistant_message)
            return content

        #step 3: preserve the assistant tool call request
        messages.append(assistant_message)

        if round_number == MAX_TOOL_ROUNDS - 1:
            raise RuntimeError("Maximum tool-call rounds exceeded")

        # step 4: execute each requested tool
        for tool_call in tool_calls:
            try:
                tool_id = tool_call["id"]
                function = tool_call["function"]

                tool_name = function["name"]
                arguments = function["arguments"]

                if not isinstance(tool_id, str) or not tool_id.strip():
                    raise ValueError("Invalid tool call ID")

                if not isinstance(tool_name, str) or not tool_name.strip():
                    raise ValueError("Invalid tool name")

                if not isinstance(arguments, str):
                    raise ValueError("Tool arguments must be a JSON string")

            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    f"Malformed tool call received from LLM: {error}"
                ) from error

            print(f"[Agent] calling tool: {tool_name} with arguments: {arguments}")

            result = execute_tool(
                tool_name=tool_name,
                arguments=arguments,
                approval_callback=request_tool_approval,
                idempotency_key=tool_id
            )

            print(f"[Agent] Tool result: {result}")

            #step 5: send the tool result back to the llm
            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": result
            })

    raise RuntimeError("Agent stopped without producing a final answer")
