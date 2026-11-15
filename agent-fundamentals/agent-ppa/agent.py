from typing import Any
from llm_interface import LLMClient
from tools.registry import (TOOL_DEFINITIONS, execute_tool)

MAX_TOOL_ROUNDS = 5

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
            tool_id = tool_call["id"]
            function = tool_call["function"]

            tool_name = function["name"]
            arguments = function["arguments"]

            print(f"[Agent] calling tool: {tool_name} with arguments: {arguments}")

            result = execute_tool(tool_name=tool_name, arguments=arguments)

            print(f"[Agent] Tool result: {result}")

            #step 5: send the tool result back to the llm
            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": result
            })

    raise RuntimeError("Agent stopped without producing a final answer")
