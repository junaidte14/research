from typing import Any
from litellm import completion
from config import settings

class LiteLLMAdapter:
    """
    Communicates with different LLM providers
    through a unified LiteLLM interface.
    """

    def __init__(self, model: str | None = None):
        self.model = model or settings.model

    def complete(self, messages: list[dict[str, str]]) -> str:
        try:
            response = completion(
                model=self.model, 
                messages=messages, 
                timeout=30
            )
            content = response.choices[0].message.content

            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("LLM returned an invalid response")
            
            return content.strip()
        except Exception as err:
            raise RuntimeError(f"LLM request failed: {err}") from err

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]]
    ) -> dict[str, Any] :
        try:

            response = completion(
                model = self.model,
                messages = messages,
                tools = tools,
                tool_choice = "auto",
                timeout=30
            )

            message = response.choices[0].message

            normalized_tool_calls = []

            for call in message.tool_calls or []:
                normalized_tool_calls.append({
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments
                    }
                })

            return {
                "role": "assistant",
                "content": message.content,
                "tool_calls": normalized_tool_calls
            }

        except Exception as error:
            raise RuntimeError(f"LLM tool request failed: {error}") from error

        