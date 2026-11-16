from dataclasses import dataclass
from typing import Callable, Any
from pydantic import BaseModel, ValidationError

@dataclass
class Tool:
    name: str
    description: str
    args_schema: type[BaseModel]
    function: Callable[..., str]
    required_permission: str
    requires_approval: bool = False

    def to_llm_definition(self) -> dict:
        """
        Generate the tool definition expected by the LLM.
        """

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema.model_json_schema()
            }
        }

    def execute(
            self, 
            arguments_json: str,
            approval_callback: Callable[[str, dict[str, Any]], bool] | None = None
        ) -> str:
        """
        Validate arguments and execute the Python function.
        """

        try:
            validated_arguments = (
                self.args_schema.model_validate_json(arguments_json)
            )

            # Require explicit approval for sensitive tools
            if self.requires_approval:
                if approval_callback is None:
                    return (
                        "Tool approval error: No approval handler "
                        "configured. Action is blocked."
                    )
                approved = approval_callback(
                    self.name,
                    validated_arguments.model_dump()
                )
                if approved is not True:
                    return "Tool execution cancelled by user."

            result = self.function(**validated_arguments.model_dump())

            if not isinstance(result, str):
                raise TypeError("Tool function must return strings")

            return result

        except ValidationError as error:
            return f"Tool execution error: Invalid arguments. {error}"

        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as error:
            return f"Tool execution error: {error}"