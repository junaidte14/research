from dataclasses import dataclass
from typing import Callable
from pydantic import BaseModel, ValidationError

@dataclass
class Tool:
    name: str
    description: str
    args_schema: type[BaseModel]
    function: Callable[..., str]
    required_permission: str

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

    def execute(self, arguments_json: str) -> str:
        """
        Validate arguments and execute the Python function.
        """

        try:
            validated_arguments = (
                self.args_schema.model_validate_json(arguments_json)
            )

            result = self.function(**validated_arguments.model_dump())

            if not isinstance(result, str):
                raise TypeError("Tool function must return strings")

            return result

        except ValidationError as error:
            return f"Tool execution error: Invalid arguments. {error}"

        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as error:
            return f"Tool execution error: {error}"