from dataclasses import dataclass
from typing import Callable, Any
from pydantic import BaseModel, ValidationError
from tools.retry import execute_with_retry, TransientToolError

@dataclass
class Tool:
    name: str
    description: str
    args_schema: type[BaseModel]
    function: Callable[..., str]
    required_permission: str
    requires_approval: bool = False
    max_attempts: int = 1
    retry_delay_seconds: float = 0.5

    def to_llm_definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema.model_json_schema()
            }
        }

    def validate_arguments(self, arguments_json: str) -> dict[str, Any]:
        validated = self.args_schema.model_validate_json(
            arguments_json
        )
        return validated.model_dump()

    def request_approval(
        self,
        arguments: dict[str, Any],
        approval_callback: Callable[
            [str, dict[str, Any]], bool
        ] | None = None
    ) -> bool:

        if not self.requires_approval:
            return True

        if approval_callback is None:
            return False

        return approval_callback(
            self.name,
            arguments
        ) is True

    def execute_validated(
        self,
        arguments: dict[str, Any]
    ) -> str:

        result = execute_with_retry(
            self.function,
            arguments,
            max_attempts=self.max_attempts,
            base_delay_seconds=self.retry_delay_seconds
        )

        if not isinstance(result, str):
            raise TypeError(
                "Tool function must return strings."
            )

        return result

    def execute(
        self,
        arguments_json: str,
        approval_callback: Callable[
            [str, dict[str, Any]], bool
        ] | None = None
    ) -> str:

        try:
            arguments = self.validate_arguments(
                arguments_json
            )

            if not self.request_approval(
                arguments,
                approval_callback
            ):
                if self.requires_approval and approval_callback is None:
                    return (
                        "Tool approval error: No approval handler "
                        "configured. Action is blocked."
                    )

                return "Tool execution cancelled by user."

            return self.execute_validated(arguments)

        except ValidationError as error:
            return (
                f"Tool execution error: Invalid arguments. {error}"
            )

        except TransientToolError as error:
            return (
                f"Tool execution error: Temporary failure. {error}"
            )

        except (
            ValueError,
            TypeError,
            ZeroDivisionError,
            OverflowError
        ) as error:
            return f"Tool execution error: {error}"
