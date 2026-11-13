from typing import Protocol

class LLMClient(Protocol):
    """
    Standard interface for all LLM providers.

    Any provider adapter implementing this interface
    must provide a complete() method.
    """

    def complete(self, messages: list[dict[str, str]]) -> str:
        ...