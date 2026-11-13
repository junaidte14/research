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
            response = completion(model=self.model, messages=messages, timeout=60)
            content = response.choices[0].message.content

            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("LLM returned an invalid response")
            
            return content.strip()
        except Exception as err:
            raise RuntimeError(f"LLM request faild: {err}") from err