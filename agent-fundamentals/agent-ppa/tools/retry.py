import time
from typing import Any, Callable

class TransientToolError(Exception):
    """Raised when a tool encounters a temporary, retryable failure."""

def execute_with_retry(
    function: Callable[..., Any],
    arguments: dict[str, Any],
    *,
    max_attempts: int = 1,
    base_delay_seconds: float = 0.5
) -> Any:
    """
    Execute a tool function with bounded retries.
    Only TransientToolError triggers a retry.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    if base_delay_seconds < 0:
        raise ValueError("base_delay_seconds cannot be negative")

    for attempt in range(1, max_attempts+1):
        try:
            return function(**arguments)
        
        except TransientToolError:
            
            if attempt == max_attempts:
                raise

            delay = base_delay_seconds * (2 ** (attempt -1))

            print(
                f"[Retry] Temporary tool failure. "
                f"Attempt {attempt}/{max_attempts}. "
                f"Retrying in {delay:.2f} seconds."
            )

            time.sleep(delay)