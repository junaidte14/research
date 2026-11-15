import json
import re

def analyze_text(text: str) -> str:
    """
    Analyzes text and returns basic statistics.
    """

    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    words = re.findall(
        r"\b[\w]+(?:['’-][\w]+)*\b",
        text,
        flags=re.UNICODE
    )

    result = {
        "character_count": len(text),
        "word_count": len(words),
        "line_count": len(text.splitlines()),
        "unique_word_count": len(
            set(word.casefold() for word in words)
        )
    }

    return json.dumps(result, ensure_ascii=False)