"""context.py: managing what the model sees: usage tracking, pruning, summarizing, saving sessions."""
import json
from pathlib import Path

SESSIONS_DIR = Path(__file__).parent / "sessions"
PRUNE_MARK = "[pruned:"    # every pruned tool result contains this text
HEAD_CHARS = 200           # how much of an old tool result we keep


# ---------------------------------------------------------------
# Plain-dict messages (easy to edit and to save as JSON)
# ---------------------------------------------------------------
def message_to_dict(msg) -> dict:
    """Turn the SDK's assistant message object into a plain dict.
    We keep only role, content and tool_calls. The model's private reasoning text is not resent (it costs tokens)."""
    d = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.function.name, "arguments": c.function.arguments},
            }
            for c in msg.tool_calls
        ]
    elif d["content"] is None:
        d["content"] = ""
    return d


# ---------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------
class Usage:
    """Tracks token usage reported by the API."""

    def __init__(self):
        self.calls = 0
        self.prompt_tokens = 0         # total INPUT tokens billed over all calls
        self.completion_tokens = 0     # total OUTPUT tokens
        self.last_prompt_tokens = 0    # size of the context on the most recent call

    def add(self, usage, is_context: bool = True) -> None:
        if usage is None:
            return
        self.calls += 1
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        if is_context:
            self.last_prompt_tokens = usage.prompt_tokens


def estimate_tokens(messages: list) -> int:
    """Rough size without calling the API: about 4 characters per token."""
    return len(json.dumps(messages, ensure_ascii=False)) // 4


# ---------------------------------------------------------------
# Pruning: cheap, and it keeps the head of every old result
# ---------------------------------------------------------------
def prune_old_tool_results(messages: list, keep_recent: int = 8, min_chars: int = 400) -> int:
    """Shorten big, OLD tool results to their first HEAD_CHARS characters. Returns how many were pruned."""
    pruned = 0
    for m in messages[: max(len(messages) - keep_recent, 0)]:
        content = m.get("content")
        if m.get("role") == "tool" and content and len(content) > min_chars and PRUNE_MARK not in content:
            head = content[:HEAD_CHARS].rstrip()
            removed = len(content) - len(head)
            m["content"] = (f"{head}\n{PRUNE_MARK} {removed:,} more characters removed to save context; "
                            "call the tool again if you need them]")
            pruned += 1
    return pruned


# ---------------------------------------------------------------
# Summarizing (compaction)
# ---------------------------------------------------------------
def _safe_cut(messages: list, keep_recent: int):
    """Index where the 'recent' part starts. It is always a user message, so a tool call is
    never separated from its tool result (the API rejects orphaned tool messages)."""
    limit = len(messages) - keep_recent
    candidates = [i for i, m in enumerate(messages) if 2 <= i <= limit and m["role"] == "user"]
    return candidates[-1] if candidates else None


def render_transcript(messages: list, max_chars: int = 600) -> str:
    """Turn messages into plain text for the summarizer, shortening long pieces."""
    lines = []
    for m in messages:
        for c in m.get("tool_calls") or []:
            lines.append(f"ASSISTANT called {c['function']['name']}({c['function']['arguments'][:200]})")
        text = m.get("content") or ""
        if text:
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            lines.append(f"{m['role'].upper()}: {text}")
    return "\n".join(lines)


def compact(messages: list, summarize_fn, keep_recent: int = 8) -> bool:
    """Replace old messages with an LLM-written summary, IN PLACE. Returns True if it did something."""
    cut = _safe_cut(messages, keep_recent)
    if cut is None:
        return False
    try:
        summary = summarize_fn(render_transcript(messages[1:cut]))
    except Exception as e:
        print(f"  [context] summarizing failed ({e}); keeping the full history")
        return False
    if not summary.strip():
        return False

    messages[:] = [
        messages[0],                                    # system prompt
        {"role": "user", "content": f"[Summary of our earlier conversation]\n{summary}"},
        {"role": "assistant", "content": "Understood. I will continue from this summary."},
        *messages[cut:],                                # recent messages, kept word for word
    ]
    return True


def manage_context(messages, last_prompt_tokens, prune_at, compact_at, summarize_fn, keep_recent=8) -> bool:
    """Called before every model call. Acts only when the context is getting big.
    Returns True if it changed the history (the caller should then reset its loop guard)."""
    if last_prompt_tokens > compact_at:
        before = len(messages)
        if compact(messages, summarize_fn, keep_recent):
            print(f"  [context] over {compact_at:,} tokens: summarized older messages ({before} -> {len(messages)})")
            return True
    if last_prompt_tokens > prune_at:
        n = prune_old_tool_results(messages, keep_recent)
        if n:
            print(f"  [context] over {prune_at:,} tokens: shortened {n} old tool outputs")
            return True
    return False


# ---------------------------------------------------------------
# Saved sessions (long-term memory, part 1)
# ---------------------------------------------------------------
def _safe_name(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in "-_") or "session"


def save_session(messages: list, name: str) -> Path:
    SESSIONS_DIR.mkdir(exist_ok=True)
    path = SESSIONS_DIR / f"{_safe_name(name)}.json"
    path.write_text(json.dumps(messages, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_session(name: str) -> list:
    path = SESSIONS_DIR / f"{_safe_name(name)}.json"
    messages = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(messages, list) or not messages or messages[0].get("role") != "system":
        raise ValueError("not a valid session file")
    return messages