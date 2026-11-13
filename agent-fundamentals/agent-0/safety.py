"""safety.py: every guard in one place: paths, secrets, command filter, approvals, audit log."""
import json
import os
import re
import time
from pathlib import Path

# The agent may only WRITE inside this folder.
WORKSPACE = (Path(__file__).parent / "workspace").resolve()
WORKSPACE.mkdir(exist_ok=True)

NOTES_FILE = WORKSPACE / "AGENT_NOTES.md"   # long-term project memory, loaded into the system prompt

# Outside the workspace on purpose: the agent's file tools can't touch its own audit trail.
AUDIT_FILE = Path(__file__).parent / "audit.log"

MAX_OUTPUT_CHARS = 8000   # protects the context window from huge outputs


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated, {len(text) - limit} more characters]"


# ---------------------------------------------------------------
# Audit log: one JSON line per event
# ---------------------------------------------------------------
def audit(event: str, **fields) -> None:
    record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **fields}
    with AUDIT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
def resolve(path: str) -> Path:
    """Relative paths start in the workspace. Absolute paths stay as they are."""
    return (WORKSPACE / (path.strip() or ".")).resolve()


def resolve_inside_workspace(path: str) -> Path:
    """Like resolve(), but refuses anything outside the workspace (used for writing)."""
    p = resolve(path)
    if not p.is_relative_to(WORKSPACE):   # resolve() already collapsed '..' and symlinks
        audit("blocked", reason="outside workspace", path=path)
        raise PermissionError(f"'{path}' is outside the workspace ({WORKSPACE}). Writing is only allowed inside it.")
    return p


# ---------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------
def is_sensitive(p: Path) -> bool:
    """A simple deny-list for files that probably contain secrets."""
    name = p.name.lower()
    if name.startswith(".env") and name != ".env.example":
        return True
    return name.endswith((".pem", ".key")) or name.startswith("id_rsa")


def read_text_safely(path: str) -> str:
    p = resolve(path)
    if is_sensitive(p):
        audit("blocked", reason="secret file", path=str(p))
        raise PermissionError("reading this file is blocked because it may contain secrets")
    return p.read_text(encoding="utf-8", errors="replace")


SECRET_ENV_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def clean_env() -> dict:
    """A copy of the environment WITHOUT secret-looking variables such as GROQ_API_KEY."""
    env = {k: v for k, v in os.environ.items() if not any(h in k.upper() for h in SECRET_ENV_HINTS)}
    env["PYTHONIOENCODING"] = "utf-8"
    return env


# ---------------------------------------------------------------
# Command filter: a seat belt, not a wall (see test_filter_is_only_a_seat_belt)
# ---------------------------------------------------------------
BLOCKED_COMMAND_PATTERNS = [
    (r"\brm\s+(-\w*\s+)*-\w*[rf]", "recursive or forced delete (rm -r / rm -f)"),
    (r"\bdel\s+.*(/s|/q)", "bulk delete (del /s or /q)"),
    (r"\b(rmdir|rd)\s+.*/s", "recursive folder delete"),
    (r"remove-item\s+.*-recurse", "recursive delete (PowerShell)"),
    (r"(?:^|[;&|]\s*)(format|diskpart|mkfs\S*|shutdown|reboot)\b", "disk or system command"),
    (r"(curl|wget|iwr|invoke-webrequest)\b.*\|\s*(sh|bash|iex|powershell)\b", "download-and-execute"),
    (r"\bgit\s+push\b", "pushing to a remote"),
]


def check_command(command: str) -> str | None:
    """Return the reason if the command matches a blocked pattern, otherwise None."""
    for pattern, reason in BLOCKED_COMMAND_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return reason
    return None


# ---------------------------------------------------------------
# Approvals: per-tool policy
# ---------------------------------------------------------------
POLICY = {"write_file": "ask", "edit_file": "ask", "delete_file": "ask", "run_command": "ask", "remember": "ask"}

def ask_permission(tool_name: str, action: str, detail: str = "", full_detail: str | None = None) -> bool:
    """Human in the loop. Obeys the tool's policy and records every decision."""
    mode = POLICY.get(tool_name, "ask")
    if mode == "deny":
        audit("approval", tool=tool_name, action=action, decision="denied by policy")
        return False
    if mode == "auto":
        audit("approval", tool=tool_name, action=action, decision="auto-approved")
        return True

    print(f"\n  [approval needed] The agent wants to {action}")
    if detail:
        print(detail)
    options = "y/N/v=view all" if full_detail else "y/N"
    while True:
        answer = input(f"  Allow? [{options}] ").strip().lower()
        if answer == "v" and full_detail:
            print(full_detail)
            continue
        break

    allowed = answer in {"y", "yes"}
    audit("approval", tool=tool_name, action=action, decision="approved" if allowed else "denied")
    return allowed