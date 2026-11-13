"""tools.py: the tool registry and every tool the agent can use. The guards live in safety.py."""
import difflib
import inspect
import json
import os
from typing import get_type_hints

import sandbox
from safety import (
    NOTES_FILE, POLICY, WORKSPACE, ask_permission, audit, check_command, read_text_safely,
    resolve, resolve_inside_workspace, truncate,
)

DENIED = "Error: the user denied this action."
MAX_LISTED = 60   # most entries list_files will show

# ---------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------
TOOL_FUNCTIONS = {}   # name -> python function
TOOLS = []            # JSON schemas sent to the model

PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean"}


def json_type(py_type) -> dict:
    """Python type hint -> JSON schema fragment."""
    if py_type == list[str]:
        return {"type": "array", "items": {"type": "string"}}
    return {"type": PY_TO_JSON.get(py_type, "string")}


def tool(description: str, params: dict | None = None):
    """Register a function as a tool. Builds the JSON schema from its type hints."""
    params = params or {}

    def decorator(func):
        hints = get_type_hints(func)
        properties, required = {}, []
        for name, p in inspect.signature(func).parameters.items():
            properties[name] = {
                **json_type(hints.get(name, str)),
                "description": params.get(name, name),
            }
            if p.default is inspect.Parameter.empty:
                required.append(name)

        TOOLS.append({
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        })
        TOOL_FUNCTIONS[func.__name__] = func
        return func

    return decorator


# ---------------------------------------------------------------
# Read-only tools
# ---------------------------------------------------------------
@tool(
    "Read a text file and return its contents. Use this before editing a file. "
    "Very large files are truncated. Files that look like secrets (.env, keys) are blocked. "
    "Works on files only; to see what is inside a folder use list_files.",
    {"path": "Absolute path to the file (relative paths start in the workspace)"},
)
def read_file(path: str) -> str:
    shown = path
    try:
        p = resolve(path)
        shown = p
        if p.is_dir():
            return f"Error: {p} is a folder, not a file. Use list_files to see what is inside it."
        return truncate(read_text_safely(path))
    except Exception as e:
        return f"Error: cannot read {shown}: {e}"


@tool(
    "List the files and folders inside a directory. Every item is shown as a FULL path that you can copy "
    "into other tools; folders end with a path separator. Use this to explore a project before reading files.",
    {"directory": "Absolute path to the folder. Defaults to the workspace root"},
)
def list_files(directory: str = ".") -> str:
    shown = directory
    try:
        p = resolve(directory)
        shown = p
        if p.is_file():
            return f"Error: {p} is a file, not a folder. Use read_file to see its contents."
        entries = sorted(p.iterdir(), key=lambda e: e.name.lower())
        if not entries:
            return f"Contents of {p} (0 items): (empty folder)"
        lines = [str(e) + (os.sep if e.is_dir() else "") for e in entries[:MAX_LISTED]]
        if len(entries) > MAX_LISTED:
            lines.append(f"... and {len(entries) - MAX_LISTED} more not shown")
        header = f"Contents of {p} ({len(entries)} items). Full paths; folders end with a path separator:"
        return truncate(header + "\n" + "\n".join(lines))
    except Exception as e:
        return f"Error: cannot list {shown}: {e}"


@tool("Return the absolute path of the workspace folder, which is the agent's working directory.")
def current_directory() -> str:
    return str(WORKSPACE)


@tool(
    "Write or update your plan for the task. Call it FIRST for any task that needs more than two tool calls, "
    "and again each time a step is finished. The plan is shown to the user.",
    {"steps": "Short imperative steps, in order", "done": "How many steps from the start are finished (0 if none)"},
)
def update_plan(steps: list[str], done: int = 0) -> str:
    try:
        if isinstance(steps, str):                    # small models sometimes send a string
            try:
                steps = json.loads(steps)
            except json.JSONDecodeError:
                steps = steps.splitlines()
        if not isinstance(steps, list):
            steps = [str(steps)]
        steps = [str(s).strip() for s in steps if str(s).strip()]
        if not steps:
            return "Error: provide at least one step."

        done = max(0, min(int(done), len(steps)))
        lines = []
        for i, step in enumerate(steps, 1):
            mark = "x" if i <= done else (">" if i == done + 1 else " ")
            lines.append(f"  [{mark}] {i}. {step}")
        plan = "\n".join(lines)
        print("\n  Plan:\n" + plan)
        return "Plan saved:\n" + plan
    except Exception as e:
        return f"Error: {e}"


@tool("Return the size of a file in bytes.", {"path": "Absolute path to the file"})
def file_size(path: str) -> str:
    shown = path
    try:
        p = resolve(path)
        shown = p
        return f"{p.stat().st_size} bytes"
    except Exception as e:
        return f"Error: cannot get the size of {shown}: {e}"


@tool(
    "Search ONE file for a piece of text (case-insensitive) and return matching line numbers and lines. "
    "Use this to locate a function or variable in a big file instead of reading all of it. "
    "It does not search folders.",
    {"path": "Absolute path to a single file", "text": "Text to look for"},
)
def search_text(path: str, text: str) -> str:
    shown = path
    try:
        p = resolve(path)
        shown = p
        if p.is_dir():
            return f"Error: {p} is a folder. search_text works on one file; use list_files to see the files in it."
        lines = read_text_safely(path).splitlines()
        hits = [f"{n}: {line.strip()}" for n, line in enumerate(lines, 1) if text.lower() in line.lower()]
        if not hits:
            return f"No matches for '{text}' in {p}."
        note = f"\n(showing first 50 of {len(hits)} matches)" if len(hits) > 50 else ""
        return truncate("\n".join(hits[:50])) + note
    except Exception as e:
        return f"Error: cannot search {shown}: {e}"


# ---------------------------------------------------------------
# Tools that change things (approval policy applies)
# ---------------------------------------------------------------
def indent(text: str, prefix: str = "    | ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def preview(text: str, max_lines: int = 40) -> str:
    lines = text.splitlines()
    shown = indent("\n".join(lines[:max_lines]))
    hidden = len(lines) - max_lines
    return shown + (f"\n    ... ({hidden} more lines not shown, press v to view all)" if hidden > 0 else "")


@tool(
    "Create a new file, or completely overwrite an existing one, with the given content. "
    "To change only part of an existing file, use edit_file instead. Requires user approval.",
    {"path": "File path inside the workspace", "content": "The full text to write"},
)
def write_file(path: str, content: str) -> str:
    try:
        p = resolve_inside_workspace(path)
        verb = "overwrite" if p.exists() else "create"
        if not ask_permission("write_file", f"{verb} {p}", preview(content), indent(content)):
            return DENIED
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {p}"
    except Exception as e:
        return f"Error: {e}"


@tool(
    "Edit an existing file by replacing one exact piece of text with new text. "
    "old_text must match the file exactly (including indentation) and appear exactly once, "
    "so include a few surrounding lines. Read the file first. Requires user approval.",
    {
        "path": "File path inside the workspace",
        "old_text": "The exact text to replace",
        "new_text": "The replacement text",
    },
)
def edit_file(path: str, old_text: str, new_text: str) -> str:
    try:
        p = resolve_inside_workspace(path)
        original = p.read_text(encoding="utf-8")
        count = original.count(old_text)
        if count == 0:
            return "Error: old_text was not found. Read the file again and copy the text exactly."
        if count > 1:
            return f"Error: old_text appears {count} times. Include more surrounding lines so it matches once."

        updated = original.replace(old_text, new_text, 1)
        diff = "".join(difflib.unified_diff(
            original.splitlines(True), updated.splitlines(True), "before", "after", n=2
        ))
        if not ask_permission("edit_file", f"edit {p}", diff):
            return DENIED
        p.write_text(updated, encoding="utf-8")
        return f"Edited {p}"
    except Exception as e:
        return f"Error: {e}"


@tool(
    "Delete a single file inside the workspace. It cannot delete folders. Requires user approval.",
    {"path": "File path inside the workspace"},
)
def delete_file(path: str) -> str:
    try:
        p = resolve_inside_workspace(path)
        if not p.exists():
            return f"Error: {path} does not exist."
        if not p.is_file():
            return "Error: delete_file only deletes single files, not folders."
        if not ask_permission("delete_file", f"delete {p} ({p.stat().st_size} bytes)"):
            return DENIED
        p.unlink()
        return f"Deleted {p}"
    except Exception as e:
        return f"Error: {e}"


@tool(
    "Save a short, durable note about the project (how to run tests, conventions, decisions) to "
    "AGENT_NOTES.md so it is available in future sessions. Never save secrets or temporary details. "
    "Requires user approval.",
    {"note": "One or two sentences"},
)
def remember(note: str) -> str:
    try:
        note = " ".join(note.split())          # force a single line
        if not note:
            return "Error: the note is empty."
        if not ask_permission("remember", "save this note to AGENT_NOTES.md", f"    + {note}"):
            return DENIED
        with NOTES_FILE.open("a", encoding="utf-8") as f:
            f.write(f"- {note}\n")
        return "Saved."
    except Exception as e:
        return f"Error: {e}"


@tool(
    "Run a shell command in the workspace folder and return its exit code, stdout and stderr. "
    "Use it to run scripts and tests (for example: python app.py, pytest). "
    "The command cannot read from the keyboard and may have no internet access, so do not install packages. "
    "Requires user approval.",
    {"command": "The shell command to run", "timeout": "Seconds before the command is stopped (default 30, max 120)"},
)
def run_command(command: str, timeout: int = 30) -> str:
    reason = check_command(command)
    if reason:
        audit("blocked", tool="run_command", command=command, reason=reason)
        return f"Error: command blocked by the safety filter ({reason}). Do not try to work around it; explain to the user instead."
    detail = f"    $ {command}\n    (runs in: {sandbox.backend()})"
    if not ask_permission("run_command", "run this command", detail):
        return DENIED
    return sandbox.run_sandboxed(command, timeout)


# ---------------------------------------------------------------
# Changing approval policy (used by the /policy command)
# ---------------------------------------------------------------
def set_policy(tool_name: str, mode: str) -> str:
    if tool_name not in POLICY or mode not in {"ask", "auto", "deny"}:
        return f"Usage: /policy <{'|'.join(POLICY)}> <ask|auto|deny>"
    if mode == "auto" and tool_name == "delete_file":
        return "delete_file can never be auto-approved."
    if mode == "auto" and tool_name == "run_command" and not (sandbox.backend() == "docker" and sandbox.docker_ready()):
        return "run_command can only be auto-approved when the Docker sandbox is active and ready."
    POLICY[tool_name] = mode
    audit("policy_change", tool=tool_name, mode=mode)
    return f"{tool_name} is now '{mode}'"


# ---------------------------------------------------------------
# Safe tool execution: bad model output becomes text, not a crash
# ---------------------------------------------------------------
def short_args(args: dict, limit: int = 80) -> dict:
    """Shorten long values (like file contents) so the trace and audit log stay readable."""
    return {k: (str(v) if len(str(v)) <= limit else str(v)[:limit] + "...") for k, v in args.items()}


def execute_tool(call) -> str:
    name = call.function.name
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return f"Error: unknown tool '{name}'. Available tools: {', '.join(TOOL_FUNCTIONS)}"

    try:
        args = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError as e:
        return f"Error: arguments were not valid JSON ({e})"
    if not isinstance(args, dict):
        return "Error: arguments must be a JSON object"

    shown = short_args(args)
    print(f"  [tool] {name}({shown})")
    try:
        result = str(func(**args))
    except TypeError as e:
        result = f"Error: wrong arguments for {name}: {e}"
    except Exception as e:
        result = f"Error: {name} failed: {e}"

    audit("tool", name=name, args=shown, result=result[:120])
    first_line = result.splitlines()[0][:100] if result else "(empty)"
    print(f"    -> {first_line}")
    return result