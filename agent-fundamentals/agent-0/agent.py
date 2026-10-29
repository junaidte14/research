import json
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"  # must be a Groq model that supports tool use

# 1. The tool: a plain Python function
def read_file(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"   # return errors as text so the LLM can react

def list_files(directory: str) -> str:
    try:
        files = os.listdir(directory)
        return "\n".join(files)
    except Exception as e:
        return f"Error: {e}"   # return errors as text so the LLM can react

# 2. A registry mapping tool names to functions
TOOL_FUNCTIONS = {"read_file": read_file, "list_files": list_files}

# 3. The tool's description: this is all the LLM sees
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files in the provided directory.",
            "parameters": {
                "type": "object",
                "properties": {"directory": {"type": "string", "description": "Folder/directory path"}},
                "required": ["directory"],
            },
        },
    }
]

# 4. The loop
def run_agent(task: str, max_steps: int = 10) -> str:
    SYSTEM_PROMPT = (
        "You are a coding assistant. Use the tools to complete the user's task. "
        "If a file or resource the user asked for does not exist, say so clearly "
        "before suggesting or trying alternatives."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    for _ in range(max_steps):                      # a cap prevents infinite loops
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",                     # the model decides whether to use a tool
        )
        msg = resp.choices[0].message
        messages.append(msg)                        # keep the assistant turn in history

        if not msg.tool_calls:                      # no tool wanted -> done
            return msg.content

        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)   # arguments arrive as a JSON *string*
            print(f"[tool] {name}({args})")

            func = TOOL_FUNCTIONS.get(name)
            output = func(**args) if func else f"Error: unknown tool {name}"  # WE run the tool

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": output,
            })

    return "Stopped: reached max_steps."

#print(run_agent("Read agent0.py and explain what it does in 3 sentences.", max_steps=20))
print(run_agent("Read all folders in the agent-fundamentals directory (a directory 1/2 steps above the current directory) and explain what each folder project does briefly."))