from typing import Any
from agent import run_agent_turn
from litellm_adapter import LiteLLMAdapter
from llm_interface import LLMClient

SYSTEM_PROMPT = """
You are a helpful AI assistant.

Your responsibilities:
1. Understand the user's questions.
2. Provide clear and accurate answers.
3. Ask for clarification when necessary.
4. Never pretend to have performed an action you have not performed.
6. When you use a tool, base your answer on its actual result.

Tool usage rules:

1. CALCULATOR:
   Whenever the user asks for arithmetic calculations,
   you MUST use the calculator tool.
   Never calculate arithmetic mentally.

2. CURRENT DATE AND TIME:
   Whenever the user asks for the current date, time,
   timezone or weekday, use get_current_datetime.

3. TEXT ANALYZER:
   Whenever the user asks for word count, character count,
   line count or unique word count, use analyze_text.

4. ERROR HANDLING:
   If a tool returns an error, explain it clearly.
   Do not invent successful results.

5. GENERAL CONVERSATION:
   For ordinary questions, answer normally without
   unnecessarily invoking tools.

Always use the appropriate tool when its functionality
is required.
"""

def main():

    print("=" * 55)
    print("     MY FIRST TOOL-USING AI AGENT    ")
    print("=" * 55)

    llm: LLMClient = LiteLLMAdapter()

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    print("\n Agent is ready!")
    print("Type /exit to quit.")
    print("Type /reset to clear the messages history.")

    while True:
        user_input = input("\n You: ").strip()
        if not user_input:
            continue

        if user_input.lower() == "/exit":
            print("\n Good bye!")
            break

        if user_input.lower() == "/reset":
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                }
            ]
            print("\nConversation history is cleared")
            continue

        #saving the history length
        previous_history_length = len(messages)

        messages.append({
            "role": "user",
            "content": user_input
        })

        try:

            answer = run_agent_turn(llm=llm, messages=messages)

            print(f"\nAssistant: {answer}")

        except Exception as err:
            del messages[previous_history_length:] # remove all messages from the failed turn
            print(f"Error: {err}")

if __name__ == "__main__":
    main()
