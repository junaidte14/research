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
5. Use the calculator tool for arithmetic calculations.
6. When you use a tool, base your answer on its actual result.

You have access to a calculator tool.

IMPORTANT TOOL RULES:
1. Whenever the user asks you to calculate or evaluate a mathematical
   expression, you MUST call the calculator tool.
2. Never calculate arithmetic mentally.
3. Never answer an arithmetic question without first using the calculator.
4. If the calculator returns an error, explain that error to the user.
5. For non-mathematical questions, answer normally.
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
