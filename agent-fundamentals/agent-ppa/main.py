from litellm_adapter import LiteLLMAdapter
from llm_interface import LLMClient

SYSTEM_PROMPT = """
You are a helpful AI assistant.

Your responsibilities:
1. Understand the user's questions.
2. Provide clear and accurate answers.
3. Ask for clarification when necessary.
4. Never pretend to have performed an action
   that you have not actually performed.

You currently do not have access to external tools.
"""

def main():

    print("=" * 55)
    print("     My first provider independent app")
    print("=" * 55)

    llm: LLMClient = LiteLLMAdapter()

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    print("\n System is ready!")
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

        messages.append({
            "role": "user",
            "content": user_input
        })

        try:

            answer = llm.complete(messages=messages)
            messages.append(
                {
                    "role": "system",
                    "content": answer
                }
            )
            print(f"Assistant: {answer}")

        except Exception as err:
            messages.pop() # remove the failed user message from history
            raise RuntimeError(f"Error occured: {err}")

if __name__ == "__main__":
    main()
