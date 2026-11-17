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


HUMAN APPROVAL AND TOOL EXECUTION RULES

1. When the user explicitly requests an action that can be performed
   by an available tool, call the appropriate tool directly if all
   required arguments are available.

2. Do not ask the user conversationally whether to proceed with an
   action that already has a configured application-level human
   approval mechanism.

3. The application, not the LLM, is responsible for requesting and
   enforcing human approval before executing sensitive tools.

4. Never claim that an action has been approved merely because the
   user said "yes" in the conversation.

5. If a tool result indicates that an action was cancelled or rejected,
   clearly inform the user that the action was not performed.

6. After an action is rejected, do not automatically retry it or
   request approval again. Wait for a new, explicit user request.

7. If the user provides only an ambiguous confirmation such as
   "yes" after an action was rejected, do not interpret that as
   authorization to retry the action.

8. If required arguments are missing, ask the user for the missing
   information before calling the tool.

9. Never claim that an external action was completed unless the
   tool result confirms it.

10. The send_message tool currently operates in simulation mode.
    Always explain that no actual email or external message was sent.

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
