# Agent PPA: A Tool-Using AI Agent with Safety and Reliability Controls

A hands-on Python project exploring how to build more reliable tool-using AI agents by combining LLM-based tool selection with application-level validation, authorization, human approval, retry handling, idempotency, operation leases, recovery, and audit logging.

> **Project status:** Educational and experimental implementation. It is not a production-ready system for payments, messaging, or other high-impact external actions.

## Why this project?

Connecting an LLM to tools is relatively straightforward. Making sure those tools are invoked safely and reliably is a much harder engineering problem.

An agent may select the wrong tool, generate malformed arguments, request an unauthorized action, repeat an operation after a timeout, or lose track of an operation when a worker crashes.

This project explores an important design principle:

**The LLM proposes actions; the application validates, authorizes, approves, executes, and records them.**

## Key concepts implemented

### 1. System prompt guidance

The agent receives instructions describing available tools and appropriate tool-use behavior. Prompts guide model behavior, but are not treated as a security boundary.

### 2. Precise tool definitions

Tools have explicit names, descriptions, and argument schemas. Their definitions are exposed to the LLM through a centralized registry.

### 3. Schema validation

Tool arguments are validated using Pydantic before execution. Schemas reject unexpected fields and malformed inputs, helping establish a predictable boundary between model-generated arguments and application code.

### 4. Permission-based access control

Each tool declares a required permission. An application-level policy checks whether that permission is allowed before the tool can execute. The design follows the principle of least privilege.

### 5. Human-in-the-loop approval

Sensitive tools can require an application-level approval callback. The approval decision is made outside the LLM, using validated tool arguments. The example messaging tool is simulated; it does not send real messages.

### 6. Timeouts and controlled retries

LLM requests have configured timeouts. Tool execution supports bounded retries for explicitly classified transient failures, with exponential backoff. Retries are not automatically safe for operations with external side effects.

### 7. Idempotency and concurrency control

SQLite stores operation records keyed by an idempotency key. Repeated requests with the same key can return a recorded result or report an operation that is already in progress, rather than blindly executing it again.

Operation leases and ownership tokens help prevent competing workers from finalizing the same operation. Expired operations are moved to a recovery-required state rather than automatically re-executed.

**Important:** A local idempotency store does not by itself guarantee exactly-once execution in an external service. Safe external side effects require cooperation from that service, such as its own idempotency support, or a reconciliation process.

### 8. Audit logging and recovery

Important execution events are recorded in JSON Lines format. SQLite separately tracks operation lifecycle state, including completion, cancellation, failure, and recovery-required outcomes.

Audit logs support investigation; operation records support execution-state management. They serve related but distinct purposes.

## Architecture overview

```text
User Request
     |
     v
LLM / Agent
     |
     v
Tool Selection
     |
     v
Tool Registry
     |
     v
Permission Check
     |
     v
Argument Validation (Pydantic)
     |
     v
Idempotency Check / Operation Claim (SQLite)
     |
     v
Human Approval (when required)
     |
     v
Tool Execution (bounded retry)
     |
     v
Operation Finalization
     |
     v
Audit Logging
```

The exact execution path depends on whether an idempotency key is supplied and whether the selected tool requires approval.

## Current example tools

The project currently includes:

- `calculator` — evaluates supported mathematical expressions using a constrained parser.
- `get_current_datetime` — returns the current date and time.
- `analyze_text` — performs basic text analysis.
- `send_message` — demonstrates a sensitive, approval-gated action using simulation rather than real message delivery.

## Reliability mechanisms

The implementation includes:

- Provider-independent LLM interface and LiteLLM adapter.
- Centralized tool registry and generated tool definitions.
- Pydantic argument validation.
- Permission allowlisting.
- Human approval callback for sensitive tools.
- Configurable tool retry policy and LLM request timeouts.
- SQLite-backed operation records and idempotency keys.
- Atomic operation claiming.
- Lease expiration, renewal, and ownership checks.
- Explicit recovery-required state for uncertain operations.
- Structured JSONL audit events.

## Testing

The project uses `pytest` to test tool behavior and execution controls.

The latest full test run reported **71 passing tests** (November 18, 2026). This is a point-in-time result; run the test suite in your own environment to verify the current checkout.

From the repository root:

```powershell
python -m pytest -v
```

Run a specific test module:

```powershell
python -m pytest tests/test_tool_architecture.py -v
python -m pytest tests/test_idempotency.py -v
python -m pytest tests/test_agent_idempotency.py -v
```

Tests cover areas including validation, authorization, approval handling, retries, idempotency, duplicate requests, lease expiration, and recovery behavior.

## Getting started

### Requirements

- Python 3.12 or compatible version
- An API key for a supported LLM provider configured through the project's environment/configuration
- Project dependencies installed in a virtual environment

### Setup

Clone the repository and enter the project directory:

```powershell
git clone https://github.com/junaidte14/research.git
cd research/agent-fundamentals/agent-ppa
```

Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Configure the required LLM provider credentials and model settings according to `config.py` and the project's environment-variable conventions. Do not commit API keys or other secrets.

Run the agent:

```powershell
python main.py
```

Run tests:

```powershell
python -m pytest -v
```

> If the repository's dependency file or configuration conventions change, follow the current files in the checkout. This README describes the project state and setup pattern, not a guarantee that every environment has identical configuration.

## Current limitations and security considerations

This project is an evolving learning and experimentation codebase. In particular:

- The permission policy is an application-level allowlist, not a complete identity-aware authorization system.
- Human approval is demonstrated through a local callback; a production approval workflow needs authenticated users, secure interfaces, and tamper-resistant records.
- The messaging tool is simulated.
- Tool retry safety depends on the operation and external service semantics.
- SQLite leases coordinate local operation ownership; they do not fence off side effects in an external system.
- An expired lease means the outcome may be uncertain. It does not prove the action did not happen.
- Prompt instructions and tool descriptions do not independently defend against prompt injection.
- Production use would require additional controls, such as secret management, identity and access management, stronger isolation, rate limiting, monitoring, external-service idempotency, and security testing.

## Roadmap

Planned learning and engineering work includes:

- Threat modeling and adversarial testing.
- Prompt-injection tests involving untrusted tool output.
- Fault injection and concurrency stress tests.
- More structured execution traces and failure attribution.
- Reproducible reliability and safety evaluation.

## Learning and research motivation

This project is also a practical foundation for studying the reliability and safety of tool-using LLM agents. The longer-term research interest is to evaluate how application-level controls affect execution reliability, safety decisions, and the consistency of explanations under repeated requests and injected failures.

The implementation should be treated as an experimental platform, not as evidence that any particular control is universally effective. Those claims require controlled comparisons and empirical evaluation.

## References

- [LiteLLM documentation](https://docs.litellm.ai/docs/)
- [Pydantic models and validation](https://docs.pydantic.dev/latest/concepts/models/)
- [Python `sqlite3` documentation](https://docs.python.org/3/library/sqlite3.html)
- [pytest documentation](https://docs.pytest.org/en/stable/)
- [AWS Builders' Library: Making retries safe with idempotent APIs](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-apis/)
- [OWASP GenAI Security Project: Excessive Agency](https://genai.owasp.org/llmrisk/llm062025-excessive-agency/)
- [AgentDojo: Evaluating prompt-injection attacks and defenses for LLM agents](https://arxiv.org/abs/2406.13352)

## Author

[Junaid Hassan](https://github.com/junaidte14)

## Repository

[View the Agent PPA source code](https://github.com/junaidte14/research/tree/main/agent-fundamentals/agent-ppa)
