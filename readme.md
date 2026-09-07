# Microlyth

> **Small tools build monumental intelligence.**

---

## The Philosophy Behind Microlyth

Human history was fundamentally shaped by our ability to craft and master tools. The invention of the earliest stone tools—**liths**—did not merely change how hominids interacted with the physical world; it triggered an evolutionary leap in cognitive complexity, strategy, and abstract reasoning. 

**Microlyth** (from *micro-* + *-lith*, "small stone tool") is built on a simple premise: **AI intelligence scales not just through model parameters, but through targeted tool usage.** 

While many modern frameworks over-complicate agentic workflows with monolithic architectures and bloated agent definitions, Microlyth returns to first principles: **hyper-focused agentic building blocks centered entirely around crisp, precise tool calling.**

---

## ⚡ Key Principles

### 1 — Model Agnostic
Microlyth seamlessly sits on top of any LLM provider (OpenAI, Anthropic, Google Gemini, Ollama, vLLM, etc.). Swap backends with zero friction and without refactoring your agent logic.

### 2 — Business Goal Driven, Not "Agentic Coding"
Frameworks shouldn't require hundreds of lines of boilerplate or deep framework abstractions just to handle logic flow. Microlyth is designed for business engineers to map workflows directly to real-world outcomes without getting bogged down in low-level loop mechanics.

### 3 — Hyper-Specialized Micro-Agents
Large toolsets pollute model context, leading to hallucinations, degraded reasoning, and soaring latency. Microlyth advocates for **Micro-Agents**: tight, context-constrained agents equipped *only* with the handful of dedicated tools necessary for their immediate function.

### 4 — Swarm Orchestration via Hand-off
Instead of a single agent wearing ten hats, Microlyth manages agent swarms through dynamic hand-offs. Agents seamlessly switch control and pass execution state to another specialized agent when the task transitions across domain boundaries.

### 5 — ReAct Loop with Instruction-Set Execution
At its core, Microlyth runs a lightweight **ReAct (Reason + Act) loop** driven by an explicit instruction set. Handled primitives include:
* **Tool Calling:** Native, typed execution of micro-tools.
* **Agent Switching:** Clean context transfer to another specialized swarm member.
* **User Notification:** Pause execution or send real-time user feedback/prompts.
* **UI Interaction:** Seamless stream hooks for frontend integrations and interactive components.

---

## 🚀 Quick Start

```python
from microlyth import MicroAgent, Swarm, tool

@tool
def CalculateDiscount(price: float, percentage: float) -> float:
    """Calculates the final price after discount."""
    return price * (1 - percentage / 100)

# Define specialized micro-agent
salesAgent = MicroAgent(
    name="SalesSpecialist",
    instructions="Help customers with pricing and discount calculations.",
    tools=[CalculateDiscount]
)

# Run agent swarm execution loop
swarm = Swarm(defaultAgent=salesAgent)
response = swarm.Run("Can you apply a 15% discount to $250?")
print(response)