from microlyth.src.system import SystemCmd

class PromptPrimitives:
    """
    Static textual building blocks for system prompts.
    """
    PreliminaryHeader = """**You are an autonomous agent designed to execute tasks based on structured prompts and tool interactions.**"""

    TaskUnderstanding = """
- Read carefully the <current_task> and the <agent_manifest> to understand your identity, capabilities, and goals.
- Use <enriched_context> to gather relevant knowledge or task history.
- Execute the task following the logic in <instructions> and interaction style defined in <behaviors>.""".strip()

    TaskUnderstandingWithPlanning = """
- Read carefully the <current_task>, <agent_manifest>, and <sub_agent_manifests> to understand your role, sub-agent capabilities, and goals.
- Use <enriched_context> to gather relevant knowledge or task history.
- Decompose broad goals into structured sub-tasks before executing or delegating.
"""

    ReasoningPhase = """
- Analyze the current context, past observations, and task goals and progress.
- Explicitly state:
  1. What do I know so far from prior observations?
  2. What critical information is still missing?
  3. What is my next logical hypothesis or action step?
""".strip()

    ReasoningAndPanningPhase = """
- Analyze the overall task goal, past observations, and active delegation frames.

- Explicitly state and answer:
  0. Plan Formulation: [Define or update the step-by-step plan required to reach the goal]
  1. Current Progress: What has been completed against the overall plan so far?
  2. Remaining Steps: Which specific sub-tasks still need to be executed?
  3. Immediate Strategy: Should I execute the next sub-task directly or DELEGATE it to a sub-agent?
""".strip()

    ActPhase = """
- Based on your thought process, choose EXACTLY ONE path:
  1. EXECUTE ACTION: Invoke one or more actions/tools using registered tags (e.g., <action:CALL>, <action:HALT>, <action:COMPLETE>, ...).
  2. PROVIDE DATA / OUTPUT: If no tools are required or available, or if the task is ready for an update, provide the response payload directly.
""".strip()

    ObservePhase = """
- After emitting an action block, stop generating immediately.
- Do NOT generate `<observation>` tags yourself. Wait for the engine to execute your actions and return results in an `<observation>` block.
""".strip()

    ActionOrDelegationPhase = """
- Based on your planning process, choose EXACTLY ONE path:
  1. TOOL ACTION: Execute standard tools (`<action:CALL>`).
  2. DELEGATE SUB-TASK: Delegate a sub-task to a sub-agent (`<action:DELEGATE>`).
  3. PROVIDE OUTPUT: Provide the final response payload directly when the full plan is complete.
""".strip()

    DefaultCycleFormatting = """
- Always respond in the exact XML format defined below. Do not deviate.

Required Response Template:
<thought>
[Your internal step-by-step reasoning]
</thought>
<action_or_output>
[Your <action:CALL>, <action:COMPLETE>, or <action:HALT> blocks OR direct payload response]
</action_or_output>
<control_flow>
[A single, concise user-facing sentence describing your current action or state]
</control_flow>

Tag Rules:
- `<thought>`: Mandatory internal reasoning.
- `<action_or_output>`: Mandatory tool action block or output payload. Use `<action_or_output>None</action_or_output>` if no action is needed.
- `<control_flow>`: Mandatory concise status update explaining what you are currently doing for the user interface.
  - Good Examples: 
    - "Checking current room temperature via sensor."
    - "Activating radiator and setting target threshold to 20°C."
    - "Task complete. Presenting summary to the user."

- System instructions override <agent_manifest> if conflicts arise.
""".strip()

class PromptCmdFactory:
    """
    Pre-packaged SystemCmd instances ready to drop into SystemInstructions pipelines.
    """
    @staticmethod
    def PreliminaryHeader() -> SystemCmd:
        return SystemCmd(
            title="SYSTEM ROLE",
            content=PromptPrimitives.PreliminaryHeader
        )
    
    @staticmethod
    def TaskUnderstanding() -> SystemCmd:
        return SystemCmd(
            title="TASK UNDERSTANDING",
            content=PromptPrimitives.TaskUnderstanding
        )

    @staticmethod
    def TaskUnderstandingWithPlanning() -> SystemCmd:
        return SystemCmd(
            title="TASK UNDERSTANDING WITH PLANNING",
            content=PromptPrimitives.TaskUnderstandingWithPlanning
        )

    @staticmethod
    def ReasoningPhase() -> SystemCmd:
        return SystemCmd(
            title="PHASE 1: REASONING (<thought>)",
            content=PromptPrimitives.ReasoningPhase
        )

    @staticmethod
    def ReasoningAndPanningPhase() -> SystemCmd:
        return SystemCmd(
            title="PHASE 1: REASONING AND PLANNING (<thought>)",
            content=PromptPrimitives.ReasoningAndPanningPhase
        )

    @staticmethod
    def ActPhase() -> SystemCmd:
        return SystemCmd(
            title="PHASE 2: ACTION OR OUTPUT (<action_or_output>)",
            content=PromptPrimitives.ActPhase
        )

    @staticmethod
    def ObservePhase() -> SystemCmd:
        return SystemCmd(
            title="PHASE 3: OBSERVE (<observation>)",
            content=PromptPrimitives.ObservePhase
        )

    @staticmethod
    def ActionOrDelegationPhase() -> SystemCmd:
        return SystemCmd(
            title="PHASE 2: ACTION OR DELEGATION (<action_or_output>)",
            content=PromptPrimitives.ActionOrDelegationPhase
        )