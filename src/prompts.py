from microlyth.src.system import SystemCmd

class PromptPrimitives:
    """
    Static textual building blocks for system prompts.
    """
    PreliminaryHeader = """**You are an autonomous agent designed to execute tasks based on a structured prompt.**"""

    TaskUnderstanding = """
* Read carefully the <current_task> and the <agent_manifest> to understand your identity, capabilities, and the specific task you need to accomplish.
* Use the <enriched_context> to gather any relevant information that can assist you in completing the task. This may include retrieved knowledge, past experiences, or any other data provided.
* Please execute the task below following the logic defined in the Agent <instructions> and the interaction style defined in the Agent <behaviors> section of the manifest.
""".strip()
    
    ReasoningPhase = """
- Analyze the current context, past observations, and task goals.
- Explicitly state:
  1. What do I know so far from prior observations?
  2. What critical information is still missing?
  3. What is my next logical hypothesis or action step?
""".strip()

    ActPhase = """
- Based on your thought process, choose EXACTLY ONE action type:
  1. EXECUTE ACTION: Invoke one or more actions/tools using registered tags (e.g., <action:CALL>, <action:HALT>, <action:COMPLETE>, ...).
  2. PROVIDE DATA / OUTPUT: If no tool call is required or the task is ready for an update/completion, provide your response payload directly.
""".strip()

    ObservePhase = """
- After outputting an action tag, wait for the engine to execute your action and return environmental results in an <observation> block before proceeding to your next thought cycle.
""".strip()

    DefaultCycleFormatting = """
* Always respond in the format defined right below. Do not deviate.
* Format: <thought>...</thought> followed by <action_or_output>...</action_or_output> and <control_flow>...</control_flow>.
* Always ensure that the <control_flow> tag is present and correctly reflects and summarizes the the current state.
* Always ensure that the <action_or_output> tag is present, even if you have no action or output to provide at the moment (use <action_or_output>None</action_or_output> if so).
* Always ensure that the <thought> tag is present, even if you have no specific thought to share at the moment (use <thought>None</thought> if so).
* Always prioritize system format over agent_manifest format. If there is a conflict, follow the system instructions and ignore the conflicting part in the manifest, but make sure to acknowledge it in your thought process.
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
    def ReasoningPhase() -> SystemCmd:
        return SystemCmd(
            title="REASONING PHASE",
            content=PromptPrimitives.ReasoningPhase
        )

    @staticmethod
    def ActPhase() -> SystemCmd:
        return SystemCmd(
            title="ACT PHASE",
            content=PromptPrimitives.ActPhase
        )

    @staticmethod
    def ObservePhase() -> SystemCmd:
        return SystemCmd(
            title="OBSERVE PHASE",
            content=PromptPrimitives.ObservePhase
        )