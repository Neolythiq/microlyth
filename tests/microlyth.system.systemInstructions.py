from microlyth.src.system import SystemInstructions, SystemCmd
from microlyth.src.system import InstructionSet

import json
from typing import Any

if __name__ == "__main__":
    customInstructions = InstructionSet(openBlockTag="<action:", endBlockTag="</action:", closeTag=">")

    # ---------------- Define custom instructions for the agent ----------------
    # 1. Tool Execution Primitive
    @customInstructions.NewInstruction(
        name="CALL",
        description="Invoke a tool. tool_name(arg1=val1)",
        example="Payload MUST be valid JSON:\n<action:CALL>\n{\"tool_name\": \"my_tool\", \"arguments\": {}}\n</action:CALL>",
        payloadParser=json.loads
    )
    def ToolCallHandler(payload: str, agentContext: Any) -> Any:
        # Logic to parse payload and call tool on agentContext
        return f"Tool output for {payload}"
    
    reactInstructions = SystemInstructions([
        SystemCmd(
            title="Role & Task Understanding",
            content="Read carefully the <current_task> and follow agent rules."
        ),
        SystemCmd(
            title="Reasoning Chain",
            content="State explicitly: What do I know so far? What is missing?"
        ),

        # Integrates directly with InstructionSet from Microlyth!
        SystemCmd(
            title="Available Actions & Protocols",
            content=lambda: customInstructions.BuildSystemPromptInstructions()
        )
    ])

    print(reactInstructions.Render())

    planAndExecuteInstructions = SystemInstructions([
        SystemCmd(
            title="System Role",
            content="You are a strategic planning agent."
        ),
        SystemCmd(
            title="Planning Rules",
            content=[
                "1. Break the task down into sub-goals.",
                "2. Assign dependent tools to each step.",
                "3. Output plan in valid JSON format."
            ]
        )
    ])

    print(planAndExecuteInstructions.Render())