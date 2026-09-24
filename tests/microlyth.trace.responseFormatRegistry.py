
from microlyth.src.system import SystemCmd, SystemInstructions
from microlyth.src.trace import ResponseFormatRegistry

if __name__ == "__main__":
    # Initialize format registry
    formatRegistry = ResponseFormatRegistry()

    # Configure response requirements
    formatRegistry.SetFormat(
        name="ReAct-Cycle-XML",
        instruction_text="""
    Always respond using the XML structure below:
    1. <thought>: Internal reasoning, task assessment, hypothesis.
    2. <action_or_output>: Action calls or final answer.
    3. <control_flow>: Next state signal ([REFINE], [INPUT], [COMPLETE], [ABORT]).
    """,
        required_tags=["thought", "action_or_output", "control_flow"]
    )

    # Inject into SystemInstructions using a callable reference
    systemPrompt = SystemInstructions([
        SystemCmd(
            title="Cycle Formatting Rules",
            content=formatRegistry.GetFormatInstructions  # Pass callable directly!
        )
    ])

    print(systemPrompt.Render())