from microlyth.src.system import InstructionSet

import json
from typing import Any, Dict

if __name__ == "__main__":
    # Initialize instruction set with custom syntax tags
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

    # 2. Agent Handoff Primitive
    @customInstructions.NewInstruction(
        name="SWITCH",
        description="Hand off execution to another agent."
    )
    def AgentSwitchHandler(payload: str, agentContext: Any) -> Any:
        agentContext.ActiveAgent = payload.strip()
        return f"Switched focus to agent: {payload}"

    # 3. Task Termination Primitive
    @customInstructions.NewInstruction(
        name="COMPLETE",
        description="Mark the task as completed."
    )

    def CompleteHandler(payload: str, agentContext: Any) -> Any:
        return {"status": "SUCCESS", "result": payload}

    # 4. Loop Abort Primitive
    @customInstructions.NewInstruction(
        name="ABORT",
        description="Stop the engine loop due to an unrecoverable issue."
    )

    def AbortHandler(payload: str, agentContext: Any) -> Any:
        raise RuntimeError(f"Engine Loop Aborted: {payload}")

    sysInstructions = customInstructions.BuildSystemPromptInstructions()
    print(sysInstructions)

    @customInstructions.SystemPromptFormatter
    def CustomPromptFormatter(instruction_specs: Dict[str, Any]) -> str:
        output = ["Coin", "Coin"]
        return "\n".join(output)

    sysInstructions = customInstructions.BuildSystemPromptInstructions()
    print(sysInstructions)

    # ---------------- Custom instructions Parsing ----------------
    sampleLlmResponse = """
    I will start by checking weather and stock prices at the same time.

    <action:PARALLEL>
        <action:CALL> {"tool_name": "get_weather", "city": "Paris"} </action:CALL>
        <action:CALL> {"tool_name": "get_stock", "ticker": "MSFT"} </action:CALL>
    </action:PARALLEL>

    Now let's hand off control to the research agent.

    <action:SWITCH> {"target_agent": "ResearchAgent"} </action:SWITCH>

    Next, let's run a pipeline where step 2 depends on step 1.

    <action:PIPELINE>
        <action:CALL> {"tool_name": "fetch_document", "id": 101} </action:CALL>
        <action:CALL> {"tool_name": "summarize_text"} </action:CALL>
    </action:PIPELINE>

    Finally, we complete the task.

    <action:COMPLETE> Everything has been processed successfully. </action:COMPLETE>
    """

    print("=== 1. Testing ParseResponse ===")
    batches = customInstructions.ParseResponse(sampleLlmResponse)

    for idx, batch in enumerate(batches, 1):
        print(f"\nBatch #{idx} | Mode: {batch.mode.value} | Action Count: {len(batch.actions)}")
        for action in batch.actions:
            print(f"  - Action: {action.actionType} | payload={action.rawPayload} | args={action.args}")

    # ---------------- Custom instructions Dispatch ----------------
    class MockAgentContext:
        def __init__(self):
            self.active_agent = "MainAgent"
            self.history = []

    print("\n=== 2. Testing DispatchBatch Execution ===")
    context = MockAgentContext()
   
    for idx, batch in enumerate(batches, 1):
        print(f"\nExecuting Batch #{idx} ({batch.mode.value})...")
        results = customInstructions.DispatchBatch(batch, context)
        print(f"Batch #{idx} Results: {results}")