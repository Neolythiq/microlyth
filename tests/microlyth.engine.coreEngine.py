

import asyncio

from microlyth.src.agents import ProxyAgentProfile
from microlyth.src.engine import CoreEngine
from microlyth.src.system import InstructionSet, SystemCmd, SystemInstructions
from microlyth.src.trace import StepType, TraceParser


async def Main():
    # 1. Setup core components
    instructionSet = InstructionSet()
    systemInstructions = SystemInstructions([
        SystemCmd(title="Instructions", content="Follow the ReAct loop carefully.")
    ])
    traceParser = TraceParser()

    # 2. Instantiate CoreEngine
    engine = CoreEngine(
        instructionSet=instructionSet,
        systemInstructions=systemInstructions,
        traceParser=traceParser
    )

    # 3. Register User Lifecycle Event Callbacks using Decorators
    @engine.OnTaskStart
    def HandleStart(eng: CoreEngine):
        print(f"🚀 [START] Task initiated for Agent: {eng.activeAgent.name}")

    @engine.BeforeStepExecution
    def HandleBeforeStep(eng: CoreEngine):
        print(f"--> [STEP {eng._stepCounter + 1}] Invoking Model...")

    @engine.AfterStepExecution
    def HandleAfterStep(eng: CoreEngine):
        latestFrame = eng.chainOfThought.CurrentFrame
        print(f"<-- [STEP {eng._stepCounter}] Complete. Total steps logged: {len(latestFrame.steps)}")

    @engine.OnHalt
    def HandleHalt(eng: CoreEngine):
        print("⏸️ [HALT] Engine paused waiting for input/event.")

    @engine.OnTaskComplete
    def HandleComplete(eng: CoreEngine):
        print("🎉 [COMPLETE] Engine loop finished successfully.")

    @engine.RegisterControlFlow("INPUT")
    def HandleInputSignal(eng: CoreEngine, signalPayload: str):
        """Custom handler for [INPUT] signal: pause engine loop."""
        print(" [CONTROL FLOW] Intercepted [INPUT]. Halting loop for user input...")
        eng.Halt(reason="Awaiting input from user interface")

    @engine.RegisterControlFlow("REFINE")
    def HandleRefineSignal(eng: CoreEngine, signalPayload: str):
        """Custom handler for [REFINE] signal: continue internal loop without tools."""
        print(" [CONTROL FLOW] Intercepted [REFINE]. Proceeding to next thought loop...")
        eng.chainOfThought.AddStep(StepType.STATE_TRANSITION, "REFINING_STATE")

    @engine.FallbackControlFlow
    def HandleUnknownSignal(eng: CoreEngine, rawSignal: str):
        """Fallback handler for unmapped control flow signals."""
        print(f"⚠️ [CONTROL FLOW WARNING] Unrecognized control flow signal: '{rawSignal}'")

    # 4. Load Agent and Run Step-by-Step Execution Loop
    agent = ProxyAgentProfile(name="SearchAgent", role="Searcher", manifest="Specialized in web search.")
    engine.SetActiveAgent(agent=agent)
    engine.StartTask(task="Find information on Microlyth framework.")

    # Explicit step iteration loop
    while engine.HasNextStep:
        trace = await engine.RunStep()
        print(f"     Step Output Thought: {trace.thought if trace else None}")

if __name__ == "__main__":
    asyncio.run(Main()) 