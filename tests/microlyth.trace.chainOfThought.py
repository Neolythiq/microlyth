from microlyth.src.trace import ChainOfThought, StepType

import asyncio

async def SimulateAgentWorkflow():
    # 1. Initialize ChainOfThought with the main agent
    cot = ChainOfThought(rootAgentName="OrchestratorAgent")
    print("Initial Stack Trace:\n", cot.GetStackTrace(), "\n")

    # 2. OrchestratorAgent performs local reasoning and a tool action
    cot.AddStep(
        StepType.THOUGHT, 
        "The user requested financial data and market sentiment. I need to gather both."
    )
    cot.AddStep(
        StepType.ACTION, 
        {"action": "CALL", "tool": "FetchStockPrice", "ticker": "MSFT"}
    )
    cot.AddStep(
        StepType.OBSERVATION, 
        "MSFT is currently trading at $420.50 (+1.2%)."
    )

    # 3. OrchestratorAgent decides to spawn/delegate a sub-task to ResearchAgent
    print("--> Delegating to ResearchAgent...")
    cot.PushDelegation(
        subAgentName="ResearchAgent", 
        subTask="Perform sentiment analysis on latest news for MSFT."
    )
    
    print("\nUpdated Stack Trace during Delegation:\n", cot.GetStackTrace(), "\n")

    # 4. ResearchAgent executes steps in its own stack frame
    cot.AddStep(
        StepType.THOUGHT, 
        "I need to query news APIs and evaluate overall headline sentiment."
    )
    cot.AddStep(
        StepType.ACTION, 
        {"action": "CALL", "tool": "SearchNews", "query": "MSFT stock updates"}
    )
    cot.AddStep(
        StepType.OBSERVATION, 
        "Found 5 articles: 4 positive, 1 neutral. Strong cloud earnings growth reported."
    )
    cot.AddStep(
        StepType.THOUGHT, 
        "Sentiment analysis complete. Summary ready to send back to Orchestrator."
    )

    # 5. ResearchAgent finishes sub-task and pops delegation stack
    subTaskResult = "Sentiment for MSFT is strongly bullish based on recent cloud revenue metrics."
    cot.PopDelegation(result=subTaskResult)

    print("\nStack Trace after popping Delegation:\n", cot.GetStackTrace(), "\n")

    # 6. OrchestratorAgent concludes the task
    cot.AddStep(
        StepType.THOUGHT, 
        "Received sentiment results. Compiling final answer for the user."
    )
    cot.AddStep(
        StepType.STATE_TRANSITION, 
        "COMPLETE"
    )

    # 7. Print total XML history recorded across all frames
    print("=== Complete Rendered Execution History (XML) ===")
    print(cot.RenderHistory())

if __name__ == "__main__":
    asyncio.run(SimulateAgentWorkflow())