from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any, Optional

from enum import Enum, auto
import inspect
import asyncio

from microlyth.src.system import InstructionSet, SystemInstructions
from microlyth.src.trace import ParsedCycleTrace, StepType, TraceParser, ChainOfThought
from microlyth.src.agents import AgentProfile

import re

#class EngineType(Enum):
#    REACT = auto()
#    PLAN_AND_EXECUTE = auto()
#    SELF_CORRECTION = auto()
#    TREE_OF_THOUGHTS = auto()
#    GRAPH_OF_THOUGHTS = auto()
#    LATS = auto()
#    ROUTER = auto()


class EngineEvent(Enum):
    # Initialization & Task Lifecycle
    ON_INITIALIZE = "OnInitialize"
    ON_TASK_START = "OnTaskStart"
    ON_TASK_COMPLETE = "OnTaskComplete"
    ON_TASK_ABORT = "OnTaskAbort"
    
    # Step Loop Execution
    BEFORE_STEP_EXECUTION = "BeforeStepExecution"
    AFTER_STEP_EXECUTION = "AfterStepExecution"
    
    # Suspension & Resumption
    ON_HALT = "OnHalt"
    ON_RESUME = "OnResume"
    
    # Model & Context Hooks
    BEFORE_PROMPT_COMPILE = "BeforePromptCompile"
    ON_LLM_RESPONSE = "OnLlmResponse"
    
    # Error Handling & Retries
    ON_TOOL_ERROR = "OnToolError"
    ON_PARSE_ERROR = "OnParseError"
    ON_RETRY = "OnRetry"
    
    # Agent Switching & Delegation
    ON_AGENT_SWITCH = "OnAgentSwitch"
    BEFORE_DELEGATION = "BeforeDelegation"
    AFTER_DELEGATION = "AfterDelegation"
    
    # Engine Teardown
    ON_CLEANUP = "OnCleanup"

ControlFlowHandler = Callable[["CoreEngine", str], None]

class CoreEngine:
    """
    Central orchestration engine that connects instructions, prompt builders,
    trace parsing, and agent execution frames.
    """
    def __init__(
        self,
        instructionSet: InstructionSet,
        systemInstructions: SystemInstructions,
        traceParser: TraceParser,
        gateway: Any = None,  # GenAIGateway instance
        defaultAgent: Optional[AgentProfile] = None,
    ):
        self.instructionSet = instructionSet
        self.systemInstructions = systemInstructions
        self.traceParser = traceParser
        self.gateway = gateway
        
        # Runtime State
        self.activeAgent: Optional[AgentProfile] = defaultAgent
        self.chainOfThought: Optional[ChainOfThought] = None
        self.currentTask: Optional[str] = None
        
        # Execution Flags
        self._isCompleted: bool = False
        self._isHalted: bool = False
        self._stepCounter: int = 0
        
        # Decorator Callbacks Registry
        self._hooks: Dict[EngineEvent, List[Callable[["CoreEngine"], Any]]] = {
            event: [] for event in EngineEvent
        }

        # Custom Control Flow Handlers Registry
        # Maps tag names (e.g., "INPUT", "REFINE") to user callbacks
        self._customControlFlowHandlers: Dict[str, ControlFlowHandler] = {}
        # Optional fallback handler for unrecognized signals
        self._fallbackControlFlowHandler: Optional[ControlFlowHandler] = None

        self._TriggerHooks(EngineEvent.ON_INITIALIZE)

    # --- DECORATOR HOOK REGISTRATION ---
    def OnInitialize(self, fn: Callable[["CoreEngine"], Any]) -> Callable:
        self._hooks[EngineEvent.ON_INITIALIZE].append(fn)
        return fn

    def OnTaskStart(self, fn: Callable[["CoreEngine"], Any]) -> Callable:
        self._hooks[EngineEvent.ON_TASK_START].append(fn)
        return fn

    def BeforeStepExecution(self, fn: Callable[["CoreEngine"], Any]) -> Callable:
        self._hooks[EngineEvent.BEFORE_STEP_EXECUTION].append(fn)
        return fn

    def AfterStepExecution(self, fn: Callable[["CoreEngine"], Any]) -> Callable:
        self._hooks[EngineEvent.AFTER_STEP_EXECUTION].append(fn)
        return fn

    def OnTaskComplete(self, fn: Callable[["CoreEngine"], Any]) -> Callable:
        self._hooks[EngineEvent.ON_TASK_COMPLETE].append(fn)
        return fn

    def OnHalt(self, fn: Callable[["CoreEngine"], Any]) -> Callable:
        self._hooks[EngineEvent.ON_HALT].append(fn)
        return fn

    def OnResume(self, fn: Callable[["CoreEngine"], Any]) -> Callable:
        self._hooks[EngineEvent.ON_RESUME].append(fn)
        return fn

    def OnTaskAbort(self, fn: Callable[["CoreEngine"], Any]) -> Callable:
        self._hooks[EngineEvent.ON_TASK_ABORT].append(fn)
        return fn

    def BeforePromptCompile(self, fn: Callable[["CoreEngine"], Any]) -> Callable:
        self._hooks[EngineEvent.BEFORE_PROMPT_COMPILE].append(fn)
        return fn

    def OnLlmResponse(self, fn: Callable[["CoreEngine"], Any]) -> Callable:
        self._hooks[EngineEvent.ON_LLM_RESPONSE].append(fn)
        return fn

    def OnToolError(self, fn: Callable[["CoreEngine"], Any]) -> Callable:
        self._hooks[EngineEvent.ON_TOOL_ERROR].append(fn)
        return fn

    def OnParseError(self, fn: Callable[["CoreEngine"], Any]) -> Callable:
        self._hooks[EngineEvent.ON_PARSE_ERROR].append(fn)
        return fn

    def OnRetry(self, fn: Callable[["CoreEngine"], Any]) -> Callable:
        self._hooks[EngineEvent.ON_RETRY].append(fn)
        return fn

    def OnAgentSwitch(self, fn: Callable[["CoreEngine"], Any]) -> Callable:
        self._hooks[EngineEvent.ON_AGENT_SWITCH].append(fn)
        return fn

    def BeforeDelegation(self, fn: Callable[["CoreEngine"], Any]) -> Callable:
        self._hooks[EngineEvent.BEFORE_DELEGATION].append(fn)
        return fn

    def AfterDelegation(self, fn: Callable[["CoreEngine"], Any]) -> Callable:
        self._hooks[EngineEvent.AFTER_DELEGATION].append(fn)
        return fn

    def OnCleanup(self, fn: Callable[["CoreEngine"], Any]) -> Callable:
        self._hooks[EngineEvent.ON_CLEANUP].append(fn)
        return fn
    
    def _TriggerHooks(self, event: EngineEvent) -> None:
        """Executes all registered sync/async callbacks for a given lifecycle event."""
        for hook in self._hooks[event]:
            if inspect.iscoroutinefunction(hook):
                asyncio.create_task(hook(self))
            else:
                hook(self)

    # --- ENGINE LIFECYCLE API ---
    def BuildSystemPrompt(self) -> str:
        """Combines system instructions and current agent manifest into a prompt."""
        manifestSection = f"# AGENT MANIFEST ({self.activeAgent.name})\n{self.activeAgent.manifest}"
        taskSection = f"# CURRENT TASK\n{self.currentTask}"
        renderedInstructions = self.systemInstructions.Render()
        
        return f"{manifestSection}\n\n{taskSection}\n\n{renderedInstructions}"

    def SetActiveAgent(self, agent: AgentProfile) -> None:
        """Loads an active agent profile, task, and initializes a new ChainOfThought."""
        self.activeAgent = agent

        self._TriggerHooks(EngineEvent.ON_TASK_START)

    def StartTask(self, task: str, agent: AgentProfile = None) -> None:
        self.activeAgent = agent or self.activeAgent
        self.currentTask = task
        self.chainOfThought = ChainOfThought(rootAgentName=self.activeAgent.name)
        self._isCompleted = False
        self._isHalted = False
        self._stepCounter = 0

        self._TriggerHooks(EngineEvent.ON_TASK_START)

    @property
    def HasNextStep(self) -> bool:
        """Evaluates whether the loop should continue executing steps."""
        return not self._isCompleted and not self._isHalted

    async def RunStep(self) -> Optional[ParsedCycleTrace]:
        """Executes a single iteration cycle of the ReAct/Engine loop."""
        if not self.HasNextStep:
            return None

        self._stepCounter += 1
        self._TriggerHooks(EngineEvent.BEFORE_STEP_EXECUTION)

        # 1. Compile System Prompt & Context History
        systemPrompt = self.BuildSystemPrompt()
        historyText = self.chainOfThought.RenderHistory(currentFrameOnly=True)
        
        fullPrompt = f"{systemPrompt}\n\n# EXECUTION HISTORY\n{historyText}"

        # 2. Invoke Gateway / Model LLM
        # (Mocked gateway call returning LLM completion text)
        rawLlmResponse = await self._InvokeModel(fullPrompt)

        # 3. Parse Response Trace
        parsedTrace = self.traceParser.ParseTrace(rawLlmResponse)

        # 4. Record Thought Step
        if parsedTrace.thought:
            self.chainOfThought.AddStep(StepType.THOUGHT, parsedTrace.thought)

        # 5. Dispatch Action Batches if actions were extracted
        if parsedTrace.action_block:
            batches = self.instructionSet.ParseResponse(parsedTrace.action_block)
            for batch in batches:
                results = await self.instructionSet.DispatchBatch(batch, self.activeAgent)
                self.chainOfThought.AddStep(StepType.OBSERVATION, results)

        # 6. Process Control Flow Signals
        if parsedTrace.control_flow:
            self._ProcessControlFlow(parsedTrace.control_flow)

        self._TriggerHooks(EngineEvent.AFTER_STEP_EXECUTION)
        
        if self._isCompleted:
            self._TriggerHooks(EngineEvent.ON_TASK_COMPLETE)

        return parsedTrace

    def Halt(self, reason: str = "Waiting for external input") -> None:
        """Pauses engine execution until Resume() is triggered."""
        self._isHalted = True
        self.chainOfThought.AddStep(StepType.STATE_TRANSITION, f"HALTED: {reason}")
        self._TriggerHooks(EngineEvent.ON_HALT)

    def Resume(self, inputData: Optional[Any] = None) -> None:
        """Resumes execution after a halt event."""
        if not self._isHalted:
            return
        
        self._isHalted = False
        if inputData:
            self.chainOfThought.AddStep(StepType.OBSERVATION, f"RESUMED with input: {inputData}")
        
        self._TriggerHooks(EngineEvent.ON_RESUME)

    # --- DECORATOR REGISTRATION API ---
    def RegisterControlFlow(self, signalTag: str) -> Callable[[ControlFlowHandler], ControlFlowHandler]:
        """
        Decorator to register a custom state handler for a specific control flow signal.
        Example: @engine.RegisterControlFlow("INPUT")
        """
        tagUpper = signalTag.strip().upper()
        
        # Guard against overriding mandatory system flags
        if tagUpper in ("COMPLETE", "ABORT"):
            raise ValueError(f"Signal '[{tagUpper}]' is a mandatory system signal and cannot be overridden.")

        def Decorator(fn: ControlFlowHandler) -> ControlFlowHandler:
            self._customControlFlowHandlers[tagUpper] = fn
            return fn
        return Decorator

    def FallbackControlFlow(self, fn: ControlFlowHandler) -> ControlFlowHandler:
        """
        Decorator for handling signals that are neither mandatory nor registered.
        """
        self._fallbackControlFlowHandler = fn
        return fn

    # --- CONTROL FLOW PROCESSOR ---
    def _ProcessControlFlow(self, signal: str) -> None:
        """
        Evaluates state transitions. 
        Mandatory signals (COMPLETE, ABORT) are hardcoded.
        Custom signals are dispatched to registered handlers.
        """
        if not signal:
            return

        cleanSignal = signal.strip().upper()

        # 1. MANDATORY FRAMEWORK SIGNALS (Cannot be overridden)
        if "[COMPLETE]" in cleanSignal:
            self._isCompleted = True
            self.chainOfThought.AddStep(StepType.STATE_TRANSITION, "COMPLETED")
            return

        if "[ABORT]" in cleanSignal:
            self._isCompleted = True
            self.chainOfThought.AddStep(StepType.STATE_TRANSITION, "ABORTED")
            return

        # 2. CUSTOM REGISTERED SIGNALS
        # Extract signal token inside brackets, e.g. [INPUT] -> INPUT
        match = re.search(r"\[([A-Z0-9_]+)\]", cleanSignal)
        if match:
            extractedTag = match.group(1)
            if extractedTag in self._customControlFlowHandlers:
                handler = self._customControlFlowHandlers[extractedTag]
                handler(self, cleanSignal)
                return

        # 3. FALLBACK HANDLER (If signal is unknown)
        if self._fallbackControlFlowHandler:
            self._fallbackControlFlowHandler(self, cleanSignal)
            return

        # Default fallback behavior if no handler matches
        self.chainOfThought.AddStep(
            StepType.STATE_TRANSITION, 
            f"UNHANDLED_SIGNAL: {cleanSignal}"
        )

    async def _InvokeModel(self, prompt: str) -> str:
        """Internal model call helper."""
        if self.gateway:
            return await self.gateway.GenerateText(prompt)
        
        # Mock LLM return for demonstration
        return """
        <thought>I need to execute a tool to fetch information.</thought>
        <action_or_output><action:CALL>FetchData(id=123)</action:CALL></action_or_output>
        <control_flow>[REFINE]</control_flow>
        """