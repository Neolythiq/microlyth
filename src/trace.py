from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union

class StepType(Enum):
    THOUGHT = "thought"           # Internal reasoning / reflection
    ACTION = "action"             # Executed action item(s) / tool call(s)
    OBSERVATION = "observation"   # Tool output / environmental feedback
    PLAN = "plan"                 # Strategic multi-step plan / agenda
    DELEGATION = "delegation"     # Handoff or sub-agent call
    OUTPUT = "output"             # Response / data payload produced for the user
    STATE_TRANSITION = "status"   # Engine control signal (COMPLETE, ABORT, INPUT, etc.)

@dataclass
class ThoughtStep:
    """
    Represents an atomic entry within a single cycle or multi-step execution loop.
    """
    stepType: StepType
    content: Union[str, Dict[str, Any], List[Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def RenderXml(self) -> str:
        """Formats the step into structured XML tag notation."""
        tagName = self.stepType.value
        if isinstance(self.content, (dict, list)):
            import json
            renderedContent = json.dumps(self.content, indent=2)
        else:
            renderedContent = str(self.content)
            
        return f"<{tagName}>\n{renderedContent}\n</{tagName}>"

    def __str__(self) -> str:
        return self.RenderXml()


@dataclass
class DelegationFrame:
    """
    Represents an active agent execution context in a delegation stack trace.
    """
    frameId: str
    agentName: str
    callerAgentName: str
    taskDescription: str
    steps: List[ThoughtStep] = field(default_factory=list)
    parentFrameId: Optional[str] = None
    status: str = "ACTIVE"  # "ACTIVE", "COMPLETED", "FAILED"
    result: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def AddStep(self, step: ThoughtStep) -> None:
        self.steps.append(step)

class ChainOfThought:
    """
    Manages cycle history, multi-step thoughts/actions, and hierarchical agent delegation stacks.
    """
    def __init__(self, rootAgentName: str = "RootAgent"):
        self.rootAgentName = rootAgentName
        
        # Initialize root delegation frame
        rootFrame = DelegationFrame(
            frameId="frame_root",
            agentName=rootAgentName,
            callerAgentName="User",
            taskDescription="Root Execution Loop"
        )
        self._frames: Dict[str, DelegationFrame] = {rootFrame.frameId: rootFrame}
        self._stack: List[str] = [rootFrame.frameId]  # Stack tracking active frames

    @property
    def CurrentFrame(self) -> DelegationFrame:
        """Returns the active delegation stack frame."""
        return self._frames[self._stack[-1]]

    # --- Step Management ---

    def AddStep(self, stepType: StepType, content: Any, metadata: Optional[Dict[str, Any]] = None) -> ThoughtStep:
        """Appends an atomic step to the current active agent frame."""
        step = ThoughtStep(
            stepType=stepType,
            content=content,
            metadata=metadata or {}
        )
        self.CurrentFrame.AddStep(step)
        return step

    def RecordCycle(
        self, 
        thought: Optional[str] = None, 
        actions: Optional[Any] = None, 
        observations: Optional[Any] = None, 
        status: Optional[str] = None
    ) -> List[ThoughtStep]:
        """Convenience method to record a complete multi-step ReAct cycle."""
        recordedSteps = []
        if thought:
            recordedSteps.append(self.AddStep(StepType.THOUGHT, thought))
        if actions:
            recordedSteps.append(self.AddStep(StepType.ACTION, actions))
        if observations:
            recordedSteps.append(self.AddStep(StepType.OBSERVATION, observations))
        if status:
            recordedSteps.append(self.AddStep(StepType.STATE_TRANSITION, status))
        return recordedSteps

    # --- Agent Delegation Stack Mechanics ---

    def PushDelegation(self, subAgentName: str, subTask: str, frameId: Optional[str] = None) -> DelegationFrame:
        """Pushes a new sub-agent onto the delegation stack trace."""
        parentFrame = self.CurrentFrame
        newFrameId = frameId or f"frame_{len(self._frames) + 1}_{subAgentName}"

        newFrame = DelegationFrame(
            frameId=newFrameId,
            agentName=subAgentName,
            callerAgentName=parentFrame.agentName,
            taskDescription=subTask,
            parentFrameId=parentFrame.frameId
        )
        
        self._frames[newFrameId] = newFrame
        self._stack.append(newFrameId)
        
        # Record delegation entry step in parent frame
        self.AddStep(
            StepType.DELEGATION,
            f"Delegated task to [{subAgentName}]: {subTask}",
            metadata={"childFrameId": newFrameId}
        )
        return newFrame

    def PopDelegation(self, result: Any, status: str = "COMPLETED") -> DelegationFrame:
        """Pops the active delegation frame and returns execution control to caller."""
        if len(self._stack) <= 1:
            raise RuntimeError("Cannot pop root delegation frame from ChainOfThought.")

        completedFrame = self.CurrentFrame
        completedFrame.status = status
        completedFrame.result = result
        self._stack.pop()

        # Record observation in parent frame upon return
        self.AddStep(
            StepType.OBSERVATION,
            f"Result from agent [{completedFrame.agentName}]: {result}",
            metadata={"sourceFrameId": completedFrame.frameId}
        )
        return completedFrame

    def AddOutput(self, content: Any, metadata: Optional[Dict[str, Any]] = None) -> ThoughtStep:
        """Records an explicit output response for the user."""
        return self.AddStep(StepType.OUTPUT, content, metadata)

    def GetLatestOutput(self) -> Optional[Any]:
        """Retrieves the most recent output step from the active frame."""
        for step in reversed(self.CurrentFrame.steps):
            if step.stepType == StepType.OUTPUT:
                return step.content
        return None

    # --- Formatting and Stack Trace Inspection ---

    def GetStackTrace(self) -> str:
        """Renders the current hierarchical stack trace across delegated agents."""
        lines = ["=== Agent Delegation Stack Trace ==="]
        for depth, frameId in enumerate(self._stack):
            frame = self._frames[frameId]
            indent = "  " * depth
            lines.append(f"{indent}└─ [{frame.agentName}] Task: '{frame.taskDescription}' (Status: {frame.status})")
        return "\n".join(lines)

    def RenderHistory(self, currentFrameOnly: bool = False) -> str:
        """Renders step history as structured XML string for system prompts or logging."""
        framesToRender = [self.CurrentFrame] if currentFrameOnly else [self._frames[fid] for fid in self._stack]
        
        output = []
        for frame in framesToRender:
            output.append(f"<!-- AGENT CONTEXT: {frame.agentName} | TASK: {frame.taskDescription} -->")
            for step in frame.steps:
                output.append(step.RenderXml())
                
        return "\n\n".join(output)

    def __str__(self) -> str:
        return self.RenderHistory()

import re
from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass

# --- 1. RESPONSE FORMAT SPECIFICATION ---

@dataclass
class ResponseFormatSpec:
    """Defines the response structure rules and tags to inject into system prompt."""
    format_name: str
    instruction_text: str
    required_tags: List[str]

class ResponseFormatRegistry:
    def __init__(self):
        self._current_format: Optional[ResponseFormatSpec] = None

    def SetFormat(self, name: str, instruction_text: str, required_tags: List[str]) -> None:
        """Sets the active response formatting rule for the engine loop."""
        self._current_format = ResponseFormatSpec(
            format_name=name,
            instruction_text=instruction_text,
            required_tags=required_tags
        )

    def GetFormatInstructions(self) -> str:
        """Callable prompt provider to be passed into SystemInstructions/SystemCmd."""
        if not self._current_format:
            return ""
        
        tags_str = ", ".join([f"<{tag}>" for tag in self._current_format.required_tags])
        return (
            f"## Response Format Guidelines ({self._current_format.format_name})\n"
            f"{self._current_format.instruction_text.strip()}\n"
            f"Mandatory Tags: {tags_str}"
        )


# --- 2. DECORATOR-DRIVEN TRACE PARSER ---

@dataclass
class ParsedCycleTrace:
    """Structured container holding extracted components of a single loop iteration."""
    thought: Optional[str] = None
    action_block: Optional[str] = None
    control_flow: Optional[str] = None
    custom_elements: Dict[str, Any] = None


class TraceParser:
    def __init__(self):
        self._parser_fn: Optional[Callable[[str], ParsedCycleTrace]] = None

    def RegisterParser(self, fn: Callable[[str], ParsedCycleTrace]) -> Callable:
        """Decorator allowing users to define custom regex or AST parsing logic."""
        self._parser_fn = fn
        return fn

    def ParseTrace(self, raw_response: str) -> ParsedCycleTrace:
        """Executes the user-defined parser or falls back to standard XML extraction."""
        if self._parser_fn:
            return self._parser_fn(raw_response)
        return self._DefaultXmlParser(raw_response)

    def _DefaultXmlParser(self, raw_response: str) -> ParsedCycleTrace:
        """Fallback parser that extracts <thought>, <action_or_output>, and <control_flow> tags."""
        def extract_tag(tag: str, text: str) -> Optional[str]:
            match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else None

        return ParsedCycleTrace(
            thought=extract_tag("thought", raw_response),
            action_block=extract_tag("action_or_output", raw_response),
            control_flow=extract_tag("control_flow", raw_response)
        )