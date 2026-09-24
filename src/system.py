import asyncio

from typing import Callable, Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
 

from enum import Enum
import json
import re

 

@dataclass

class ActionSpec:
    name: str
    description: str
    callback: Callable[..., Any]
    openingToken: str  # Formatted as opening_block_tag + name + closing_tag
    closingToken: str  # Formatted as end_block_tag + name + closing_tag
    example: Optional[str] = None

class ExecMode(Enum):
    PARALLEL = "parallel"  # Run concurrently using asyncio.gather
    SEQUENTIAL = "sequential"  # Run step-by-step
    PIPELINE = "pipeline"  # Step N receives output of Step N-1

@dataclass
class ActionItem:
    actionType: str          # e.g., "CALL", "SWITCH", "ABORT"
    rawPayload: str          # Raw string between opening and closing tags
    args: Any                # Parsed representation (dict, list, str, etc.)
    actionId: Optional[str] = None
    dependsOn: Optional[str] = None

@dataclass
class ActionSpec:
    name: str
    description: str
    callback: Callable[..., Any]
    openingToken: str
    closingToken: str
    payloadParser: Optional[Callable[[str], Any]] = None
    example: Optional[str] = None

@dataclass
class ActionBatch:
    mode: ExecMode
    actions: List[ActionItem] = field(default_factory=list)

class InstructionSet:
    def __init__(self, openBlockTag: str = "<action:",  endBlockTag="</action:", closeTag: str = ">"):
        self.openBlockTag = openBlockTag
        self.endBlockTag = endBlockTag
        self.closeTag = closeTag
        self._instructions: Dict[str, ActionSpec] = {}
        self._custom_prompt_formatter: Optional[Callable[[Dict[str, ActionSpec]], str]] = None

    def NewInstruction(
            self,
            name: str,
            description: str,
            example: Optional[str] = None,
            payloadParser: Optional[Callable[[str], Any]] = None,
            ) -> Callable:
        """
        Decorator to register an instruction, its system prompt description,
        and its engine loop callback.
        """
        # Default payload parser converts JSON strings to dicts
        defaultParser = payloadParser or (lambda raw: json.loads(raw) if raw.startswith("{") else raw)

        def Decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            # Wrap the user callback to parse payload before execution
            async def ExecWrapper(payload: str, agentContext: Any) -> Any:
                try:
                    parsedPayload = defaultParser(payload)
                except Exception as e:
                    raise ValueError(f"Failed to parse payload for {name}: {e}")

                if asyncio.iscoroutinefunction(fn):
                    return await fn(parsedPayload, agentContext)
                return fn(parsedPayload, agentContext)

            self._instructions[name] = ActionSpec(
                name=name,
                description=description,
                callback=ExecWrapper,  # Wrapped callback handles parsing transparently
                openingToken=f"{self.openBlockTag}{name}{self.closeTag}",
                closingToken=f"{self.endBlockTag}{name}{self.closeTag}",
                payloadParser=defaultParser,
                example=example
            )
            return fn
        return Decorator

    def SystemPromptFormatter(self, fn: Callable[[Dict[str, ActionSpec]], str]) -> Callable:
        """Decorator to override default system prompt generation."""
        self._custom_prompt_formatter = fn
        return fn

    def BuildSystemPromptInstructions(self) -> str:
        if self._custom_prompt_formatter:
            return self._custom_prompt_formatter(self._instructions)

        return self._DefaultFormatter()

    def _DefaultFormatter(self) -> str:
        lines = ["## Available Actions"]
        for spec in self._instructions.values():
            lines.append(f"- Syntax: `{spec.openingToken} <payload> {spec.closingToken}` | Description: {spec.description}")

        lines.append("\n## Execution Mode Wrappers (Optional)")
        lines.append(f"- Parallel: `{self.openBlockTag}PARALLEL{self.closeTag} ... {self.endBlockTag}PARALLEL{self.closeTag}`")
        lines.append(f"- Pipeline: `{self.openBlockTag}PIPELINE{self.closeTag} ... {self.endBlockTag}PIPELINE{self.closeTag}`")
        lines.append(f"- Sequential: `{self.openBlockTag}SEQUENTIAL{self.closeTag} ... {self.endBlockTag}SEQUENTIAL{self.closeTag}` (Default)")

        lines.append("\n## Payload Formatting Rules & Examples")
        for spec in self._instructions.values():
            if spec.example:
                lines.append(f"### {spec.name}\n{spec.example}\n")

        return "\n".join(lines)

    # --- PARSING ENGINE ---
    def ParseResponse(self, rawText: str) -> List[ActionBatch]:
        """
        Scans LLM output for block mode wrappers (PARALLEL/PIPELINE/SEQUENTIAL).
        Extracts embedded action tags and defaults to ExecMode.SEQUENTIAL if unwrapped.
        """
        escapedOpen = re.escape(self.openBlockTag)
        escapedEnd = re.escape(self.endBlockTag)
        escapedClose = re.escape(self.closeTag)

        # Regex for block wrappers: <action:PARALLEL> ... </action:PARALLEL>
        wrapperPattern = re.compile(
            rf"{escapedOpen}(PARALLEL|PIPELINE|SEQUENTIAL){escapedClose}(.*?){escapedEnd}\1{escapedClose}",
            re.DOTALL | re.IGNORECASE
        )

        batches: List[ActionBatch] = []
        lastEndIndex = 0

        for match in wrapperPattern.finditer(rawText):
            # Parse any unwrapped actions appearing before this execution block as SEQUENTIAL
            precedingText = rawText[lastEndIndex:match.start()]
            standaloneActions = self._ParseActionsFromText(precedingText)
            if standaloneActions:
                batches.append(ActionBatch(mode=ExecMode.SEQUENTIAL, actions=standaloneActions))

            # Parse wrapped actions inside the block
            modeStr = match.group(1).upper()
            blockContent = match.group(2)
            mode = ExecMode[modeStr]

            blockActions = self._ParseActionsFromText(blockContent)
            if blockActions:
                batches.append(ActionBatch(mode=mode, actions=blockActions))

            lastEndIndex = match.end()

        # Parse trailing unwrapped content
        remainingText = rawText[lastEndIndex:]
        remainingActions = self._ParseActionsFromText(remainingText)

        if remainingActions:
            batches.append(ActionBatch(mode=ExecMode.SEQUENTIAL, actions=remainingActions))

        return batches

    def _ParseActionsFromText(self, text: str) -> List[ActionItem]:
        actions: List[ActionItem] = []
        escapedOpen = re.escape(self.openBlockTag)
        escapedEnd = re.escape(self.endBlockTag)
        escapedClose = re.escape(self.closeTag)

        registeredNames = "|".join(re.escape(name) for name in self._instructions.keys())
        if not registeredNames:
            return actions

        actionPattern = re.compile(
            rf"{escapedOpen}({registeredNames}){escapedClose}(.*?){escapedEnd}\1{escapedClose}",
            re.DOTALL
        )

        for match in actionPattern.finditer(text):
            actionType = match.group(1)
            rawPayload = match.group(2).strip()
            spec = self._instructions[actionType]

            # 1. Parse payload using instruction-specific parser or default fallback
            parsedArgs = rawPayload
            if spec.payloadParser:
                try:
                    parsedArgs = spec.payloadParser(rawPayload)
                except Exception as e:
                    # Keep raw payload on parsing failure or handle error strategy
                    parsedArgs = {"raw": rawPayload, "error": str(e)}

            # 2. Safely extract metadata if parsedArgs is a dictionary
            actionId = None
            dependsOn = None
            if isinstance(parsedArgs, dict):
                actionId = parsedArgs.get("id")
                dependsOn = parsedArgs.get("depends_on")

            # 3. Instantiate completely generic ActionItem
            actions.append(ActionItem(
                actionType=actionType,
                rawPayload=rawPayload,
                args=parsedArgs,
                actionId=actionId,
                dependsOn=dependsOn
            ))

        return actions

    # --- ASYNC DISPATCH ENGINE ---
    async def DispatchBatch(self, batch: ActionBatch, agentContext: Any) -> List[Any]:
        """Executes an ActionBatch according to its configured ExecMode."""
        if batch.mode == ExecMode.PARALLEL:
            tasks = [
                self._ExecuteSingleAction(action, agentContext)
                for action in batch.actions
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

        elif batch.mode == ExecMode.PIPELINE:
            results = []
            pipedInput = None
            for action in batch.actions:
                if pipedInput is not None:
                    action.args["piped_input"] = pipedInput
                result = await self._ExecuteSingleAction(action, agentContext)
                results.append(result)
                pipedInput = result

            return results

        else:  # ExecMode.SEQUENTIAL (Fallback)
            results = []
            for action in batch.actions:
                result = await self._ExecuteSingleAction(action, agentContext)
                results.append(result)
            return results

    async def _ExecuteSingleAction(self, action: ActionItem, agentContext: Any) -> Any:
        if action.actionType not in self._instructions:
            raise KeyError(f"Instruction '{action.actionType}' is not registered.")

        handler = self._instructions[action.actionType].callback

        if asyncio.iscoroutinefunction(handler):
            return await handler(payload=action.rawPayload, agentContext=agentContext)
        return handler(payload=action.rawPayload, agentContext=agentContext)

PromptComponent = Union[
    str,
    "SystemCmd",
    Callable[[], str],
    List[Any]
]

class SystemCmd:
    """
    A modular prompt primitive that wraps an instruction section.
    Supports dynamic activation via conditional flags or predicates.
    """
    def __init__(
        self,
        content: PromptComponent,
        title: Optional[str] = None,
        enabled: Union[bool, Callable[[], bool]] = True
    ):
        self.content = content
        self.title = title
        self.enabled = enabled

    def IsEnabled(self) -> bool:
        """Evaluates whether this command should be included in the final prompt."""
        if callable(self.enabled):
            return self.enabled()
        return bool(self.enabled)

    def Render(self) -> str:
        """Renders the instruction block, resolving callables and nested structures."""
        if not self.IsEnabled():
            return ""

        renderedText = ""

        # Handle different component types cleanly
        if callable(self.content):
            renderedText = str(self.content())
        elif isinstance(self.content, list):
            items = []
            for item in self.content:
                if isinstance(item, SystemCmd):
                    items.append(item.Render())
                else:
                    items.append(str(item))
            renderedText = "\n".join(filter(None, items))
        elif isinstance(self.content, SystemCmd):
            renderedText = self.content.Render()
        else:
            renderedText = str(self.content).strip()

        if not renderedText:
            return ""

        # Format optional title block
        if self.title:
            return f"# {self.title}\n{renderedText}"
        return renderedText

    def __str__(self) -> str:
        return self.Render()

class SystemInstructions:
    """
    A composable prompt builder that concatenates arbitrary instruction components.
    """
    def __init__(self, components: Optional[List[PromptComponent]] = None):
        self.components: List[PromptComponent] = components or []

    def Add(self, component: PromptComponent) -> "SystemInstructions":
        """Appends a new instruction, string, or SystemCmd block (fluent builder)."""
        self.components.append(component)
        return self

    def Extend(self, components: List[PromptComponent]) -> "SystemInstructions":
        """Extends the prompt pipeline with multiple instruction components."""
        self.components.extend(components)
        return self

    def Render(self) -> str:
        """Concatenates all active components into a single system prompt string."""
        sections = []
        for component in self.components:
            if isinstance(component, SystemCmd):
                text = component.Render()
            elif callable(component):
                text = str(component())
            elif isinstance(component, list):
                text = "\n".join(str(i) for i in component if i)
            else:
                text = str(component).strip()
            if text:
                sections.append(text)

        return "\n\n".join(sections)

    def __str__(self) -> str:
        return self.Render()


if __name__ == "__main__":
    # Initialize instruction set with custom syntax tags
    customInstructions = InstructionSet(openBlockTag="<action:", endBlockTag="</action:", closeTag=">")

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

    #@customInstructions.SystemPromptFormatter
    def CustomPromptFormatter(instruction_specs: Dict[str, Any]) -> str:
        output = ["Coin", "Coin"]
        return "\n".join(output)

    print(customInstructions.BuildSystemPromptInstructions())

    sample_llm_response = """
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
    batches = customInstructions.ParseResponse(sample_llm_response)

    for idx, batch in enumerate(batches, 1):
        print(f"\nBatch #{idx} | Mode: {batch.mode.value} | Action Count: {len(batch.actions)}")
        for action in batch.actions:
            print(f"  - Action: {action.actionType} | payload={action.rawPayload} | args={action.args}")

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

    # Create ReAct System Prompt with conditional tool support
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