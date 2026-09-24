from microlyth.src.trace import TraceParser
from microlyth.src.trace import ParsedCycleTrace

import re

traceParser = TraceParser()

if __name__ == "__main__":
    # Custom user-defined trace parser registered with a decorator
    @traceParser.RegisterParser
    def CustomReActTraceParser(raw_response: str) -> ParsedCycleTrace:
        # Custom regex or parsing pipeline defined by the user
        thought_match = re.search(r"<thought>(.*?)</thought>", raw_response, re.DOTALL)
        action_match = re.search(r"<action_or_output>(.*?)</action_or_output>", raw_response, re.DOTALL)
        flow_match = re.search(r"<control_flow>(.*?)</control_flow>", raw_response, re.DOTALL)

        return ParsedCycleTrace(
            thought=thought_match.group(1).strip() if thought_match else None,
            action_block=action_match.group(1).strip() if action_match else None,
            control_flow=flow_match.group(1).strip() if flow_match else None
        )

    # Execution inside engine loop
    raw_llm_output = """
    <thought>I should search for weather data.</thought>
    <action_or_output><action:CALL>GetWeather(city="Paris")</action:CALL></action_or_output>
    <control_flow>[REFINE]</control_flow>
    """

    parsed_trace = traceParser.ParseTrace(raw_llm_output)
    print(f"Parsed Thought: {parsed_trace.thought}")
    print(f"Parsed Actions: {parsed_trace.action_block}")
    print(f"Parsed Control Flow: {parsed_trace.control_flow}")