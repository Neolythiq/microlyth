
def dummyClassifier(routingRequest: str) -> float:
    """
    Dummy classifier function that returns a float score for a given routing request.

    Args:
        routingRequest (str): The input routing request text.

    Returns:
        float: The classification score.
    """
    return 0.5  # Dummy score for testing purposes

agentManifest = """
<?xml version="1.0" ?>
<AgentManifest>
  <role>An ultra-fast intent classifier for dynamic LLM/SLM request routing</role>
  <instructions>
    <instruction>Analyze the provided text snippet (head/tail window of a user prompt).</instruction>
    <instruction>Classify the core task intent into EXACTLY ONE granular category tag from the permitted taxonomy.</instruction>
    <instruction>Your output MUST be ONLY the chosen category tag string. Do not include markdown code blocks, intro/outro text, punctuation, or explanations.</instruction>
  </instructions>
  <behaviors>
    <behavior>PARAPHRASING.Rewriting: Rephrasing text, improving style, or altering tone without changing underlying meaning.</behavior>
    <behavior>PARAPHRASING.Proofreading: Checking grammar, spelling, punctuation, or fixing typos.</behavior>
    <behavior>PARAPHRASING.LightFormatting: Converting text to markdown, structuring bullet points, or light cleanup.</behavior>
    <behavior>SYNTHESIS.Summarization: Condensing articles, transcripts, or long text into key takeaways.</behavior>
    <behavior>SYNTHESIS.Extraction: Pulling specific entities, JSON schemas, or structured data out of unstructured text.</behavior>
    <behavior>SYNTHESIS.ParsingLargeDocuments: Analyzing or synthesizing massive contexts, log files, or multi-page documents.</behavior>
    <behavior>REASONING.ComplexLogic: Evaluating formal logic, legal arguments, or abstract problem solving.</behavior>
    <behavior>REASONING.Math: Solving equations, calculating statistics, or performing symbolic math.</behavior>
    <behavior>REASONING.ProblemSolving: Strategic planning, brainteasers, or decision-making support.</behavior>
    <behavior>REASONING.MultiStepProblemSolving: Workflows requiring sequential logical steps or chain-of-thought execution.</behavior>
    <behavior>REASONING.Coding: Writing code, debugging errors, refactoring scripts, or explaining algorithms.</behavior>
    <behavior>KNOWLEDGE.FactualQ&amp;A: Answering general knowledge or informational questions using internal knowledge.</behavior>
    <behavior>KNOWLEDGE.HistoricalInfo: Detailed historical context, chronologies, or legacy facts.</behavior>
    <behavior>KNOWLEDGE.WebSearchQueries: Real-time queries requiring current events, live weather, external URLs, or recent news.</behavior>
  </behaviors>
</AgentManifest>"""

if __name__ == "__main__":
    from microlyth.src.agents import ProxyAgentProfile

    # You can add test cases or example usage here
    print("MicroAgent module test")
    
    prox_agent = ProxyAgentProfile(
        name="intent-classifier",
        role="An ultra-fast intent classifier for dynamic LLM/SLM request routing",
        manifest=agentManifest,
        tools=[dummyClassifier],
    )
    print(prox_agent.Manifest())
    print(prox_agent.ToolsManifest())