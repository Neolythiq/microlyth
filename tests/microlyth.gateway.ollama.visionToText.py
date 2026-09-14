from microlyth.src.gateway import OllamaTextGenClient, OllamaVisionClient

task = """
You are a precise data analyst tasked with providing a factual, objective description of the provided chart. 
Your goal is to convert the visual information into a clear verbal summary without introducing speculation, external domain knowledge, or premature conclusions.
Execution Instructions

1. Extract Metadata & Labels
- Chart Type: Identify the visual format (e.g., Bar Chart, Stacked Bar Chart, Waterfall Chart, Time-Series Line Graph, Scatter Plot).
- Title & Subtitle: Transcribe exact wording.
- Axes & Units: State the X-axis label, Y-axis label, scale unit (e.g., USD in millions, percentages, date ranges), and any secondary axes.

2. Verbal Data Description (Step-by-Step)
- Describe what the data is showing sequentially (e.g., chronologically along a time series, or from highest to lowest in a bar chart).
- For Waterfall Charts: Explicitly state the starting baseline value, itemize each positive (upward) and negative (downward) delta step-by-step with its exact or approximate numerical value, and state the final ending baseline.
- For Time-Series Graphs: Describe movement phase by phase (e.g., "From [Date A] to [Date B], the value increased from [X] to [Y], reached a peak of [Z] at [Date C], and subsequently declined to [W]").
- State exact numbers, percentages, or data points where explicitly visible. If values must be estimated from the gridlines, explicitly state that they are visual approximations.

3. Strict Boundaries (What NOT to do)
- NO Causal Speculation: Do not explain why a change happened unless the reason is explicitly written inside a annotation box on the chart image itself.
- NO Business Conclusions: Avoid phrases like "strong performance," "concerning drop," "healthy growth," or "failed target." Describe only the raw directional movement and values.
- NO Extrapolation: Do not predict future trends or extrapolate missing data points.
"""

if __name__ == "__main__":
    client = OllamaVisionClient()

    response = client.AnalyzeImage(r"C:\\Users\\jerem\\Downloads\\LinkedIn-revenues.jpeg", task, systemPrompt="You are a helpful assistant.")
    print(response)