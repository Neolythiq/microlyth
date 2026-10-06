
import asyncio

from microlyth.src.engine import EngineFactory
from microlyth.src.agents import ProxyAgentProfile

def Main():
    reactEngine = EngineFactory.ReActLoop()

    agent = ProxyAgentProfile(name="HomeAssistantAgent", role="Home assistant", manifest="Specialized in providing assistance.")
    reactEngine.SetActiveAgent(agent=agent)
    #reactEngine.StartTask(task="Find information on Microlyth framework.")

    prompt = reactEngine.BuildSystemPrompt()
    print(prompt)

if __name__ == "__main__":
    Main()