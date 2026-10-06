from microlyth.src.engine import EngineFactory
from microlyth.src.agents import AgentProfile, ProxyAgentProfile

from microlyth.src.gateway import OllamaTextGenClient

import time

class RealTimer:
    def __init__(self) -> None:
        self._startTime: float = 0.0

    def Start(self) -> None:
        """Initializes or resets the timer."""
        self._startTime = time.perf_counter()

    def ElapsedSeconds(self) -> float:
        """Calculates time passed since Start() was called."""
        if self._startTime == 0.0:
            return 0.0
        return time.perf_counter() - self._startTime

    def Reset(self) -> None:
        """Resets the timer to zero."""
        self._startTime = 0.0

increase_temperature = 0.5
current_temperature = 17.5
isHeaterOn = False
timer = RealTimer()
timer.Start()

print(f"Elapsed time: {timer.ElapsedSeconds():.4f} seconds")

def check_temprature():
    """
    Checks the temperature in the home.
    check_temprature()
    Returns:
        str: The current temperature in Celsius.
    """
    global current_temperature
    global increase_temperature
    global isHeaterOn
    global timer
    elapsed_time = timer.ElapsedSeconds()
    if isHeaterOn and elapsed_time >= 20.0:
        current_temperature += increase_temperature
        timer.Reset()
    return current_temperature

def check_humidity():
    """
    Checks the humidity in the home.
    function:check_humidity()
    Returns:
        str: The current humidity level.
    """
    return "60%"

def turn_on_heater():
    """
    Turns on the heater.
    function: turn_on_heater()
    Returns:
        str: Confirmation that the heater is turned on.
    """
    global isHeaterOn
    isHeaterOn = True
    return "Heater turned on."

def turn_off_heater():
    """
    Turns off the heater.
    function: turn_off_heater()
    Returns:
        str: Confirmation that the heater is turned off.
    """
    global isHeaterOn
    isHeaterOn = False
    return "Heater turned off."

def wait_for_temperature_treshold(threshold: int = 20):
    """
    Waits for the temperature to reach a certain threshold.
    function: wait_for_temperature_treshold(threshold)
    Args:
        threshold (int): The temperature threshold to wait for.
    Returns:
        str: Confirmation that the temperature threshold has been reached.
    """
    return f"Current temperature is below threshold. Waiting for it to rise above {threshold}°C. An event will be raised when the temperature reaches the desired level."

def IdleDuringNMinutes(minutes: int):
    """
    Simulates the agent being idle for a specified number of minutes.
    function: IdleDuringNMinutes(minutes)
    Args:
        minutes (int): The number of minutes to remain idle.
    Returns:
        str: Confirmation that the agent has been idle for the specified duration.
    """
    return f"Agent has been idle for {minutes} minutes."

def CreateReactEngine():
    client = OllamaTextGenClient()
    reactEngine = EngineFactory.ReActLoop(gateway=client)

    #agent = ProxyAgentProfile(name="HomeAssistantAgent", role="Home assistant", manifest="Specialized in providing assistance.")
    homeAgent = AgentProfile(
        name="HomeAssistantAgent",
        role="Home assistant",
        instructions=["Check temperature in home", "Turn on heater if below 20°C", "Turn off heater if above 22°C"],
        behaviors=["Be helpfull", "Be reactive", "Be polite"],
        tools=[check_temprature, check_humidity, turn_on_heater, turn_off_heater, IdleDuringNMinutes, wait_for_temperature_treshold]
    )

    reactEngine.SetActiveAgent(agent=homeAgent)

    return reactEngine

if __name__ == "__main__":
    reactEngine = CreateReactEngine()
    reactEngine.StartTask(task="Check temperature in home and act accordingly to maintain comfort.")

    prompt = reactEngine.BuildSystemPrompt()
    print(prompt)

    import asyncio
    while reactEngine.HasNextStep:
        trace = asyncio.run(reactEngine.RunStep())
        print(f"     Step Output Thought: {trace.thought if trace else None}")