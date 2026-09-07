
def CalculateDiscount(price: float, percentage: float) -> float:
    """Calculates the final price after discount.
    
    Args:
        price (float): The original price.
        percentage (float): The discount percentage.

    Returns:
        float: The final price after discount.
    """
    return price * (1 - percentage / 100)

if __name__ == "__main__":
    from microlyth.src.worker import MicroAgent

    # You can add test cases or example usage here
    print("MicroAgent module test")
    
    micro_agent = MicroAgent(
        name="TestMicroAgent",
        role="Tester",
        instructions=["Do something", "Do something else"],
        behaviors=["Be proactive", "Be reactive"],
        tools=[CalculateDiscount]
    )
    print(micro_agent.Manifest())
    print(micro_agent.ToolsManifest())