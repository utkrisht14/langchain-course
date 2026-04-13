from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import requests

load_dotenv()


@tool
def get_weather(city: str) -> str:
    """
    Tool that gets real weather for a city.

    Args:
        city: The city name

    Returns:
        Current weather information
    """
    print(f"Fetching weather for {city}")

    try:
        response = requests.get(f"https://wttr.in/{city}?format=3", timeout=10)
        response.raise_for_status()
        return response.text.strip()
    except Exception as e:
        return f"Error fetching weather: {str(e)}"


llm = ChatOpenAI(model="gpt-4.1-mini")
tools = [get_weather]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from lang-chain course.")
    result = agent.invoke(
        {"messages": [HumanMessage(content="What is the weather in Amsterdam?")]}
    )
    print(result)


if __name__ == "__main__":
    main()