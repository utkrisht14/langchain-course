from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()


@tool
def search(query: str) -> str:
    """
    Tool that searches over internet.

    Args:
        query: The query to search for.

    Returns:
        The search result.
    """
    print(f"Searching for {query}")
    return "Amsterdam weather is sunny"


llm = ChatOpenAI(model="gpt-4.1-mini")
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from lang-chain course.")
    result = agent.invoke(
        {"messages": [HumanMessage(content="What is the weather in Amsterdam?")]}
    )
    print(result)


if __name__ == "__main__":
    main()