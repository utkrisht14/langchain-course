from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

load_dotenv()

llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    print("In this we will explore Tavily API for real-world search information.")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What are the top 3 jobs in Agentic AI in the Netherlands?"
                )
            ]
        }
    )
    print(result)


if __name__ == "__main__":
    main()