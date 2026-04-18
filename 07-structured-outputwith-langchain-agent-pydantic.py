from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


class Source(BaseModel):
    """ Schema for a source used by the agent. """

    url:str = Field(description="The url of the source.")

class AgentResponse(BaseModel):
    """ Schema for the agent's response. """
    answer: str = Field(description="The agent's answer to the question.")
    sources : List[Source] = Field(default_factory=List ,description="The sources used to answer the question.")


llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format= AgentResponse)


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