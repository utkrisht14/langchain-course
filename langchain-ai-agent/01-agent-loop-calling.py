from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

MAX_ITERATIONS = 10
model = "qwen3:1.7b"

# ---- Tools (Langchain @tool decorator)

@tool
def get_product_price(product: str):
    """ Look up the price of a product in the catalog """
    print(f"    >>> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier:str) -> float:
    """ Apply a discount to the price and return the final price.
    Available tiers: "bronze", "silver", "gold"""
    print(f"    >>> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze":5, "silver":12, "gold":23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

@traceable(name="Langchain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama: {MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f" Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            "You're a helpful shopping assistant."
            "You have access to product catalog tool and a discount tool."
            "STRICT RULES - you must follow these exactly:\n"
            "1. Never Guess or Assume any product price. \n"
            "You must call get_product_price first to get the real price. \n"
            "2. Only call appy_discount AFTER you have received "
            "a price from get_product_price. Pass the exact price "
            "returned by get_product_price - do NOT pass a made-up number. \n"
            "3. NEVER calculate discounts yourself using math."
            "Always use the apply_discount tool.\n"
        )
    ]



if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)")
    print()
    run_agent("What is the price of a laptop after applying a gold discount?")