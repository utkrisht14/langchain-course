import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables from .env file
load_dotenv()

print("Initializing components....")

# Creates embedding model for converting text into vectors
embeddings = OpenAIEmbeddings()

# Creates chat model
llm = ChatOpenAI(model="gpt-5")

# Connects to existing Pinecone index
# This assumes you already ingested documents into Pinecone
vector_store = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"],
    embedding=embeddings
)

# Creates retriever from Pinecone vector store
# k=3 means retrieve top 3 most relevant chunks
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Prompt template for RAG answer generation
prompt_template = ChatPromptTemplate.from_template(
    """
Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:
"""
)


def format_docs(docs):
    """Convert retrieved documents into one context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def retrieval_chain_without_lcel(query: str):
    """
    Simple RAG chain without LCEL.

    Steps:
    1. Retrieve relevant documents from Pinecone
    2. Format retrieved documents into context
    3. Insert context and question into prompt
    4. Send prompt to LLM
    5. Return final answer
    """

    # Step 1: Retrieve relevant documents from Pinecone
    docs = retriever.invoke(query)

    # Step 2: Format retrieved documents into a single context string
    context = format_docs(docs)

    # Step 3: Format the prompt with context and question
    # FIX: context was misspelled as conetxt
    messages = prompt_template.format_messages(
        context=context,
        question=query
    )

    # Step 4: Invoke LLM with formatted messages
    response = llm.invoke(messages)

    # Step 5: Return only the text content from AIMessage
    return response.content


if __name__ == "__main__":
    print("Retrieving....")

    # User question
    query = "What is Pinecone in Machine Learning"

    ########################################
    # Option 0: Raw invocation without RAG
    ########################################

    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw invocation without RAG")
    print("=" * 70)

    # HumanMessage must be inside a list
    result_raw = llm.invoke([HumanMessage(content=query)])

    print("\nAnswer")
    print(result_raw.content)

    ###########################################
    # Option 1: RAG without LCEL
    ###########################################

    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: RAG without LCEL")
    print("=" * 70)

    # This answer uses retrieved Pinecone context
    result_without_lcel = retrieval_chain_without_lcel(query)

    print("\nAnswer")
    print(result_without_lcel)