import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


# Load environment variables from the .env file.
# Expected variables:
# - OPENAI_API_KEY
# - INDEX_NAME
load_dotenv()


print("Initializing components...")


# Embedding model used to convert user queries into vectors.
# Pinecone uses these vectors to find semantically similar document chunks.
embeddings = OpenAIEmbeddings()


# Chat model used to generate final answers.
# You can specify a model explicitly, for example:
# ChatOpenAI(model="gpt-5")
llm = ChatOpenAI()


# Connect to an existing Pinecone index.
# This assumes your documents were already ingested into Pinecone.
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"],
    embedding=embeddings,
)


# Convert the vector store into a retriever.
# k=3 means: return the top 3 most relevant document chunks.
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# Prompt template used for RAG.
# The LLM is instructed to answer only from retrieved context.
prompt_template = ChatPromptTemplate.from_template(
    """
Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:
"""
)


def format_docs(docs):
    """
    Convert retrieved LangChain Document objects into one text block.

    Input:
        docs = [Document(page_content="..."), Document(page_content="...")]

    Output:
        "document text 1\n\ndocument text 2"
    """
    return "\n\n".join(doc.page_content for doc in docs)


# ============================================================================
# IMPLEMENTATION 1: RAG WITHOUT LCEL
# ============================================================================
def retrieval_chain_without_lcel(query: str):
    """
    Run a simple RAG pipeline manually.

    Flow:
    1. Retrieve relevant chunks from Pinecone
    2. Convert retrieved documents into context text
    3. Insert context and question into prompt
    4. Send prompt to LLM
    5. Return final answer text
    """

    # Step 1: Search Pinecone for document chunks related to the query.
    docs = retriever.invoke(query)

    # Step 2: Combine retrieved document chunks into a single context string.
    context = format_docs(docs)

    # Step 3: Fill the prompt template with retrieved context and user question.
    messages = prompt_template.format_messages(
        context=context,
        question=query,
    )

    # Step 4: Send the formatted prompt messages to the LLM.
    response = llm.invoke(messages)

    # Step 5: Return only the text content from the AIMessage response.
    return response.content


# ============================================================================
# IMPLEMENTATION 2: RAG WITH LCEL
# ============================================================================
def create_retrieval_chain_with_lcel():
    """
    Create the same RAG pipeline using LCEL.

    LCEL = LangChain Expression Language.

    The pipe operator `|` connects steps together, like a data pipeline.

    Final chain input:
        {"question": "what is Pinecone in machine learning?"}

    Final chain output:
        string answer
    """

    retrieval_chain = (
        # RunnablePassthrough keeps the original input dictionary.
        #
        # Input:
        #   {"question": "..."}
        #
        # assign(...) adds a new key called "context".
        #
        # After this step:
        #   {
        #     "question": "...",
        #     "context": "retrieved document text..."
        #   }
        RunnablePassthrough.assign(
            # itemgetter("question") extracts the question value.
            #
            # Then:
            # question -> retriever -> list of docs -> format_docs -> context string
            context=itemgetter("question") | retriever | format_docs
        )

        # Insert "context" and "question" into the prompt template.
        | prompt_template

        # Send the formatted prompt to the chat model.
        | llm

        # Convert AIMessage output into a plain string.
        | StrOutputParser()
    )

    return retrieval_chain


if __name__ == "__main__":
    print("Retrieving...")

    # User question for all three examples.
    query = "What is Pinecone in machine learning?"

    # =========================================================================
    # OPTION 0: RAW LLM CALL WITHOUT RAG
    # =========================================================================
    # This does NOT use Pinecone.
    # The model answers only from its own internal knowledge.
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)

    result_raw = llm.invoke([HumanMessage(content=query)])

    print("\nAnswer:")
    print(result_raw.content)

    # =========================================================================
    # OPTION 1: MANUAL RAG WITHOUT LCEL
    # =========================================================================
    # This uses Pinecone, but every step is written manually.
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: RAG Without LCEL")
    print("=" * 70)

    result_without_lcel = retrieval_chain_without_lcel(query)

    print("\nAnswer:")
    print(result_without_lcel)

    # =========================================================================
    # OPTION 2: RAG WITH LCEL
    # =========================================================================
    # This uses Pinecone and LCEL to express the pipeline declaratively.
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: RAG With LCEL")
    print("=" * 70)

    chain_with_lcel = create_retrieval_chain_with_lcel()

    # Input must be a dictionary because the chain expects a "question" key.
    result_with_lcel = chain_with_lcel.invoke(
        {"question": query}
    )

    print("\nAnswer:")
    print(result_with_lcel)