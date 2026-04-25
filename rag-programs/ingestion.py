# Import built-in module to interact with environment variables and file paths
import os

# Import dotenv to load variables from .env file (like API keys)
from dotenv import load_dotenv

# Loader to read text files into LangChain document format
from langchain_community.document_loaders import TextLoader

# Splits large text into smaller chunks for embeddings
from langchain_text_splitters import CharacterTextSplitter

# OpenAI embedding model (converts text → vectors)
from langchain_openai import OpenAIEmbeddings

# Pinecone vector database integration for storing embeddings
from langchain_pinecone import PineconeVectorStore


# Load environment variables from .env file into os.environ
load_dotenv()

# Print current working directory (helps debug file path issues)
print(os.getcwd())


# Main entry point of the program
if __name__ == "__main__":

    # Step 1: Start ingestion process
    print("Ingesting....")

    # Load text file into LangChain documents
    # encoding="utf-8" ensures proper reading of text
    # autodetect_encoding=True tries fallback encodings if needed
    loader = TextLoader(
        r"C:\Users\utkri\PycharmProjects\LangChainCourse\rag-programs\mediumblog1.txt",
        encoding="utf-8",
        autodetect_encoding=True
    )

    # Actually read the file → returns list of Document objects
    documents = loader.load()

    # Step 2: Split documents into chunks
    print("Splitting....")

    # Create text splitter:
    # chunk_size = max size of each chunk
    # chunk_overlap = overlap between chunks (0 here)
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )

    # Split documents into smaller chunks
    texts = text_splitter.split_documents(documents)

    # Print how many chunks were created
    print(f"Created {len(texts)} chunks.")

    # Step 3: Create embeddings model
    # This will convert each chunk into a vector
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

    # Step 4: Store embeddings in Pinecone vector database
    print("Ingesting into Pinecone...")

    # from_documents does:
    # 1. Convert chunks → embeddings
    # 2. Store embeddings in Pinecone index
    vector_store = PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=os.environ["INDEX_NAME"]   # Make sure this exists in .env
    )

    # Print confirmation
    print("Vector store created:", vector_store)

    # Final message
    print("Finished ingestion successfully.")