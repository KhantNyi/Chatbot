# Import basics
import os
import time
from dotenv import load_dotenv

# Import Pinecone
from pinecone import Pinecone, ServerlessSpec

# Import LangChain
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Documents
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Debug environment variables
print("PINECONE_API_KEY:", os.environ.get("PINECONE_API_KEY"))
print("PINECONE_INDEX_NAME:", os.environ.get("PINECONE_INDEX_NAME"))

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Initialize Pinecone database
index_name = os.environ.get("PINECONE_INDEX_NAME")
if not index_name:
    raise ValueError("PINECONE_INDEX_NAME is not set in .env file")

# Check whether index exists, and create if not
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
print("Existing indexes:", existing_indexes)

if index_name in existing_indexes:
    print(f"Deleting existing index: {index_name}")
    pc.delete_index(index_name)

print(f"Creating new index: {index_name}")
pc.create_index(
    name=index_name,
    dimension=384,  # For all-MiniLM-L6-v2
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

print(f"Waiting for index {index_name} to be ready...")
while not pc.describe_index(index_name).status["ready"]:
    time.sleep(1)
print(f"Index {index_name} is ready!")

# Access the index
try:
    index = pc.Index(index_name)
    print(f"Successfully accessed index: {index_name}")
except Exception as e:
    print(f"Failed to access index {index_name}: {e}")
    raise

# Initialize embeddings model + vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Loading the PDF document
loader = PyPDFDirectoryLoader("/Users/khantnyi/Desktop/NLP/LangChain-Pinecone-RAG-main/documents/")
raw_documents = loader.load()
print("Number of raw documents loaded:", len(raw_documents))

# Splitting the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=400,
    length_function=len,
    is_separator_regex=False,
)
documents = text_splitter.split_documents(raw_documents)
print("Number of document chunks:", len(documents))

# Generate unique IDs
uuids = [f"id{i+1}" for i in range(len(documents))]

# Add to database
try:
    vector_store.add_documents(documents=documents, ids=uuids)
    print(f"Added {len(documents)} documents to Pinecone index '{index_name}'")
except Exception as e:
    print(f"Error adding documents: {e}")

# Check index stats
stats = index.describe_index_stats()
print("Pinecone index stats:", stats)