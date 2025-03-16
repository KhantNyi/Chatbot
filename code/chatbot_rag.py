import streamlit as st
import os
import time
from dotenv import load_dotenv

# Import Pinecone
from pinecone import Pinecone, ServerlessSpec

# Import LangChain
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Load environment variables
load_dotenv()

# Ensure Hugging Face token is set
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACE_TOKEN is not set in the environment. Please check your .env file.")

# Authenticate with Hugging Face
login(token=hf_token)

# Streamlit Title
st.title("Chatbot")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Set the Pinecone index
index_name = os.getenv("PINECONE_INDEX_NAME")
if not index_name:
    raise ValueError("PINECONE_INDEX_NAME is not set in the environment.")

# Check and create Pinecone index
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    print(f"Index '{index_name}' not found. Creating a new index with dimension 384...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
else:
    index_desc = pc.describe_index(index_name)
    if index_desc.dimension != 384:
        print(f"Index '{index_name}' has dimension {index_desc.dimension}, expected 384. Recreating index...")
        pc.delete_index(index_name)
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

index = pc.Index(index_name)

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage("You are an assistant for question-answering tasks.")]

# Display chat history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# User input
prompt = st.chat_input("How can I assist you?")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append(HumanMessage(prompt))

    # Load Mistral-7B model
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)

    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        temperature=1.0,
        do_sample=True
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    # Retrieve relevant documents
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )

    docs = retriever.invoke(prompt)
    docs_text = " ".join(d.page_content for d in docs)

    # Construct system prompt
    system_prompt = f"""You are an assistant for question-answering tasks.
    Use the following retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Keep your answer concise (maximum 3 sentences).
    Context: {docs_text}"""

    st.session_state.messages.append(SystemMessage(system_prompt))

    # Generate response
    result = llm.invoke(st.session_state.messages)

    # Display chatbot response
    with st.chat_message("assistant"):
        st.markdown(result)
        st.session_state.messages.append(AIMessage(result))
