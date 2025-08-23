import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile"
)

# Embeddings for vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load existing FAISS index if it exists, else None
try:
    vector_store = FAISS.load_local("faiss_index", embeddings)
except:
    vector_store = None

# Setup retriever if vector_store exists
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
) if vector_store else None

# Build RAG chain if retriever exists, else fallback to LLM only
if retriever:
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
else:
    prompt = PromptTemplate(template="{question}", input_variables=["question"])
    rag_chain = LLMChain(llm=llm, prompt=prompt)
