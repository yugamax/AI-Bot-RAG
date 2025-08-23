import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.embeddings.base import Embeddings
import hashlib

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("gr_api_key1")

# Initialize Groq LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile"
)

# Ultra-lightweight hash-based embeddings (no model downloads)
class TinyEmbeddings(Embeddings):
    def __init__(self, dimension=128):
        self.dimension = dimension
    
    def _text_to_vector(self, text):
        # Create deterministic vector from text hash
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to vector
        vector = []
        for i in range(self.dimension):
            byte_idx = i % len(hash_bytes)
            vector.append((hash_bytes[byte_idx] / 255.0) - 0.5)
        return vector
    
    def embed_documents(self, texts):
        return [self._text_to_vector(text) for text in texts]
    
    def embed_query(self, text):
        return self._text_to_vector(text)

# Use tiny embeddings - no downloads, minimal size
embeddings = TinyEmbeddings(dimension=128)

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
