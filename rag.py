import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.embeddings.base import Embeddings
import hashlib
from sqlalchemy.orm import Session
from models import Document
from database import filter_documents

load_dotenv()
GROQ_API_KEY = os.getenv("gr_api_key1")

llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")

# Tiny hash-based embeddings
class TinyEmbeddings(Embeddings):
    def __init__(self, dimension=128):
        self.dimension = dimension

    def _text_to_vector(self, text):
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        vector = [(hash_bytes[i % len(hash_bytes)] / 255.0) - 0.5 for i in range(self.dimension)]
        return vector

    def embed_documents(self, texts):
        return [self._text_to_vector(text) for text in texts]

    def embed_query(self, text):
        return self._text_to_vector(text)

embeddings = TinyEmbeddings(dimension=128)

# Load FAISS index if exists
try:
    vector_store = FAISS.load_local("faiss_index", embeddings)
except:
    vector_store = None

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3}) if vector_store else None

# Build RAG chain
if retriever:
    rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
else:
    prompt = PromptTemplate(template="{question}", input_variables=["question"])
    rag_chain = LLMChain(llm=llm, prompt=prompt)

# Query documents from DB with admin control
def query_documents(db: Session, is_admin: bool = False):
    query = db.query(Document)
    query = filter_documents(query, is_admin)
    return query.all()

# Insert document with optional admin/public flag
def insert_document(db: Session, title: str, content: str, user_id: int = None, is_public: bool = False):
    doc = Document(title=title, content=content, user_id=user_id, is_public=is_public)
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc
