import os
import uuid
import tempfile
from fastapi import FastAPI, Depends, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session
from database import get_db, engine
from models import Base, Document, User
from rag import rag_chain, vector_store, embeddings, insert_document
from langchain.docstore.document import Document as LC_Document
from langchain.vectorstores import FAISS
from groq import Groq
from dotenv import load_dotenv
import uvicorn

load_dotenv()

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize Groq clients
clients = [Groq(api_key=os.getenv(f"gr_api_key{i}")) for i in range(1,3)]

app = FastAPI(title="Groq LLM RAG Chatbot with Memory")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "https://status-ten-gules.vercel.app",
    "http://localhost:5173",
    "http://127.0.0.1:8000"
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_memory = {}  # session_id -> conversation history

# ---------------------- Pydantic Schemas ----------------------
class UserCreate(BaseModel):
    username: str
    email: str
    is_admin: Optional[bool] = False

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    is_admin: bool

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    user_id: Optional[int] = None

class ChatResponse(BaseModel):
    answer: str
    session_id: str

class DocumentCreate(BaseModel):
    title: str
    content: str
    is_public: Optional[bool] = False

# ---------------------- Helper Functions ----------------------
def get_accessible_documents(db: Session, user_id: Optional[int] = None, is_admin: bool = False):
    if is_admin:
        return db.query(Document).all()
    elif user_id:
        return db.query(Document).filter(
            (Document.user_id == user_id) | (Document.is_public == True)
        ).all()
    else:
        return db.query(Document).filter(Document.is_public == True).all()

def generate_text_response(query: str, session_id: Optional[str], db: Session, user_id: Optional[int] = None):
    session_id = session_id or str(uuid.uuid4())
    if session_id not in chat_memory:
        chat_memory[session_id] = []

    history = chat_memory[session_id]
    conversation_context = "\n".join([f"{h['role']}: {h['text']}" for h in history])

    system_prompt = (
        "You are a helpful assistant. Answer the user's question in a natural, conversational manner. "
        "Use the provided context from the database as closely as possible. "
        "When referencing information from the context, mention the relevant lines or phrases in double quotes. "
        "If the answer is not found in the context, let the user know."
    )

    user = None
    is_admin = False
    if user_id:
        user = db.query(User).filter(User.id == user_id).first()
        is_admin = user.is_admin if user else False

    db_docs = get_accessible_documents(db, user_id, is_admin)
    db_context = "\n".join([f"Title: {doc.title}\nContent: {doc.content}" for doc in db_docs]) if db_docs else ""

    prompt = (
        f"{system_prompt}\n{db_context}\n{conversation_context}\nUser: {query}"
        if history else f"{system_prompt}\n{db_context}\nUser: {query}"
    )

    if hasattr(rag_chain, "run"):
        answer = rag_chain.run(prompt)
    else:
        answer = str(rag_chain.predict(question=prompt))

    history.append({"role": "User", "text": query})
    history.append({"role": "Bot", "text": answer})

    return answer, session_id

# ---------------------- API Endpoints ----------------------
@app.post("/create_user", response_model=UserResponse)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = User(username=user.username, email=user.email, is_admin=user.is_admin)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/users", response_model=list[UserResponse])
def get_users(db: Session = Depends(get_db)):
    return db.query(User).all()

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    answer, session_id = generate_text_response(request.query, request.session_id, db, request.user_id)
    return {"answer": answer, "session_id": session_id}

@app.post("/add_document")
def add_document(doc: DocumentCreate, user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    document = insert_document(db, doc.title, doc.content, user_id, doc.is_public)

    lc_doc = LC_Document(page_content=doc.content, metadata={"title": doc.title})
    global vector_store
    if vector_store:
        vector_store.add_documents([lc_doc])
    else:
        vector_store = FAISS.from_documents([lc_doc], embeddings)

    FAISS.save_local(vector_store, "faiss_index")
    return {"id": document.id, "title": document.title}

@app.get("/documents")
def get_documents(user_id: Optional[int] = None, db: Session = Depends(get_db)):
    user = None
    is_admin = False
    if user_id:
        user = db.query(User).filter(User.id == user_id).first()
        is_admin = user.is_admin if user else False

    docs = get_accessible_documents(db, user_id, is_admin)
    return [{"id": doc.id, "title": doc.title, "content": doc.content, "is_public": doc.is_public, "owner": doc.owner.username if doc.owner else None} for doc in docs]

@app.post("/talk", response_class=StreamingResponse)
async def talk(audio_file: UploadFile, session_id: Optional[str] = None, db: Session = Depends(get_db)):
    session_id = session_id or str(uuid.uuid4())
    if session_id not in chat_memory:
        chat_memory[session_id] = []

    history = chat_memory[session_id]
    conversation_context = "\n".join([f"{h['role']}: {h['text']}" for h in history])
    chat_hist = [{"role": h["role"].lower(), "content": h["text"]} for h in history]

    try:
        ext = ".webm" if audio_file.content_type == "audio/webm" else ".m4a"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_audio:
            temp_audio.write(await audio_file.read())
            temp_audio_path = temp_audio.name

        with open(temp_audio_path, "rb") as file:
            transcription = clients[-1].audio.transcriptions.create(
                file=(temp_audio_path, file.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
            )
        os.remove(temp_audio_path)
        chat_hist.append({"role": "user", "content": transcription.text})

        answer, _ = generate_text_response(transcription.text, session_id, db)
        chat_hist.append({"role": "assistant", "content": answer})

        tts_error = None
        for client in clients:
            try:
                response = client.audio.speech.create(
                    model="playai-tts",
                    voice="Nia-PlayAI",
                    response_format="wav",
                    input=answer
                )
                audio_data = b"".join([chunk for chunk in response.iter_bytes()])
                return StreamingResponse(
                    iter([audio_data]),
                    media_type="audio/wav",
                    headers={"Content-Disposition": "attachment; filename=medbot_response.wav"}
                )
            except Exception as e:
                print(f"TTS client failed: {e}")
                tts_error = e
                continue

        raise HTTPException(status_code=503, detail=f"All TTS models unavailable. Last error: {tts_error}")

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"General error: {e}")
        raise HTTPException(status_code=500, detail=f"General error: {e}")

# ---------------------- Main ----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
