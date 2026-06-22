from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langchain_core.documents import Document
import subprocess
import shutil
import json
import uvicorn

from ingest import load_pdf_documents, ingest_pdf_documents, load_and_split_pdf
from vectorstore import get_embeddings, get_vector_store
from retriever import get_hybrid_retriever
from chat import run_chat, ChatSession
from rag_chain import answer, get_llm, refresh_retriever, get_retriever
from router import decide_route, Route
from config import Config
from database import init_db, SessionLocal, ensure_chat, add_message, list_chats, get_messages, delete_chat



cfg = Config()

# In-memory conversation store: session_id -> ChatSession
sessions = {}


def get_session(session_id: str) -> ChatSession:
    if session_id in sessions:
        return sessions[session_id]

    session = ChatSession()
    db = SessionLocal()
    try:
        session.load_history(get_messages(db, session_id))
    finally:
        db.close()
    sessions[session_id] = session
    return session




async def lifespan(app: FastAPI):
    init_db()
    try:
        retriever = get_retriever()
    except Exception as e:
        retriever = None
        print(f"Retriever not ready — ingest documents first via POST /ingest. ({e})")
        docs = load_pdf_documents(cfg.pdf_dir)
        ingest_pdf_documents(docs)
        
    app.state.retriever = retriever
    yield
    sessions.clear()
    
app = FastAPI(title="DocuMind AI", version="1.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"
    model: str | None = None


class Source(BaseModel):
    source: str
    page: int | str
    score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    route: str
    sources: list[Source] = []


def title_from_query(query: str, max_len: int = 90) -> str:
    title = " ".join(query.split())
    if len(title) <= max_len:
        return title

    clipped = title[:max_len].rsplit(" ", 1)[0].rstrip()
    return (clipped or title[:max_len].rstrip()) + "..."


def selected_model(model: str | None) -> str:
    return (model or cfg.llm_model).strip()


def is_chat_model(model: str) -> bool:
    return "embed" not in model.casefold()


def list_ollama_models() -> list[str]:
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        timeout=8,
        check=True,
    )
    models = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.upper().startswith("NAME "):
            continue
        model = line.split()[0]
        if is_chat_model(model):
            models.append(model)
    return sorted(set(models), key=str.casefold)
    
    
@app.get("/health")
def health():
    return {"status": "ok", "model": cfg.llm_model, "embedding_model": cfg.embed_model}


@app.get("/models")
def get_models():
    try:
        models = list_ollama_models()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise HTTPException(status_code=503, detail=f"Could not read ollama list: {e}") from e
    return {"models": models, "default_model": cfg.llm_model}



@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session = get_session(request.session_id)
    
    result = session.ask(request.query, selected_model(request.model))
    chat_response = ChatResponse(
        answer=result.text,
        route=result.route.value,
        sources=[Source(source=s["source"], page=s["page"], score=s.get("score")) for s in result.sources]
    )
    return chat_response


@app.post("/ingest")
def ingest():
    docs = load_pdf_documents(cfg.pdf_dir)
    ingest_pdf_documents(docs)
    return {"status": "ingested", "documents": len(docs)}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    is_pdf = file.filename.lower().endswith(".pdf")
    if not is_pdf:
        return {"error": "only PDF files are accepted"}
    
    destination = f"{cfg.pdf_dir}/{file.filename}"
    
    # save the file
    with open(destination, "wb") as f:
        shutil.copyfileobj(file.file, f)                      

    # load + split one file
    chunks = load_and_split_pdf(str(destination))                    

    # ADD (not from_documents)
    embeddings = get_embeddings(cfg.embed_model, cfg.ollama_url)
    store = get_vector_store(str(cfg.chroma_dir), cfg.collection_name, embeddings)
    store.add_documents(chunks)                               

    # drop the stale BM25 cache
    refresh_retriever()
    return {"status": "uploaded", "file": file.filename, "chunks": len(chunks)}


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    session = get_session(request.session_id)

    # Persist the user turn synchronously in the request thread (a Session must
    # not span the generator's threadpool hops, or its commits get dropped).
    db = SessionLocal()
    try:
        ensure_chat(db, request.session_id, title_from_query(request.query))
        add_message(db, request.session_id, "user", request.query)
    finally:
        db.close()

    model_name = selected_model(request.model)

    def event_stream():
        full, route, sources = "", None, []
        for evt in session.ask_stream(request.query, model_name):
            if evt["type"] == "meta":
                route, sources = evt["route"], evt["sources"]
            elif evt["type"] == "token":
                full += evt["text"]
            yield f"data: {json.dumps(evt)}\n\n"

        # Persist the assistant turn with its own fresh session.
        db2 = SessionLocal()
        try:
            add_message(db2, request.session_id, "assistant", full, route, sources)
        finally:
            db2.close()

    return StreamingResponse(event_stream(), media_type="text/event-stream")



@app.get("/chats")
def get_chats():
    db = SessionLocal()
    try:
        chats = list_chats(db)
        return [{"id": chat.id, "title": chat.title, "created_at": chat.created_at, "updated_at": chat.updated_at} for chat in chats]
    finally:
        db.close()
        
@app.get("/chats/{chat_id}")
def get_chat_messages(chat_id: str):
    db = SessionLocal()
    try:
        messages = get_messages(db, chat_id)
        return [{"role": msg.role, "content": msg.content, "route": msg.route, "sources": msg.sources, "created_at": msg.created_at} for msg in messages]
    finally:
        db.close()
        
@app.delete("/chats/{chat_id}")
def remove_chat(chat_id: str):
    db = SessionLocal()
    try:
        delete_chat(db, chat_id)
        return {"status": "deleted"}
    finally:
        db.close()
        

app.mount("/images", StaticFiles(directory="images"), name="images")
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)
