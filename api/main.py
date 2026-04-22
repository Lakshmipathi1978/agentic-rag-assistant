"""
FastAPI Backend for Agentic RAG Assistant
Endpoints:
  GET  /health  - health check
  POST /ingest  - upload and index a PDF
  POST /ask     - run agentic RAG pipeline for a question
"""
import os, shutil, time, sys
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest import ingest_pdf
from src.agent  import run_agent

app = FastAPI(
    title="Agentic RAG Assistant",
    description="Upload PDFs and ask questions — powered by LangGraph self-correcting agent",
    version="1.0.0"
)

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class QuestionRequest(BaseModel):
    question: str


class AgentResponse(BaseModel):
    answer:     str
    sources:    list
    steps:      list
    retries:    int
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "service": "agentic-rag-assistant"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Upload a PDF and index it into ChromaDB."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = ingest_pdf(str(save_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "filename": file.filename,
        "pages":    result["pages"],
        "chunks":   result["chunks"],
        "message":  "PDF indexed! The agent is ready to answer questions."
    }


@app.post("/ask", response_model=AgentResponse)
def ask(request: QuestionRequest):
    """Run the agentic RAG pipeline for a question."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    start = time.time()
    try:
        result = run_agent(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = round((time.time() - start) * 1000, 1)

    return AgentResponse(
        answer=result["answer"],
        sources=result["sources"],
        steps=result["steps"],
        retries=result["retries"],
        latency_ms=latency_ms
    )
