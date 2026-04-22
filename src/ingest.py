"""
PDF Ingestion Module
Loads PDF, splits into chunks, embeds and stores in ChromaDB
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

CHROMA_DIR = str(PROJECT_ROOT / "data" / "chroma_db")


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )


def ingest_pdf(pdf_path: str) -> dict:
    """Load PDF, chunk it, embed and store in ChromaDB."""
    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(pages)

    embeddings = get_embeddings()
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    return {
        "pages":  len(pages),
        "chunks": len(chunks),
        "status": "ingested"
    }


def get_retriever(k: int = 4):
    """Return a ChromaDB retriever."""
    embeddings = get_embeddings()
    vectordb   = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    return vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
