"""
Agentic RAG - LangGraph State Machine
Nodes:
  1. retrieve        - fetch top-4 chunks from ChromaDB
  2. grade_chunks    - LLM decides if chunks are relevant
  3. rewrite_query   - LLM rewrites question if chunks irrelevant
  4. generate        - LLM generates answer from chunks
  5. grade_answer    - LLM checks if answer addresses the question
Flow:
  retrieve → grade_chunks → [relevant] generate → grade_answer → [good] END
                          → [irrelevant] rewrite_query → retrieve (retry)
                                                generate → [bad] rewrite_query → retrieve (retry)
"""
import os
from pathlib import Path
from typing import List, Literal
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

MAX_RETRIES = 2  # Maximum number of rewrite+retry cycles


# ── State Definition ──────────────────────────────────────────────────────────
class AgentState(TypedDict):
    question:       str
    documents:      List[Document]
    answer:         str
    retry_count:    int
    steps:          List[str]   # Track what the agent did — visible in UI


# ── LLM ──────────────────────────────────────────────────────────────────────
def get_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.0
    )


# ── Node 1: Retrieve ──────────────────────────────────────────────────────────
def retrieve(state: AgentState) -> AgentState:
    """Retrieve top-4 chunks from ChromaDB for the current question."""
    from src.ingest import get_retriever
    retriever = get_retriever(k=4)
    docs      = retriever.invoke(state["question"])
    state["documents"] = docs
    state["steps"].append(f"Retrieved {len(docs)} chunks for: '{state['question']}'")
    return state


# ── Node 2: Grade Chunks ──────────────────────────────────────────────────────
def grade_chunks(state: AgentState) -> AgentState:
    """Use LLM to decide if retrieved chunks are relevant to the question."""
    llm = get_llm()

    prompt = PromptTemplate.from_template("""You are a relevance grader.
Given a question and a document chunk, decide if the chunk is relevant.
Answer with a single word: 'yes' or 'no'.

Question: {question}
Document chunk: {document}
Relevant (yes/no):""")

    chain = prompt | llm | StrOutputParser()

    relevant_docs = []
    for doc in state["documents"]:
        score = chain.invoke({
            "question": state["question"],
            "document": doc.page_content
        }).strip().lower()
        if "yes" in score:
            relevant_docs.append(doc)

    state["documents"] = relevant_docs
    state["steps"].append(
        f"Chunk grading: {len(relevant_docs)}/{len(state['documents']) + (len(state['documents']) - len(relevant_docs))} chunks relevant"
        if relevant_docs else
        f"Chunk grading: 0 relevant chunks found — will rewrite question"
    )
    return state


# ── Node 3: Rewrite Query ─────────────────────────────────────────────────────
def rewrite_query(state: AgentState) -> AgentState:
    """Use LLM to rewrite the question for better retrieval."""
    llm = get_llm()

    prompt = PromptTemplate.from_template("""You are a query rewriter for a document search system.
Rewrite the following question to improve document retrieval.
Make it more specific and use different keywords.
Return ONLY the rewritten question, nothing else.

Original question: {question}
Rewritten question:""")

    chain = prompt | llm | StrOutputParser()
    new_question = chain.invoke({"question": state["question"]}).strip()

    state["retry_count"] += 1
    state["steps"].append(f"Query rewritten (attempt {state['retry_count']}): '{new_question}'")
    state["question"] = new_question
    return state


# ── Node 4: Generate ──────────────────────────────────────────────────────────
def generate(state: AgentState) -> AgentState:
    """Generate answer from relevant chunks using Groq LLM."""
    llm = get_llm()

    context = "\n\n".join(doc.page_content for doc in state["documents"])

    prompt = PromptTemplate.from_template("""You are a helpful assistant. Use ONLY the context below to answer.
If the answer is not in the context, say "I don't have enough information in the document to answer that."

Context:
{context}

Question: {question}

Answer:""")

    chain  = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context":  context,
        "question": state["question"]
    })

    state["answer"] = answer
    state["steps"].append("Answer generated from relevant chunks")
    return state


# ── Node 5: Grade Answer ──────────────────────────────────────────────────────
def grade_answer(state: AgentState) -> AgentState:
    """Use LLM to verify the answer actually addresses the question."""
    llm = get_llm()

    prompt = PromptTemplate.from_template("""You are an answer quality checker.
Does the following answer properly address the question?
Answer with a single word: 'yes' or 'no'.

Question: {question}
Answer: {answer}
Addresses the question (yes/no):""")

    chain  = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "question": state["question"],
        "answer":   state["answer"]
    }).strip().lower()

    quality = "good" if "yes" in result else "poor"
    state["steps"].append(f"Answer quality check: {quality}")
    return state


# ── Conditional Edges ─────────────────────────────────────────────────────────
def should_rewrite_or_generate(state: AgentState) -> Literal["generate", "rewrite_query"]:
    """After grading chunks: generate if relevant docs found, else rewrite."""
    if state["documents"] and len(state["documents"]) > 0:
        return "generate"
    if state["retry_count"] >= MAX_RETRIES:
        # Give up retrying — generate with empty context
        state["documents"] = []
        return "generate"
    return "rewrite_query"


def should_end_or_retry(state: AgentState) -> Literal["end", "rewrite_query"]:
    """After grading answer: end if good, retry if poor."""
    last_step = state["steps"][-1] if state["steps"] else ""
    if "good" in last_step or state["retry_count"] >= MAX_RETRIES:
        return "end"
    return "rewrite_query"


# ── Build Graph ───────────────────────────────────────────────────────────────
def build_agent():
    """Build and compile the LangGraph agent."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("retrieve",      retrieve)
    graph.add_node("grade_chunks",  grade_chunks)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate",      generate)
    graph.add_node("grade_answer",  grade_answer)

    # Set entry point
    graph.set_entry_point("retrieve")

    # Add edges
    graph.add_edge("retrieve",      "grade_chunks")
    graph.add_conditional_edges(
        "grade_chunks",
        should_rewrite_or_generate,
        {"generate": "generate", "rewrite_query": "rewrite_query"}
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate",      "grade_answer")
    graph.add_conditional_edges(
        "grade_answer",
        should_end_or_retry,
        {"end": END, "rewrite_query": "rewrite_query"}
    )

    return graph.compile()


# ── Run Agent ─────────────────────────────────────────────────────────────────
def run_agent(question: str) -> dict:
    """Run the agentic RAG pipeline for a given question."""
    agent = build_agent()

    initial_state: AgentState = {
        "question":    question,
        "documents":   [],
        "answer":      "",
        "retry_count": 0,
        "steps":       []
    }

    final_state = agent.invoke(initial_state)

    sources = sorted(set(
        doc.metadata.get("page", 0) + 1
        for doc in final_state["documents"]
    )) if final_state["documents"] else []

    return {
        "answer":  final_state["answer"],
        "sources": sources,
        "steps":   final_state["steps"],
        "retries": final_state["retry_count"]
    }
