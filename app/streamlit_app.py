"""
Streamlit UI for Agentic RAG Assistant
Shows the agent's thinking steps in real time
"""
import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Agentic RAG Assistant")
st.markdown("Upload a PDF and ask questions. The AI agent **self-corrects** if it cannot find a good answer.")

# Health check
try:
    r = requests.get(f"{API_URL}/health", timeout=3)
    if r.status_code == 200:
        st.success("✅ Agent API is online")
    else:
        st.error("❌ API error")
except:
    st.error("❌ Cannot connect to API. Make sure FastAPI server is running.")

st.divider()

# ── Step 1: Upload PDF ────────────────────────────────────────────────────────
st.subheader("Step 1 — Upload your PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    if st.button("📥 Index PDF", key="index_btn"):
        with st.spinner("Reading and indexing PDF..."):
            files    = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            response = requests.post(f"{API_URL}/ingest", files=files)

        if response.status_code == 200:
            data = response.json()
            st.success(
                f"✅ **{data['filename']}** indexed!\n\n"
                f"📃 Pages: **{data['pages']}** | 🔢 Chunks: **{data['chunks']}**"
            )
            st.session_state["pdf_ready"] = True
            st.session_state["pdf_name"]  = data["filename"]
        else:
            st.error(f"❌ {response.json().get('detail', 'Unknown error')}")

st.divider()

# ── Step 2: Ask Question ──────────────────────────────────────────────────────
st.subheader("Step 2 — Ask a question")

if st.session_state.get("pdf_ready"):
    st.info(f"🤖 Agent is ready to answer from: **{st.session_state.get('pdf_name')}**")

question = st.text_input("Type your question", placeholder="e.g. What is the main topic of this document?")

if st.button("🔍 Ask Agent", key="ask_btn"):
    if not question.strip():
        st.warning("Please type a question first.")
    else:
        with st.spinner("Agent is thinking... (may take 15-30 seconds)"):
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question},
                timeout=120
            )

        if response.status_code == 200:
            data = response.json()

            # ── Answer ────────────────────────────────────────────────────────
            st.markdown("### 💡 Answer")
            st.write(data["answer"])

            # ── Metrics ───────────────────────────────────────────────────────
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱ Latency", f"{data['latency_ms']} ms")
            with col2:
                pages = ", ".join(str(p) for p in data["sources"]) if data["sources"] else "N/A"
                st.metric("📖 Source Pages", pages)
            with col3:
                st.metric("🔄 Retries", data["retries"])

            # ── Agent Steps ───────────────────────────────────────────────────
            st.markdown("### 🧠 Agent Thinking Steps")
            st.caption("This shows exactly what the agent did to find your answer:")
            for i, step in enumerate(data["steps"], 1):
                if "rewritten" in step.lower():
                    st.warning(f"Step {i}: {step}")
                elif "0 relevant" in step.lower():
                    st.error(f"Step {i}: {step}")
                elif "good" in step.lower():
                    st.success(f"Step {i}: {step}")
                else:
                    st.info(f"Step {i}: {step}")

        else:
            st.error(f"❌ {response.json().get('detail', 'Unknown error')}")

st.divider()
st.caption("Built with LangGraph · LangChain · ChromaDB · Groq LLM · FastAPI · Streamlit")
