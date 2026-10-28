"""
app.py
Production Academic Research Agent UI
Features: Multi-Document Management, Document-Isolated Conversation Threads,
SQLite Persistence, and Langfuse Observability Tracing.
"""

import hashlib
import uuid
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import get_langfuse_handler
from src.indexing.loader import extract_metadata_and_chunks
from src.indexing.hybrid_retriever import build_hybrid_retriever
from src.agent.graph import build_agent_graph

# --- Imports ---
from src.auth import render_login_barrier

# --- Enforce Authentication Gatekeeper ---
if not render_login_barrier():
    st.stop()  # Halts further UI rendering until authentication passes

# --- Setup Directories & Database ---
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "raw_pdfs"
CHROMA_BASE = DATA_DIR / "chroma_indices"
DB_PATH = DATA_DIR / "checkpoints.db"

PDF_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_BASE.mkdir(parents=True, exist_ok=True)

# --- SQLite Session & Document Metadata Store ---
def init_db():
    """Initializes tables for tracking documents and conversation threads."""
    with sqlite3.connect(str(DB_PATH)) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_registry (
                file_hash TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                uploaded_at TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_threads (
                thread_id TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                messages_json TEXT NOT NULL,
                FOREIGN KEY (file_hash) REFERENCES document_registry (file_hash)
            )
        """)
        conn.commit()

init_db()

# --- Database Helper Functions ---
def register_document(file_hash: str, filename: str, filepath: str):
    with sqlite3.connect(str(DB_PATH)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO document_registry (file_hash, filename, filepath, uploaded_at) VALUES (?, ?, ?, ?)",
            (file_hash, filename, filepath, datetime.now().isoformat())
        )
        conn.commit()

def get_registered_documents() -> List[Dict[str, str]]:
    with sqlite3.connect(str(DB_PATH)) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT file_hash, filename, filepath FROM document_registry ORDER BY uploaded_at DESC")
        rows = cursor.fetchall()
        return [{"file_hash": r[0], "filename": r[1], "filepath": r[2]} for r in rows]

def get_document_threads(file_hash: str) -> List[Dict[str, Any]]:
    with sqlite3.connect(str(DB_PATH)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT thread_id, title, messages_json FROM conversation_threads WHERE file_hash = ? ORDER BY created_at DESC",
            (file_hash,)
        )
        rows = cursor.fetchall()
        return [{"thread_id": r[0], "title": r[1], "messages": json.loads(r[2])} for r in rows]

def save_thread_messages(thread_id: str, file_hash: str, title: str, messages: list):
    with sqlite3.connect(str(DB_PATH)) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO conversation_threads (thread_id, file_hash, title, created_at, messages_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                messages_json = excluded.messages_json,
                title = excluded.title
        """, (thread_id, file_hash, title, datetime.now().isoformat(), json.dumps(messages)))
        conn.commit()

def delete_thread(thread_id: str):
    with sqlite3.connect(str(DB_PATH)) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM conversation_threads WHERE thread_id = ?", (thread_id,))
        conn.commit()

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Academic Research Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern cards and pills
st.markdown("""
    <style>
    .metric-card {
        background-color: #1e2129;
        border: 1px solid #313642;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #2b313e;
        color: #6ee7b7;
        margin-right: 6px;
    }
    .stButton>button {
        border-radius: 6px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Indexing document and compiling LangGraph state machine...")
def setup_agent(pdf_path_str: str, file_hash: str):
    pdf_path = Path(pdf_path_str)
    metadata_profile, chunks = extract_metadata_and_chunks(pdf_path)

    # 1. Dynamic Embeddings Switch
    if os.getenv("GROQ_API_KEY"):
        # Runs lightweight CPU embeddings inside Render container
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    else:
        # Runs local Ollama embeddings on your laptop
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

    chroma_dir = CHROMA_BASE / f"chroma_{file_hash}"
    retriever = build_hybrid_retriever(
        chunks=chunks,
        chroma_dir=chroma_dir,
        collection_name=f"col_{file_hash}",
        embedding_model=embeddings,  # Pass the dynamic embeddings object here
    )

    # 2. Dynamic LLM Switch
    if os.getenv("GROQ_API_KEY"):
        # Runs cloud LLM via Groq on Render
        llm = ChatGroq(
            model_name="qwen-2.5-32b",
            temperature=0.0,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    else:
        # Runs local Ollama on your computer
        llm = ChatOllama(
            model="qwen2.5:7b",
            temperature=0.0,
            num_ctx=4096,
        )

    # Return or compile graph using the dynamic `llm` and `retriever`
    # ... rest of your graph compilation code ...

    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    agent_app = build_agent_graph(
        retriever=retriever,
        metadata_profile=metadata_profile,
        llm=llm,
        checkpointer=checkpointer,
    )
    return agent_app, metadata_profile, len(chunks)

# --- State Management ---
if "active_file_hash" not in st.session_state:
    st.session_state.active_file_hash = None
if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: Document Ingestion & Document-Isolated Chat History ---
with st.sidebar:
    st.title("📚 Documents")
    
    # Inside `with st.sidebar:`
    if st.button("🔒 Log Out", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()

    # Upload Section
    uploaded_file = st.file_uploader("Upload new PDF document", type=["pdf"])
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        f_hash = hashlib.sha256(file_bytes).hexdigest()[:12]
        saved_path = PDF_DIR / f"{f_hash}_{uploaded_file.name}"

        if not saved_path.exists():
            with open(saved_path, "wb") as f:
                f.write(file_bytes)

        register_document(f_hash, uploaded_file.name, str(saved_path))
        if st.session_state.active_file_hash != f_hash:
            st.session_state.active_file_hash = f_hash
            st.session_state.active_thread_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

    # Document Switcher
    registered_docs = get_registered_documents()
    if registered_docs:
        doc_options = {d["file_hash"]: d["filename"] for d in registered_docs}
        
        if st.session_state.active_file_hash not in doc_options:
            st.session_state.active_file_hash = registered_docs[0]["file_hash"]

        selected_doc_hash = st.selectbox(
            "Active Research Document",
            options=list(doc_options.keys()),
            format_func=lambda h: doc_options[h],
            index=list(doc_options.keys()).index(st.session_state.active_file_hash)
        )

        if selected_doc_hash != st.session_state.active_file_hash:
            st.session_state.active_file_hash = selected_doc_hash
            st.session_state.active_thread_id = None
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("Upload a thesis or research paper to begin.")
        st.stop()

    st.markdown("---")

    # Document-Isolated Conversation Threads
    st.subheader("💬 Chat History")
    col_new_chat, _ = st.columns([1, 0.05])
    with col_new_chat:
        if st.button("➕ New Chat Thread", use_container_width=True):
            st.session_state.active_thread_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

    threads = get_document_threads(st.session_state.active_file_hash)

    # Initialize thread if none selected
    if not st.session_state.active_thread_id:
        if threads:
            st.session_state.active_thread_id = threads[0]["thread_id"]
            st.session_state.messages = threads[0]["messages"]
        else:
            st.session_state.active_thread_id = str(uuid.uuid4())
            st.session_state.messages = []

    # Display past threads for the active document
    if threads:
        for th in threads:
            col_th, col_del = st.columns([0.85, 0.15])
            is_active = th["thread_id"] == st.session_state.active_thread_id
            button_label = f"• {th['title'][:26]}..." if len(th['title']) > 26 else f"• {th['title']}"

            if col_th.button(
                button_label,
                key=f"btn_{th['thread_id']}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.active_thread_id = th["thread_id"]
                st.session_state.messages = th["messages"]
                st.rerun()

            if col_del.button("✕", key=f"del_{th['thread_id']}", help="Delete thread"):
                delete_thread(th["thread_id"])
                if st.session_state.active_thread_id == th["thread_id"]:
                    st.session_state.active_thread_id = str(uuid.uuid4())
                    st.session_state.messages = []
                st.rerun()
    else:
        st.caption("No conversations yet for this document.")

    st.markdown("---")
    st.caption("⚙️ **Engine Status**")
    st.markdown("""
        <span class="badge">LangGraph DAG</span>
        <span class="badge">Hybrid RRF</span>
        <span class="badge">SQLite Active</span>
    """, unsafe_allow_html=True)


# --- Load Current Document & Initialize Engine ---
active_doc_record = next(d for d in registered_docs if d["file_hash"] == st.session_state.active_file_hash)
agent, metadata_profile, chunk_count = setup_agent(active_doc_record["filepath"], active_doc_record["file_hash"])


# --- Main Work Area ---
st.title("🎓 Academic Research Assistant")

# Document Header Bar
with st.container():
    st.markdown(f"""
        <div class="metric-card">
            <b>📄 Current Document:</b> {active_doc_record['filename']} &nbsp;|&nbsp;
            <b>Total Chunks:</b> {chunk_count} &nbsp;|&nbsp;
            <b>Thread ID:</b> <code>{st.session_state.active_thread_id[:8]}</code>
        </div>
    """, unsafe_allow_html=True)

# Starter Suggestion Chips (Shown only when the conversation is empty)
prompt_to_send = None
if len(st.session_state.messages) == 0:
    st.markdown("##### 💡 Suggested Research Inquiries")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👤 Who is the author and what is the document title?", use_container_width=True):
            prompt_to_send = "Who is the author of this document and what is the official title?"
        if st.button("📖 What are the main chapters covered in this work?", use_container_width=True):
            prompt_to_send = "List all the main chapters and structural sections included in this document."
    with col2:
        if st.button("📐 Explain the core mathematical equations and formulas", use_container_width=True):
            prompt_to_send = "Explain the key mathematical models, formulas, and equations described in this research."
        if st.button("📊 Summarize the primary experimental results and conclusions", use_container_width=True):
            prompt_to_send = "Summarize the primary experimental results, findings, and final conclusions."

# Render Existing Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "traces" in msg and msg["traces"]:
            with st.expander("🔍 Inspection & Decision Trace"):
                for trace in msg["traces"]:
                    st.text(trace)
        if "docs" in msg and msg["docs"]:
            with st.expander("📚 View Cited Sources"):
                for idx, doc_info in enumerate(msg["docs"]):
                    st.caption(f"**Source {idx+1} — {doc_info['chapter']} (Page {doc_info['page']})**")
                    st.text(doc_info["text"][:300] + "...")

# Chat Input Bar
user_input = st.chat_input("Ask a research question about the active document...")
query_to_execute = prompt_to_send or user_input

if query_to_execute:
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": query_to_execute})
    with st.chat_message("user"):
        st.markdown(query_to_execute)

    initial_state = {
        "question": query_to_execute,
        "current_query": "",
        "routing_reason": "",
        "sub_queries": [],
        "documents": [],
        "is_relevant": False,
        "generation": "",
        "retry_count": 0,
    }

    traces = []
    final_generation = ""
    final_docs = []

    # Observability & Checkpointing Config
    langfuse_handler = get_langfuse_handler()
    callbacks = [langfuse_handler] if langfuse_handler else []
    config = {
        "configurable": {"thread_id": st.session_state.active_thread_id},
        "callbacks": callbacks,
    }

    with st.chat_message("assistant"):
        status_box = st.status("Agent analyzing inquiry...", expanded=True)

        for step in agent.stream(initial_state, config=config):
            for node_name, output_dict in step.items():
                if node_name == "router":
                    chosen = output_dict.get("current_query", "")
                    reason = output_dict.get("routing_reason", "")
                    msg = f"Intent classified: Routed to '{chosen}' tool\nReason: {reason}"
                    traces.append(msg)
                    status_box.write(f"🧠 **Router Decision:** `{chosen}` — *{reason}*")

                elif node_name == "decomposer":
                    sqs = output_dict.get("sub_queries", [])
                    msg = f"Decomposed into {len(sqs)} sub-queries: {sqs}"
                    traces.append(msg)
                    status_box.write(f"🔀 **Query Decomposition:** ({len(sqs)} search vectors)")
                    for q in sqs:
                        status_box.write(f"   • *{q}*")

                elif node_name == "metadata_tool":
                    docs = output_dict.get("documents", [])
                    final_docs = docs
                    msg = "Accessed front-matter and structural metadata profile"
                    traces.append(msg)
                    status_box.write(f"📋 Accessing title & chapter front-matter")

                elif node_name == "search_tool":
                    docs = output_dict.get("documents", [])
                    final_docs = docs
                    msg = f"Executed hybrid RRF search ({len(docs)} deduplicated chunks)"
                    traces.append(msg)
                    status_box.write(f"📥 Hybrid RRF search retrieved {len(docs)} chunks")

                elif node_name == "generate":
                    final_generation = output_dict.get("generation", "")
                    status_box.update(label="Response generated", state="complete", expanded=False)

        st.markdown(final_generation)

        docs_summary = []
        if final_docs:
            with st.expander("📚 View Cited Sources"):
                for idx, doc in enumerate(final_docs):
                    ch = doc.metadata.get("chapter", "General Content")
                    pg = doc.metadata.get("page_number", doc.metadata.get("page", 0) + 1)
                    docs_summary.append({
                        "chapter": ch,
                        "page": pg,
                        "text": doc.page_content
                    })
                    st.caption(f"**Source {idx+1} — {ch} (Page {pg})**")
                    st.text(doc.page_content[:300] + "...")

    # Save Assistant Message
    assistant_record = {
        "role": "assistant",
        "content": final_generation,
        "traces": traces,
        "docs": docs_summary
    }
    st.session_state.messages.append(assistant_record)

    # Determine thread title (uses the first question asked in this thread)
    thread_title = st.session_state.messages[0]["content"] if st.session_state.messages else "New Conversation"

    # Persist the thread and messages to SQLite
    save_thread_messages(
        thread_id=st.session_state.active_thread_id,
        file_hash=st.session_state.active_file_hash,
        title=thread_title,
        messages=st.session_state.messages
    )
    st.rerun()