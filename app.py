"""
app.py
Production Academic Research Agent UI Entrypoint
"""
import sys
from pathlib import Path

import streamlit as st

from src.auth import render_login_barrier
from src.config import get_langfuse_handler
from src.database import init_db, save_thread_messages
from src.services.agent_service import setup_agent
from ui.components import apply_custom_css, render_starter_suggestions
from ui.sidebar import render_sidebar

# --- Enforce Authentication Gatekeeper ---
if not render_login_barrier():
  st.stop()

# --- Initialize Core Storage ---
init_db()

# --- Page Setup & Styles ---
st.set_page_config(
    page_title="Academic Research Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_custom_css()

# --- State Initialization ---
if "active_file_hash" not in st.session_state:
  st.session_state.active_file_hash = None
if "active_thread_id" not in st.session_state:
  st.session_state.active_thread_id = None
if "messages" not in st.session_state:
  st.session_state.messages = []

# --- Render Sidebar & Retrieve Active Document List ---
registered_docs = render_sidebar()

# --- Load Active Agent Engine ---
active_doc_record = next(
    d
    for d in registered_docs
    if d["file_hash"] == st.session_state.active_file_hash
)
agent, metadata_profile, chunk_count = setup_agent(
    active_doc_record["filepath"], active_doc_record["file_hash"]
)

# --- Main Work Area ---
st.title("🎓 Academic Research Assistant")

# Document Header Bar
st.markdown(
    f"""
    <div class="metric-card">
        <b>📄 Current Document:</b> {active_doc_record['filename']} &nbsp;|&nbsp;
        <b>Total Chunks:</b> {chunk_count} &nbsp;|&nbsp;
        <b>Thread ID:</b> <code>{st.session_state.active_thread_id[:8]}</code>
    </div>
""",
    unsafe_allow_html=True,
)

# Starter Suggestion Chips
prompt_to_send = None
if len(st.session_state.messages) == 0:
  prompt_to_send = render_starter_suggestions()

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
          st.caption(
              f"**Source {idx+1} — {doc_info['chapter']} (Page"
              f" {doc_info['page']})**"
          )
          st.text(doc_info["text"][:300] + "...")

# Chat Input & Execution
user_input = st.chat_input("Ask a research question about the active document...")
query_to_execute = prompt_to_send or user_input

if query_to_execute:
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
          status_box.write(
              f"🔀 **Query Decomposition:** ({len(sqs)} search vectors)"
          )
          for q in sqs:
            status_box.write(f"   • *{q}*")

        elif node_name == "metadata_tool":
          docs = output_dict.get("documents", [])
          final_docs = docs
          msg = "Accessed front-matter and structural metadata profile"
          traces.append(msg)
          status_box.write("📋 Accessing title & chapter front-matter")

        elif node_name == "search_tool":
          docs = output_dict.get("documents", [])
          final_docs = docs
          msg = (
              f"Executed hybrid RRF search ({len(docs)} deduplicated chunks)"
          )
          traces.append(msg)
          status_box.write(
              f"📥 Hybrid RRF search retrieved {len(docs)} chunks"
          )

        elif node_name == "generate":
          final_generation = output_dict.get("generation", "")
          status_box.update(
              label="Response generated", state="complete", expanded=False
          )

    st.markdown(final_generation)

    docs_summary = []
    if final_docs:
      with st.expander("📚 View Cited Sources"):
        for idx, doc in enumerate(final_docs):
          ch = doc.metadata.get("chapter", "General Content")
          pg = doc.metadata.get("page_number", doc.metadata.get("page", 0) + 1)
          docs_summary.append(
              {"chapter": ch, "page": pg, "text": doc.page_content}
          )
          st.caption(f"**Source {idx+1} — {ch} (Page {pg})**")
          st.text(doc.page_content[:300] + "...")

  # Persist Assistant Output
  st.session_state.messages.append({
      "role": "assistant",
      "content": final_generation,
      "traces": traces,
      "docs": docs_summary,
  })

  thread_title = (
      st.session_state.messages[0]["content"]
      if st.session_state.messages
      else "New Conversation"
  )

  save_thread_messages(
      thread_id=st.session_state.active_thread_id,
      file_hash=st.session_state.active_file_hash,
      title=thread_title,
      messages=st.session_state.messages,
  )
  st.rerun()