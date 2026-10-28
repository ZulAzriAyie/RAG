"""
ui/sidebar.py
Sidebar rendering components for file management and conversation history.
"""

import hashlib
from pathlib import Path
import uuid
import streamlit as st

from src.database import (
    delete_thread,
    get_document_threads,
    get_registered_documents,
    register_document,
)

BASE_DIR = Path(__file__).parent.parent.resolve()
PDF_DIR = BASE_DIR / "data" / "raw_pdfs"


def render_sidebar():
  with st.sidebar:
    st.title("Documents")

    if st.button("Log Out", use_container_width=True):
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
          index=list(doc_options.keys()).index(
              st.session_state.active_file_hash
          ),
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

    # Thread Management
    st.subheader("Chat History")
    col_new_chat, _ = st.columns([1, 0.05])
    with col_new_chat:
      if st.button("➕ New Chat Thread", use_container_width=True):
        st.session_state.active_thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    threads = get_document_threads(st.session_state.active_file_hash)

    if not st.session_state.active_thread_id:
      if threads:
        st.session_state.active_thread_id = threads[0]["thread_id"]
        st.session_state.messages = threads[0]["messages"]
      else:
        st.session_state.active_thread_id = str(uuid.uuid4())
        st.session_state.messages = []

    if threads:
      for th in threads:
        col_th, col_del = st.columns([0.85, 0.15])
        is_active = th["thread_id"] == st.session_state.active_thread_id
        button_label = (
            f"• {th['title'][:26]}..."
            if len(th["title"]) > 26
            else f"• {th['title']}"
        )

        if col_th.button(
            button_label,
            key=f"btn_{th['thread_id']}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
          st.session_state.active_thread_id = th["thread_id"]
          st.session_state.messages = th["messages"]
          st.rerun()

        if col_del.button(
            "✕", key=f"del_{th['thread_id']}", help="Delete thread"
        ):
          delete_thread(th["thread_id"])
          if st.session_state.active_thread_id == th["thread_id"]:
            st.session_state.active_thread_id = str(uuid.uuid4())
            st.session_state.messages = []
          st.rerun()
    else:
      st.caption("No conversations yet for this document.")

    st.markdown("---")
    st.caption("**Engine Status**")
    st.markdown(
        """
        <span class="badge">LangGraph DAG</span>
        <span class="badge">Hybrid RRF</span>
        <span class="badge">SQLite Active</span>
    """,
        unsafe_allow_html=True,
    )

  return registered_docs