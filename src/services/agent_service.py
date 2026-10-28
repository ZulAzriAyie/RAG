"""
src/services/agent_service.py
Service functions for bootstrapping embeddings, retrievers, LLMs, and compiled agent graphs.
"""

import os
from pathlib import Path
import sqlite3
from typing import Any, Tuple

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
import streamlit as st

from src.agent.graph import build_agent_graph
from src.database import DB_PATH
from src.indexing.hybrid_retriever import build_hybrid_retriever
from src.indexing.loader import extract_metadata_and_chunks

BASE_DIR = Path(__file__).parent.parent.parent.resolve()
CHROMA_BASE = BASE_DIR / "data" / "chroma_indices"


@st.cache_resource(
    show_spinner="Indexing document and compiling LangGraph state machine..."
)
def setup_agent(pdf_path_str: str, file_hash: str) -> Tuple[Any, Any, int]:
  pdf_path = Path(pdf_path_str)
  metadata_profile, chunks = extract_metadata_and_chunks(pdf_path)

  # Dynamic Embeddings Selection
  if os.getenv("GROQ_API_KEY"):
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
  else:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

  chroma_dir = CHROMA_BASE / f"chroma_{file_hash}"
  retriever = build_hybrid_retriever(
      chunks=chunks,
      chroma_dir=chroma_dir,
      collection_name=f"col_{file_hash}",
      embedding_model=embeddings,
  )

  # Dynamic LLM Selection
  if os.getenv("GROQ_API_KEY"):
    llm = ChatGroq(
        model_name="qwen/qwen3.8-27b",
        temperature=0.0,
        api_key=os.getenv("GROQ_API_KEY"),
    )
  else:
    llm = ChatOllama(
        model="qwen2.5:7b",
        temperature=0.0,
        num_ctx=4096,
    )

  conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
  checkpointer = SqliteSaver(conn)

  agent_app = build_agent_graph(
      retriever=retriever,
      metadata_profile=metadata_profile,
      llm=llm,
      checkpointer=checkpointer,
  )
  return agent_app, metadata_profile, len(chunks)