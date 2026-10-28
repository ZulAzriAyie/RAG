"""
src/indexing/hybrid_retriever.py
Hybrid retrieval engine combining BM25 keyword search with Chroma vector search
via Reciprocal Rank Fusion (RRF) and page diversity constraints.
"""

from pathlib import Path
from typing import List, Any, Union
from pydantic import Field
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


class SimpleHybridRetriever(BaseRetriever):
    """Combines BM25 and Chroma vector retrieval using Reciprocal Rank Fusion (RRF)."""

    bm25: Any = Field(description="BM25 keyword retriever")
    chroma: Any = Field(description="Chroma vector retriever")
    k: int = Field(default=8, description="Number of documents to return")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        # 1. Fetch candidates from both search mechanisms
        bm25_docs = self.bm25.invoke(query)
        chroma_docs = self.chroma.invoke(query)

        # 2. Score candidates using Reciprocal Rank Fusion (RRF)
        scores = {}
        doc_map = {}
        rrf_c = 60

        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content
            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + (1.0 / (rrf_c + rank + 1))

        for rank, doc in enumerate(chroma_docs):
            key = doc.page_content
            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + (1.0 / (rrf_c + rank + 1))

        # 3. Sort by aggregated score
        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # 4. Enforce page diversity (max 2 chunks per page)
        final_docs = []
        page_counts = {}
        for key in sorted_keys:
            doc = doc_map[key]
            page_num = doc.metadata.get("page_number", 0)
            if page_counts.get(page_num, 0) < 2:
                final_docs.append(doc)
                page_counts[page_num] = page_counts.get(page_num, 0) + 1
            if len(final_docs) >= self.k:
                break

        return final_docs


def build_hybrid_retriever(
    chunks: List[Document],
    chroma_dir: Path,
    collection_name: str,
    embedding_model: Union[str, Embeddings] = "nomic-embed-text",
) -> SimpleHybridRetriever:
    """Initializes or loads Chroma vector store and builds the hybrid retriever."""
    # Check whether a string name or pre-initialized Embeddings object was passed
    if isinstance(embedding_model, str):
        embeddings = OllamaEmbeddings(model=embedding_model)
    else:
        embeddings = embedding_model

    # Initialize Chroma store
    vector_store = Chroma(
        persist_directory=str(chroma_dir),
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    # Populate vector store if newly created
    if not (chroma_dir.exists() and any(chroma_dir.iterdir())):
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            vector_store.add_documents(chunks[i : i + batch_size])

    chroma_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20},
    )

    # Initialize BM25 keyword index
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 8

    return SimpleHybridRetriever(bm25=bm25_retriever, chroma=chroma_retriever, k=8)