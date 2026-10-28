"""
src/indexing/loader.py
Handles high-fidelity PDF loading, chapter auto-tagging, metadata extraction, and text splitting.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def auto_tag_chapters(docs: List[Document]) -> List[Document]:
    """
    Scans pages dynamically to detect chapter starts, titles, and appendices.
    Populates chapter metadata without relying on hardcoded page numbers.
    """
    chapter_pattern = re.compile(
        r"^(chapter\s+[0-9ivx]+[\s\:\-\.]*.*|appendix\s+[a-z0-9]+[\s\:\-\.]*.*)",
        re.IGNORECASE,
    )

    current_chapter = "Front Matter / Preliminary"

    for doc in docs:
        page_num = doc.metadata.get("page", 0) + 1
        doc.metadata["page_number"] = page_num

        lines = [line.strip() for line in doc.page_content.split("\n") if line.strip()]
        for line in lines[:6]:
            match = chapter_pattern.match(line)
            if match:
                current_chapter = line[:60].strip()
                break

        doc.metadata["chapter"] = current_chapter

    return docs


def extract_metadata_and_chunks(pdf_path: Path) -> Tuple[str, List[Document]]:
    """
    Loads a PDF, extracts front-matter metadata (pages 1-3 + detected chapters),
    and produces chunked documents for search.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF document not found at: {pdf_path}")

    # 1. Load document pages
    loader = PyMuPDFLoader(str(pdf_path))
    raw_docs = loader.load()

    # 2. Tag structural chapters
    tagged_docs = auto_tag_chapters(raw_docs)

    # 3. Extract Front Matter & Chapter Index for the metadata tool
    front_matter_pages = [d for d in tagged_docs if d.metadata.get("page_number", 0) <= 3]
    front_matter_text = "\n\n---\n\n".join(
        f"[Page {d.metadata.get('page_number')}]\n{d.page_content}"
        for d in front_matter_pages
    )

    detected_chapters = sorted(
        list(set(d.metadata.get("chapter", "") for d in tagged_docs if "chapter" in d.metadata.get("chapter", "").lower()))
    )
    chapter_summary = "\n".join(f"- {c}" for c in detected_chapters)

    metadata_profile = (
        f"=== THESIS METADATA & FRONT MATTER (PAGES 1-3) ===\n"
        f"{front_matter_text}\n\n"
        f"=== DETECTED DOCUMENT CHAPTERS ===\n"
        f"{chapter_summary}"
    )

    # 4. Split into manageable chunks for vector/BM25 search
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(tagged_docs)

    # 5. Inject structural context into chunk text
    for chunk in chunks:
        ch = chunk.metadata.get("chapter", "General Content")
        pg = chunk.metadata.get("page_number", "")
        chunk.page_content = f"[{ch} | Page {pg}]\n{chunk.page_content}"

    return metadata_profile, chunks