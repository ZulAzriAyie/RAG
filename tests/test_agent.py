"""
tests/test_agent.py
Automated testing suite evaluating intent routing and retrieval hit rate.
"""

import json
from pathlib import Path
import pytest
from langchain_ollama import ChatOllama

from src.indexing.loader import extract_metadata_and_chunks
from src.indexing.hybrid_retriever import build_hybrid_retriever
from src.agent.nodes import router_node, search_tool_node, decompose_node

# --- Test Fixtures ---
BASE_DIR = Path(__file__).parent.parent.resolve()
RAW_PDF_DIR = BASE_DIR / "data" / "raw_pdfs"
DATASET_PATH = BASE_DIR / "tests" / "eval_dataset.json"


@pytest.fixture(scope="session")
def golden_dataset():
    """Loads the golden dataset questions and expected benchmarks."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def setup_environment():
    """Initializes LLM, document indexing, and hybrid retriever once for the test session."""
    pdf_files = list(RAW_PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        pytest.skip("No PDF document located in data/raw_pdfs for testing.")

    test_pdf = pdf_files[0]
    metadata_profile, chunks = extract_metadata_and_chunks(test_pdf)

    chroma_dir = BASE_DIR / "data" / "chroma_indices" / "eval_test_store"
    retriever = build_hybrid_retriever(
        chunks=chunks,
        chroma_dir=chroma_dir,
        collection_name="eval_test_col",
        embedding_model="nomic-embed-text",
    )

    llm = ChatOllama(
        model="qwen2.5:7b",
        temperature=0.0,
        num_ctx=4096,
    )

    return {
        "llm": llm,
        "retriever": retriever,
        "metadata_profile": metadata_profile,
    }


# --- Test Cases ---

def test_intent_routing_accuracy(setup_environment, golden_dataset):
    """
    Validates that the Router node maps queries to the correct tool.
    Target: 100% accuracy on golden routing benchmarks.
    """
    llm = setup_environment["llm"]
    correct_routes = 0

    for item in golden_dataset:
        state = {
            "question": item["question"],
            "current_query": "",
            "sub_queries": [],
            "documents": [],
            "is_relevant": False,
            "generation": "",
            "retry_count": 0,
        }
        result = router_node(state, llm)
        actual_route = result["current_query"]

        print(f"\n[Routing Test] '{item['question'][:45]}...' -> Expected: {item['expected_route']} | Got: {actual_route}")
        if actual_route == item["expected_route"]:
            correct_routes += 1

    accuracy = correct_routes / len(golden_dataset)
    print(f"\n---> Routing Accuracy: {accuracy * 100:.1f}% ({correct_routes}/{len(golden_dataset)})")
    assert accuracy >= 0.83, f"Routing accuracy fell below threshold: {accuracy:.2f}"


def test_retrieval_ground_truth_recall(setup_environment, golden_dataset):
    """
    Evaluates whether the expected page numbers are captured in retrieved context.
    """
    llm = setup_environment["llm"]
    retriever = setup_environment["retriever"]

    tested_queries = 0
    passed_queries = 0

    for item in golden_dataset:
        if item["required_page"] is None or item["expected_route"] != "search":
            continue

        tested_queries += 1
        state = {
            "question": item["question"],
            "current_query": "search",
            "sub_queries": [item["question"]],
            "documents": [],
            "is_relevant": False,
            "generation": "",
            "retry_count": 0,
        }

        # Execute decomposition and search
        decomp = decompose_node(state, llm)
        state["sub_queries"] = decomp["sub_queries"]
        search_res = search_tool_node(state, retriever)

        retrieved_pages = [
            doc.metadata.get("page_number", doc.metadata.get("page", 0) + 1)
            for doc in search_res["documents"]
        ]

        target_page = item["required_page"]
        hit = target_page in retrieved_pages
        print(f"\n[Retrieval Test] Target Page {target_page} -> Found in: {retrieved_pages} | Hit: {hit}")

        if hit:
            passed_queries += 1

    recall = passed_queries / tested_queries if tested_queries > 0 else 1.0
    print(f"\n---> Retrieval Ground-Truth Recall: {recall * 100:.1f}% ({passed_queries}/{tested_queries})")
    assert recall >= 0.80, f"Retrieval recall fell below target threshold: {recall:.2f}"