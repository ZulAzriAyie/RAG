"""
Houses all prompt templates used across agent decision nodes.
"""

from langchain_core.prompts import ChatPromptTemplate

# 1. Grader Prompt: Evaluates retrieval quality with binary output
DOC_GRADER_PROMPT = ChatPromptTemplate.from_template(
    """You are an evaluator assessing whether the provided document context contains information to help answer the user's question.

Context:
{context}

Question:
{question}

Instructions:
- If the question asks for metadata (such as the author, title, institution, supervisor, date, or chapter titles), evaluate if title pages, front matter, or author names appear in the context.
- If the context contains even partial or direct answers (like a name or title), score 'yes'.
- Only score 'no' if the text is completely unrelated to the topic of the question.

Provide ONLY a binary score: 'yes' or 'no'.
Score:"""
)


# 2. Rewriter Prompt: Generates targeted search terms if initial search missed
QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """You are a research search specialist. An initial search query failed to retrieve relevant sections from the thesis.
Analyze the original query and the user's intent, then rephrase it into 3 to 6 precise academic search terms 
optimized for BM25 keyword matching and vector similarity.

Original Query:
{query}

Provide ONLY the rewritten search terms without explanation:"""
)

# 3. Answer Generation Prompt: Synthesizes final response strictly from verified context
from langchain_core.prompts import ChatPromptTemplate

RAG_GENERATE_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert research evaluator. Answer the question using ONLY the provided context excerpts.

Context:
{context}

Question: {question}

Instructions:
- If the question targets a specific chapter, section, or conclusion, synthesize information from the matching chapter headers in the excerpts.
- Cite specific metrics, findings, and statements verbatim when available.
- If the context does not contain sufficient details to answer, state clearly what is missing.

Answer:"""
)