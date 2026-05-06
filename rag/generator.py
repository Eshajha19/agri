"""
RAG Generator — synthesises retrieved documents into a structured response with citations.
"""
from .retriever import get_retriever


def generate_response(query: str, top_k: int = 3) -> dict:
    retriever = get_retriever()
    docs = retriever.retrieve(query, top_k=top_k)

    if not docs:
        return {
            "answer": "I could not find specific research-backed information for your query. Please consult your local Krishi Vigyan Kendra (KVK) or agricultural extension officer for personalised advice.",
            "citations": [],
            "sources_used": 0,
        }

    # Synthesise answer from retrieved docs
    paragraphs = []
    for i, doc in enumerate(docs, 1):
        paragraphs.append(f"[{i}] {doc['content']}")

    answer = " ".join(paragraphs)

    citations = [
        {
            "index": i,
            "title": doc["title"],
            "citation": doc["citation"],
            "source": doc["source"],
            "year": doc["year"],
            "topic": doc["topic"],
            "relevance": doc["relevance_score"],
        }
        for i, doc in enumerate(docs, 1)
    ]

    return {
        "answer": answer,
        "citations": citations,
        "sources_used": len(docs),
    }
