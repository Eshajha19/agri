"""
RAG Retriever — TF-IDF based document retrieval using scikit-learn.
No external vector DB required.
"""
import threading

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .knowledge_base import KNOWLEDGE_BASE


class RAGRetriever:
    def __init__(self):
        self.docs = KNOWLEDGE_BASE
        # Build corpus: title + content + tags joined
        corpus = [
            f"{d['title']} {d['content']} {' '.join(d['tags'])}"
            for d in self.docs
        ]
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 3) -> dict:
        """
        Return top_k most relevant knowledge base entries along with
        confidence and retrieval metadata.
        """

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        match_scores = []

        for idx in top_indices:
            if scores[idx] > 0.01:  # minimum relevance threshold
                score = round(float(scores[idx]), 4)

                if score >= 0.75:
                    confidence_level = "high"
                    confidence_explanation = (
                        "Strong semantic match with highly relevant content."
                    )
                elif score >= 0.40:
                    confidence_level = "medium"
                    confidence_explanation = (
                        "Moderately relevant content matched the query."
                    )
                else:
                    confidence_level = "low"
                    confidence_explanation = (
                        "Weak match returned due to limited relevant content."
                    )

                entry = dict(self.docs[idx])

                entry["relevance_score"] = score
                entry["confidence_level"] = confidence_level
                entry["confidence_explanation"] = confidence_explanation

                results.append(entry)
                match_scores.append(score)

        if match_scores:
            top_match_score = max(match_scores)

            average_match_score = round(
                sum(match_scores) / len(match_scores),
                4,
            )

            if top_match_score >= 0.75:
                retrieval_quality = "high"
                retrieval_summary = (
                    "Strong retrieval confidence across matched documents."
                )
            elif top_match_score >= 0.40:
                retrieval_quality = "medium"
                retrieval_summary = (
                    "Moderate retrieval confidence with relevant matches."
                )
            else:
                retrieval_quality = "low"
                retrieval_summary = (
                    "Low retrieval confidence; consider refining the query."
                )
        else:
            top_match_score = 0.0
            average_match_score = 0.0
            retrieval_quality = "none"
            retrieval_summary = (
                "No sufficiently relevant documents were retrieved."
            )

        retrieval_metadata = {
            "query": query,
            "requested_results": top_k,
            "matched_documents": len(results),
            "minimum_relevance_threshold": 0.01,
            "top_match_score": round(top_match_score, 4),
            "average_match_score": average_match_score,
            "retrieval_quality": retrieval_quality,
            "retrieval_summary": retrieval_summary,
            "confidence_generated": True,
            "confidence_distribution": {
                "high": sum(
                    1 for r in results
                    if r["confidence_level"] == "high"
                ),
                "medium": sum(
                    1 for r in results
                    if r["confidence_level"] == "medium"
                ),
                "low": sum(
                    1 for r in results
                    if r["confidence_level"] == "low"
                ),
            },
        }

        return {
            "results": results,
            "retrieval_metadata": retrieval_metadata,
        }

# Singleton instance — loaded once at startup
_retriever_instance: RAGRetriever | None = None
_retriever_lock = threading.Lock()


def get_retriever() -> RAGRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        with _retriever_lock:
            if _retriever_instance is None:
                _retriever_instance = RAGRetriever()
    return _retriever_instance