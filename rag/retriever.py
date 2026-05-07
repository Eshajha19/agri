"""
RAG Retriever — TF-IDF based document retrieval using scikit-learn.
No external vector DB required.
"""
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

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Return the top_k most relevant knowledge base entries for a query."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0.01:   # minimum relevance threshold
                entry = dict(self.docs[idx])
                entry["relevance_score"] = round(float(scores[idx]), 4)
                results.append(entry)
        return results


# Singleton instance — loaded once at startup
_retriever_instance: RAGRetriever | None = None


def get_retriever() -> RAGRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RAGRetriever()
    return _retriever_instance
