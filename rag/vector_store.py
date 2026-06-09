"""Lightweight vector store abstraction for the RAG advisor.

This implementation keeps the dependency footprint small while still providing
production-friendly behavior: persistent documents, deterministic embeddings,
and metadata filters for privacy controls.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, List, Optional


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _hash_token(token: str, dimensions: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(digest, 16) % dimensions


def _vector_norm(values: List[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    numerator = sum(l * r for l, r in zip(left, right))
    denominator = _vector_norm(left) * _vector_norm(right)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _embed_text(text: str, dimensions: int = 128) -> List[float]:
    vector = [0.0] * dimensions
    for token in _tokenize(text):
        vector[_hash_token(token, dimensions)] += 1.0
    return vector


@dataclass
class VectorRecord:
    id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class InMemoryVectorStore:
    """Simple persistent vector store with metadata filters."""

    def __init__(self, storage_path: str | None = None, dimensions: int = 128):
        self.storage_path = storage_path
        self.dimensions = dimensions
        self._records: Dict[str, VectorRecord] = {}
        self._persist_lock = threading.Lock()
        if storage_path and os.path.exists(storage_path):
            self.load()

    def upsert(self, record_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._records[record_id] = VectorRecord(
            id=record_id,
            text=text,
            embedding=_embed_text(text, self.dimensions),
            metadata=metadata or {},
        )

    def delete(self, record_id: str) -> None:
        self._records.pop(record_id, None)

    def query(
        self,
        text: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        query_embedding = _embed_text(text, self.dimensions)
        ranked: List[tuple[float, VectorRecord]] = []

        for record in self._records.values():
            if metadata_filter and not self._matches_filter(record.metadata, metadata_filter):
                continue
            score = _cosine_similarity(query_embedding, record.embedding)
            ranked.append((score, record))

        ranked.sort(key=lambda item: (item[0], item[1].id), reverse=True)
        results = []
        for score, record in ranked[:top_k]:
            if score <= 0:
                continue
            results.append(
                {
                    "id": record.id,
                    "text": record.text,
                    "metadata": record.metadata,
                    "score": round(float(score), 4),
                }
            )
        return results

    def persist(self) -> None:
        if not self.storage_path:
            return
        payload = [asdict(record) for record in self._records.values()]
        tmp_path = self.storage_path + ".tmp"
        with self._persist_lock:
            try:
                with open(tmp_path, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=True, indent=2)
                os.replace(tmp_path, self.storage_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise

    def load(self) -> None:
        if not self.storage_path or not os.path.exists(self.storage_path):
            return
        with open(self.storage_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self._records = {
            item["id"]: VectorRecord(
                id=item["id"],
                text=item["text"],
                embedding=item["embedding"],
                metadata=item.get("metadata", {}),
            )
            for item in payload
        }

    @staticmethod
    def _matches_filter(metadata: Dict[str, Any], metadata_filter: Dict[str, Any]) -> bool:
        for key, expected in metadata_filter.items():
            if metadata.get(key) != expected:
                return False
        return True