"""Production-style RAG advisor service with privacy controls."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .safety import RAGSafetyValidator
from .vector_store import InMemoryVectorStore

logger = logging.getLogger(__name__)


PII_PATTERNS = [
    (re.compile(r"\b\d{10}\b"), "[redacted-phone]"),
    (re.compile(r"\b\d{12}\b"), "[redacted-id]"),
    (re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}"), "[redacted-email]"),
    (re.compile(r"\b(?:\+?91[-\s]?)?[6-9]\d{9}\b"), "[redacted-phone]"),
]


def redact_pii(text: str) -> str:
    redacted = text
    for pattern, replacement in PII_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def has_privacy_opt_out(metadata: Optional[Dict[str, Any]]) -> bool:
    if not metadata:
        return False
    tags = metadata.get("tags") or []
    privacy = metadata.get("privacy") or {}
    return bool(privacy.get("opt_out") or "no_index" in tags or "private" in tags)


@dataclass
class AdvisorDocument:
    id: str
    text: str
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAGAdvisorService:
    """RAG advisor that ingests docs, retrieves context, and generates answers."""

    def __init__(self, store: InMemoryVectorStore | None = None):
        self.store = store or InMemoryVectorStore()
        self.safety = RAGSafetyValidator()

    def ingest(self, documents: Iterable[AdvisorDocument]) -> int:
        count = 0
        for document in documents:
            if has_privacy_opt_out(document.metadata):
                logger.info("Skipping opted-out document: %s", document.id)
                continue
            cleaned = redact_pii(document.text)
            metadata = dict(document.metadata)
            metadata["source"] = document.source
            metadata["privacy_redacted"] = cleaned != document.text
            self.store.upsert(document.id, cleaned, metadata=metadata)
            count += 1
        self.store.persist()
        return count

    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        safety_result = self.safety.validate_query(query_text)
        if not safety_result.is_safe:
            return {
                "answer": "",
                "citations": [],
                "sources_used": 0,
                "llm_used": False,
                "blocked": True,
                "reason": safety_result.details,
            }

        scoped, scope_reason = self.safety.is_rag_scoped(query_text)
        if not scoped:
            return {
                "answer": (
                    "This advisor is scoped to agricultural guidance. Please ask a crop, soil, "
                    "irrigation, pest, or farm management question."
                ),
                "citations": [],
                "sources_used": 0,
                "llm_used": False,
                "blocked": False,
                "reason": scope_reason,
            }

        matches = self.store.query(query_text, top_k=top_k)
        if not matches:
            return {
                "answer": (
                    "I could not find a strong match in the indexed knowledge base. "
                    "Please consult your local agricultural extension officer or KVK."
                ),
                "citations": [],
                "sources_used": 0,
                "llm_used": False,
                "blocked": False,
            }

        answer_parts = []
        citations = []
        for index, match in enumerate(matches, start=1):
            answer_parts.append(f"[{index}] {match['text']}")
            citations.append(
                {
                    "index": index,
                    "id": match["id"],
                    "source": match["metadata"].get("source", "unknown"),
                    "score": match["score"],
                    "privacy_redacted": match["metadata"].get("privacy_redacted", False),
                }
            )

        return {
            "answer": " ".join(answer_parts),
            "citations": citations,
            "sources_used": len(matches),
            "llm_used": False,
            "blocked": False,
        }
