"""
RAG Module — Retrieval-Augmented Generation with Safety & Citations

Components:
- Retriever: TF-IDF based document retrieval
- Safety: Prompt injection defense and query validation
- Citations: Citation integrity and source tracking
- Generator: Response generation with citations
"""

from .retriever import RAGRetriever, get_retriever
from .safety import (
    PromptInjectionDetector,
    RAGSafetyValidator,
    SafetyResult,
    ThreatLevel,
)
from .citation_manager import (
    Citation,
    CitationContext,
    CitationIntegrityChecker,
    CitationManager,
)

__all__ = [
    # Retriever
    "RAGRetriever",
    "get_retriever",
    # Safety
    "PromptInjectionDetector",
    "RAGSafetyValidator",
    "SafetyResult",
    "ThreatLevel",
    # Citations
    "Citation",
    "CitationContext",
    "CitationIntegrityChecker",
    "CitationManager",
]
