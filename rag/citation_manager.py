"""
Citation Manager — Citation Integrity & Source Tracking

Manages:
- Citation extraction and validation
- Source attribution and tracking
- Citation format standardization
- Circular reference detection
- Citation completeness verification
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import hashlib
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Represents a single citation/source attribution."""
    source_id: str  # Unique identifier for source
    title: str
    url: Optional[str] = None
    author: Optional[str] = None
    date_retrieved: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = 0.95  # Confidence score (0.0 to 1.0)
    section: Optional[str] = None  # Specific section cited
    relevance_score: Optional[float] = None
    
    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
    
    def to_dict(self) -> dict:
        """Convert citation to dictionary."""
        return {
            "source_id": self.source_id,
            "title": self.title,
            "url": self.url,
            "author": self.author,
            "date_retrieved": self.date_retrieved.isoformat(),
            "confidence": self.confidence,
            "section": self.section,
            "relevance_score": self.relevance_score,
        }
    
    def to_markdown(self) -> str:
        """Format citation as markdown."""
        if self.url:
            return f"[{self.title}]({self.url})"
        return f"{self.title}"
    
    def get_hash(self) -> str:
        """Get unique hash of this citation for deduplication."""
        content = f"{self.source_id}{self.title}{self.url}"
        return hashlib.sha256(content.encode()).hexdigest()[:8]


@dataclass
class CitationContext:
    """Context for a single answer/response."""
    response_id: str
    citations: list[Citation] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    query: Optional[str] = None
    response_text: Optional[str] = None
    
    def add_citation(self, citation: Citation) -> bool:
        """Add citation if not duplicate."""
        citation_hash = citation.get_hash()
        for existing in self.citations:
            if existing.get_hash() == citation_hash:
                logger.info(f"Duplicate citation detected: {citation.title}")
                return False
        
        self.citations.append(citation)
        return True
    
    def get_citations_for_claim(self, claim: str) -> list[Citation]:
        """Find citations relevant to specific claim."""
        claim_lower = claim.lower()
        relevant = []
        
        for citation in self.citations:
            if any(keyword in citation.title.lower() for keyword in claim_lower.split()):
                relevant.append(citation)
        
        return relevant


class CitationIntegrityChecker:
    """Validates citation integrity and completeness."""
    
    def __init__(self, min_confidence: float = 0.80, max_citations: int = 10):
        self.min_confidence = min_confidence
        self.max_citations = max_citations
        self.source_registry: dict[str, Citation] = {}
    
    def validate_citations(self, citations: list[Citation]) -> tuple[bool, list[str]]:
        issues = []
        
        if not citations:
            issues.append("No citations provided")
            return False, issues
        
        if len(citations) > self.max_citations:
            issues.append(f"Too many citations ({len(citations)} > {self.max_citations})")
        
        seen_hashes = set()
        for citation in citations:
            h = citation.get_hash()
            if h in seen_hashes:
                issues.append(f"Duplicate citation: {citation.title}")
            seen_hashes.add(h)
        
        for citation in citations:
            if citation.confidence < self.min_confidence:
                issues.append(
                    f"Low confidence citation: {citation.title} ({citation.confidence})"
                )
        
        for citation in citations:
            if not citation.source_id:
                issues.append("Citation missing source_id")
            if not citation.title:
                issues.append("Citation missing title")
        
        return len(issues) == 0, issues
    
    def detect_circular_references(self, citations: list[Citation]) -> list[tuple[str, str]]:
        circular = []
        
        graph: dict[str, set[str]] = {}
        for citation in citations:
            if citation.source_id not in graph:
                graph[citation.source_id] = set()
            
            for other in citations:
                if other.source_id != citation.source_id:
                    if other.source_id in citation.title.lower():
                        graph[citation.source_id].add(other.source_id)
        
        def has_cycle(node: str, visited: set, rec_stack: set) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    circular.append((node, neighbor))
            
            rec_stack.remove(node)
            return False
        
        visited = set()
        for node in graph:
            if node not in visited:
                has_cycle(node, visited, set())
        
        return circular
    
    def check_citation_coverage(self, response_text: str, citations: list[Citation]) -> tuple[float, list[str]]:
        uncited_topics = []
        topics = response_text.split()[:20]
        
        covered = 0
        for topic in topics:
            if any(topic.lower() in c.title.lower() for c in citations):
                covered += 1
            else:
                uncited_topics.append(topic)
        
        coverage_score = covered / len(topics) if topics else 0.0
        return coverage_score, uncited_topics
    
    def register_source(self, citation: Citation) -> None:
        self.source_registry[citation.source_id] = citation
    
    def get_source(self, source_id: str) -> Optional[Citation]:
        return self.source_registry.get(source_id)


class CitationManager:
    """
    Main citation manager.
    Manages citation lifecycle and integrity.
    """

    MAX_CONTEXTS = 1000

    def __init__(self, min_confidence: float = 0.80):
        self.integrity_checker = CitationIntegrityChecker(min_confidence=min_confidence)
        self.contexts: dict[str, CitationContext] = {}
        self.citation_index: dict[str, list[Citation]] = {}

    def create_context(self, response_id: str, query: str = "") -> CitationContext:
        """Create new citation context for response, evicting oldest if over cap."""
        if len(self.contexts) >= self.MAX_CONTEXTS:
            oldest_id = next(iter(self.contexts))
            del self.contexts[oldest_id]
            logger.warning(
                "CitationManager.contexts reached cap (%d); evicted oldest context: %s",
                self.MAX_CONTEXTS,
                oldest_id,
            )
        context = CitationContext(response_id=response_id, query=query)
        self.contexts[response_id] = context
        return context
    
    def add_citation(self, response_id: str, citation: Citation) -> bool:
        if response_id not in self.contexts:
            logger.error(f"Context {response_id} not found")
            return False
        
        added = self.contexts[response_id].add_citation(citation)
        
        if added:
            if citation.source_id not in self.citation_index:
                self.citation_index[citation.source_id] = []
            self.citation_index[citation.source_id].append(citation)
            self.integrity_checker.register_source(citation)
        
        return added
    
    def validate_response_citations(self, response_id: str) -> tuple[bool, list[str]]:
        if response_id not in self.contexts:
            return False, ["Response context not found"]
        
        context = self.contexts[response_id]
        return self.integrity_checker.validate_citations(context.citations)
    
    def check_citation_integrity(self, response_id: str) -> dict:
        if response_id not in self.contexts:
            return {
                "is_valid": False,
                "issues": ["Response context not found"],
                "circular_refs": [],
                "coverage_score": 0.0,
                "uncited_topics": [],
            }
        
        context = self.contexts[response_id]
        citations = context.citations
        
        is_valid, issues = self.integrity_checker.validate_citations(citations)
        
        circular_refs = self.integrity_checker.detect_circular_references(citations)
        if circular_refs:
            is_valid = False
            issues.append(f"Circular references detected: {circular_refs}")
        
        coverage_score, uncited = (
            self.integrity_checker.check_citation_coverage(context.response_text or "", citations)
            if context.response_text
            else (0.0, [])
        )
        
        return {
            "is_valid": is_valid,
            "issues": issues,
            "circular_refs": circular_refs,
            "coverage_score": coverage_score,
            "uncited_topics": uncited,
        }
    
    def format_citations(self, response_id: str, format_style: str = "markdown") -> str:
        if response_id not in self.contexts:
            return ""
        
        citations = self.contexts[response_id].citations
        
        if format_style == "markdown":
            lines = ["## Sources\n"]
            for i, citation in enumerate(citations, 1):
                lines.append(f"{i}. {citation.to_markdown()}")
                if citation.author:
                    lines.append(f"   Author: {citation.author}")
                if citation.confidence < 0.95:
                    lines.append(f"   Confidence: {citation.confidence:.1%}")
            return "\n".join(lines)
        
        elif format_style == "json":
            return json.dumps([c.to_dict() for c in citations], indent=2, default=str)
        
        elif format_style == "html":
            html = ["<div class='citations'>\n<h3>Sources</h3>\n<ol>\n"]
            for citation in citations:
                if citation.url:
                    html.append(f"<li><a href='{citation.url}'>{citation.title}</a></li>\n")
                else:
                    html.append(f"<li>{citation.title}</li>\n")
            html.append("</ol>\n</div>")
            return "".join(html)
        
        return ""
    
    def get_context(self, response_id: str) -> Optional[CitationContext]:
        return self.contexts.get(response_id)
    
    def cleanup_context(self, response_id: str) -> None:
        if response_id in self.contexts:
            del self.contexts[response_id]
            logger.info(f"Cleaned up context: {response_id}")