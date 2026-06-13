"""
RAG Safety Layer — Prompt Injection Defense + Query Validation

Detects and prevents:
- Prompt injection attacks (role override, data exfiltration)
- Jailbreak attempts
- Malformed or suspicious queries
- Out-of-scope requests
"""

import re
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity classification."""
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class SafetyResult:
    """Result of safety analysis on a query."""
    is_safe: bool
    threat_level: ThreatLevel
    details: str
    threat_detected: str | None = None
    remediation: str | None = None


class PromptInjectionDetector:
    """Detects common prompt injection attack patterns."""

    INJECTION_KEYWORDS = [
        r"ignore.*previous",
        r"ignore.*your",
        r"ignore.*instruction",
        r"forget.*instruction",
        r"disregard.*prompt",
        r"override.*system",
        r"bypass.*security",
        r"system.*break",
        r"role.*play.*as",
        r"pretend.*to.*be",
        r"act.*as.*if",
        r"simulate.*being",
        r"assume.*role",
        r"you.*are.*now",
        r"from.*now.*on",
        r"new.*instructions",
        r"execute.*command",
        r"run.*code",
        r"eval",
        r"exec",
        r"__import__",
        r"import.*os",
        r"import.*sys",
    ]

    EXFILTRATION_PATTERNS = [
        r"show.*password",
        r"reveal.*secret",
        r"dump.*database",
        r"export.*data",
        r"list.*all.*users",
        r"get.*admin",
        r"backdoor",
        r"api.*key",
        r"auth.*token",
    ]

    JAILBREAK_PATTERNS = [
        r"do.*anything.*now",
        r"without.*restrictions",
        r"no.*ethical",
        r"ignore.*safety",
        r"harmful.*but",
        r"pretend.*safety",
    ]

    # ── Heuristic signals (paraphrase-resistant word lists) ──────────────
    # These catch paraphrased / reworded variants that the keyword regexes
    # would miss.

    INSTRUCTION_OVERRIDE_WORDS = {
        "forget", "ignore", "skip", "disregard", "overlook", "neglect",
        "bypass", "dismiss", "erase", "delete", "remove", "clear",
        "reset", "restart", "undo", "drop", "cancel", "abort",
        "don't follow", "do not follow", "stop following",
        "don't listen", "do not listen", "stop listening",
        "don't obey", "do not obey", "stop obeying",
        "don't answer", "do not answer",
        "don't respond", "do not respond",
        "new instruction", "different instruction",
        "change your", "alter your", "modify your",
        "switch your", "update your",
    }

    ROLE_PLAY_WORDS = {
        "act as", "behave as", "behave like", "act like",
        "pretend to", "pretend you", "imagine you",
        "you are a", "you're a", "you are an", "you're an",
        "become", "transform", "convert", "turn into",
        "you are now", "you're now", "now you are",
        "your new role", "your new identity",
        "speak as", "talk as", "respond as", "reply as",
        "you will act", "you will behave",
        "you will pretend", "you will become",
        "take on the role", "assume the identity",
        "step into the shoes",
    }

    DATA_EXFIL_WORDS = {
        "passwords", "secrets", "credentials", "tokens",
        "admin panel", "admin url", "admin page",
        "internal", "confidential", "classified",
        "private key", "private data", "private info",
        "customer data", "user data", "user list",
        "all users", "all records", "all entries",
        "access key", "access secret",
        "connection string", "endpoint",
        "show me", "tell me", "give me", "output",
        "print", "display", "leak", "expose", "reveal",
        "dump", "extract", "pull", "fetch", "retrieve",
    }

    JAILBREAK_WORDS = {
        "unrestricted", "unfiltered", "uncensored", "unlimited",
        "no restrictions", "no limits", "no boundaries",
        "no rules", "no constraints", "no barriers",
        "anything you want", "whatever you want",
        "without limits", "without boundaries",
        "without rules", "without constraints",
        "do anything", "say anything", "answer anything",
        "full access", "complete access", "total access",
        "you are free", "you can do", "you may do",
        "i have permission", "i have authorization",
        "for research", "for educational",
        "hypothetical", "fictional", "simulation",
        "in a test", "in a lab", "in a sandbox",
        "don't worry", "it's safe", "it's okay",
        "this is for", "this is a",
        "i need you to", "i want you to",
        "can you please", "could you please",
    }

    # ── Sensitivity → threshold map ──────────────────────────────────────
    #  heuristic_score >= threshold is flagged
    THRESHOLDS = {
        "low":     0.85,
        "medium":  0.65,
        "high":    0.45,
    }

    def __init__(self, sensitivity: str = "medium"):
        self.sensitivity = sensitivity
        self.compiled_injection = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_KEYWORDS]
        self.compiled_exfiltration = [re.compile(p, re.IGNORECASE) for p in self.EXFILTRATION_PATTERNS]
        self.compiled_jailbreak = [re.compile(p, re.IGNORECASE) for p in self.JAILBREAK_PATTERNS]

    # ── Heuristic scoring (paraphrase-resistant) ─────────────────────────

    def _heuristic_score(self, query: str) -> float:
        """Return a suspicion score in [0.0, 1.0] based on word-level signals.

        The score is an average of per-category signal densities so that a
        query hitting many signals in one category (e.g. role-play) scores
        as highly as a query spreading signals across categories.
        """
        words_lower = query.lower().split()
        text_lower = query.lower()

        def _signal_ratio(word_set: set[str]) -> float:
            """Fraction of query word-boundary signals present in *word_set*."""
            hits = sum(1 for signal in word_set if signal in text_lower)
            return hits / max(len(word_set), 1)

        # 1) instruction-override signal
        override_score = _signal_ratio(self.INSTRUCTION_OVERRIDE_WORDS)

        # 2) role-play signal
        role_score = _signal_ratio(self.ROLE_PLAY_WORDS)

        # 3) data-exfiltration signal
        exfil_score = _signal_ratio(self.DATA_EXFIL_WORDS)

        # 4) jailbreak signal
        jailbreak_score = _signal_ratio(self.JAILBREAK_WORDS)

        # 5) structural signal — commands (imperatives) without question words
        question_words = {"how", "what", "why", "when", "where", "which", "who", "is", "are", "can", "could", "would", "should", "do", "does"}
        command_verbs = {"tell", "give", "show", "provide", "output", "list", "print", "display", "send", "return", "write", "create", "generate", "produce", "do", "make", "change", "set", "update", "delete", "remove"}
        word_set = set(words_lower)
        has_question = bool(word_set & question_words)
        has_command = bool(word_set & command_verbs)
        # Imperative-heavy + no question words → suspicious
        structural_score = 0.0
        if has_command and not has_question:
            # Count how many command verbs present
            cmd_count = sum(1 for w in words_lower if w in command_verbs)
            structural_score = min(cmd_count * 0.15, 0.6)

        # 6) length / density bonus — very long queries with many signals
        density_bonus = 0.0
        if len(words_lower) > 50:
            density_bonus = 0.05
        if len(words_lower) > 100:
            density_bonus = 0.10

        # Weighted combination
        score = (
            override_score * 0.30 +
            role_score * 0.25 +
            exfil_score * 0.20 +
            jailbreak_score * 0.15 +
            structural_score * 0.10 +
            density_bonus
        )
        return min(score, 1.0)

    def detect_injection(self, query: str) -> tuple[bool, str | None]:
        """
        Detect prompt injection attempts — regex baseline + heuristic scoring.

        Returns:
            (is_injection, threat_type) - True if injection detected
        """
        query_lower = query.lower().strip()

        # Layer 1 — regex baseline (catches exact keyword hits)
        for pattern in self.compiled_injection:
            if pattern.search(query_lower):
                return True, "injection_keyword"

        for pattern in self.compiled_exfiltration:
            if pattern.search(query_lower):
                return True, "data_exfiltration"

        for pattern in self.compiled_jailbreak:
            if pattern.search(query_lower):
                return True, "jailbreak_attempt"

        # Layer 2 — heuristic scoring (catches paraphrased / reworded variants)
        score = self._heuristic_score(query)
        threshold = self.THRESHOLDS.get(self.sensitivity, 0.65)
        if score >= threshold:
            return True, "heuristic_injection"

        return False, None

    def detect_sql_injection(self, query: str) -> bool:
        sql_patterns = [
            r"'\s*or\s*'",
            r"'[^']*=[^']*'",
            r"\"[^\"]*or[^\"]*\"",
            r"\bunion\b.{0,50}select",
            r"\bdrop\b.{0,50}table",
            r"\binsert\b.{0,50}into",
            r"\bdelete\b.{0,50}from",
            r"\bupdate\b.{0,50}set",
            r"\bexec\b.{0,50}proc",
        ]
        for pattern in sql_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False

    def detect_command_injection(self, query: str) -> bool:
        command_patterns = [
            r";\s*(?:rm|del|drop|kill|stop)",
            r"\$\{[^}]*\}",
            r"\$\([^)]*\)",
            r"`[^`]*`",
            r"\|\s*(?:nc|ncat|curl|wget)",
        ]
        for pattern in command_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False

    def get_threat_level(self, query: str) -> ThreatLevel:
        if self.detect_injection(query)[0]:
            return ThreatLevel.CRITICAL

        if self.detect_sql_injection(query) or self.detect_command_injection(query):
            return ThreatLevel.CRITICAL

        if self.sensitivity == "high":
            if len(query) > 2000:
                return ThreatLevel.WARNING
            if query.count("\"") > 5 or query.count("'") > 5:
                return ThreatLevel.WARNING

        return ThreatLevel.SAFE


class RAGSafetyValidator:
    """
    Main safety validator for RAG queries and responses.
    Combines injection detection, validation, and safety checks.
    """

    def __init__(self, sensitivity: str = "medium", max_query_length: int = 2000, heuristic_threshold: float | None = None):
        """
        Initialize validator.
        
        Args:
            sensitivity: Detection sensitivity level
            max_query_length: Maximum allowed query length
            heuristic_threshold: Override heuristic score threshold (0.0–1.0).
                                 If None, uses sensitivity-based default.
        """
        self.injection_detector = PromptInjectionDetector(sensitivity)
        self.max_query_length = max_query_length
        self.sensitivity = sensitivity
        self._heuristic_threshold = heuristic_threshold

    def validate_query(self, query: str) -> SafetyResult:
        if not query or not isinstance(query, str):
            return SafetyResult(
                is_safe=False,
                threat_level=ThreatLevel.WARNING,
                details="Query must be non-empty string",
                threat_detected="invalid_input",
            )

        query = query.strip()

        if len(query) > self.max_query_length:
            return SafetyResult(
                is_safe=False,
                threat_level=ThreatLevel.WARNING,
                details=f"Query exceeds maximum length of {self.max_query_length} characters",
                threat_detected="length_exceeded",
            )

        if len(query) < 3:
            return SafetyResult(
                is_safe=False,
                threat_level=ThreatLevel.WARNING,
                details="Query too short (minimum 3 characters)",
                threat_detected="too_short",
            )

        # Injection detection — regex baseline + heuristic scoring
        is_injection, threat_type = self.injection_detector.detect_injection(query)
        if is_injection:
            heuristic_score = self.injection_detector._heuristic_score(query)
            logger.warning(
                "Injection detected: %s in query (heuristic=%.2f): %s",
                threat_type, heuristic_score, query[:80],
            )
            return SafetyResult(
                is_safe=False,
                threat_level=ThreatLevel.CRITICAL,
                details=f"Potential {threat_type} detected in query",
                threat_detected=threat_type,
                remediation="Query contains suspicious patterns. Please rephrase your question.",
            )

        if self.injection_detector.detect_sql_injection(query):
            logger.warning(f"SQL injection attempt detected")
            return SafetyResult(
                is_safe=False,
                threat_level=ThreatLevel.CRITICAL,
                details="SQL injection pattern detected",
                threat_detected="sql_injection",
                remediation="Query contains SQL patterns. Please use natural language.",
            )

        if self.injection_detector.detect_command_injection(query):
            logger.warning(f"Command injection attempt detected")
            return SafetyResult(
                is_safe=False,
                threat_level=ThreatLevel.CRITICAL,
                details="Command injection pattern detected",
                threat_detected="command_injection",
                remediation="Query contains command patterns. Please use natural language.",
            )

        threat_level = self.injection_detector.get_threat_level(query)

        return SafetyResult(
            is_safe=True,
            threat_level=threat_level,
            details="Query passed all safety checks",
            threat_detected=None,
        )

    # ── Citation marker pattern ───────────────────────────────────────────
    # Matches [Source N], [source N], (Source N), [N], (N), Source N only
    # when followed by punctuation/whitespace/end-of-string (not embedded in
    # numbers like "10:26:26").
    _CITATION_RE = re.compile(r"""
        \[(?:[Ss]ource\s+)?(\d+)\]
        |\((\d+)\)
        |(?<!\d)\b(?:[Ss]ource\s+)(\d+)\b(?!\d)
    """, re.VERBOSE)

    @staticmethod
    def _normalize_citation_markers(text: str) -> str:
        """Normalise all citation variants to ``[Source N]`` canonical form."""
        def _replacer(m: re.Match) -> str:
            num = next(g for g in m.groups() if g is not None)
            return f"[Source {num}]"
        return RAGSafetyValidator._CITATION_RE.sub(_replacer, text)

    @staticmethod
    def _extract_citation_indices(text: str) -> set[int]:
        """Return the set of source numbers cited in *text* (1‑based)."""
        indices: set[int] = set()
        for m in RAGSafetyValidator._CITATION_RE.finditer(text):
            num = int(next(g for g in m.groups() if g is not None))
            indices.add(num)
        return indices

    def validate_response(
        self,
        query: str,
        response: str,
        citations: list[dict] | None = None,
    ) -> SafetyResult:
        """
        Validate RAG response for information leakage and citation integrity.

        Args:
            query: Original user query
            response: Generated response
            citations: Structured citation metadata (list of dicts with at
                       least an ``"index"`` key). When provided the validator
                       checks that the response actually references at least
                       one source and that every referenced index exists.

        Returns:
            SafetyResult indicating if response is safe to return
        """
        # Check response length
        if len(response) > 10000:
            logger.warning("Response exceeds safe length")
            return SafetyResult(
                is_safe=False,
                threat_level=ThreatLevel.WARNING,
                details="Response too long (>10000 characters). May indicate data dump.",
                threat_detected="response_too_long",
            )

        # ── Citation integrity checks (structured metadata, not regex) ──
        cited: set[int] = set()
        if citations is not None:
            cited = self._extract_citation_indices(response)
            available = {c["index"] for c in citations}

            # Response must contain at least one citation marker
            if not cited:
                return SafetyResult(
                    is_safe=False,
                    threat_level=ThreatLevel.WARNING,
                    details=(
                        "Response lacks any citation markers. "
                        "Advice must reference the provided sources."
                    ),
                    threat_detected="missing_citations",
                    remediation="The generated response does not cite any sources.",
                )

            # Every referenced source must exist in the citation metadata
            out_of_range = cited - available
            if out_of_range:
                return SafetyResult(
                    is_safe=False,
                    threat_level=ThreatLevel.CRITICAL,
                    details=(
                        f"Response references non‑existent source indices: "
                        f"{sorted(out_of_range)}. Available: {sorted(available)}."
                    ),
                    threat_detected="invalid_citation_reference",
                    remediation="Generated response contains fabricated source references.",
                )

        # Check for system information leakage
        sensitive_patterns = [
            r"password",
            r"api.*key",
            r"secret.*key",
            r"private.*key",
            r"auth.*token",
            r"database.*url",
            r"config.*secret",
        ]
        for pattern in sensitive_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                logger.warning(f"Sensitive info pattern detected in response: {pattern}")
                return SafetyResult(
                    is_safe=False,
                    threat_level=ThreatLevel.CRITICAL,
                    details="Response contains sensitive information patterns",
                    threat_detected="info_leakage",
                    remediation="Response appears to contain sensitive data. Blocked for safety.",
                )

        return SafetyResult(
            is_safe=True,
            threat_level=ThreatLevel.SAFE,
            details="Response passed safety validation",
            threat_detected=None,
        )

    def sanitize_query(self, query: str) -> str:
        query = re.sub(r"['\";`$]{2,}", "", query)
        query = re.sub(r"(?:;\s*)?(?:rm|del|drop|kill)\s+", "", query, flags=re.IGNORECASE)
        query = " ".join(query.split())
        return query

    def is_rag_scoped(self, query: str) -> tuple[bool, str]:
        ag_keywords = [
            "crop", "farm", "soil", "irrigation", "fertilizer", "yield",
            "weather", "climate", "pest", "disease", "harvest", "season",
            "agriculture", "farming", "plant", "seed", "water", "rain",
            "temperature", "pesticide", "fertilization", "rotation",
            "yield", "produce", "grain", "rice", "wheat", "corn",
        ]

        query_lower = query.lower()
        found_keywords = sum(1 for kw in ag_keywords if kw in query_lower)

        if found_keywords >= 1:
            return True, "Query matches agricultural domain"

        general_indicators = [
            "how", "what", "why", "when", "where", "explain",
            "help", "guide", "recommend", "suggest", "calculate",
        ]

        has_indicator = any(query_lower.startswith(ind) for ind in general_indicators)

        if has_indicator:
            return True, "Query is general informational"

        return False, "Query appears outside agricultural domain"