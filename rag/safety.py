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

    def __init__(self, sensitivity: str = "medium"):
        self.sensitivity = sensitivity
        self.compiled_injection = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_KEYWORDS]
        self.compiled_exfiltration = [re.compile(p, re.IGNORECASE) for p in self.EXFILTRATION_PATTERNS]
        self.compiled_jailbreak = [re.compile(p, re.IGNORECASE) for p in self.JAILBREAK_PATTERNS]

    def detect_injection(self, query: str) -> tuple[bool, str | None]:
        query_lower = query.lower().strip()

        for pattern in self.compiled_injection:
            if pattern.search(query_lower):
                return True, "injection_keyword"

        for pattern in self.compiled_exfiltration:
            if pattern.search(query_lower):
                return True, "data_exfiltration"

        for pattern in self.compiled_jailbreak:
            if pattern.search(query_lower):
                return True, "jailbreak_attempt"

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

    # Citation marker pattern — [1], [2], etc.
    _CITATION_RE = re.compile(r"\[\d+\]")

    def __init__(self, sensitivity: str = "medium", max_query_length: int = 2000):
        self.injection_detector = PromptInjectionDetector(sensitivity)
        self.max_query_length = max_query_length
        self.sensitivity = sensitivity

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

        is_injection, threat_type = self.injection_detector.detect_injection(query)
        if is_injection:
            logger.warning(f"Injection detected: {threat_type} in query: {query[:100]}")
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

    def validate_response(self, query: str, response: str) -> SafetyResult:
        if len(response) > 10000:
            logger.warning("Response exceeds safe length")
            return SafetyResult(
                is_safe=False,
                threat_level=ThreatLevel.WARNING,
                details="Response too long (>10000 characters). May indicate data dump.",
                threat_detected="response_too_long",
            )

        # Block dosage recommendations only when they are NOT backed by a citation marker.
        # Cited responses (containing [1], [2], etc.) come from the verified knowledge base
        # and must not be blocked — doing so causes valid agricultural advice to be suppressed.
        ag_unsafe_patterns = [
            r"\d+\s*(?:ml|g|kg|l)\s*(?:per|/)\s*(?:acre|hectare|litre|plant)",
            r"apply\s+\d+\s*(?:ml|g|kg|l)\s+of\s+\w+",
        ]
        response_has_citation = bool(self._CITATION_RE.search(response))
        if not response_has_citation:
            for pattern in ag_unsafe_patterns:
                if re.search(pattern, response, re.IGNORECASE):
                    logger.warning("Unvalidated agricultural dosage recommendation detected in response")
                    return SafetyResult(
                        is_safe=False,
                        threat_level=ThreatLevel.WARNING,
                        details="Response contains unvalidated agricultural dosage recommendations",
                        threat_detected="unvalidated_ag_advice",
                        remediation="Agricultural dosages should not be provided without verification. Blocked for safety.",
                    )

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