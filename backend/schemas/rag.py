"""Shared request schema for RAG-style query routes."""

from html.parser import HTMLParser
import re

import bleach
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core import PydanticCustomError

_DANGEROUS_HTML_BLOCK_TAGS = {"script", "style", "iframe", "object", "embed"}


class QuerySanitizationError(ValueError):
    """Validation error with stable code/reason for sanitized query failures."""

    def __init__(
        self,
        error_code: str,
        message: str,
        reason: str,
        *,
        threat_type: str | None = None,
        threat_match: str | None = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.reason = reason
        self.threat_type = threat_type
        self.threat_match = threat_match

    def to_pydantic_error(self) -> PydanticCustomError:
        context = {
            "error_code": self.error_code,
            "error_message": self.message,
            "reason": self.reason,
        }
        if self.threat_type is not None:
            context["threat_type"] = self.threat_type
        if self.threat_match is not None:
            context["threat_match"] = self.threat_match

        return PydanticCustomError(
            "query_sanitization_error",
            self.message,
            context,
        )


class _DangerousHtmlBlockStripper(HTMLParser):
    """Remove dangerous block elements and their contents from a string."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._drop_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in _DANGEROUS_HTML_BLOCK_TAGS:
            self._drop_depth += 1

    def handle_startendtag(self, tag, attrs):
        if tag.lower() not in _DANGEROUS_HTML_BLOCK_TAGS:
            return

    def handle_endtag(self, tag):
        if tag.lower() in _DANGEROUS_HTML_BLOCK_TAGS and self._drop_depth:
            self._drop_depth -= 1

    def handle_data(self, data):
        if self._drop_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def _strip_dangerous_html_blocks(value: str) -> str:
    parser = _DangerousHtmlBlockStripper()
    parser.feed(value)
    parser.close()
    return parser.get_text()


def _rewrite_markdown_links_safe(text: str) -> str:
    """Rewrite well-formed markdown links as 'label (url)'.

    Uses balanced scanning for both [] and () so nested parentheses in URLs are
    supported. Malformed patterns are left unchanged.
    """
    result: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        if text[i] != "[":
            result.append(text[i])
            i += 1
            continue

        # Parse link text: [ ... ] with bracket-depth tracking
        j = i + 1
        bracket_depth = 1
        while j < n and bracket_depth > 0:
            if text[j] == "[":
                bracket_depth += 1
            elif text[j] == "]":
                bracket_depth -= 1
            j += 1

        if bracket_depth != 0:
            # Malformed opening bracket; keep source unchanged.
            result.append(text[i])
            i += 1
            continue

        close_bracket = j - 1
        if j >= n or text[j] != "(":
            # Not a markdown link pattern.
            result.append(text[i])
            i += 1
            continue

        # Parse URL: ( ... ) with parenthesis-depth tracking.
        k = j + 1
        paren_depth = 1
        while k < n and paren_depth > 0:
            if text[k] == "(":
                paren_depth += 1
            elif text[k] == ")":
                paren_depth -= 1
            k += 1

        if paren_depth != 0:
            # Malformed URL section; keep source unchanged.
            result.append(text[i])
            i += 1
            continue

        label = text[i + 1:close_bracket].strip()
        url = text[j + 1:k - 1].strip()

        if label and url:
            result.append(f"{label} ({url})")
        else:
            # Preserve malformed/empty pieces as-is.
            result.append(text[i:k])

        i = k

    return "".join(result)


def _normalize_for_injection_checks(value: str) -> str:
    """Normalize text for prompt-injection checks while preserving safe punctuation."""
    safe_punctuation = "#*"
    value = re.sub(rf"[^a-z0-9{re.escape(safe_punctuation)}]+", " ", value.lower())
    return re.sub(r"\s+", " ", value).strip()


_PROMPT_INJECTION_PHRASES = (
    "ignore previous instructions",
    "ignore prior instructions",
    "ignore prior msgs",
    "ignore prior messages",
    "ignore system prompt",
    "ignore the system prompt",
    "override system constraints",
    "developer mode",
    "bypass safety filter",
    "disregard prior instructions",
    "disregard all prior instructions",
    "act as different ai",
    "act as unrestricted ai",
    "act as unfiltered ai",
    "pretend you are different",
    "pretend to be different",
    "jailbreak",
    "prompt injection",
)


_PROMPT_INJECTION_TOKEN_SETS = (
    {"ignore", "previous", "instructions"},
    {"ignore", "prior", "instructions"},
    {"ignore", "prior", "msgs"},
    {"ignore", "prior", "messages"},
    {"ignore", "system", "prompt"},
    {"ignore", "the", "system", "prompt"},
    {"override", "system", "constraints"},
    {"developer", "mode"},
    {"bypass", "safety", "filter"},
    {"disregard", "prior", "instructions"},
    {"disregard", "all", "prior", "instructions"},
    {"act", "as", "different", "ai"},
    {"act", "as", "unrestricted", "ai"},
    {"act", "as", "unfiltered", "ai"},
    {"pretend", "you", "are", "different"},
    {"pretend", "to", "be", "different"},
    {"jailbreak"},
    {"prompt", "injection"},
)


def _prompt_injection_match(value: str) -> str | None:
    if any(phrase in value for phrase in _PROMPT_INJECTION_PHRASES):
        return "phrase_match"

    words = set(value.split())
    for token_set in _PROMPT_INJECTION_TOKEN_SETS:
        if token_set.issubset(words):
            return "token_cluster"

    suspicious_tokens = {"ignore", "override", "bypass", "disregard", "jailbreak", "pretend"}
    context_tokens = {"instructions", "instruction", "prompt", "system", "assistant", "developer", "mode", "msgs", "messages", "prior", "previous"}
    if bool(words & suspicious_tokens) and len(words & context_tokens) >= 2:
        return "heuristic_cluster"

    return None


class RAGQuery(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    query: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)

    @field_validator("query", mode="before")
    @classmethod
    def sanitize_and_normalize_query(cls, value):
        if not value or not isinstance(value, str):
            raise QuerySanitizationError(
                "invalid_query_type",
                "Query must be a non-empty string.",
                "non_string_or_empty_input",
            ).to_pydantic_error()

        value = _strip_dangerous_html_blocks(value)
        value = bleach.clean(
            value,
            tags=[],
            attributes={},
            protocols=[],
            strip=True,
            strip_comments=True,
        )
        value = re.sub(r"&lt;(?=\s)", "<", value)
        value = re.sub(r"&gt;(?=\s)", ">", value)
        value = _rewrite_markdown_links_safe(value)
        value = re.sub(r"\s+", " ", value.strip())

        normalized = _normalize_for_injection_checks(value)
        threat_match = _prompt_injection_match(normalized)
        if threat_match is not None:
            raise QuerySanitizationError(
                "disallowed_prompt_injection",
                "Query contains disallowed phrases or prompt injection attempts.",
                "prompt_injection_detected",
                threat_type="prompt_injection",
                threat_match=threat_match,
            ).to_pydantic_error()

        if len(value) < 3:
            raise QuerySanitizationError(
                "length_exceeded_after_sanitization",
                "Query must be at least 3 characters long after sanitization.",
                "min_length_not_met_after_sanitization",
            ).to_pydantic_error()

        return value
