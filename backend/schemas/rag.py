"""Shared request schema for RAG-style query routes."""

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator


_DANGEROUS_HTML_BLOCK_RE = re.compile(
    r"<(script|style|iframe|object|embed)\b[^>]*>.*?</\1\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9:-]*(?:\s+[^<>]*?)?/?>")


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


class RAGQuery(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    query: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)

    @field_validator("query", mode="before")
    @classmethod
    def sanitize_and_normalize_query(cls, value):
        if not value or not isinstance(value, str):
            raise ValueError("Query must be a non-empty string.")

        value = _DANGEROUS_HTML_BLOCK_RE.sub("", value)
        value = re.sub(r"</?script.*?>", "", value, flags=re.IGNORECASE)
        value = re.sub(r"on\w+\s*=", "", value, flags=re.IGNORECASE)
        value = re.sub(r"javascript:", "", value, flags=re.IGNORECASE)
        value = re.sub(r"data:", "", value, flags=re.IGNORECASE)
        value = re.sub(r"vbscript:", "", value, flags=re.IGNORECASE)
        value = _HTML_TAG_RE.sub("", value)
        value = _rewrite_markdown_links_safe(value)
        value = re.sub(r"[*_~`#]", "", value)
        value = re.sub(r"\s+", " ", value.strip())

        forbidden_patterns = [
            r"ignore\s+(?:all\s+)?previous\s+instructions",
            r"ignore\s+(?:the\s+)?system\s+prompt",
            r"override\s+system\s+constraints",
            r"developer\s+mode",
            r"bypass\s+safety\s+filter",
            r"disregard\s+(?:all\s+)?prior\s+instructions",
            r"act\s+as\s+(?:a\s+)?(?:different|unrestricted|unfiltered)\s+(?:ai|model|assistant)",
            r"pretend\s+(?:you\s+are|to\s+be)\s+(?:a\s+)?(?:different|unrestricted)",
            r"jailbreak",
            r"prompt\s+injection",
        ]

        lowered = value.lower()
        for pattern in forbidden_patterns:
            if re.search(pattern, lowered):
                raise ValueError("Query contains disallowed phrases or prompt injection attempts.")

        if len(value) < 3:
            raise ValueError("Query must be at least 3 characters long after sanitization.")

        return value
