"""
RAG Generator — synthesises retrieved documents into a structured response with citations.

Synthesis strategy:
  1. If GEMINI_API_KEY is set, use the Gemini LLM to produce a context-aware,
     human-readable answer from the retrieved documents.
  2. If the key is absent or the API call fails, fall back to the original
     document-concatenation approach so the endpoint always returns a response.
"""

import logging
import os
import re

from .retriever import get_retriever
from .safety import RAGSafetyValidator

logger = logging.getLogger(__name__)

_safety = RAGSafetyValidator()

# ---------------------------------------------------------------------------
# Gemini client — initialised lazily so missing keys don't crash the import
# ---------------------------------------------------------------------------
_gemini_model = None


def _get_gemini_model():
    """Return a cached Gemini GenerativeModel, or None if unavailable."""
    global _gemini_model
    if _gemini_model is not None:
        return _gemini_model

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        logger.info(
            "GEMINI_API_KEY not set — RAG generator will use fallback concatenation mode."
        )
        return None

    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        logger.info("Gemini model initialised successfully (gemini-1.5-flash).")
        return _gemini_model
    except ImportError:
        logger.warning(
            "google-generativeai package not installed — falling back to concatenation mode."
        )
        return None
    except Exception as exc:
        logger.error("Failed to initialise Gemini model: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(query: str, docs: list[dict]) -> str:
    """Construct a structured prompt for Gemini from the query and retrieved docs."""
    context_blocks = []
    for i, doc in enumerate(docs, 1):
        context_blocks.append(
            f"[Source {i}] {doc['title']} ({doc['source']}, {doc['year']})\n"
            f"{doc['content']}"
        )

    context_text = "\n\n".join(context_blocks)

    prompt = (
        "You are Fasal Sathi, an expert agricultural advisor for Indian farmers. "
        "Answer the farmer's question enclosed inside the `<question>` tag using ONLY the research-backed context provided below. "
        "Be concise, practical, and use simple language. "
        "Cite sources by their [Source N] number where relevant. "
        "If the context does not fully address the question, say so and suggest consulting "
        "a local Krishi Vigyan Kendra (KVK) or agricultural extension officer.\n\n"
        f"### Farmer's Question\n<question>\n{query}\n</question>\n\n"
        f"### Research Context\n{context_text}\n\n"
        "### Answer"
    )
    return prompt


# ---------------------------------------------------------------------------
# LLM synthesis
# ---------------------------------------------------------------------------

def _synthesise_with_gemini(query: str, docs: list[dict]) -> str | None:
    """
    Call Gemini to synthesise an answer.
    Returns the answer string, or None if synthesis fails.
    """
    model = _get_gemini_model()
    if model is None:
        return None

    prompt = _build_prompt(query, docs)

    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,       # factual, low creativity
                "max_output_tokens": 512,
                "top_p": 0.9,
            },
        )
        answer = response.text.strip()
        if not answer:
            logger.warning("Gemini returned an empty response — falling back.")
            return None
        logger.info("Gemini synthesis successful (%d chars).", len(answer))
        return answer
    except Exception as exc:
        logger.error("Gemini API call failed: %s — falling back to concatenation.", exc)
        return None


# ---------------------------------------------------------------------------
# Fallback: original concatenation approach
# ---------------------------------------------------------------------------

def _sanitise_doc_content(raw: str) -> str:
    """Strip internal metadata, markup, control chars, and annotations."""
    text = re.sub(r"<!--.*?-->", "", raw, flags=re.DOTALL)
    text = re.sub(r"\[/?[a-z_]+\]", "", text)
    text = re.sub(r"\{\s*#\s*\w+\s*\}", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:2000]


def _synthesise_fallback(docs: list[dict]) -> str:
    """Concatenate sanitised document summaries as a plain-text answer."""
    paragraphs = []
    for i, doc in enumerate(docs, 1):
        content = _sanitise_doc_content(doc.get("content", ""))
        title = doc.get("title", "")
        if title:
            paragraphs.append(f"[{i}] {title}: {content}")
        else:
            paragraphs.append(f"[{i}] {content}")
    return " ".join(paragraphs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_response(query: str, top_k: int = 3) -> dict:
    """
    Retrieve relevant documents and synthesise a response.

    Returns a dict with keys:
      - answer        : str  — the synthesised (or fallback) answer
      - citations     : list — source metadata for each retrieved document
      - sources_used  : int  — number of documents used
      - llm_used      : bool — True if Gemini synthesis was used
    """

    retriever = get_retriever()

    retrieval_result = retriever.retrieve(
        query,
        top_k=top_k,
    )

    docs = retrieval_result["results"]
    retrieval_metadata = retrieval_result["retrieval_metadata"]

    if not docs:
        return {
            "answer": (
                "I could not find specific research-backed information for your query. "
                "Please consult your local Krishi Vigyan Kendra (KVK) or agricultural "
                "extension officer for personalised advice."
            ),
            "citations": [],
            "sources_used": 0,
            "llm_used": False,
            "retrieval_metadata": retrieval_metadata,
        }

    # Attempt Gemini synthesis; fall back to concatenation on failure / missing key
    llm_answer = _synthesise_with_gemini(query, docs)
    if llm_answer is not None:
        answer = llm_answer
        llm_used = True
    else:
        answer = _synthesise_fallback(docs)
        llm_used = False

    # Validate the response before returning — reject unsafe content.
    result = _safety.validate_response(query, answer)
    if not result.is_safe:
        logger.warning(
            "LLM response failed safety validation (threat=%s, detail=%s). Returning safe fallback.",
            result.threat_detected, result.details,
        )
        answer = (
            "I'm unable to provide a safe answer to this query right now. "
            "Please consult your local Krishi Vigyan Kendra (KVK) or "
            "agricultural extension officer for personalised advice."
        )

    citations = [
        {
            "rank": i,
            "index": i,
            "title": doc["title"],
            "citation": doc["citation"],
            "source": doc["source"],
            "year": doc["year"],
            "topic": doc["topic"],
            "relevance": doc["relevance_score"],
            "confidence_level": doc["confidence_level"],
            "confidence_explanation": doc["confidence_explanation"],
        }
        for i, doc in enumerate(docs, 1)
    ]

    return {
        "answer": answer,
        "citations": citations,
        "sources_used": len(docs),
        "llm_used": llm_used,
        "retrieval_metadata": retrieval_metadata,
    }
