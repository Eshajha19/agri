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

from .retriever import get_retriever

logger = logging.getLogger(__name__)

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
        "Answer the farmer's question using ONLY the research-backed context provided below. "
        "Be concise, practical, and use simple language. "
        "Cite sources by their [Source N] number where relevant. "
        "If the context does not fully address the question, say so and suggest consulting "
        "a local Krishi Vigyan Kendra (KVK) or agricultural extension officer.\n\n"
        f"### Farmer's Question\n{query}\n\n"
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

def _synthesise_fallback(docs: list[dict]) -> str:
    """Concatenate retrieved document contents as a plain-text answer."""
    paragraphs = [f"[{i}] {doc['content']}" for i, doc in enumerate(docs, 1)]
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
    docs = retriever.retrieve(query, top_k=top_k)

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
        }

    # Attempt Gemini synthesis; fall back to concatenation on failure / missing key
    llm_answer = _synthesise_with_gemini(query, docs)
    if llm_answer is not None:
        answer = llm_answer
        llm_used = True
    else:
        answer = _synthesise_fallback(docs)
        llm_used = False

    citations = [
        {
            "index": i,
            "title": doc["title"],
            "citation": doc["citation"],
            "source": doc["source"],
            "year": doc["year"],
            "topic": doc["topic"],
            "relevance": doc["relevance_score"],
        }
        for i, doc in enumerate(docs, 1)
    ]

    return {
        "answer": answer,
        "citations": citations,
        "sources_used": len(docs),
        "llm_used": llm_used,
    }
