"""Rewrite user queries for semantic / embedding search."""

import sys
from pathlib import Path

from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_client import get_client


MODEL_NAME = "gemini-3.5-flash-lite"

REWRITE_PROMPT = """\
You optimize search queries for semantic vector retrieval (dense embeddings).

Given a user query, rewrite it into ONE clear, self-contained question or statement that:
- preserves the user's exact intent
- removes ambiguity, filler words, and conversational noise
- uses precise, domain-neutral wording an embedding model can match well
- stays concise (one or two sentences at most)

Rules:
- Do not invent facts or narrow the scope beyond what the user asked.
- Do not add keyword lists — write natural language suited for meaning-based search.
- Do not explain your reasoning.
- Return ONLY the rewritten query — no quotes, labels, or markdown.
"""


def _clean_llm_output(text: str) -> str:
    cleaned = text.strip().strip('"').strip("'")
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").strip()
        if cleaned.lower().startswith("text"):
            cleaned = cleaned[4:].strip()
        cleaned = cleaned.removesuffix("```").strip()
    return cleaned


def rewrite_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("rewrite_query expects a non-empty query string.")

    client = get_client()
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=f"{REWRITE_PROMPT}\n\nUser query:\n{query.strip()}",
        config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=256),
    )
    rewritten = _clean_llm_output(response.text or "")
    if not rewritten:
        raise RuntimeError("Query rewriting returned an empty response.")
    return rewritten


if __name__ == "__main__":
    query = "how many of those drones minimum should be used for them tasks my man"
    print(f"Original: {query}")
    print(f"Rewritten: {rewrite_query(query)}")
