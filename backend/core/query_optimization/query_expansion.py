"""Expand user queries with extra keywords for BM25 / full-text search."""

import sys
from pathlib import Path

from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_client import get_client


MODEL_NAME = "gemini-3.5-flash-lite"

EXPANSION_PROMPT = """\
You optimize search queries for BM25 keyword retrieval over technical documents.

Given a user query, produce ONE expanded query string that improves recall by adding:
- synonyms and common alternate spellings
- acronyms and their expanded forms (both directions)
- closely related terms likely to appear in the source text
- concrete nouns that clarify vague wording

Rules:
- Keep the original intent and all important original terms.
- Prefer short keyword phrases separated by spaces, not full sentences.
- Do not add unrelated topics.
- Do not explain your reasoning.
- Return ONLY the expanded query string — no quotes, labels, or markdown.
"""


def _clean_llm_output(text: str) -> str:
    cleaned = text.strip().strip('"').strip("'")
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").strip()
        if cleaned.lower().startswith("text"):
            cleaned = cleaned[4:].strip()
        cleaned = cleaned.removesuffix("```").strip()
    return cleaned


def expand_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("expand_query expects a non-empty query string.")

    client = get_client()
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=f"{EXPANSION_PROMPT}\n\nUser query:\n{query.strip()}",
        config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=256),
    )
    expanded = _clean_llm_output(response.text or "")
    if not expanded:
        raise RuntimeError("Query expansion returned an empty response.")
    return expanded


if __name__ == "__main__":
    query = "how many drones minimum should be used for the tasks"
    print(f"Original: {query}")
    print(f"Expanded: {expand_query(query)}")
