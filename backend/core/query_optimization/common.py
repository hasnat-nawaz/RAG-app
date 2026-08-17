"""Shared helpers for query optimization prompts and LLM output cleanup."""

import bootstrap

NO_QUERY = 'NO_QUERY'
MIN_HYPOTHETICAL_DOC_CHARS = 40


def is_no_query(text: str) -> bool:
    """Return True when the model refused to produce an optimized query."""
    return text.strip().upper() == NO_QUERY


def with_user_input_tags(query: str) -> str:
    """Wrap user text so prompts can treat it as untrusted data."""
    return f'<user_input>\n{query.strip()}\n</user_input>'


def fallback_to_original(original: str, optimized: str) -> str:
    """Use the original query when optimization fails or returns NO_QUERY."""
    if is_no_query(optimized) or not optimized.strip():
        return original.strip()
    return optimized


def clean_llm_output(text: str) -> str:
    """Strip fences and surrounding quotes from raw model output."""
    cleaned = text.strip().strip('"').strip("'")
    if cleaned.startswith('```markdown'):
        cleaned = cleaned.removeprefix('```markdown').strip()
    if cleaned.startswith('```'):
        cleaned = cleaned.removeprefix('```').strip()
        if cleaned.lower().startswith('text'):
            cleaned = cleaned[4:].strip()
    if cleaned.endswith('```'):
        cleaned = cleaned.removesuffix('```').strip()
    return cleaned
