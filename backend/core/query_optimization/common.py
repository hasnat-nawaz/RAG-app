"""Shared helpers for query optimization steps."""

NO_QUERY = "NO_QUERY"
MIN_HYPOTHETICAL_DOC_CHARS = 40


def is_no_query(text: str) -> bool:
    return text.strip().upper() == NO_QUERY


def with_user_input_tags(query: str) -> str:
    return f"<user_input>\n{query.strip()}\n</user_input>"


def fallback_to_original(original: str, optimized: str) -> str:
    if is_no_query(optimized):
        return original.strip()
    return optimized
