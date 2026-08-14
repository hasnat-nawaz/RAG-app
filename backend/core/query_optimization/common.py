import bootstrap  # noqa: F401

NO_QUERY = "NO_QUERY"
MIN_HYPOTHETICAL_DOC_CHARS = 40


def is_no_query(text: str) -> bool:
    return text.strip().upper() == NO_QUERY


def with_user_input_tags(query: str) -> str:
    return f"<user_input>\n{query.strip()}\n</user_input>"


def fallback_to_original(original: str, optimized: str) -> str:
    if is_no_query(optimized) or not optimized.strip():
        return original.strip()
    return optimized


def clean_llm_output(text: str) -> str:
    cleaned = text.strip().strip('"').strip("'")
    if cleaned.startswith("```markdown"):
        cleaned = cleaned.removeprefix("```markdown").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").strip()
        if cleaned.lower().startswith("text"):
            cleaned = cleaned[4:].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned.removesuffix("```").strip()
    return cleaned
