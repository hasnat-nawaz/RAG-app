"""Cross-encoder reranking for retrieved RAG documents."""

import bootstrap  # noqa: F401

from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
TOP_K = 10

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    # Lazy-load the cross-encoder so the model is only downloaded once.
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


def _dedupe_documents(documents: list[dict]) -> list[dict]:
    # Drop duplicate chunks that appear across multiple retrievers (by id).
    seen: set = set()
    unique: list[dict] = []
    for doc in documents:
        key = doc.get("id")
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        unique.append(doc)
    return unique


def _extract_content(doc: dict, content_key: str) -> str:
    text = doc.get(content_key)
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Document is missing non-empty '{content_key}'.")
    return text.strip()


def rerank(
    query: str,
    documents: list[dict],
    top_k: int = TOP_K,
    content_key: str = "content",
) -> list[dict]:
    """Rerank documents and return the top_k highest-scoring results."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("rerank expects a non-empty query string.")
    if not documents:
        return []
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    unique_docs = _dedupe_documents(documents)

    # Score every query–document pair with the cross-encoder.
    pairs = [(query.strip(), _extract_content(doc, content_key)) for doc in unique_docs]
    raw_scores = _get_model().predict(pairs)

    scored_docs = [
        {**doc, "rerank_score": float(score)}
        for doc, score in zip(unique_docs, raw_scores)
    ]
    scored_docs.sort(key=lambda doc: doc["rerank_score"], reverse=True)

    return scored_docs[:top_k]
