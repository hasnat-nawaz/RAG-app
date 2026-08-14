"""Cross-encoder reranking for retrieved RAG documents."""

import sys

sys.dont_write_bytecode = True

import bootstrap  # noqa: F401

import logging
import os
import threading
import warnings
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

# Silence Hugging Face download / weight-loading noise before importing the model stack.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")

from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
TOP_K = 10

_model: CrossEncoder | None = None
_warmup_thread: threading.Thread | None = None


def _hub_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _get_model() -> CrossEncoder:
    # Lazy-load the cross-encoder so weights are only read from disk once per process.
    global _model
    if _model is None:
        token = _hub_token()
        kwargs = {"token": token} if token else {}
        try:
            _model = CrossEncoder(MODEL_NAME, local_files_only=True, **kwargs)
        except (OSError, ValueError):
            _model = CrossEncoder(MODEL_NAME, **kwargs)
    return _model


def warmup_reranker() -> None:
    # Load weights and run one dummy pair so the first real rerank is inference-only.
    model = _get_model()
    model.predict([("warmup query", "warmup document")], show_progress_bar=False)


def start_reranker_warmup() -> threading.Thread:
    # Start loading the cross-encoder on a background thread (e.g. during retrieval).
    global _warmup_thread
    if _model is not None:
        return threading.Thread()
    if _warmup_thread is not None and _warmup_thread.is_alive():
        return _warmup_thread
    _warmup_thread = threading.Thread(target=warmup_reranker, daemon=True)
    _warmup_thread.start()
    return _warmup_thread


def wait_for_reranker_warmup() -> None:
    if _warmup_thread is not None and _warmup_thread.is_alive():
        _warmup_thread.join()


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

    wait_for_reranker_warmup()
    unique_docs = _dedupe_documents(documents)

    pairs = [(query.strip(), _extract_content(doc, content_key)) for doc in unique_docs]
    raw_scores = _get_model().predict(pairs, show_progress_bar=False, batch_size=64)

    scored_docs = [
        {**doc, "rerank_score": float(score)}
        for doc, score in zip(unique_docs, raw_scores)
    ]
    scored_docs.sort(key=lambda doc: doc["rerank_score"], reverse=True)

    return scored_docs[:top_k]
