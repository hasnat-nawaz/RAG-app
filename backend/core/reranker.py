import bootstrap  # noqa: F401

import logging
import os
import threading
import warnings

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")

from sentence_transformers import CrossEncoder

from models.schemas import DEFAULT_TOP_K, RerankInput, RetrievedDocument

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

_model: CrossEncoder | None = None
_warmup_thread: threading.Thread | None = None


def _hub_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _get_model() -> CrossEncoder:
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
    _get_model().predict(
        [("warmup query", "warmup document")],
        show_progress_bar=False,
    )


def start_reranker_warmup() -> None:
    global _warmup_thread
    if _model is not None:
        return
    if _warmup_thread is not None and _warmup_thread.is_alive():
        return
    _warmup_thread = threading.Thread(target=warmup_reranker, daemon=True)
    _warmup_thread.start()


def wait_for_reranker_warmup() -> None:
    if _warmup_thread is not None and _warmup_thread.is_alive():
        _warmup_thread.join()


def _dedupe_documents(documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
    seen: set[int] = set()
    unique: list[RetrievedDocument] = []
    for doc in documents:
        if doc.id is not None:
            if doc.id in seen:
                continue
            seen.add(doc.id)
        unique.append(doc)
    return unique


def rerank(
    query: str,
    documents: list[dict],
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    payload = RerankInput(query=query, documents=documents, top_k=top_k)
    if not payload.documents:
        return []

    wait_for_reranker_warmup()
    unique_docs = _dedupe_documents(payload.documents)
    pairs = [(payload.query, doc.content) for doc in unique_docs]
    raw_scores = _get_model().predict(pairs, show_progress_bar=False, batch_size=64)

    scored_docs = [
        doc.model_copy(update={"rerank_score": float(score)})
        for doc, score in zip(unique_docs, raw_scores)
    ]
    scored_docs.sort(key=lambda doc: doc.rerank_score or 0.0, reverse=True)
    return [doc.model_dump(by_alias=True) for doc in scored_docs[: payload.top_k]]
