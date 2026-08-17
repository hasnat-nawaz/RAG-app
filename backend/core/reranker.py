"""Local cross-encoder reranker for reordering retrieved chunks by relevance."""

import bootstrap
import logging
import os
import warnings

os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*unauthenticated requests to the HF Hub.*')

from sentence_transformers import CrossEncoder
from models.schemas import DEFAULT_TOP_K, RerankInput, RetrievedDocument

MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L6-v2'
_reranker: 'Reranker | None' = None


class Reranker:
    """Score query-chunk pairs with a local cross-encoder model."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.model = self._load_model(model_name)

    @staticmethod
    def _hub_token() -> str | None:
        """Return an optional Hugging Face token for model download."""
        return os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')

    @classmethod
    def _load_model(cls, model_name: str) -> CrossEncoder:
        """Load the cross-encoder, preferring a cached local copy."""
        token = cls._hub_token()
        kwargs = {'token': token} if token else {}
        try:
            return CrossEncoder(model_name, local_files_only=True, **kwargs)
        except (OSError, ValueError):
            return CrossEncoder(model_name, **kwargs)

    @staticmethod
    def _dedupe_documents(documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
        """Drop duplicate chunks by database id before scoring."""
        seen: set[int] = set()
        unique: list[RetrievedDocument] = []
        for doc in documents:
            if doc.id is not None:
                if doc.id in seen:
                    continue
                seen.add(doc.id)
            unique.append(doc)
        return unique

    def rerank(self, query: str, documents: list[dict], top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """Return the top_k chunks sorted by cross-encoder relevance score."""
        payload = RerankInput(query=query, documents=documents, top_k=top_k)
        if not payload.documents:
            return []
        unique_docs = self._dedupe_documents(payload.documents)
        pairs = [(payload.query, doc.content) for doc in unique_docs]
        raw_scores = self.model.predict(pairs, show_progress_bar=False, batch_size=64)
        scored_docs = [
            doc.model_copy(update={'rerank_score': float(score)})
            for doc, score in zip(unique_docs, raw_scores)
        ]
        scored_docs.sort(key=lambda doc: doc.rerank_score or 0.0, reverse=True)
        return [doc.model_dump(by_alias=True) for doc in scored_docs[:payload.top_k]]


def get_reranker() -> Reranker:
    """Return the shared Reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
