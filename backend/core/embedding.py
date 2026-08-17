"""Gemini embedding client for query and ingest vectorization."""

import bootstrap
from google.genai import types
from gemini_retry import run_with_retries
from llm_client import EMBEDDING_MODEL, get_client
from models.schemas import QueryInput

OUTPUT_DIMENSIONALITY = 768

_embedder: 'Embedder | None' = None


class Embedder:
    """Wraps Gemini embed_content calls for text batches and queries."""

    def __init__(self) -> None:
        self.client = get_client()

    def _as_embed_contents(self, texts: list[str]) -> list[types.Content]:
        """Convert plain strings into Gemini Content objects."""
        return [types.Content(parts=[types.Part(text=text)]) for text in texts]

    def embed_texts(self, texts: list[str], *, label: str = 'embed') -> list[list[float]]:
        """Embed a list of strings and return one vector per input."""
        if not texts:
            return []
        response = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=self._as_embed_contents(texts),
            config=types.EmbedContentConfig(output_dimensionality=OUTPUT_DIMENSIONALITY),
        )
        embeddings = [list(item.values) for item in response.embeddings or []]
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f'{label}: expected {len(texts)} embeddings, got {len(embeddings)}.'
            )
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query with retries."""
        payload = QueryInput(query=query)
        return run_with_retries(
            'query', lambda: self.embed_texts([payload.query], label='query'),
        )[0]


def get_embedder() -> Embedder:
    """Return the shared Embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
