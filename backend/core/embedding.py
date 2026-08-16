import bootstrap
from google.genai import types
from gemini_retry import run_with_retries
from llm_client import EMBEDDING_MODEL, get_client
from models.schemas import EmbedChunksInput, MAX_EMBED_BATCH_SIZE, QueryInput
DEFAULT_BATCH_SIZE = MAX_EMBED_BATCH_SIZE
OUTPUT_DIMENSIONALITY = 768
_embedder: 'Embedder | None' = None

class Embedder:

    def __init__(self) -> None:
        self.client = get_client()

    def _as_embed_contents(self, texts: list[str]) -> list[types.Content]:
        return [types.Content(parts=[types.Part(text=text)]) for text in texts]

    def embed_texts(self, texts: list[str], *, label: str='embed') -> list[list[float]]:
        if not texts:
            return []
        response = self.client.models.embed_content(model=EMBEDDING_MODEL, contents=self._as_embed_contents(texts), config=types.EmbedContentConfig(output_dimensionality=OUTPUT_DIMENSIONALITY))
        embeddings = [list(item.values) for item in response.embeddings or []]
        if len(embeddings) != len(texts):
            raise RuntimeError(f'{label}: expected {len(texts)} embeddings, got {len(embeddings)}. Some chunks were not embedded.')
        return embeddings

    def embed_chunks(self, chunks: list[dict], batch_size: int=DEFAULT_BATCH_SIZE) -> list[list[float]]:
        if not chunks:
            return []
        payload = EmbedChunksInput(chunks=chunks, batch_size=batch_size)
        texts = [chunk.embedding_text for chunk in payload.chunks]
        vectors: list[list[float]] = []
        total_batches = (len(texts) + payload.batch_size - 1) // payload.batch_size
        for batch_index, start in enumerate(range(0, len(texts), payload.batch_size), start=1):
            end = min(start + payload.batch_size, len(texts))
            batch_label = f'Batch {batch_index}/{total_batches} (chunks {start + 1}-{end})'
            slice_texts = texts[start:end]
            vectors.extend(run_with_retries(batch_label, lambda t=slice_texts, lbl=batch_label: self.embed_texts(t, label=lbl)))
        return vectors

    def embed_query(self, query: str) -> list[float]:
        payload = QueryInput(query=query)
        return run_with_retries('query', lambda: self.embed_texts([payload.query], label='query'))[0]

def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
