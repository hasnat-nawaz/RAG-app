import bootstrap  # noqa: F401

import time

from google.genai import types
from google.genai import errors as genai_errors

from llm_client import EMBEDDING_MODEL, get_client
from models.schemas import EmbedChunksInput, MAX_EMBED_BATCH_SIZE, QueryInput

DEFAULT_BATCH_SIZE = MAX_EMBED_BATCH_SIZE
OUTPUT_DIMENSIONALITY = 768
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 15
REQUEST_DELAY_SECONDS = 1

_embedder: "Embedder | None" = None


class Embedder:
    def __init__(self) -> None:
        self.client = get_client()

    def _as_embed_contents(self, texts: list[str]) -> list[types.Content]:
        return [types.Content(parts=[types.Part(text=text)]) for text in texts]

    def _embed_batch(self, texts: list[str], batch_label: str) -> list[list[float]]:
        last_error: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=self._as_embed_contents(texts),
                    config=types.EmbedContentConfig(
                        output_dimensionality=OUTPUT_DIMENSIONALITY,
                    ),
                )
                embeddings = [list(item.values) for item in (response.embeddings or [])]
                if len(embeddings) != len(texts):
                    raise RuntimeError(
                        f"{batch_label}: expected {len(texts)} embeddings, "
                        f"got {len(embeddings)}. Some chunks were not embedded."
                    )
                return embeddings

            except genai_errors.ClientError as e:
                status_code = getattr(e, "code", None)
                if status_code == 429 and attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                    last_error = e
                    continue
                raise RuntimeError(
                    f"{batch_label}: client error ({status_code}): {e}"
                ) from e

            except genai_errors.ServerError as e:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                    last_error = e
                    continue
                raise RuntimeError(
                    f"{batch_label}: server error after {MAX_RETRIES} attempts: {e}"
                ) from e

        raise RuntimeError(
            f"{batch_label} failed after {MAX_RETRIES} attempts. Last error: {last_error}"
        )

    def embed_chunks(
        self,
        chunks: list[dict],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> list[list[float]]:
        payload = EmbedChunksInput(chunks=chunks, batch_size=batch_size)
        texts = [chunk.content for chunk in payload.chunks]
        vectors: list[list[float]] = []
        total_batches = (len(texts) + payload.batch_size - 1) // payload.batch_size

        for batch_index, start in enumerate(
            range(0, len(texts), payload.batch_size), start=1
        ):
            end = min(start + payload.batch_size, len(texts))
            batch_label = f"Batch {batch_index}/{total_batches} (chunks {start + 1}-{end})"
            vectors.extend(self._embed_batch(texts[start:end], batch_label))
            if end < len(texts):
                time.sleep(REQUEST_DELAY_SECONDS)

        return vectors

    def embed_query(self, query: str) -> list[float]:
        payload = QueryInput(query=query)
        return self._embed_batch([payload.query], "query")[0]


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
