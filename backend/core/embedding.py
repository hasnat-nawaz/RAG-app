"""Batch embedding for RAG chunks using Gemini Embedding 2."""

import bootstrap  # noqa: F401

import time

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from llm_client import get_client

MODEL_NAME = "gemini-embedding-2"
DEFAULT_BATCH_SIZE = 100
MAX_BATCH_SIZE = 100
OUTPUT_DIMENSIONALITY = 768
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 15
REQUEST_DELAY_SECONDS = 1


def _extract_contents(chunks: list[dict]) -> list[str]:
    contents: list[str] = []
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise TypeError(f"Chunk {index} is not a dict.")
        text = chunk.get("content")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Chunk {index} is missing non-empty 'content'.")
        contents.append(text)
    return contents


def _as_embed_contents(texts: list[str]) -> list[types.Content]:
    return [types.Content(parts=[types.Part(text=text)]) for text in texts]


def _embed_batch(client: genai.Client, texts: list[str], batch_label: str) -> list[list[float]]:
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.embed_content(
                model=MODEL_NAME,
                contents=_as_embed_contents(texts),
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
            raise RuntimeError(f"{batch_label}: client error ({status_code}): {e}") from e

        except genai_errors.ServerError as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                last_error = e
                continue
            raise RuntimeError(
                f"{batch_label}: server error after {MAX_RETRIES} attempts: {e}"
            ) from e

    raise RuntimeError(f"{batch_label} failed after {MAX_RETRIES} attempts. Last error: {last_error}")


def embed_chunks(chunks: list[dict], batch_size: int = DEFAULT_BATCH_SIZE) -> list[list[float]]:
    if not chunks:
        raise ValueError("embed_chunks expects a non-empty list of chunk dicts.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if batch_size > MAX_BATCH_SIZE:
        raise ValueError(
            f"batch_size cannot exceed {MAX_BATCH_SIZE} "
            f"(Gemini Embedding 2 limit)."
        )

    texts = _extract_contents(chunks)
    client = get_client()
    vectors: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_index, start in enumerate(range(0, len(texts), batch_size), start=1):
        end = min(start + batch_size, len(texts))
        batch_label = f"Batch {batch_index}/{total_batches} (chunks {start + 1}-{end})"
        batch_vectors = _embed_batch(client, texts[start:end], batch_label)
        vectors.extend(batch_vectors)

        if end < len(texts):
            time.sleep(REQUEST_DELAY_SECONDS)

    if len(vectors) != len(texts):
        raise RuntimeError(
            f"Embedding incomplete: {len(vectors)}/{len(texts)} vectors returned."
        )
    return vectors


def embed_query(query: str) -> list[float]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("embed_query expects a non-empty query string.")

    client = get_client()
    vectors = _embed_batch(client, [query], "query")
    return vectors[0]
