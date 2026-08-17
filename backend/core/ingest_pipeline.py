"""Parallel PDF ingest pipeline: LLM parse, chunk, and embed in three async workers."""

from __future__ import annotations

import asyncio
from pathlib import Path

from chunking import get_chunker
from document_loader import DocumentLoader, get_document_loader
from embedding import get_embedder
from gemini_retry import is_quota_error, should_retry
from models.schemas import MAX_EMBED_BATCH_SIZE
from pipeline_log import log
from vector_store import get_vector_store

BATCH_SIZE = 15
MAX_RETRIES = 5
COOLDOWN_SECONDS = 65.0

STORAGE_DIR = Path(__file__).resolve().parents[1] / "storage"
PDF_DIR = STORAGE_DIR / "pdfs"
MARKDOWN_DIR = STORAGE_DIR / "markdown"

_SENTINEL = None


class IngestFailed(RuntimeError):
    """Raised when ingest cannot complete after retries or worker failure."""


async def _parse_single_chunk(
    loader: DocumentLoader,
    pdf_bytes: bytes,
    chunk_idx: int,
    total_chunks: int,
    batch_num: int,
) -> str:
    """Parse one 4-page PDF slice to markdown with per-chunk retries."""
    label = f"chunk {chunk_idx + 1}/{total_chunks}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            md = await asyncio.to_thread(loader.parse_chunk, pdf_bytes)
            return loader.clean_markdown(md)
        except Exception as exc:
            if not should_retry(exc) or attempt >= MAX_RETRIES:
                raise
            log("LLM", f"batch {batch_num} {label} — retry {attempt}/{MAX_RETRIES}")
            await asyncio.sleep(COOLDOWN_SECONDS if is_quota_error(exc) else 8.0)
    return ""


async def _llm_system(
    loader: DocumentLoader,
    pdf_chunks: list[bytes],
    md_queue: asyncio.Queue[list[str] | None],
    source: str,
) -> None:
    """Send PDF slices to Gemini in batches of 15; enqueue full markdown batches."""
    total = len(pdf_chunks)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in range(1, total_batches + 1):
        start = (batch_num - 1) * BATCH_SIZE
        end = min(start + BATCH_SIZE, total)
        batch = pdf_chunks[start:end]

        log("LLM", f"batch {batch_num}/{total_batches} — sending {len(batch)} chunks")

        tasks = [
            _parse_single_chunk(loader, b, start + i, total, batch_num)
            for i, b in enumerate(batch)
        ]
        results = await asyncio.gather(*tasks)
        batch_markdown = [md for md in results if md]

        log(
            "LLM",
            f"batch {batch_num}/{total_batches} — received {len(batch_markdown)}/{len(batch)} responses",
        )

        if batch_markdown:
            await md_queue.put(batch_markdown)

        if batch_num < total_batches:
            log("LLM", f"cooldown {COOLDOWN_SECONDS:.0f}s before next batch")
            await asyncio.sleep(COOLDOWN_SECONDS)

    await md_queue.put(_SENTINEL)


async def _chunker_system(
    md_queue: asyncio.Queue[list[str] | None],
    chunk_queue: asyncio.Queue[list[dict] | None],
    source: str,
) -> None:
    """Chunk each completed LLM batch in order and forward all chunks at once."""
    chunker = get_chunker()
    batches_processed = 0

    while True:
        markdown_batch = await md_queue.get()
        if markdown_batch is _SENTINEL:
            await chunk_queue.put(_SENTINEL)
            break

        batches_processed += 1
        batch_chunks: list[dict] = []
        for md in markdown_batch:
            piece_chunks = await asyncio.to_thread(chunker.chunk_markdown, md, source)
            batch_chunks.extend(piece_chunks)

        if batch_chunks:
            log(
                "CHUNKER",
                f"batch {batches_processed} — produced {len(batch_chunks)} chunks "
                f"from {len(markdown_batch)} markdown pieces",
            )
            await chunk_queue.put(batch_chunks)

    log("CHUNKER", f"finished — processed {batches_processed} batches")


async def _embed_batch_with_retry(
    embedder,
    texts: list[str],
    batch_label: str,
) -> list[list[float]]:
    """Embed a text batch with retries on transient API errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await asyncio.to_thread(
                embedder.embed_texts, texts, label=batch_label,
            )
        except Exception as exc:
            if not should_retry(exc) or attempt >= MAX_RETRIES:
                raise
            log("EMBEDDER", f"{batch_label} — retry {attempt}/{MAX_RETRIES}")
            await asyncio.sleep(COOLDOWN_SECONDS if is_quota_error(exc) else 8.0)
    return []


async def _embedder_system(
    chunk_queue: asyncio.Queue[list[dict] | None],
    source: str,
) -> None:
    """Embed chunk batches (up to 90 per API call) and store vectors in LanceDB."""
    embedder = get_embedder()
    store = get_vector_store()
    embed_round = 0
    total_stored = 0
    batches_processed = 0

    while True:
        batch_chunks = await chunk_queue.get()
        if batch_chunks is _SENTINEL:
            break

        batches_processed += 1
        pending = list(batch_chunks)

        while pending:
            slice_size = min(len(pending), MAX_EMBED_BATCH_SIZE)
            to_embed = pending[:slice_size]
            pending = pending[slice_size:]
            embed_round += 1
            total_stored += await _embed_and_store(
                embedder,
                store,
                to_embed,
                embed_round,
                source,
            )
            if pending:
                log("EMBEDDER", f"cooldown {COOLDOWN_SECONDS:.0f}s")
                await asyncio.sleep(COOLDOWN_SECONDS)

        log(
            "EMBEDDER",
            f"batch {batches_processed} — stored {len(batch_chunks)} chunks",
        )

    log("EMBEDDER", f"finished — stored {total_stored} chunks total")


async def _embed_and_store(
    embedder,
    store,
    chunks: list[dict],
    round_num: int,
    source: str,
) -> int:
    """Embed one slice of chunks and write them to the vector store."""
    texts = [c.get("embedding_text") or c.get("content", "") for c in chunks]
    label = f"round {round_num} ({len(chunks)} chunks)"
    log("EMBEDDER", f"{label} — embedding")

    vectors = await _embed_batch_with_retry(embedder, texts, label)
    await asyncio.to_thread(store.add, chunks, vectors)
    return len(chunks)


def _cleanup_file(source: str) -> None:
    """Remove stored PDF, markdown, and vector rows for a failed upload."""
    pdf_path = PDF_DIR / source
    md_path = MARKDOWN_DIR / (Path(source).stem + ".md")
    for p in (pdf_path, md_path):
        if p.is_file():
            p.unlink(missing_ok=True)

    store = get_vector_store()
    store.delete_by_source(source)


async def ingest_pdf(pdf_path: Path, source: str) -> dict:
    """Ingest a PDF by running LLM, chunker, and embedder workers in parallel."""
    loader = get_document_loader()

    log("LLM", f"splitting {source} into 4-page chunks")
    total_pages, pdf_chunks = await asyncio.to_thread(loader.split_pdf, pdf_path)
    total_chunks = len(pdf_chunks)

    log("LLM", f"{source} — {total_pages} pages, {total_chunks} chunks, "
        f"{(total_chunks + BATCH_SIZE - 1) // BATCH_SIZE} batches")

    md_queue: asyncio.Queue[list[str] | None] = asyncio.Queue()
    chunk_queue: asyncio.Queue[list[dict] | None] = asyncio.Queue()

    llm_task = asyncio.create_task(_llm_system(loader, pdf_chunks, md_queue, source))
    chunker_task = asyncio.create_task(_chunker_system(md_queue, chunk_queue, source))
    embedder_task = asyncio.create_task(_embedder_system(chunk_queue, source))

    try:
        await asyncio.gather(llm_task, chunker_task, embedder_task)
    except Exception as exc:
        for t in (llm_task, chunker_task, embedder_task):
            t.cancel()
        await asyncio.gather(llm_task, chunker_task, embedder_task, return_exceptions=True)
        _cleanup_file(source)
        raise IngestFailed(f"Upload failed for {source}") from exc

    md_path = MARKDOWN_DIR / (Path(source).stem + ".md")
    stored = get_vector_store().count_by_source(source)

    log("LLM", f"{source} — complete ({stored} chunks stored)")

    return {
        "source": source,
        "pages": total_pages,
        "chunks_stored": stored,
        "markdown_path": str(md_path) if md_path.is_file() else None,
    }
