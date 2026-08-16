from __future__ import annotations

import asyncio
import time
from collections import deque
from pathlib import Path

from document_loader import (
    MAX_BATCH_REQUESTS,
    MAX_CONCURRENT_PARSES,
    PAGES_PER_CHUNK,
    DocumentLoader,
)
from embedding import Embedder
from gemini_retry import MAX_RETRIES, arun_with_retries
from logutil import Timer, plog
from models.schemas import MAX_EMBED_BATCH_SIZE

EMBED_BATCH_SIZE = MAX_EMBED_BATCH_SIZE


class IngestFailed(RuntimeError):
    pass


async def ingest_pdf(
    *,
    pdf_path: Path,
    loader: DocumentLoader,
    chunker,
    embedder: Embedder,
    vector_store,
    markdown_dir: Path,
) -> dict:
    pipeline = _DualIngestPipeline(
        pdf_path=pdf_path,
        loader=loader,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        markdown_dir=markdown_dir,
    )
    return await pipeline.run()


async def collect_markdown_only(loader: DocumentLoader, pdf_path: Path) -> str:
    page_count, pieces = await asyncio.to_thread(loader.split_pdf, pdf_path)
    total_waves = max(1, (len(pieces) + MAX_BATCH_REQUESTS - 1) // MAX_BATCH_REQUESTS)
    plog(
        "markdown",
        event="split",
        file=pdf_path.name,
        pages=page_count,
        pieces=len(pieces),
        waves=total_waves,
    )
    parts: list[str] = []
    for wave_number, start in enumerate(
        range(0, len(pieces), MAX_BATCH_REQUESTS), start=1
    ):
        wave = pieces[start : start + MAX_BATCH_REQUESTS]
        md = await _parse_wave(
            loader=loader,
            wave=wave,
            wave_offset=start,
            wave_number=wave_number,
            total_waves=total_waves,
            timer=Timer(),
        )
        if md:
            parts.append(md)
    cleaned = loader.clean_markdown("\n\n".join(parts))
    if not cleaned:
        raise IngestFailed(
            f"Document loader produced empty markdown for {pdf_path.name}."
        )
    return cleaned


class _DualIngestPipeline:
    def __init__(
        self,
        *,
        pdf_path: Path,
        loader: DocumentLoader,
        chunker,
        embedder: Embedder,
        vector_store,
        markdown_dir: Path,
    ) -> None:
        self.pdf_path = pdf_path
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.markdown_dir = markdown_dir
        self._pending: deque[dict] = deque()
        self._cond = asyncio.Condition()
        self._producer_done = False
        self._markdown_parts: list[str] = []
        self._all_chunks: list[dict] = []
        self._all_embeddings: list[list[float]] = []
        self._timer = Timer()

    async def run(self) -> dict:
        plog(
            "ingest",
            event="start",
            file=self.pdf_path.name,
            pages_per_piece=PAGES_PER_CHUNK,
            wave_size=MAX_BATCH_REQUESTS,
            concurrency=MAX_CONCURRENT_PARSES,
            embed_pack=EMBED_BATCH_SIZE,
            retries=MAX_RETRIES,
        )

        producer = asyncio.create_task(
            self._markdown_producer(),
            name=f"markdown-producer:{self.pdf_path.name}",
        )
        worker = asyncio.create_task(
            self._embed_worker(),
            name=f"embed-worker:{self.pdf_path.name}",
        )

        try:
            await asyncio.gather(producer, worker)
        except Exception as exc:
            producer.cancel()
            worker.cancel()
            await asyncio.gather(producer, worker, return_exceptions=True)
            if isinstance(exc, IngestFailed):
                raise
            raise IngestFailed(str(exc)) from exc

        t_systems = self._timer.elapsed()

        full_markdown = self.loader.clean_markdown("\n\n".join(self._markdown_parts))
        if not full_markdown:
            raise IngestFailed(
                f"Document loader produced empty markdown for {self.pdf_path.name}."
            )
        if not self._all_chunks:
            raise IngestFailed(f"No chunks produced for {self.pdf_path.name}")
        if len(self._all_embeddings) != len(self._all_chunks):
            raise IngestFailed(
                f"Embed mismatch: {len(self._all_embeddings)} vectors for "
                f"{len(self._all_chunks)} chunks."
            )

        plog(
            "db",
            event="write_start",
            file=self.pdf_path.name,
            chunks=len(self._all_chunks),
            elapsed_s=self._timer.elapsed(),
        )
        t_db0 = time.perf_counter()
        try:
            rows_added = await asyncio.to_thread(
                self.vector_store.add, self._all_chunks, self._all_embeddings
            )
        except Exception as exc:
            try:
                await asyncio.to_thread(
                    self.vector_store.delete_by_source, self.pdf_path.name
                )
            except Exception as cleanup_exc:
                plog("db", event="rollback_fail", error=str(cleanup_exc)[:160])
            raise IngestFailed(f"Database write failed: {exc}") from exc
        db_seconds = time.perf_counter() - t_db0
        plog(
            "db",
            event="write_done",
            file=self.pdf_path.name,
            rows=rows_added,
            elapsed_s=db_seconds,
        )

        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        md_path = self.markdown_dir / f"{self.pdf_path.stem}.md"
        try:
            await asyncio.to_thread(
                md_path.write_text, full_markdown, encoding="utf-8"
            )
        except Exception as exc:
            try:
                await asyncio.to_thread(
                    self.vector_store.delete_by_source, self.pdf_path.name
                )
            except Exception as cleanup_exc:
                plog("db", event="rollback_fail", error=str(cleanup_exc)[:160])
            md_path.unlink(missing_ok=True)
            raise IngestFailed(f"Failed to save markdown file: {exc}") from exc

        total_seconds = self._timer.elapsed()
        timings = {
            "systems_seconds": round(t_systems, 3),
            "db_seconds": round(db_seconds, 3),
            "total_seconds": round(total_seconds, 3),
            "markdown_waves": len(self._markdown_parts),
            "chunks": len(self._all_chunks),
        }
        plog(
            "ingest",
            event="done",
            file=self.pdf_path.name,
            waves=len(self._markdown_parts),
            chunks=len(self._all_chunks),
            rows=rows_added,
            md_chars=len(full_markdown),
            systems_s=t_systems,
            db_s=db_seconds,
            total_s=total_seconds,
        )
        return {
            "markdown_path": md_path.name,
            "chunks": len(self._all_chunks),
            "rows_added": rows_added,
            "timings": timings,
        }

    async def _markdown_producer(self) -> None:
        try:
            page_count, pieces = await asyncio.to_thread(
                self.loader.split_pdf, self.pdf_path
            )
            total_waves = max(
                1, (len(pieces) + MAX_BATCH_REQUESTS - 1) // MAX_BATCH_REQUESTS
            )
            plog(
                "markdown",
                event="split",
                file=self.pdf_path.name,
                pages=page_count,
                pieces=len(pieces),
                waves=total_waves,
                elapsed_s=self._timer.elapsed(),
            )

            for wave_number, start in enumerate(
                range(0, len(pieces), MAX_BATCH_REQUESTS), start=1
            ):
                wave = pieces[start : start + MAX_BATCH_REQUESTS]
                plog(
                    "markdown",
                    event="wave_start",
                    wave=f"{wave_number}/{total_waves}",
                    pieces=len(wave),
                    elapsed_s=self._timer.elapsed(),
                )
                markdown = await _parse_wave(
                    loader=self.loader,
                    wave=wave,
                    wave_offset=start,
                    wave_number=wave_number,
                    total_waves=total_waves,
                    timer=self._timer,
                )
                if markdown:
                    self._markdown_parts.append(markdown)
                    await self._enqueue_markdown(markdown, wave_number, total_waves)
                else:
                    plog(
                        "markdown",
                        event="wave_empty",
                        wave=f"{wave_number}/{total_waves}",
                        elapsed_s=self._timer.elapsed(),
                    )

            plog(
                "markdown",
                event="done",
                waves_nonempty=len(self._markdown_parts),
                elapsed_s=self._timer.elapsed(),
            )
        finally:
            async with self._cond:
                self._producer_done = True
                self._cond.notify_all()
            plog("markdown", event="producer_signal_done", elapsed_s=self._timer.elapsed())

    async def _enqueue_markdown(
        self,
        markdown: str,
        wave_number: int,
        total_waves: int,
    ) -> None:
        t0 = time.perf_counter()
        chunks = await asyncio.to_thread(
            self.chunker.chunk_markdown, markdown, self.pdf_path.name
        )
        chunk_s = time.perf_counter() - t0
        if not chunks:
            plog(
                "chunker",
                event="empty",
                wave=f"{wave_number}/{total_waves}",
                md_chars=len(markdown),
                elapsed_s=chunk_s,
            )
            return

        for i, chunk in enumerate(chunks):
            if not (chunk.get("embedding_text") or chunk.get("content")):
                raise IngestFailed(
                    f"[chunker] wave {wave_number}: chunk {i} missing text fields"
                )

        async with self._cond:
            self._pending.extend(chunks)
            pending_count = len(self._pending)
            self._cond.notify_all()

        avg_chars = int(sum(len(c.get("content") or "") for c in chunks) / len(chunks))
        plog(
            "chunker",
            event="queued",
            wave=f"{wave_number}/{total_waves}",
            chunks=len(chunks),
            avg_chars=avg_chars,
            fifo_pending=pending_count,
            elapsed_s=chunk_s,
        )

    async def _embed_worker(self) -> None:
        embed_batch_index = 0
        plog("embed", event="worker_start", elapsed_s=self._timer.elapsed())
        try:
            while True:
                async with self._cond:
                    while True:
                        can_full = len(self._pending) >= EMBED_BATCH_SIZE
                        can_drain = self._producer_done and bool(self._pending)
                        if can_full or can_drain:
                            break
                        if self._producer_done and not self._pending:
                            plog(
                                "embed",
                                event="worker_done",
                                total_embedded=len(self._all_chunks),
                                elapsed_s=self._timer.elapsed(),
                            )
                            return
                        await self._cond.wait()

                    take = min(EMBED_BATCH_SIZE, len(self._pending))
                    batch = [self._pending.popleft() for _ in range(take)]
                    still_pending = len(self._pending)

                embed_batch_index += 1
                texts = [
                    (c.get("embedding_text") or c.get("content") or "").strip()
                    for c in batch
                ]
                if any(not t for t in texts):
                    raise IngestFailed(
                        f"[embed] pack {embed_batch_index}: empty embedding_text"
                    )

                plog(
                    "embed",
                    event="pack_start",
                    pack=embed_batch_index,
                    size=len(batch),
                    fifo_left=still_pending,
                    elapsed_s=self._timer.elapsed(),
                )
                t0 = time.perf_counter()
                vectors = await arun_with_retries(
                    f"embed_pack_{embed_batch_index}",
                    lambda t=texts: self.embedder.embed_texts(
                        t, label=f"embed_pack_{embed_batch_index}"
                    ),
                    max_retries=MAX_RETRIES,
                    pipeline="embed",
                )
                embed_s = time.perf_counter() - t0
                if len(vectors) != len(batch):
                    raise IngestFailed(
                        f"[embed] pack {embed_batch_index}: got {len(vectors)} "
                        f"vectors for {len(batch)} chunks"
                    )

                self._all_chunks.extend(batch)
                self._all_embeddings.extend(vectors)
                plog(
                    "embed",
                    event="pack_done",
                    pack=embed_batch_index,
                    size=len(batch),
                    total_embedded=len(self._all_chunks),
                    elapsed_s=embed_s,
                )
        except asyncio.CancelledError:
            plog("embed", event="worker_cancelled", elapsed_s=self._timer.elapsed())
            raise
        except Exception as exc:
            plog("embed", event="worker_failed", error=str(exc)[:200])
            raise


async def _parse_wave(
    *,
    loader: DocumentLoader,
    wave: list[bytes],
    wave_offset: int,
    wave_number: int,
    total_waves: int,
    timer: Timer,
) -> str:
    parts: list[str] = []
    total_groups = (len(wave) + MAX_CONCURRENT_PARSES - 1) // MAX_CONCURRENT_PARSES
    t_wave = time.perf_counter()

    for group_start in range(0, len(wave), MAX_CONCURRENT_PARSES):
        group = wave[group_start : group_start + MAX_CONCURRENT_PARSES]
        group_number = group_start // MAX_CONCURRENT_PARSES + 1
        plog(
            "markdown",
            event="group_start",
            wave=f"{wave_number}/{total_waves}",
            group=f"{group_number}/{total_groups}",
            concurrent=len(group),
            elapsed_s=timer.elapsed(),
        )
        t0 = time.perf_counter()
        group_parts = await asyncio.gather(
            *[
                _parse_piece(
                    loader=loader,
                    pdf_bytes=piece,
                    chunk_index=wave_offset + group_start + offset,
                    wave_number=wave_number,
                    total_waves=total_waves,
                )
                for offset, piece in enumerate(group)
            ]
        )
        non_empty = sum(1 for p in group_parts if p)
        plog(
            "markdown",
            event="group_done",
            wave=f"{wave_number}/{total_waves}",
            group=f"{group_number}/{total_groups}",
            nonempty=f"{non_empty}/{len(group)}",
            elapsed_s=time.perf_counter() - t0,
        )
        parts.extend(group_parts)

    joined = "\n\n".join(part for part in parts if part)
    cleaned = loader.clean_markdown(joined)
    plog(
        "markdown",
        event="wave_done",
        wave=f"{wave_number}/{total_waves}",
        chars=len(cleaned),
        elapsed_s=time.perf_counter() - t_wave,
    )
    return cleaned


async def _parse_piece(
    *,
    loader: DocumentLoader,
    pdf_bytes: bytes,
    chunk_index: int,
    wave_number: int,
    total_waves: int,
) -> str:
    label = f"md_w{wave_number}_p{chunk_index + 1}"
    return await arun_with_retries(
        label,
        lambda: loader.generate_markdown_piece(pdf_bytes, chunk_index),
        max_retries=MAX_RETRIES,
        pipeline="markdown",
    )
