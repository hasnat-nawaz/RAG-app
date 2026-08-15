import bootstrap  # noqa: F401

import asyncio
import time
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

router = APIRouter()

STORAGE_DIR = Path(__file__).resolve().parents[1] / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploaded_docs"
MARKDOWN_DIR = STORAGE_DIR / "markdown_docs"


async def _ingest(request: Request, pdf_path: Path) -> dict:
    loader = request.app.state.document_loader
    chunker = request.app.state.chunker
    embedder = request.app.state.embedder
    vector_store = request.app.state.vector_store

    t0 = time.perf_counter()
    markdown = await loader.aload_as_markdown(pdf_path)
    t_md = time.perf_counter()

    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    md_path = MARKDOWN_DIR / f"{pdf_path.stem}.md"
    await asyncio.to_thread(md_path.write_text, markdown, encoding="utf-8")
    t_save = time.perf_counter()

    chunks = await asyncio.to_thread(chunker.chunk_markdown, markdown, pdf_path.name)
    t_chunk = time.perf_counter()

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail=f"No chunks produced for {pdf_path.name}",
        )

    embeddings = await asyncio.to_thread(embedder.embed_chunks, chunks)
    t_embed = time.perf_counter()

    rows_added = await asyncio.to_thread(vector_store.add, chunks, embeddings)
    t_db = time.perf_counter()

    timings = {
        "markdown_seconds": round(t_md - t0, 3),
        "markdown_save_seconds": round(t_save - t_md, 3),
        "chunk_seconds": round(t_chunk - t_save, 3),
        "embed_seconds": round(t_embed - t_chunk, 3),
        "db_seconds": round(t_db - t_embed, 3),
        "total_seconds": round(t_db - t0, 3),
    }
    print(
        f"Ingest done: {pdf_path.name} → {len(chunks)} chunks, "
        f"{rows_added} rows added | {timings}"
    )
    return {
        "markdown_path": md_path.name,
        "chunks": len(chunks),
        "rows_added": rows_added,
        "timings": timings,
    }


@router.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)) -> dict:
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / Path(filename).name

    if dest.exists():
        await file.close()
        print(f"Upload discarded: {dest.name} already present")
        return {"filename": dest.name, "status": "already_present"}

    start = time.perf_counter()
    content = await file.read()
    await asyncio.to_thread(dest.write_bytes, content)
    upload_elapsed = time.perf_counter() - start
    print(f"Upload done: {dest.name} ({len(content)} bytes) in {upload_elapsed:.3f}s")

    try:
        ingest = await _ingest(request, dest)
    except Exception:
        # Failed ingest leaves orphans that block retries via already_present.
        dest.unlink(missing_ok=True)
        (MARKDOWN_DIR / f"{dest.stem}.md").unlink(missing_ok=True)
        raise

    return {
        "filename": dest.name,
        "bytes": len(content),
        "upload_seconds": round(upload_elapsed, 3),
        "status": "ingested",
        **ingest,
    }
