import bootstrap  # noqa: F401

import asyncio
import time
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from ingest_pipeline import IngestFailed, ingest_pdf
from logutil import Timer, plog

router = APIRouter()

STORAGE_DIR = Path(__file__).resolve().parents[1] / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploaded_docs"
MARKDOWN_DIR = STORAGE_DIR / "markdown_docs"
MAX_UPLOAD_BYTES = 80 * 1024 * 1024


def _force_unlink(path: Path) -> bool:
    try:
        if path.exists():
            path.unlink()
        return not path.exists()
    except Exception as exc:
        plog("upload", event="scrub_unlink_fail", path=str(path), error=str(exc)[:120])
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return not path.exists()


async def _scrub_failed_upload(*, pdf_path: Path, vector_store=None) -> None:
    md_path = MARKDOWN_DIR / f"{pdf_path.stem}.md"
    removed: list[str] = []

    if pdf_path.exists() or pdf_path.is_symlink():
        if _force_unlink(pdf_path):
            removed.append(f"pdf={pdf_path.name}")
        else:
            plog("upload", event="scrub_pdf_orphan", path=str(pdf_path))

    if md_path.exists() or md_path.is_symlink():
        if _force_unlink(md_path):
            removed.append(f"md={md_path.name}")
        else:
            plog("upload", event="scrub_md_orphan", path=str(md_path))

    if vector_store is not None:
        try:
            deleted = await asyncio.to_thread(
                vector_store.delete_by_source, pdf_path.name
            )
            if deleted:
                removed.append(f"db_rows={deleted}")
        except Exception as exc:
            plog(
                "upload",
                event="scrub_db_fail",
                source=pdf_path.name,
                error=str(exc)[:160],
            )

    plog(
        "upload",
        event="scrubbed",
        source=pdf_path.name,
        removed=",".join(removed) if removed else "none",
    )


@router.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)) -> dict:
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail={"message": "Only PDF files are accepted"},
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / Path(filename).name
    vector_store = request.app.state.vector_store
    timer = Timer()

    if dest.exists():
        indexed = await asyncio.to_thread(vector_store.has_source, dest.name)
        if indexed:
            await file.close()
            plog("upload", event="already_present", file=dest.name, indexed=True)
            return {"filename": dest.name, "status": "already_present"}
        plog("upload", event="orphan_rescrub", file=dest.name)
        await _scrub_failed_upload(pdf_path=dest, vector_store=vector_store)

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=400,
            detail={"message": "Uploaded PDF is empty."},
        )
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"message": "PDF exceeds the 80MB upload limit."},
        )

    await asyncio.to_thread(dest.write_bytes, content)
    upload_elapsed = timer.elapsed()
    plog(
        "upload",
        event="saved",
        file=dest.name,
        bytes=len(content),
        elapsed_s=upload_elapsed,
    )

    committed = False
    try:
        try:
            ingest = await ingest_pdf(
                pdf_path=dest,
                loader=request.app.state.document_loader,
                chunker=request.app.state.chunker,
                embedder=request.app.state.embedder,
                vector_store=vector_store,
                markdown_dir=MARKDOWN_DIR,
            )
        except HTTPException:
            raise
        except IngestFailed as exc:
            plog("upload", event="ingest_failed", error=str(exc)[:200])
            raise HTTPException(
                status_code=502,
                detail={"message": str(exc) or "Ingest failed after retries."},
            ) from exc
        except Exception as exc:
            plog("upload", event="ingest_failed", error=str(exc)[:200])
            text = str(exc).lower()
            if "429" in text or "resource_exhausted" in text or "quota" in text:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "message": (
                            "Model quota was hit while processing the PDF. "
                            "Please wait a minute and try again."
                        )
                    },
                ) from exc
            if (
                "503" in text
                or "unavailable" in text
                or "deadline" in text
                or "timeout" in text
            ):
                raise HTTPException(
                    status_code=503,
                    detail={
                        "message": (
                            "The PDF parser was temporarily unavailable. "
                            "Please try uploading again."
                        )
                    },
                ) from exc
            raise HTTPException(
                status_code=500,
                detail={
                    "message": (
                        "Something went wrong while ingesting the PDF. "
                        "Please try again."
                    )
                },
            ) from exc

        indexed = await asyncio.to_thread(vector_store.has_source, dest.name)
        if not indexed:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": (
                        "Ingest finished but the database has no rows for this "
                        "file. Cleaned up — please try again."
                    )
                },
            )
        committed = True
    finally:
        if not committed:
            plog("upload", event="rollback", file=dest.name)
            await _scrub_failed_upload(pdf_path=dest, vector_store=vector_store)

    wall_total = timer.elapsed()
    ingest_timings = dict(ingest.get("timings") or {})
    ingest_timings["upload_save_seconds"] = round(upload_elapsed, 3)
    ingest_timings["wall_total_seconds"] = round(wall_total, 3)

    plog(
        "upload",
        event="done",
        file=dest.name,
        chunks=ingest.get("chunks"),
        rows=ingest.get("rows_added"),
        save_s=upload_elapsed,
        ingest_s=ingest_timings.get("total_seconds"),
        total_s=wall_total,
    )

    return {
        "filename": dest.name,
        "bytes": len(content),
        "upload_seconds": round(upload_elapsed, 3),
        "total_seconds": round(wall_total, 3),
        "status": "ingested",
        **{**ingest, "timings": ingest_timings},
    }
