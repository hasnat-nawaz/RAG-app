"""FastAPI upload route: save PDF, run ingest pipeline, handle failures."""

import bootstrap  # noqa: F401

import time
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from ingest_pipeline import MARKDOWN_DIR, PDF_DIR, IngestFailed, ingest_pdf
from pipeline_log import log
from vector_store import get_vector_store

router = APIRouter()

UPLOAD_FAILED_MESSAGE = "Upload failed. Please try again."


def _format_duration(seconds: float) -> str:
    """Format elapsed seconds as a short human-readable duration."""
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


@router.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    """Accept a PDF upload, ingest it, and return indexing stats."""
    started_at = time.monotonic()

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail={"message": "Only PDF files are accepted."})

    source = file.filename
    store = get_vector_store()

    if store.has_source(source):
        raise HTTPException(
            status_code=409,
            detail={"message": f"{source} is already uploaded."},
        )

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = PDF_DIR / source
    try:
        contents = await file.read()
        pdf_path.write_bytes(contents)
    except Exception as exc:
        log("UPLOAD", f"save failed — {exc}")
        raise HTTPException(
            status_code=500,
            detail={"message": "Failed to save uploaded file."},
        ) from exc
    finally:
        await file.close()

    log("UPLOAD", f"saved {source} ({len(contents)} bytes)")

    try:
        result = await ingest_pdf(pdf_path, source)
    except IngestFailed as exc:
        log("UPLOAD", f"failed — {exc}")
        raise HTTPException(status_code=500, detail={"message": UPLOAD_FAILED_MESSAGE}) from exc
    except Exception as exc:
        log("UPLOAD", f"unexpected error — {exc}")
        _cleanup_on_failure(source, pdf_path)
        raise HTTPException(
            status_code=500,
            detail={"message": UPLOAD_FAILED_MESSAGE},
        ) from exc

    elapsed = time.monotonic() - started_at
    log("UPLOAD", f"{source} — finished in {_format_duration(elapsed)}")
    return {
        "message": f"{source} uploaded and indexed successfully.",
        **result,
    }


def _cleanup_on_failure(source: str, pdf_path: Path) -> None:
    """Remove partial upload artifacts so the same file can be retried."""
    for p in (pdf_path, MARKDOWN_DIR / (Path(source).stem + ".md")):
        if p.is_file():
            p.unlink(missing_ok=True)
    try:
        get_vector_store().delete_by_source(source)
    except Exception:
        pass
