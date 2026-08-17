"""FastAPI query route: retrieve, rerank, and generate grounded answers."""

import bootstrap  # noqa: F401

import asyncio
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from gemini_retry import is_quota_error
from models.schemas import QueryRequest, QueryResponse
from pipeline_log import log
from query_optimization.query_expansion import expand_query
from query_optimization.query_rewriting import rewrite_query

router = APIRouter()

EMPTY_DB_ANSWER = "Database is empty."
QUOTA_MESSAGE = "API limit reached. Please try again in a minute."
UNEXPECTED_MESSAGE = "An unexpected error occurred. Please try again."


def _detail(message: str) -> dict:
    """Wrap a user-facing message in the API error detail shape."""
    return {"message": message}


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


def _empty_db_response() -> QueryResponse:
    """Return the standard response when no documents are indexed yet."""
    return QueryResponse(
        answer=EMPTY_DB_ANSWER,
        methods=[],
        documents_retrieved=0,
        documents_used=0,
    )


def _map_retrieval_error(exc: BaseException) -> HTTPException:
    """Map retrieval exceptions to safe HTTP responses."""
    if isinstance(exc, BaseExceptionGroup):
        for inner in exc.exceptions:
            mapped = _map_retrieval_error(inner)
            if mapped.status_code == 404:
                return mapped
        exc = exc.exceptions[0]

    text = str(exc).strip()
    low = text.lower()
    if "database is empty" in low or "no table" in low:
        return HTTPException(status_code=404, detail=_detail(EMPTY_DB_ANSWER))
    if (
        "not found" in low
        or "no such file" in low
        or ".lance" in low
        or "fragment" in low
    ):
        return HTTPException(status_code=404, detail=_detail(EMPTY_DB_ANSWER))
    return HTTPException(
        status_code=500,
        detail=_detail(
            "Something went wrong while searching the database. Please try again."
        ),
    )


def _is_empty_db_error(exc: BaseException) -> bool:
    """Return True when an exception represents an empty vector database."""
    mapped = _map_retrieval_error(exc)
    if mapped.status_code != 404:
        return False
    detail = mapped.detail
    return isinstance(detail, dict) and detail.get("message") == EMPTY_DB_ANSWER


def _exception_chain(exc: BaseException) -> list[BaseException]:
    """Collect the exception and its __cause__ chain."""
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__
    return chain


def _map_generation_error(exc: BaseException) -> HTTPException:
    """Map generation failures to user-safe messages."""
    for err in _exception_chain(exc):
        if is_quota_error(err):
            return HTTPException(status_code=429, detail=_detail(QUOTA_MESSAGE))
    return HTTPException(status_code=500, detail=_detail(UNEXPECTED_MESSAGE))


def _preview(text: str, limit: int = 80) -> str:
    """Truncate long strings for log output."""
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


async def _run_expand(query: str) -> str:
    """Run query expansion and log when keywords are ready."""
    result = await asyncio.to_thread(expand_query, query)
    log("EXPAND", f"ready — \"{_preview(result)}\"")
    return result


async def _run_rewrite(query: str) -> str:
    """Run query rewriting and log when the rewrite is ready."""
    result = await asyncio.to_thread(rewrite_query, query)
    log("REWRITE", f"ready — \"{_preview(result)}\"")
    return result


async def _hybrid_retrieval(
    vector_store,
    top_k: int,
    expand_task: asyncio.Task[str],
    rewrite_task: asyncio.Task[str],
) -> list[dict]:
    """BM25 on expanded keywords and semantic search on rewritten query, in parallel."""

    async def bm25_branch() -> list[dict]:
        expanded = await expand_task
        docs = await asyncio.to_thread(vector_store.bm25, expanded, top_k)
        log("HYBRID", f"BM25 found {len(docs)} chunks")
        return docs

    async def semantic_branch() -> list[dict]:
        rewritten = await rewrite_task
        vector = await asyncio.to_thread(vector_store.embedder.embed_query, rewritten)
        docs = await asyncio.to_thread(vector_store.semantic_search, vector, top_k)
        log("HYBRID", f"semantic found {len(docs)} chunks")
        return docs

    bm25_docs, semantic_docs = await asyncio.gather(bm25_branch(), semantic_branch())
    merged = vector_store.merge_documents(bm25_docs, semantic_docs)
    log("HYBRID", f"merged {len(merged)} unique chunks")
    return merged


async def _hyde_retrieval(
    vector_store,
    top_k: int,
    rewrite_task: asyncio.Task[str],
) -> list[dict]:
    """HyDE retrieval using the rewritten query as soon as it is ready."""
    rewritten = await rewrite_task
    return await asyncio.to_thread(vector_store.hyde, rewritten, top_k)


async def _retrieve_documents(
    vector_store,
    payload: QueryRequest,
) -> tuple[list[dict], list[str]]:
    """Optimize the query, then run retrieval branches as each input becomes ready."""
    methods: list[str] = []
    retrieval_tasks: list = []

    log("QUERY", "optimizing — expansion + rewrite in parallel")
    expand_task = asyncio.create_task(_run_expand(payload.query))
    rewrite_task = asyncio.create_task(_run_rewrite(payload.query))

    if payload.hybrid:
        methods.append("hybrid")
        retrieval_tasks.append(
            _hybrid_retrieval(vector_store, payload.top_k, expand_task, rewrite_task)
        )
    if payload.hyde:
        methods.append("hyde")
        retrieval_tasks.append(_hyde_retrieval(vector_store, payload.top_k, rewrite_task))

    log("QUERY", f"retrieving — methods: {', '.join(methods)}, top_k={payload.top_k}")
    results = await asyncio.gather(*retrieval_tasks)

    if not payload.hybrid:
        try:
            await expand_task
        except Exception as exc:
            log("EXPAND", f"unused (hyde-only) — {exc}")

    documents = vector_store.merge_documents(*results)
    log("QUERY", f"retrieved {len(documents)} unique chunks")
    return documents, methods


@router.post("/query")
async def query(request: Request) -> QueryResponse:
    """Handle a RAG query: retrieve chunks, rerank, then generate an answer."""
    started_at = time.monotonic()
    vector_store = request.app.state.vector_store
    reranker = request.app.state.reranker
    generator = request.app.state.generator

    try:
        vector_store.ensure_queryable()
    except Exception as exc:
        if _is_empty_db_error(exc):
            log("QUERY", "database is empty")
            return _empty_db_response()
        raise _map_retrieval_error(exc) from exc

    try:
        raw = await request.json()
        body = QueryRequest.model_validate(raw)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=_detail("Invalid query request body."),
        ) from exc

    preview = body.query if len(body.query) <= 80 else f"{body.query[:77]}..."
    log("QUERY", f"started — \"{preview}\"")

    try:
        documents, methods = await _retrieve_documents(vector_store, body)
    except Exception as exc:
        if _is_empty_db_error(exc):
            log("QUERY", "database is empty")
            return _empty_db_response()
        log("QUERY", f"retrieval failed — {exc}")
        raise _map_retrieval_error(exc) from exc

    if not documents:
        log("QUERY", "no matching documents found")
        raise HTTPException(
            status_code=404,
            detail=_detail("No matching documents were found for that query."),
        )

    try:
        log("RERANK", f"reranking {len(documents)} chunks")
        reranked = await asyncio.to_thread(
            reranker.rerank, body.query, documents, body.top_k
        )
    except Exception as exc:
        log("RERANK", f"failed — {exc}")
        raise HTTPException(
            status_code=500,
            detail=_detail(
                "Something went wrong while reranking documents. Please try again."
            ),
        ) from exc

    if not reranked:
        log("RERANK", "no chunks passed reranking")
        raise HTTPException(
            status_code=404,
            detail=_detail("No matching documents were found for that query."),
        )

    log("RERANK", f"selected top {len(reranked)} chunks")

    try:
        log("GENERATE", f"generating answer from {len(reranked)} chunks")
        answer = await asyncio.to_thread(
            generator.generate_response, body.query, reranked
        )
    except Exception as exc:
        log("GENERATE", f"failed — {exc}")
        raise _map_generation_error(exc) from exc

    elapsed = time.monotonic() - started_at
    log(
        "QUERY",
        f"finished in {_format_duration(elapsed)} — "
        f"{len(documents)} retrieved, {len(reranked)} used",
    )

    return QueryResponse(
        answer=answer,
        methods=methods,
        documents_retrieved=len(documents),
        documents_used=len(reranked),
    )
