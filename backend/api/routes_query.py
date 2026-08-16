import bootstrap  # noqa: F401

import asyncio
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from google.genai import errors as genai_errors
from pydantic import ValidationError

from logutil import Timer, plog
from models.schemas import QueryRequest, QueryResponse

router = APIRouter()

EMPTY_DB_ANSWER = "Database is empty."


def _detail(message: str) -> dict:
    return {"message": message}


def _empty_db_response() -> QueryResponse:
    return QueryResponse(
        answer=EMPTY_DB_ANSWER,
        methods=[],
        documents_retrieved=0,
        documents_used=0,
    )


def _map_retrieval_error(exc: BaseException) -> HTTPException:
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
    plog("query", event="retrieve_error", error=str(exc)[:200])
    return HTTPException(
        status_code=500,
        detail=_detail(
            "Something went wrong while searching the database. Please try again."
        ),
    )


def _is_empty_db_error(exc: BaseException) -> bool:
    mapped = _map_retrieval_error(exc)
    if mapped.status_code != 404:
        return False
    detail = mapped.detail
    return isinstance(detail, dict) and detail.get("message") == EMPTY_DB_ANSWER


async def _retrieve_documents(
    vector_store,
    payload: QueryRequest,
) -> tuple[list[dict], list[str]]:
    methods: list[str] = []
    tasks: list = []
    if payload.hybrid:
        methods.append("hybrid")
        tasks.append(vector_store.ahybrid_search(payload.query, payload.top_k))
    if payload.hyde:
        methods.append("hyde")
        tasks.append(vector_store.ahyde(payload.query, payload.top_k))
    results = await asyncio.gather(*tasks)
    documents = vector_store.merge_documents(*results)
    return documents, methods


@router.post("/query")
async def query(request: Request) -> QueryResponse:
    vector_store = request.app.state.vector_store
    reranker = request.app.state.reranker
    generator = request.app.state.generator
    timer = Timer()

    try:
        vector_store.ensure_queryable()
    except Exception as exc:
        if _is_empty_db_error(exc):
            plog("query", event="empty_db", elapsed_s=timer.elapsed())
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

    plog(
        "query",
        event="start",
        hybrid=body.hybrid,
        hyde=body.hyde,
        top_k=body.top_k,
        query=body.query[:100],
    )

    try:
        t_ret = time.perf_counter()
        documents, methods = await _retrieve_documents(vector_store, body)
        retrieval_seconds = time.perf_counter() - t_ret
    except Exception as exc:
        if _is_empty_db_error(exc):
            plog("query", event="empty_db_mid_retrieve", elapsed_s=timer.elapsed())
            return _empty_db_response()
        raise _map_retrieval_error(exc) from exc

    plog(
        "query",
        event="retrieved",
        methods="+".join(methods) or "none",
        docs=len(documents),
        elapsed_s=retrieval_seconds,
    )

    if not documents:
        raise HTTPException(
            status_code=404,
            detail=_detail("No matching documents were found for that query."),
        )

    try:
        t_rr = time.perf_counter()
        reranked = await asyncio.to_thread(
            reranker.rerank, body.query, documents, body.top_k
        )
        rerank_seconds = time.perf_counter() - t_rr
    except Exception as exc:
        plog("query", event="rerank_error", error=str(exc)[:200])
        raise HTTPException(
            status_code=500,
            detail=_detail(
                "Something went wrong while reranking documents. Please try again."
            ),
        ) from exc

    plog(
        "query",
        event="reranked",
        kept=len(reranked),
        from_docs=len(documents),
        elapsed_s=rerank_seconds,
    )

    if not reranked:
        raise HTTPException(
            status_code=404,
            detail=_detail("No matching documents were found for that query."),
        )

    try:
        t_gen = time.perf_counter()
        answer = await asyncio.to_thread(
            generator.generate_response, body.query, reranked
        )
        gen_seconds = time.perf_counter() - t_gen
    except Exception as exc:
        if isinstance(exc, genai_errors.APIError):
            code = int(exc.code) if exc.code else 502
            if code < 400 or code > 599:
                code = 502
            message = (exc.message or "").strip() or "The model is temporarily unavailable."
            plog("query", event="generate_api_error", status=code, error=message[:160])
            raise HTTPException(
                status_code=code,
                detail={"message": message, "status": exc.status},
            ) from exc
        plog("query", event="generate_error", error=str(exc)[:200])
        raise HTTPException(
            status_code=500,
            detail=_detail(
                "Something went wrong while generating the answer. Please try again."
            ),
        ) from exc

    total_seconds = timer.elapsed()
    plog(
        "query",
        event="done",
        methods="+".join(methods),
        retrieved=len(documents),
        used=len(reranked),
        retrieve_s=retrieval_seconds,
        rerank_s=rerank_seconds,
        generate_s=gen_seconds,
        total_s=total_seconds,
        answer_chars=len(answer),
    )

    return QueryResponse(
        answer=answer,
        methods=methods,
        documents_retrieved=len(documents),
        documents_used=len(reranked),
    )
