import bootstrap  # noqa: F401

import asyncio
import time

from fastapi import APIRouter, HTTPException, Request

from models.schemas import QueryRequest, QueryResponse

router = APIRouter()


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
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    t0 = time.perf_counter()
    print(
        f"Query received: hybrid={body.hybrid} hyde={body.hyde} "
        f"top_k={body.top_k} | {body.query[:120]!r}..."
    )

    vector_store = request.app.state.vector_store
    reranker = request.app.state.reranker
    generator = request.app.state.generator

    if vector_store.table is None:
        raise HTTPException(status_code=400, detail="No documents indexed yet.")

    try:
        t_ret = time.perf_counter()
        documents, methods = await _retrieve_documents(vector_store, body)
        retrieval_seconds = time.perf_counter() - t_ret
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    print(
        f"Query retrieval done in {retrieval_seconds:.3f}s "
        f"({len(documents)} docs, methods={methods})"
    )

    if not documents:
        raise HTTPException(status_code=404, detail="No documents retrieved.")

    reranked = await asyncio.to_thread(
        reranker.rerank, body.query, documents, body.top_k
    )
    if not reranked:
        raise HTTPException(status_code=404, detail="No documents after reranking.")

    answer = await asyncio.to_thread(
        generator.generate_response, body.query, reranked
    )

    total_seconds = time.perf_counter() - t0
    print(
        f"Query done in {total_seconds:.3f}s total "
        f"(retrieval={retrieval_seconds:.3f}s, "
        f"used={len(reranked)}/{len(documents)} docs)"
    )

    return QueryResponse(
        answer=answer,
        methods=methods,
        documents_retrieved=len(documents),
        documents_used=len(reranked),
    )
