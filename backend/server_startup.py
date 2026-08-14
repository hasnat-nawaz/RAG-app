import bootstrap  # noqa: F401

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from chunking import get_chunker
from document_loader import get_document_loader
from embedding import get_embedder
from generation import get_generator
from reranker import get_reranker
from vector_store import get_vector_store


async def _init_app_state(app: FastAPI) -> None:
    loader, chunker, embedder, reranker, generator = await asyncio.gather(
        asyncio.to_thread(get_document_loader),
        asyncio.to_thread(get_chunker),
        asyncio.to_thread(get_embedder),
        asyncio.to_thread(get_reranker),
        asyncio.to_thread(get_generator),
    )
    vector_store = await asyncio.to_thread(get_vector_store, embedder=embedder)

    app.state.document_loader = loader
    app.state.chunker = chunker
    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.reranker = reranker
    app.state.generator = generator


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _init_app_state(app)
    yield
