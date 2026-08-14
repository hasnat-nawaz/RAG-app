import bootstrap  # noqa: F401

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from document_loader import get_document_loader
from embedding import get_embedder
from generation import get_generator
from reranker import start_reranker_warmup


async def _warmup(app: FastAPI) -> None:
    loader, generator, embedder = await asyncio.gather(
        asyncio.to_thread(get_document_loader),
        asyncio.to_thread(get_generator),
        asyncio.to_thread(get_embedder),
    )
    app.state.document_loader = loader
    app.state.generator = generator
    app.state.embedder = embedder
    start_reranker_warmup()


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_warmup(app))
    yield
