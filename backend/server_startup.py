"""FastAPI lifespan hook: preload models and attach services to app state."""

import bootstrap
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from embedding import get_embedder
from generation import get_generator
from reranker import get_reranker
from vector_store import get_vector_store


async def _init_app_state(app: FastAPI) -> None:
    """Warm up heavy clients and store query-time services on the app."""
    embedder, reranker, generator = await asyncio.gather(
        asyncio.to_thread(get_embedder),
        asyncio.to_thread(get_reranker),
        asyncio.to_thread(get_generator),
    )
    vector_store = await asyncio.to_thread(get_vector_store, embedder=embedder)
    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.reranker = reranker
    app.state.generator = generator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize app state on startup and release resources on shutdown."""
    await _init_app_state(app)
    yield
