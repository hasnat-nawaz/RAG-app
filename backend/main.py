"""FastAPI application entry point and dev server."""

from pathlib import Path
import sys

sys.dont_write_bytecode = True
BACKEND_DIR = Path(__file__).resolve().parent
CORE_DIR = BACKEND_DIR / 'core'
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import bootstrap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from api import router as api_router
from server_startup import lifespan

app = FastAPI(title='RAG API', lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
app.include_router(api_router)


@app.get('/')
def root() -> str:
    """Simple root probe."""
    return 'hello from rag api'


@app.get('/health')
def health() -> dict:
    """Health check used by the frontend online indicator."""
    return {'status': 'ok'}


if __name__ == '__main__':
    uvicorn.run('main:app', host='127.0.0.1', port=8000, reload=True)
