from pathlib import Path
import sys

sys.dont_write_bytecode = True

BACKEND_DIR = Path(__file__).resolve().parent
CORE_DIR = BACKEND_DIR / "core"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import bootstrap  # noqa: F401

from fastapi import FastAPI
import uvicorn

from api import router as api_router
from server_startup import lifespan

app = FastAPI(title="RAG API", lifespan=lifespan)
app.include_router(api_router)


@app.get("/")
def root() -> str:
    return "hello from rag api"


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
