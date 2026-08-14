import bootstrap  # noqa: F401

from fastapi import APIRouter

router = APIRouter()


@router.post("/query")
def query() -> str:
    return "hello from query"
