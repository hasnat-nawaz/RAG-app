import bootstrap  # noqa: F401

from fastapi import APIRouter

router = APIRouter()


@router.post("/upload")
def upload() -> str:
    return "hello from upload"
