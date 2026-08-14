import bootstrap  # noqa: F401

from fastapi import APIRouter

router = APIRouter()


@router.get("/methods")
def methods() -> str:
    return "hello from methods"
