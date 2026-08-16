import bootstrap
from fastapi import APIRouter
from api.routes_query import router as query_router
from api.routes_upload import router as upload_router
router = APIRouter()
router.include_router(upload_router)
router.include_router(query_router)
