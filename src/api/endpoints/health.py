from datetime import datetime
from fastapi import APIRouter

from ...config import settings

router = APIRouter()


@router.get("/")
async def read_root():
    return {
        "message": "Legal Document Assistant API",
        "version": "2.1.0",
        "status": "ready",
    }


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_available": True,
        "vector_store_available": True,
        "project": settings.google_cloud_project,
        "timestamp": datetime.utcnow(),
    }