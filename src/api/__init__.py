from fastapi import APIRouter
from .endpoints import health, parsing, extraction, vector_store, qa, monitoring, vertex_vector_search

api_router = APIRouter()

api_router.include_router(health.router, tags=["health"])
api_router.include_router(parsing.router, tags=["parsing"])
api_router.include_router(extraction.router, tags=["extraction"])
api_router.include_router(vector_store.router, tags=["vector-store"])
api_router.include_router(vertex_vector_search.router, tags=["vertex-vector-search"])
api_router.include_router(qa.router, tags=["qa"])
api_router.include_router(monitoring.router, tags=["monitoring"])