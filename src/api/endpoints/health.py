from datetime import datetime
from fastapi import APIRouter

from ...config import settings

router = APIRouter()

@router.get("/")
async def read_root():
    return {
        "message": "Legal Document Assistant API",
        "version": "3.3.0",
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

@router.get("/database-entries")
async def get_database_entries():
    """Get database entries for debugging"""
    from ...services.extraction_db_service import extraction_db_service
    if not extraction_db_service.connection_pool:
        return {"error": "Database not connected", "entries": []}
    try:
        from psycopg2.extras import RealDictCursor
        conn = extraction_db_service.connection_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                query = (
                    "SELECT id, filename, LEFT(file_hash, 12) as "
                    "hash_prefix, created_at, file_size_bytes FROM "
                    "extraction_cache ORDER BY created_at DESC LIMIT 10"
                )
                cursor.execute(query)
                rows = cursor.fetchall()
                entries = []
                for row in rows:
                    entries.append({
                        "id": row["id"],
                        "filename": row["filename"],
                        "hash_prefix": row["hash_prefix"],
                        "created_at": (
                            row["created_at"].isoformat()
                            if row["created_at"] else None
                        ),
                        "file_size_bytes": row["file_size_bytes"]
                    })
                return {
                    "status": "success",
                    "database_connected": True,
                    "count": len(entries),
                    "entries": entries
                }
        finally:
            extraction_db_service.connection_pool.putconn(conn)
    except Exception as e:
        return {
            "error": f"Database query failed: {str(e)}",
            "database_connected": (
                extraction_db_service.connection_pool is not None
            ),
            "entries": []
        }
