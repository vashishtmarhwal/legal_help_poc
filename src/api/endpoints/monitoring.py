import logging

from fastapi import APIRouter, HTTPException, status

from ...monitoring.simple_token_counter import simple_counter

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/monitoring/tokens/current")
async def get_current_token_stats():
    """Get current session token statistics"""
    try:
        stats = simple_counter.get_current_stats()
        return {
            "status": "active",
            "session_stats": stats,
            "note": "Counters reset on API restart"
        }
    except Exception as e:
        logger.error(f"Failed to get current token stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve current token statistics",
        )
