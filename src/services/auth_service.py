from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..config import settings

security = HTTPBearer()

def verify_admin_key(authorization: HTTPAuthorizationCredentials = Depends(security)):
    if not settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Admin authentication is not configured. Set ADMIN_API_KEY environment variable.",
        )

    if authorization.credentials != settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return authorization.credentials
