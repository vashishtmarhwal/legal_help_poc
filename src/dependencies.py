from typing import Optional

from fastapi import HTTPException, status, Header
from llama_index.core.node_parser import SimpleNodeParser
from vertexai.generative_models import GenerativeModel

from .services.file_service import file_service

model: Optional[GenerativeModel] = None
node_parser: Optional[SimpleNodeParser] = None
is_initialized = False
vector_store_initialized = False


def get_ai_model():
    if not is_initialized or not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is still initializing",
        )
    return model


def get_node_parser():
    if not is_initialized or not node_parser:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is still initializing",
        )
    return node_parser


def get_vector_store_status():
    if not is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is still initializing",
        )
    return vector_store_initialized


def get_admin_api_key(authorization: Optional[str] = Header(None)):
    """Validate admin API key from Authorization header"""
    from .config import settings

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Extract the key from "Bearer <key>" format
    try:
        scheme, key = authorization.split()
        if scheme.lower() != "bearer":
            raise ValueError()
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Use: Bearer <admin_key>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not settings.admin_api_key or key != settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin API key",
        )

    return key


def get_file_service():
    """Get file service instance"""
    return file_service
