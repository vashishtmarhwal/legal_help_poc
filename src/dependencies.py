from typing import Optional

from fastapi import HTTPException, status
from llama_index.core.node_parser import SimpleNodeParser
from vertexai.generative_models import GenerativeModel

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