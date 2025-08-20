"""
Synchronous wrappers for async service functions to use in Celery tasks
"""

import asyncio
import functools
import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


def sync_wrapper(async_func: Callable) -> Callable:
    """Decorator to convert async functions to sync for use in Celery tasks"""
    @functools.wraps(async_func)
    def wrapper(*args, **kwargs):
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, create a new one in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, async_func(*args, **kwargs))
                    return future.result()
            else:
                # If no loop is running, we can use the current one
                return loop.run_until_complete(async_func(*args, **kwargs))
        except RuntimeError:
            # No event loop exists, create a new one
            return asyncio.run(async_func(*args, **kwargs))
    
    return wrapper


class SyncVectorSearchService:
    """Synchronous wrapper for VertexVectorSearchService"""
    
    def __init__(self, async_service):
        self.async_service = async_service
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Sync wrapper for _generate_embeddings"""
        return sync_wrapper(self.async_service._generate_embeddings)(texts)
    
    def upsert_datapoints(self, datapoints: List) -> bool:
        """Sync wrapper for _upsert_datapoints"""
        return sync_wrapper(self.async_service._upsert_datapoints)(datapoints)
    
    def store_chunks_in_gcs(self, document_id: str, filename: str, chunks: List[str], 
                           embeddings: List[List[float]], datapoints: List):
        """Sync wrapper for _store_chunks_in_gcs"""
        return sync_wrapper(self.async_service._store_chunks_in_gcs)(
            document_id, filename, chunks, embeddings, datapoints
        )
    
    def delete_document_from_vector_search(self, document_id: str) -> Dict[str, Any]:
        """Sync wrapper for delete_document_from_vector_search"""
        return sync_wrapper(self.async_service.delete_document_from_vector_search)(document_id)
    
    def delete_document_from_gcs(self, document_id: str) -> Dict[str, Any]:
        """Sync wrapper for delete_document_from_gcs"""
        return sync_wrapper(self.async_service.delete_document_from_gcs)(document_id)
    
    def delete_all_documents_from_vector_search(self) -> Dict[str, Any]:
        """Sync wrapper for delete_all_documents_from_vector_search"""
        return sync_wrapper(self.async_service.delete_all_documents_from_vector_search)()
    
    def delete_all_documents_from_gcs(self) -> Dict[str, Any]:
        """Sync wrapper for delete_all_documents_from_gcs"""
        return sync_wrapper(self.async_service.delete_all_documents_from_gcs)()


class SyncPDFService:
    """Synchronous wrappers for PDF service functions"""
    
    @staticmethod
    def extract_text_from_pdf(file_bytes: bytes) -> str:
        """Sync wrapper for extract_text_from_pdf"""
        from ..services.pdf_service import extract_text_from_pdf
        return sync_wrapper(extract_text_from_pdf)(file_bytes)
    
    @staticmethod
    def parse_pdf_with_blocks(file_bytes: bytes, filename: str):
        """Sync wrapper for parse_pdf_with_blocks"""
        from ..services.pdf_service import parse_pdf_with_blocks
        return sync_wrapper(parse_pdf_with_blocks)(file_bytes, filename)


class SyncAIService:
    """Synchronous wrappers for AI service functions"""
    
    @staticmethod
    def extract_entities_with_ai(text: str, model):
        """Sync wrapper for extract_entities_with_ai"""
        from ..services.ai_service import extract_entities_with_ai
        return sync_wrapper(extract_entities_with_ai)(text=text, model=model)