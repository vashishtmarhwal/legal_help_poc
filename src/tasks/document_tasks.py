"""
Document processing tasks for background execution
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any

from celery import current_task
from celery.exceptions import Retry

from .celery_app import celery_app
from .sync_wrappers import SyncVectorSearchService, SyncPDFService, SyncAIService
from ..models.tasks import TaskType, TaskProgress
from ..services.vertex_vector_search_service import vertex_vector_search_service
from ..utils.validators import validate_pdf_file_bytes

logger = logging.getLogger(__name__)

# Create sync service wrappers
sync_vector_service = SyncVectorSearchService(vertex_vector_search_service)
sync_pdf_service = SyncPDFService()
sync_ai_service = SyncAIService()


def update_task_progress(current_step: int, total_steps: int, description: str):
    """Update task progress information"""
    percentage = (current_step / total_steps) * 100
    
    progress = TaskProgress(
        current=current_step,
        total=total_steps,
        description=description,
        percentage=round(percentage, 2)
    )
    
    current_task.update_state(
        state="PROGRESS",
        meta={
            "progress": progress.model_dump(),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60})
def upload_document_to_vector_search_task(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Background task for uploading a document to Vector Search
    
    Args:
        file_bytes: PDF file content as bytes
        filename: Original filename
        
    Returns:
        Upload result dictionary
    """
    try:
        logger.info(f"Starting vector search upload task for {filename}")
        document_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Step 1: Validate PDF file
        update_task_progress(1, 6, f"Validating PDF file: {filename}")
        validate_pdf_file_bytes(file_bytes, filename)
        
        # Step 2: Extract text from PDF
        update_task_progress(2, 6, f"Extracting text from PDF: {filename}")
        text = sync_pdf_service.extract_text_from_pdf(file_bytes)
        
        if not text.strip():
            raise ValueError(f"No text found in PDF: {filename}")
        
        # Step 3: Chunk the text
        update_task_progress(3, 6, f"Chunking text for {filename}")
        chunks = vertex_vector_search_service._chunk_text(text)
        logger.info(f"Created {len(chunks)} chunks for {filename}")
        
        # Step 4: Generate embeddings
        update_task_progress(4, 6, f"Generating embeddings for {len(chunks)} chunks")
        embeddings = sync_vector_service.generate_embeddings(chunks)
        
        # Step 5: Create datapoints and upload to Vector Search
        update_task_progress(5, 6, f"Uploading {len(chunks)} chunks to Vector Search")
        
        from google.cloud.aiplatform_v1.types import IndexDatapoint
        datapoints = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{document_id}_chunk_{i}"
            metadata = {
                "filename": filename,
                "document_id": document_id,
                "chunk_id": chunk_id,
                "chunk_index": i,
                "chunk_text": chunk,
                "upload_timestamp": datetime.utcnow().isoformat(),
            }
            
            datapoint = vertex_vector_search_service._create_datapoint(chunk_id, embedding, metadata)
            datapoints.append(datapoint)
        
        sync_vector_service.upsert_datapoints(datapoints)
        
        # Step 6: Store in GCS for QA service
        update_task_progress(6, 6, f"Storing metadata in GCS for {filename}")
        sync_vector_service.store_chunks_in_gcs(document_id, filename, chunks, embeddings, datapoints)
        
        processing_time = time.time() - start_time
        
        result = {
            "filename": filename,
            "document_id": document_id,
            "total_chunks": len(chunks),
            "successful_chunks": len(chunks),
            "failed_chunks": 0,
            "processing_time": processing_time,
            "is_duplicate": False,
            "duplicate_of_document_id": None,
            "file_hash": "",
        }
        
        logger.info(f"Successfully completed vector search upload for {filename} in {processing_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Failed to upload {filename} to Vector Search: {e}")
        processing_time = time.time() - start_time if 'start_time' in locals() else 0
        
        # Return error result
        error_result = {
            "filename": filename,
            "document_id": document_id if 'document_id' in locals() else str(uuid.uuid4()),
            "total_chunks": 0,
            "successful_chunks": 0,
            "failed_chunks": 1,
            "processing_time": processing_time,
            "is_duplicate": False,
            "duplicate_of_document_id": None,
            "file_hash": "",
            "error": str(e)
        }
        
        raise self.retry(countdown=60, max_retries=3, exc=e)


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 30})
def extract_entities_task(self, file_bytes: bytes, filename: str, model_name: str = None) -> Dict[str, Any]:
    """
    Background task for extracting entities from a document
    
    Args:
        file_bytes: PDF file content as bytes
        filename: Original filename
        model_name: AI model name to use
        
    Returns:
        Extracted entities result
    """
    try:
        logger.info(f"Starting entity extraction task for {filename}")
        start_time = time.time()
        
        # Step 1: Validate PDF file
        update_task_progress(1, 3, f"Validating PDF file: {filename}")
        validate_pdf_file_bytes(file_bytes, filename)
        
        # Step 2: Extract text from PDF
        update_task_progress(2, 3, f"Extracting text from PDF: {filename}")
        text = sync_pdf_service.extract_text_from_pdf(file_bytes)
        
        if not text.strip():
            raise ValueError(f"No text found in PDF: {filename}")
        
        # Step 3: Extract entities using AI
        update_task_progress(3, 3, f"Extracting entities using AI for {filename}")
        
        # Initialize model for background task
        import src.dependencies as deps
        if not hasattr(deps, 'model') or deps.model is None:
            from ..config import settings
            from vertexai.generative_models import GenerativeModel, SafetySetting, HarmCategory, HarmBlockThreshold
            import vertexai
            
            vertexai.init(project=settings.google_cloud_project, location=settings.location)
            safety_settings = [
                SafetySetting(category=HarmCategory.HARM_CATEGORY_UNSPECIFIED, threshold=HarmBlockThreshold.BLOCK_NONE),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_NONE),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.BLOCK_NONE),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.BLOCK_NONE),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_NONE),
            ]
            model = GenerativeModel(model_name=settings.model_name, safety_settings=safety_settings)
        else:
            model = deps.model
        
        extracted_data = sync_ai_service.extract_entities_with_ai(text=text, model=model)
        
        processing_time = time.time() - start_time
        
        result = {
            "filename": filename,
            "extracted_data": extracted_data.model_dump(),
            "processing_time": processing_time,
            "text_length": len(text)
        }
        
        logger.info(f"Successfully extracted entities from {filename} in {processing_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Failed to extract entities from {filename}: {e}")
        raise self.retry(countdown=30, max_retries=3, exc=e)


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 30})
def parse_pdf_task(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Background task for parsing PDF structure
    
    Args:
        file_bytes: PDF file content as bytes
        filename: Original filename
        
    Returns:
        Parsed document result
    """
    try:
        logger.info(f"Starting PDF parsing task for {filename}")
        start_time = time.time()
        
        # Step 1: Validate PDF file
        update_task_progress(1, 2, f"Validating PDF file: {filename}")
        validate_pdf_file_bytes(file_bytes, filename)
        
        # Step 2: Parse PDF with blocks
        update_task_progress(2, 2, f"Parsing PDF structure for {filename}")
        parsed_doc = sync_pdf_service.parse_pdf_with_blocks(file_bytes, filename)
        
        processing_time = time.time() - start_time
        
        result = {
            "filename": filename,
            "parsed_document": parsed_doc.model_dump(),
            "processing_time": processing_time
        }
        
        logger.info(f"Successfully parsed {filename} in {processing_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Failed to parse {filename}: {e}")
        raise self.retry(countdown=30, max_retries=3, exc=e)


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 2, 'countdown': 30})
def delete_document_task(self, document_id: str) -> Dict[str, Any]:
    """
    Background task for deleting a document from Vector Search and GCS
    
    Args:
        document_id: Document UUID to delete
        
    Returns:
        Deletion result
    """
    try:
        logger.info(f"Starting document deletion task for {document_id}")
        start_time = time.time()
        
        # Step 1: Delete from Vector Search
        update_task_progress(1, 2, f"Deleting document {document_id} from Vector Search")
        vector_result = sync_vector_service.delete_document_from_vector_search(document_id)
        
        # Step 2: Delete from GCS
        update_task_progress(2, 2, f"Deleting document {document_id} from GCS")
        gcs_result = sync_vector_service.delete_document_from_gcs(document_id)
        
        processing_time = time.time() - start_time
        
        result = {
            "document_id": document_id,
            "vector_search_result": vector_result,
            "gcs_result": gcs_result,
            "processing_time": processing_time,
            "message": f"Document {document_id} deleted successfully"
        }
        
        logger.info(f"Successfully deleted document {document_id} in {processing_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {e}")
        raise self.retry(countdown=30, max_retries=2, exc=e)


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 1, 'countdown': 60})
def cleanup_all_documents_task(self) -> Dict[str, Any]:
    """
    Background task for cleaning up all documents from Vector Search and GCS
    
    Returns:
        Cleanup result
    """
    try:
        logger.warning("Starting bulk cleanup task - this will delete ALL documents")
        start_time = time.time()
        
        # Step 1: Cleanup Vector Search (limited functionality)
        update_task_progress(1, 2, "Attempting Vector Search cleanup")
        vector_result = sync_vector_service.delete_all_documents_from_vector_search()
        
        # Step 2: Cleanup GCS
        update_task_progress(2, 2, "Cleaning up all documents from GCS")
        gcs_result = sync_vector_service.delete_all_documents_from_gcs()
        
        processing_time = time.time() - start_time
        
        result = {
            "vector_search_result": vector_result,
            "gcs_result": gcs_result,
            "processing_time": processing_time,
            "message": "Bulk cleanup completed"
        }
        
        logger.info(f"Successfully completed bulk cleanup in {processing_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Failed to complete bulk cleanup: {e}")
        raise self.retry(countdown=60, max_retries=1, exc=e)