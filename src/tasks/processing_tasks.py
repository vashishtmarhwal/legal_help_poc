"""
General processing tasks for background execution
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Any

from celery import current_task

from .celery_app import celery_app
from .sync_wrappers import SyncVectorSearchService, SyncAIService
from ..models.tasks import TaskProgress
from ..services.vertex_vector_search_service import vertex_vector_search_service
from ..services.vector_search_qa_service import search_relevant_documents_vector_search, check_vector_search_readiness

logger = logging.getLogger(__name__)

# Create sync service wrappers
sync_vector_service = SyncVectorSearchService(vertex_vector_search_service)
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


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 30})
def document_qa_task(self, question: str, max_results: int = 5, include_context: bool = True) -> Dict[str, Any]:
    """
    Background task for document Q&A processing
    
    Args:
        question: User question
        max_results: Maximum number of results to return
        include_context: Whether to include context in response
        
    Returns:
        Q&A result dictionary
    """
    try:
        logger.info(f"Starting document Q&A task for question: {question[:50]}...")
        start_time = time.time()
        
        # Step 1: Check Vector Search readiness
        update_task_progress(1, 4, "Checking Vector Search readiness")
        
        # Use sync wrapper for async function
        from ..tasks.sync_wrappers import sync_wrapper
        vector_search_ready = sync_wrapper(check_vector_search_readiness)()
        
        if not vector_search_ready:
            raise Exception("Vector Search service is not ready. Please check the configuration.")
        
        # Step 2: Search for relevant documents
        update_task_progress(2, 4, f"Searching for relevant documents")
        
        relevant_chunks, documents_searched = sync_wrapper(search_relevant_documents_vector_search)(
            question, max_results=max_results
        )
        
        if not relevant_chunks:
            logger.warning(f"No relevant documents found for question: {question}")
            return {
                "question": question,
                "answer": "I couldn't find any relevant documents to answer your question.",
                "confidence_score": 0.0,
                "sources": [],
                "context_used": [],
                "documents_searched": documents_searched,
                "processing_time": time.time() - start_time
            }
        
        # Step 3: Generate contextual answer
        update_task_progress(3, 4, "Generating contextual answer using AI")
        
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
        
        from ..services.ai_service import generate_contextual_answer
        answer, confidence_score = sync_wrapper(generate_contextual_answer)(
            question, relevant_chunks, include_context, model
        )
        
        # Step 4: Prepare response
        update_task_progress(4, 4, "Preparing response")
        
        # Extract sources information
        sources = []
        context_used = []
        
        for chunk in relevant_chunks:
            source_info = {
                "filename": chunk.get("filename", "Unknown"),
                "document_id": chunk.get("document_id", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "similarity_score": chunk.get("similarity_score", 0.0)
            }
            sources.append(source_info)
            
            if include_context:
                context_used.append({
                    "text": chunk.get("text", ""),
                    "filename": chunk.get("filename", "Unknown"),
                    "similarity_score": chunk.get("similarity_score", 0.0)
                })
        
        processing_time = time.time() - start_time
        
        result = {
            "question": question,
            "answer": answer,
            "confidence_score": confidence_score,
            "sources": sources,
            "context_used": context_used if include_context else [],
            "documents_searched": documents_searched,
            "processing_time": processing_time
        }
        
        logger.info(f"Successfully completed Q&A for question in {processing_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Failed to process Q&A for question '{question}': {e}")
        processing_time = time.time() - start_time if 'start_time' in locals() else 0
        
        # Return error result
        error_result = {
            "question": question,
            "answer": f"Sorry, I encountered an error while processing your question: {str(e)}",
            "confidence_score": 0.0,
            "sources": [],
            "context_used": [],
            "documents_searched": 0,
            "processing_time": processing_time,
            "error": str(e)
        }
        
        raise self.retry(countdown=30, max_retries=3, exc=e)


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 2, 'countdown': 60})
def bulk_processing_task(self, task_type: str, file_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Background task for bulk processing multiple files
    
    Args:
        task_type: Type of processing (vector_upload, entity_extraction, pdf_parsing)
        file_data_list: List of file data dictionaries with 'file_bytes' and 'filename'
        
    Returns:
        Bulk processing result
    """
    try:
        logger.info(f"Starting bulk {task_type} task for {len(file_data_list)} files")
        start_time = time.time()
        
        results = []
        errors = []
        total_files = len(file_data_list)
        
        for i, file_data in enumerate(file_data_list):
            current_step = i + 1
            filename = file_data.get('filename', f'file_{i}')
            file_bytes = file_data.get('file_bytes')
            
            update_task_progress(
                current_step, 
                total_files, 
                f"Processing file {current_step}/{total_files}: {filename}"
            )
            
            try:
                if task_type == "vector_upload":
                    from .document_tasks import upload_document_to_vector_search_task
                    # Execute the task synchronously within this bulk task
                    result = upload_document_to_vector_search_task.apply(args=[file_bytes, filename]).get()
                    results.append(result)
                    
                elif task_type == "entity_extraction":
                    from .document_tasks import extract_entities_task
                    result = extract_entities_task.apply(args=[file_bytes, filename]).get()
                    results.append(result)
                    
                elif task_type == "pdf_parsing":
                    from .document_tasks import parse_pdf_task
                    result = parse_pdf_task.apply(args=[file_bytes, filename]).get()
                    results.append(result)
                    
                else:
                    raise ValueError(f"Unknown task type: {task_type}")
                    
            except Exception as file_error:
                logger.error(f"Failed to process {filename}: {file_error}")
                errors.append({
                    "filename": filename,
                    "error": str(file_error)
                })
        
        processing_time = time.time() - start_time
        successful_files = len(results)
        failed_files = len(errors)
        
        bulk_result = {
            "task_type": task_type,
            "total_files": total_files,
            "successful_files": successful_files,
            "failed_files": failed_files,
            "results": results,
            "errors": errors,
            "processing_time": processing_time
        }
        
        logger.info(f"Completed bulk {task_type}: {successful_files} successful, {failed_files} failed in {processing_time:.2f}s")
        return bulk_result
        
    except Exception as e:
        logger.error(f"Failed bulk {task_type} processing: {e}")
        raise self.retry(countdown=60, max_retries=2, exc=e)