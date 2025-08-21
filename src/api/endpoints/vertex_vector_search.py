import logging
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from ...models.responses import BulkVectorStoreResponse
from ...services.auth_service import verify_admin_key
from ...services.vertex_vector_search_service import vertex_vector_search_service
from ...utils.validators import validate_pdf_file

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/vertex-vector-search/status")
async def get_vertex_vector_search_status():
    """Get the status of Vertex AI Vector Search components"""
    try:
        status_info = await vertex_vector_search_service.check_vector_search_status()
        
        return {
            "service": "Vertex AI Vector Search",
            "status": status_info,
            "ready": (
                status_info["embedding_model_ready"] and
                status_info["index_configured"] and
                status_info["index_exists"]
            ),
            "notes": {
                "deployment": "Index can be used without endpoint deployment for direct uploads",
                "endpoint_required": "Endpoint deployment required for search queries"
            }
        }
    except Exception as e:
        logger.error(f"Failed to get Vector Search status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve Vector Search status: {e}"
        )


@router.post("/upload-to-vertex-vector-search/", response_model=BulkVectorStoreResponse)
async def upload_to_vertex_vector_search(
    files: List[UploadFile] = File(...),
):
    """
    Upload PDF files to Vertex AI Vector Search Index
    
    This endpoint:
    1. Processes PDF files and extracts text
    2. Chunks text into smaller segments
    3. Generates embeddings using Vertex AI embedding model
    4. Uploads embeddings to Vector Search index using upsertDatapoints
    
    Note: Index must exist but does not need to be deployed to an endpoint
    """
    logger.info(f"Processing {len(files)} PDF files for Vertex AI Vector Search upload")
    
    # Check service status
    status_info = await vertex_vector_search_service.check_vector_search_status()
    
    if not status_info["embedding_model_ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding model is not ready"
        )
    
    if not status_info["index_configured"] or not status_info["index_exists"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector Search index is not configured or does not exist"
        )
    
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided"
        )
    
    results = []
    errors = []
    successful_files = 0
    total_chunks_processed = 0
    
    # Validate files first
    validated_files = []
    for file in files:
        try:
            file_bytes = await validate_pdf_file(file)
            validated_files.append((file, file_bytes))
            logger.debug(f"Validated {file.filename}")
        except HTTPException as e:
            error_detail = {
                "filename": file.filename,
                "error": e.detail,
            }
            errors.append(error_detail)
            logger.error(f"Failed to validate {file.filename}: {e.detail}")
        except Exception as e:
            error_detail = {
                "filename": file.filename,
                "error": f"Validation error: {e!s}",
            }
            errors.append(error_detail)
            logger.error(f"Unexpected validation error for {file.filename}: {e!s}")
    
    if not validated_files:
        return BulkVectorStoreResponse(
            total_files=len(files),
            successful_files=0,
            failed_files=len(files),
            duplicate_files=0,
            total_chunks_processed=0,
            results=results,
            errors=errors,
        )
    
    # Process each validated file
    for file, file_bytes in validated_files:
        try:
            upload_result = await vertex_vector_search_service.upload_document_to_vector_search(
                file_bytes, file.filename
            )
            
            if upload_result.failed_chunks == 0:
                results.append(upload_result)
                successful_files += 1
                total_chunks_processed += upload_result.total_chunks
                logger.info(
                    f"Successfully uploaded {file.filename} to Vector Search "
                    f"({upload_result.total_chunks} chunks)"
                )
            else:
                error_detail = {
                    "filename": file.filename,
                    "error": f"Vector Search upload failed after {upload_result.processing_time:.2f}s",
                }
                errors.append(error_detail)
                logger.error(f"Failed to upload {file.filename} to Vector Search")
        
        except HTTPException as e:
            error_detail = {
                "filename": file.filename,
                "error": e.detail,
            }
            errors.append(error_detail)
            logger.error(f"Failed to process {file.filename}: {e.detail}")
        
        except Exception as e:
            error_detail = {
                "filename": file.filename,
                "error": f"Unexpected error: {e!s}",
            }
            errors.append(error_detail)
            logger.error(f"Unexpected error processing {file.filename}: {e!s}")
    
    logger.info(
        f"Vertex Vector Search upload completed: "
        f"{successful_files} successful, {len(errors)} errors"
    )
    
    return BulkVectorStoreResponse(
        total_files=len(files),
        successful_files=successful_files,
        failed_files=len(errors),
        duplicate_files=0,  # No deduplication implemented for Vector Search yet
        total_chunks_processed=total_chunks_processed,
        results=results,
        errors=errors,
    )




@router.delete("/delete-document/{document_id}")
async def delete_document(
    document_id: str,
    _: str = Depends(verify_admin_key)
):
    """
    Delete a specific document from both Vector Search and GCS storage (Synchronous)
    
    This endpoint requires authentication with admin API key.
    It will:
    1. Remove all chunks of the document from Vector Search index
    2. Delete associated files from GCS bucket
    
    Args:
        document_id: The UUID of the document to delete
    """
    logger.info(f"Deleting document {document_id} from Vector Search and GCS")
    
    try:
        # Delete from Vector Search index
        vector_result = await vertex_vector_search_service.delete_document_from_vector_search(document_id)
        
        # Delete from GCS bucket
        gcs_result = await vertex_vector_search_service.delete_document_from_gcs(document_id)
        
        logger.info(f"Successfully deleted document {document_id}")
        
        return {
            "status": "success",
            "document_id": document_id,
            "vector_search_result": vector_result,
            "gcs_result": gcs_result,
            "message": f"Document {document_id} deleted from both Vector Search and GCS"
        }
        
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {e}"
        )




@router.delete("/cleanup-all")
async def cleanup_all_documents(
    _: str = Depends(verify_admin_key)
):
    """
    Clean up all uploaded document data from both Vector Search and GCS storage (Synchronous)
    
    This endpoint requires authentication with admin API key.
    WARNING: This will delete ALL document data!
    
    It will:
    1. Attempt to clean Vector Search index (limited functionality)
    2. Delete all document files from GCS bucket
    """
    logger.warning("Cleaning up ALL document data from Vector Search and GCS")
    
    try:
        # Clean up Vector Search index
        vector_result = await vertex_vector_search_service.delete_all_documents_from_vector_search()
        
        # Clean up GCS bucket
        gcs_result = await vertex_vector_search_service.delete_all_documents_from_gcs()
        
        logger.info("Cleanup operation completed")
        
        return {
            "status": "success",
            "vector_search_result": vector_result,
            "gcs_result": gcs_result,
            "message": "Cleanup completed - all document data removed from GCS, Vector Search cleanup has limitations"
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup all documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup documents: {e}"
        )


