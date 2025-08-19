import logging
import time
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from ...dependencies import get_node_parser, get_vector_store_status
from ...models.responses import BulkVectorStoreResponse, ClearVectorStoreResponse
from ...services.auth_service import verify_admin_key
from ...services.vector_service import (
    check_batch_for_duplicates,
    clear_vector_store_data,
    find_existing_document_by_hash,
    upload_document_to_vector_store,
)
from ...utils.helpers import calculate_file_hash
from ...utils.validators import validate_pdf_file

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload-to-vector-store/", response_model=BulkVectorStoreResponse)
async def upload_to_vector_store(
    files: List[UploadFile] = File(...),
    node_parser=Depends(get_node_parser),
    vector_store_initialized: bool = Depends(get_vector_store_status),
):
    logger.info(f"Processing {len(files)} PDF files for vector store upload with deduplication")

    if not vector_store_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store is not available. Check configuration.",
        )

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided",
        )

    results = []
    errors = []
    successful_files = 0
    duplicate_files = 0
    total_chunks_processed = 0

    validated_files = []
    for file in files:
        try:
            file_bytes = await validate_pdf_file(file)
            file_hash = calculate_file_hash(file_bytes)
            validated_files.append((file, file_bytes, file_hash))
            logger.debug(f"Validated {file.filename} with hash {file_hash[:16]}...")
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

    files_with_hashes = [(file.filename, file_hash) for file, _, file_hash in validated_files]
    batch_duplicates = await check_batch_for_duplicates(files_with_hashes)

    for file, file_bytes, file_hash in validated_files:
        start_time = time.time()

        try:
            if file.filename in batch_duplicates:
                duplicate_filename = batch_duplicates[file.filename]
                processing_time = time.time() - start_time

                logger.info(f"Skipped {file.filename} - duplicate of {duplicate_filename} in current batch")

                from ...models.responses import VectorStoreUploadResult

                result = VectorStoreUploadResult(
                    filename=file.filename,
                    document_id="",
                    total_chunks=0,
                    successful_chunks=0,
                    failed_chunks=0,
                    processing_time=processing_time,
                    is_duplicate=True,
                    duplicate_of_document_id="batch_duplicate",
                    file_hash=file_hash,
                )
                results.append(result)
                duplicate_files += 1
                continue

            existing_doc = await find_existing_document_by_hash(file_hash)
            if existing_doc:
                processing_time = time.time() - start_time

                logger.info(f"Skipped {file.filename} - duplicate of existing document {existing_doc['filename']} (ID: {existing_doc['document_id']})")

                from ...models.responses import VectorStoreUploadResult

                result = VectorStoreUploadResult(
                    filename=file.filename,
                    document_id="",
                    total_chunks=0,
                    successful_chunks=0,
                    failed_chunks=0,
                    processing_time=processing_time,
                    is_duplicate=True,
                    duplicate_of_document_id=existing_doc["document_id"],
                    file_hash=file_hash,
                )
                results.append(result)
                duplicate_files += 1
                continue

            upload_result = await upload_document_to_vector_store(file_bytes, file.filename, node_parser, vector_store_initialized)

            if upload_result.failed_chunks == 0:
                results.append(upload_result)
                successful_files += 1
                total_chunks_processed += upload_result.total_chunks
                logger.info(f"Successfully uploaded {file.filename} ({upload_result.total_chunks} chunks)")
            else:
                error_detail = {
                    "filename": file.filename,
                    "error": f"Vector store upload failed after {upload_result.processing_time:.2f}s",
                }
                errors.append(error_detail)
                logger.error(f"Failed to upload {file.filename} to vector store")

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

    logger.info(f"Upload completed: {successful_files} successful, {duplicate_files} duplicates, {len(errors)} errors")

    return BulkVectorStoreResponse(
        total_files=len(files),
        successful_files=successful_files,
        failed_files=len(errors),
        duplicate_files=duplicate_files,
        total_chunks_processed=total_chunks_processed,
        results=results,
        errors=errors,
    )


@router.delete("/clear-vector-store/", response_model=ClearVectorStoreResponse)
async def clear_vector_store(
    api_key: str = Depends(verify_admin_key),
    vector_store_initialized: bool = Depends(get_vector_store_status),
):
    start_time = time.time()

    logger.info("Starting vector store clear operation")

    if not vector_store_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store is not available",
        )

    try:
        documents_deleted = await clear_vector_store_data()
        processing_time = time.time() - start_time

        logger.info(f"Successfully cleared vector store: {documents_deleted} documents deleted in {processing_time:.2f}s")

        return ClearVectorStoreResponse(
            message=f"Successfully cleared vector store. {documents_deleted} documents deleted.",
            documents_deleted=documents_deleted,
            processing_time=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Vector store clear operation failed: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector store clear failed: {e!s}",
        )