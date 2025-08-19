import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import HTTPException, status
from google.cloud import storage
from llama_index.core import Document, Settings

from ..config import settings
from ..models.responses import VectorStoreUploadResult
from ..services.pdf_service import extract_text_from_pdf
from ..utils.helpers import calculate_file_hash

logger = logging.getLogger(__name__)


async def upload_document_to_vector_store(file_bytes: bytes, filename: str, node_parser, vector_store_initialized: bool) -> VectorStoreUploadResult:
    start_time = time.time()
    document_id = str(uuid.uuid4())

    if not vector_store_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document processing is not available",
        )

    try:
        text = await extract_text_from_pdf(file_bytes)

        if not text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"No text found in PDF: {filename}",
            )

        document = Document(
            text=text,
            doc_id=document_id,
            metadata={
                "filename": filename,
                "document_id": document_id,
                "upload_timestamp": datetime.utcnow().isoformat(),
            },
        )

        logger.info(f"Processing document for storage: {filename}")

        nodes = node_parser.get_nodes_from_documents([document])
        logger.info(f"Document chunked into {len(nodes)} pieces")

        embed_model = Settings.embed_model
        if embed_model is None:
            raise Exception("Embedding model is not configured")

        logger.info(f"Generating embeddings for {len(nodes)} chunks...")
        chunks_with_embeddings = []

        for i, node in enumerate(nodes):
            embedding = embed_model.get_text_embedding(node.get_content())

            chunk_data = {
                "chunk_id": node.node_id,
                "document_id": document_id,
                "filename": filename,
                "chunk_index": i,
                "text": node.get_content(),
                "embedding": embedding,
                "metadata": {
                    "page_number": node.metadata.get("page_number"),
                    "upload_timestamp": datetime.utcnow().isoformat(),
                },
            }
            chunks_with_embeddings.append(chunk_data)
            logger.debug(f"Generated embedding for chunk {i+1}/{len(nodes)}, length: {len(embedding)}")

        storage_client = storage.Client(project=settings.google_cloud_project)
        bucket = storage_client.bucket(settings.gcs_staging_bucket)

        storage_path = f"documents/{document_id}/chunks.json"
        blob = bucket.blob(storage_path)
        blob.upload_from_string(json.dumps(chunks_with_embeddings, indent=2))

        file_hash = calculate_file_hash(file_bytes)

        doc_metadata = {
            "document_id": document_id,
            "filename": filename,
            "total_chunks": len(chunks_with_embeddings),
            "upload_timestamp": datetime.utcnow().isoformat(),
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "file_hash": file_hash,
        }

        metadata_path = f"documents/{document_id}/metadata.json"
        metadata_blob = bucket.blob(metadata_path)
        metadata_blob.upload_from_string(json.dumps(doc_metadata, indent=2))

        num_chunks = len(chunks_with_embeddings)
        logger.info(f"Successfully stored document with {num_chunks} chunks in GCS at {storage_path}")

        processing_time = time.time() - start_time

        logger.info(f"✅ Successfully uploaded {filename} with {num_chunks} chunks to vector store")

        return VectorStoreUploadResult(
            filename=filename,
            document_id=document_id,
            total_chunks=num_chunks,
            successful_chunks=num_chunks,
            failed_chunks=0,
            processing_time=processing_time,
            file_hash=file_hash,
        )

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Vector store upload failed for {filename}: {e!s}")

        return VectorStoreUploadResult(
            filename=filename,
            document_id=document_id,
            total_chunks=0,
            successful_chunks=0,
            failed_chunks=1,
            processing_time=processing_time,
        )


async def search_relevant_documents(question: str, max_sources: int = 5, similarity_threshold: float = 0.3) -> Tuple[List[Dict], int]:
    try:
        embed_model = Settings.embed_model
        if embed_model is None:
            raise Exception("Embedding model is not configured")

        question_embedding = embed_model.get_text_embedding(question)
        logger.info(f"Generated question embedding: {len(question_embedding)} dimensions")

        storage_client = storage.Client(project=settings.google_cloud_project)
        bucket = storage_client.bucket(settings.gcs_staging_bucket)

        document_prefixes = set()
        for blob in bucket.list_blobs(prefix="documents/"):
            path_parts = blob.name.split("/")
            if len(path_parts) >= 2 and path_parts[0] == "documents":
                document_prefixes.add(f"documents/{path_parts[1]}")

        logger.info(f"Found {len(document_prefixes)} documents to search")

        all_chunks = []
        documents_searched = 0

        for doc_prefix in document_prefixes:
            try:
                chunks_blob = bucket.blob(f"{doc_prefix}/chunks.json")
                if not chunks_blob.exists():
                    continue

                chunks_data = json.loads(chunks_blob.download_as_text())

                metadata_blob = bucket.blob(f"{doc_prefix}/metadata.json")
                metadata = {}
                if metadata_blob.exists():
                    metadata = json.loads(metadata_blob.download_as_text())

                for chunk in chunks_data:
                    if chunk.get("embedding"):
                        chunk_embedding = np.array(chunk["embedding"])
                        question_arr = np.array(question_embedding)

                        chunk_norm = chunk_embedding / np.linalg.norm(chunk_embedding)
                        question_norm = question_arr / np.linalg.norm(question_arr)
                        similarity = np.dot(chunk_norm, question_norm)

                        if similarity >= similarity_threshold:
                            chunk_result = {
                                "document_id": chunk.get("document_id", ""),
                                "filename": chunk.get("filename", metadata.get("filename", "Unknown")),
                                "chunk_id": chunk.get("chunk_id", ""),
                                "chunk_index": chunk.get("chunk_index", 0),
                                "text": chunk.get("text", ""),
                                "similarity_score": float(similarity),
                                "page_number": chunk.get("metadata", {}).get("page_number"),
                                "upload_timestamp": chunk.get("metadata", {}).get("upload_timestamp"),
                            }
                            all_chunks.append(chunk_result)

                documents_searched += 1

            except Exception as doc_error:
                logger.warning(f"Error processing document {doc_prefix}: {doc_error}")
                continue

        all_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
        relevant_chunks = all_chunks[:max_sources]

        logger.info(f"Found {len(relevant_chunks)} relevant chunks from {documents_searched} documents")
        return relevant_chunks, documents_searched

    except Exception as e:
        logger.error(f"Document search failed: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document search failed: {e!s}",
        )


async def find_existing_document_by_hash(file_hash: str) -> Optional[Dict[str, str]]:
    try:
        storage_client = storage.Client(project=settings.google_cloud_project)
        bucket = storage_client.bucket(settings.gcs_staging_bucket)

        for blob in bucket.list_blobs(prefix="documents/"):
            if blob.name.endswith("/metadata.json"):
                try:
                    metadata_content = blob.download_as_text()
                    metadata = json.loads(metadata_content)

                    if metadata.get("file_hash") == file_hash:
                        return {
                            "document_id": metadata.get("document_id"),
                            "filename": metadata.get("filename"),
                            "upload_timestamp": metadata.get("upload_timestamp"),
                        }
                except Exception as blob_error:
                    logger.warning(f"Error reading metadata from {blob.name}: {blob_error}")
                    continue

        return None

    except Exception as e:
        logger.error(f"Error checking for existing documents: {e!s}")
        return None


async def check_batch_for_duplicates(files_with_hashes: List[tuple]) -> Dict[str, str]:
    duplicates = {}
    seen_hashes = {}

    for filename, file_hash in files_with_hashes:
        if file_hash in seen_hashes:
            duplicates[filename] = seen_hashes[file_hash]
        else:
            seen_hashes[file_hash] = filename

    return duplicates


async def clear_vector_store_data() -> int:
    try:
        storage_client = storage.Client(project=settings.google_cloud_project)
        bucket = storage_client.bucket(settings.gcs_staging_bucket)

        blobs = list(bucket.list_blobs(prefix="documents/"))
        documents_deleted = 0

        document_ids = set()
        for blob in blobs:
            path_parts = blob.name.split("/")
            if len(path_parts) >= 2 and path_parts[0] == "documents":
                document_ids.add(path_parts[1])

        documents_deleted = len(document_ids)

        if blobs:
            bucket.delete_blobs(blobs)
            logger.info(f"Deleted {len(blobs)} files from {documents_deleted} documents in vector store")
        else:
            logger.info("No documents found in vector store to delete")

        return documents_deleted

    except Exception as e:
        logger.error(f"Error clearing vector store: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear vector store: {e!s}",
        )