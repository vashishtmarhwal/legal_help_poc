"""
Vector Search QA Service
Uses Vertex AI Vector Search for similarity matching and GCS for metadata/text retrieval
"""

import json
import logging
from typing import Dict, List, Tuple

from google.cloud import storage

from ..config import settings
from .vertex_vector_search_service import vertex_vector_search_service

logger = logging.getLogger(__name__)


async def search_relevant_documents_vector_search(
    question: str, 
    max_sources: int = 5, 
    similarity_threshold: float = 0.3
) -> Tuple[List[Dict], int]:
    """
    Search for relevant documents using Vertex AI Vector Search
    
    This function:
    1. Uses Vector Search to find similar embeddings
    2. Retrieves text content and metadata from GCS
    3. Returns enriched results with full chunk information
    """
    try:
        logger.info(f"Searching Vector Search index for question: {question[:100]}...")
        
        # Step 1: Use Vector Search to find similar chunks
        vector_results = await vertex_vector_search_service.search_similar_documents(
            question=question,
            max_results=max_sources * 2,  # Get more results to account for filtering
            similarity_threshold=similarity_threshold
        )
        
        if not vector_results:
            logger.info("No similar chunks found in Vector Search index")
            return [], 0
        
        logger.info(f"Vector Search returned {len(vector_results)} similar chunks")
        
        # Step 2: Retrieve full metadata and text content from GCS
        storage_client = storage.Client(project=settings.google_cloud_project)
        bucket = storage_client.bucket(settings.gcs_staging_bucket)
        
        enriched_results = []
        documents_searched = set()
        
        for vector_result in vector_results:
            document_id = vector_result["document_id"]
            chunk_index = vector_result["chunk_index"]
            similarity_score = vector_result["similarity_score"]
            
            try:
                # Try to get chunk data from either Vector Search metadata or GCS fallback
                chunks_blob_path = f"vector_search/{document_id}/chunks.json"
                chunks_blob = bucket.blob(chunks_blob_path)
                
                # If Vector Search specific storage doesn't exist, try the original format
                if not chunks_blob.exists():
                    chunks_blob_path = f"documents/{document_id}/chunks.json"
                    chunks_blob = bucket.blob(chunks_blob_path)
                
                if not chunks_blob.exists():
                    logger.warning(f"No chunk data found for document {document_id}")
                    continue
                
                chunks_data = json.loads(chunks_blob.download_as_text())
                
                # Find the specific chunk by index
                target_chunk = None
                for chunk in chunks_data:
                    if chunk.get("chunk_index") == chunk_index:
                        target_chunk = chunk
                        break
                
                if not target_chunk:
                    logger.warning(f"Chunk index {chunk_index} not found for document {document_id}")
                    continue
                
                # Get document metadata
                metadata_blob_path = f"vector_search/{document_id}/metadata.json"
                metadata_blob = bucket.blob(metadata_blob_path)
                
                # Fallback to original format
                if not metadata_blob.exists():
                    metadata_blob_path = f"documents/{document_id}/metadata.json"
                    metadata_blob = bucket.blob(metadata_blob_path)
                
                metadata = {}
                if metadata_blob.exists():
                    metadata = json.loads(metadata_blob.download_as_text())
                
                # Create enriched result
                enriched_result = {
                    "document_id": document_id,
                    "filename": target_chunk.get("filename") or metadata.get("filename", "Unknown Document"),
                    "chunk_id": target_chunk.get("chunk_id", vector_result["chunk_id"]),
                    "chunk_index": chunk_index,
                    "text": target_chunk.get("text", ""),
                    "similarity_score": similarity_score,
                    "page_number": target_chunk.get("metadata", {}).get("page_number"),
                    "upload_timestamp": target_chunk.get("metadata", {}).get("upload_timestamp"),
                }
                
                enriched_results.append(enriched_result)
                documents_searched.add(document_id)
                
                logger.debug(f"Enriched chunk {chunk_index} from document {document_id[:8]}...")
                
            except Exception as chunk_error:
                logger.warning(f"Error retrieving chunk data for {document_id}: {chunk_error}")
                continue
        
        # Sort by similarity score (descending)
        enriched_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Limit to requested number
        final_results = enriched_results[:max_sources]
        
        logger.info(f"Retrieved {len(final_results)} enriched chunks from {len(documents_searched)} documents")
        
        return final_results, len(documents_searched)
        
    except Exception as e:
        logger.error(f"Vector Search document search failed: {e}")
        raise Exception(f"Vector Search document search failed: {e}")


async def check_vector_search_readiness() -> bool:
    """Check if Vector Search is ready for QA operations"""
    try:
        status = await vertex_vector_search_service.check_vector_search_status()
        return (
            status.get("embedding_model_ready", False) and
            status.get("index_exists", False) and
            status.get("endpoint_exists", False) and
            status.get("index_deployed", False)
        )
    except Exception as e:
        logger.warning(f"Failed to check Vector Search readiness: {e}")
        return False