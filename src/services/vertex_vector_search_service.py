import logging
import re
import time
import uuid
from datetime import datetime
from typing import Dict, List

from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types import IndexDatapoint
from google.cloud.aiplatform import MatchingEngineIndexEndpoint

from ..config import settings
from ..models.responses import VectorStoreUploadResult
from ..monitoring.simple_token_counter import simple_counter
from ..services.pdf_service import extract_text_from_pdf

logger = logging.getLogger(__name__)


class VertexVectorSearchService:
    """Service for uploading embeddings to Vertex AI Vector Search"""

    def __init__(self):
        self.project_id = settings.google_cloud_project
        self.location = settings.location
        self.embedding_model_name = settings.embedding_model
        self.index_id = settings.vector_search_index_id
        self.endpoint_id = settings.vector_search_endpoint_id
        self.deployed_index_id = settings.vector_search_deployed_index_id

        # Initialize Vertex AI
        aiplatform.init(project=self.project_id, location=self.location)

        # Initialize embedding model
        self.embedding_model = None
        self._initialize_embedding_model()

    def _initialize_embedding_model(self):
        """Initialize the embedding model using Vertex AI"""
        try:
            # Force the correct project in environment
            import os
            os.environ["GOOGLE_CLOUD_PROJECT"] = self.project_id

            logger.info(f"Initializing Vertex AI embedding client with project: {self.project_id}")

            # Initialize Vertex AI
            aiplatform.init(project=self.project_id, location=self.location)

            # Use Vertex AI TextEmbeddingModel
            from vertexai.language_models import TextEmbeddingModel
            self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
            logger.info(f"Initialized Vertex AI embedding client for project: {self.project_id}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.embedding_model = None

    def _chunk_text(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[Dict]:
        """Split text into semantic chunks for embedding"""
        chunk_size = chunk_size or settings.chunk_size
        min_chunk_size = settings.semantic_min_chunk_size
        overlap_percentage = settings.semantic_overlap_percentage

        if len(text) <= chunk_size:
            return [self._create_chunk_dict(text, 0, len(text), self._classify_chunk_type(text))]

        # Step 1: Preprocess text for legal documents
        cleaned_text = self._preprocess_legal_text(text)

        # Step 2: Find semantic boundaries
        boundaries = self._find_semantic_boundaries(cleaned_text, chunk_size)

        # Step 3: Create chunks from boundaries
        chunks = self._create_chunks_from_boundaries(cleaned_text, boundaries, chunk_size, min_chunk_size)

        # Step 4: Add intelligent overlaps
        final_chunks = self._add_semantic_overlaps(chunks, overlap_percentage)

        return final_chunks
    
    def _create_chunk_dict(self, text: str, start: int, end: int, semantic_type: str) -> Dict:
        """Create a standardized chunk dictionary"""
        return {
            'text': text.strip(),
            'semantic_type': semantic_type,
            'start_position': start,
            'end_position': end
        }

    def _preprocess_legal_text(self, text: str) -> str:
        """Clean and normalize legal document text"""
        # Remove excessive whitespace while preserving structure
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        # Normalize spaces but preserve paragraph structure
        text = re.sub(r'[ \t]+', ' ', text)
        # Preserve important legal clause separators
        text = re.sub(r'(WHEREAS[^:]*:)', r'\n\n\1', text, flags=re.IGNORECASE)
        text = re.sub(r'(NOW THEREFORE[^:]*:)', r'\n\n\1', text, flags=re.IGNORECASE)
        return text.strip()

    def _find_semantic_boundaries(self, text: str, max_chunk_size: int) -> List[int]:
        """Find optimal positions to split text while preserving legal meaning"""
        boundaries = [0] 

        # Legal document patterns for section detection
        section_patterns = [
            r'^\d+\.\s+[A-Z][^.]*$',
            r'^Section\s+\d+[:.\s]',
            r'^Article\s+[IVX\d]+',
            r'^[A-Z][A-Z\s]{5,}:',
        ]

        # Find section headers
        lines = text.split('\n')
        current_pos = 0

        for line in lines:
            line_start = current_pos
            current_pos += len(line) + 1

            for pattern in section_patterns:
                if re.match(pattern, line.strip(), re.IGNORECASE):
                    boundaries.append(line_start)
                    break

        paragraph_breaks = [m.start() for m in re.finditer(r'\n\s*\n', text)]
        boundaries.extend(paragraph_breaks)

        sentence_ends = [m.end() for m in re.finditer(r'[.!?]\s+[A-Z]', text)]
        for sentence_pos in sentence_ends:
            recent_boundary = max([b for b in boundaries if b <= sentence_pos], default=0)
            if sentence_pos - recent_boundary > max_chunk_size * 0.8:
                boundaries.append(sentence_pos)

        boundaries = sorted(list(set(boundaries)))
        boundaries.append(len(text))

        return boundaries

    def _create_chunks_from_boundaries(
        self, text: str, boundaries: List[int], max_size: int, min_size: int
    ) -> List[Dict]:
        """Create chunks respecting semantic boundaries"""
        chunks = []
        i = 0

        while i < len(boundaries) - 1:
            start = boundaries[i]
            end = boundaries[i + 1]
            chunk_text = text[start:end].strip()

            # Skip very small chunks by merging with next if possible
            if len(chunk_text) < min_size and i + 2 < len(boundaries):
                next_end = boundaries[i + 2]
                merged_text = text[start:next_end].strip()
                if len(merged_text) <= max_size:
                    chunk_text = merged_text
                    end = next_end
                    i += 1

            # Only process if chunk is meaningful
            if len(chunk_text.strip()) > 0:
                if len(chunk_text) > max_size:
                    sub_chunks = self._split_large_chunk(chunk_text, start, max_size)
                    chunks.extend(sub_chunks)
                elif len(chunk_text) >= min_size:
                    chunks.append(self._create_chunk_dict(
                        chunk_text, start, end, self._classify_chunk_type(chunk_text)
                    ))
                else:
                    if len(chunk_text.strip()) >= 20:  # Minimum meaningful content
                        chunks.append(self._create_chunk_dict(
                            chunk_text, start, end, self._classify_chunk_type(chunk_text)
                        ))

            i += 1

        return chunks

    def _split_large_chunk(self, text: str, start_offset: int, max_size: int) -> List[Dict]:
        """Split oversized chunks at sentence boundaries"""
        sentences = list(re.finditer(r'[.!?]\s+', text))

        if not sentences:
            # Fallback to word boundaries
            words = text.split()
            mid_point = len(words) // 2
            split_pos = len(' '.join(words[:mid_point]))
        else:
            # Find best sentence boundary near middle
            target_pos = len(text) // 2
            best_sentence = min(sentences, key=lambda s: abs(s.end() - target_pos))
            split_pos = best_sentence.end()

        return [
            self._create_chunk_dict(
                text[:split_pos], start_offset, start_offset + split_pos, 
                self._classify_chunk_type(text[:split_pos])
            ),
            self._create_chunk_dict(
                text[split_pos:], start_offset + split_pos, start_offset + len(text), 
                self._classify_chunk_type(text[split_pos:])
            )
        ]

    def _classify_chunk_type(self, text: str) -> str:
        """Classify the semantic type of a text chunk"""
        text_lower = text.lower()

        if re.search(r'\bwhereas\b', text_lower):
            return 'whereas_clause'
        elif re.search(r'\bnow therefore\b', text_lower):
            return 'therefore_clause'
        elif re.search(r'\b(payment|fee|cost|amount|\$)\b', text_lower):
            return 'financial_clause'
        elif re.search(r'\b(termination|expire|end|cancel)\b', text_lower):
            return 'termination_clause'
        elif re.search(r'\b(liability|responsible|damages|indemnif)\b', text_lower):
            return 'liability_clause'
        elif re.search(r'\b(definition|means|defined as|shall mean)\b', text_lower):
            return 'definition'
        elif re.search(r'^\d+\.', text.strip()):
            return 'numbered_section'
        elif re.search(r'\b(confidential|proprietary|intellectual property)\b', text_lower):
            return 'confidentiality_clause'
        else:
            return 'general_clause'

    def _add_semantic_overlaps(self, chunks: List[Dict], overlap_percentage: float) -> List[Dict]:
        """Add intelligent overlaps between chunks"""
        if not chunks:
            return chunks

        for i in range(len(chunks)):
            chunk_text = chunks[i]['text']

            # Add overlap with previous chunk
            if i > 0:
                prev_chunk = chunks[i - 1]
                overlap_size = int(len(prev_chunk['text']) * overlap_percentage)
                if overlap_size > 0:
                    prev_overlap = prev_chunk['text'][-overlap_size:].strip()
                    # Find word boundary for clean overlap
                    first_space = prev_overlap.find(' ')
                    if first_space > 0:
                        prev_overlap = prev_overlap[first_space:].strip()
                    if prev_overlap:
                        chunks[i]['text'] = f"{prev_overlap} {chunk_text}"

            # Add overlap with next chunk
            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                overlap_size = int(len(next_chunk['text']) * overlap_percentage)
                if overlap_size > 0:
                    next_overlap = next_chunk['text'][:overlap_size].strip()
                    # Find word boundary for clean overlap
                    last_space = next_overlap.rfind(' ')
                    if last_space > 0:
                        next_overlap = next_overlap[:last_space].strip()
                    if next_overlap:
                        chunks[i]['text'] = f"{chunks[i]['text']} {next_overlap}"

        return chunks

    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using Vertex AI"""
        if not self.embedding_model:
            raise Exception("Embedding model not initialized")

        try:
            # Generate embeddings using Vertex AI TextEmbeddingModel
            embeddings = []

            # Process texts in batches to avoid rate limits
            batch_size = 5
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                batch_embeddings = self.embedding_model.get_embeddings(batch_texts)

                for embedding in batch_embeddings:
                    embeddings.append(embedding.values)

            # Track tokens for embedding generation
            # Estimate tokens (rough approximation for monitoring)
            estimated_tokens = sum(len(text.split()) for text in texts) * 1.3
            simple_counter.total_tokens += int(estimated_tokens)
            simple_counter.request_count += 1

            logger.info(f"Generated {len(embeddings)} embeddings using Vertex AI")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    async def _upsert_datapoints(self, datapoints: List[IndexDatapoint]) -> bool:
        """Upload datapoints to Vertex AI Vector Search index"""
        if not self.index_id:
            raise Exception("Vector Search index ID not configured")

        try:
            index = aiplatform.MatchingEngineIndex(self.index_id)

            # Upsert datapoints
            logger.info(f"Upserting {len(datapoints)} datapoints to index {self.index_id}")
            index.upsert_datapoints(datapoints=datapoints)

            logger.info("Successfully upserted datapoints to index")
            return True

        except Exception as e:
            logger.error(f"Failed to upsert datapoints: {e}")
            raise

    def _create_datapoint(self, datapoint_id: str, embedding: List[float], metadata: Dict) -> IndexDatapoint:
        """Create an IndexDatapoint for Vector Search"""
        return IndexDatapoint(
            datapoint_id=datapoint_id,
            feature_vector=embedding,
            restricts=[],
            crowding_tag=None
        )

    async def check_vector_search_status(self) -> Dict[str, any]:
        """Check the status of Vector Search components"""
        status = {
            "embedding_model_ready": self.embedding_model is not None,
            "index_configured": self.index_id is not None,
            "endpoint_configured": self.endpoint_id is not None,
            "index_exists": False,
            "endpoint_exists": False,
            "index_deployed": False
        }

        try:
            if self.index_id:
                aiplatform.MatchingEngineIndex(self.index_id)
                status["index_exists"] = True
                logger.info(f"Index {self.index_id} exists")
        except Exception as e:
            logger.warning(f"Index {self.index_id} not accessible: {e}")

        try:
            if self.endpoint_id:
                endpoint = aiplatform.MatchingEngineIndexEndpoint(self.endpoint_id)
                status["endpoint_exists"] = True

                # Check if index is deployed to endpoint
                deployed_indexes = endpoint.deployed_indexes
                status["index_deployed"] = any(
                    deployed.id == self.deployed_index_id for deployed in deployed_indexes
                )
                logger.info(f"Endpoint {self.endpoint_id} exists, deployed: {status['index_deployed']}")
        except Exception as e:
            logger.warning(f"Endpoint {self.endpoint_id} not accessible: {e}")

        return status

    async def _store_chunks_in_gcs(
        self,
        document_id: str,
        filename: str,
        chunk_dicts: List[Dict],
        embeddings: List[List[float]],
        datapoints: List
    ):
        """Store semantic chunk data and metadata in GCS for QA service retrieval"""
        try:
            import json
            from google.cloud import storage
            from datetime import datetime

            storage_client = storage.Client(project=self.project_id)
            bucket = storage_client.bucket(settings.gcs_staging_bucket)

            # Prepare enhanced chunks data in the format expected by QA service
            chunks_data = []
            for i, (chunk_dict, embedding) in enumerate(zip(chunk_dicts, embeddings)):
                chunk_data = {
                    "chunk_id": f"{document_id}_chunk_{i}",
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": i,
                    "text": chunk_dict['text'],
                    "embedding": embedding,
                    "semantic_type": chunk_dict['semantic_type'],
                    "start_position": chunk_dict['start_position'],
                    "end_position": chunk_dict['end_position'],
                    "metadata": {
                        "page_number": None,  # Could be enhanced to extract page numbers
                        "upload_timestamp": datetime.utcnow().isoformat(),
                        "chunking_strategy": "semantic"
                    }
                }
                chunks_data.append(chunk_data)

            # Store chunks data
            chunks_path = f"vector_search/{document_id}/chunks.json"
            chunks_blob = bucket.blob(chunks_path)
            chunks_blob.upload_from_string(json.dumps(chunks_data, indent=2))

            # Store enhanced document metadata
            semantic_types = list(set(chunk_dict['semantic_type'] for chunk_dict in chunk_dicts))
            doc_metadata = {
                "document_id": document_id,
                "filename": filename,
                "total_chunks": len(chunk_dicts),
                "upload_timestamp": datetime.utcnow().isoformat(),
                "embedding_model": "text-embedding-005",
                "chunk_size": settings.chunk_size,
                "min_chunk_size": settings.semantic_min_chunk_size,
                "overlap_percentage": settings.semantic_overlap_percentage,
                "chunking_strategy": "semantic",
                "semantic_types": semantic_types,
                "upload_method": "vector_search"
            }

            metadata_path = f"vector_search/{document_id}/metadata.json"
            metadata_blob = bucket.blob(metadata_path)
            metadata_blob.upload_from_string(json.dumps(doc_metadata, indent=2))

            logger.info(f"Stored chunk data and metadata in GCS for document {document_id}")

        except Exception as e:
            logger.warning(f"Failed to store chunks in GCS for {document_id}: {e}")
            # Don't fail the upload if GCS storage fails

    async def search_similar_documents(
        self,
        question: str,
        max_results: int = 5,
        similarity_threshold: float = 0.3
    ) -> List[Dict]:
        """Search for similar documents in Vector Search index using question embeddings"""
        if not self.embedding_model:
            raise Exception("Embedding model not initialized")

        try:
            # Generate embedding for the question
            question_embeddings = await self._generate_embeddings([question])
            question_embedding = question_embeddings[0]

            logger.info(f"Generated question embedding with {len(question_embedding)} dimensions")

            # Initialize aiplatform if not already done
            aiplatform.init(project=self.project_id, location=self.location)

            endpoint_resource_name = (
                f"projects/{self.project_id}/locations/{self.location}/"
                f"indexEndpoints/{self.endpoint_id}"
            )

            endpoint = MatchingEngineIndexEndpoint(endpoint_resource_name)

            # Execute the query using the find_neighbors method
            response = endpoint.find_neighbors(
                deployed_index_id=self.deployed_index_id,
                queries=[question_embedding],  # List of query vectors
                num_neighbors=max_results * 2
            )

            results = []
            if response:
                for query_neighbors in response:
                    for neighbor in query_neighbors:
                        # Get the distance/similarity score
                        similarity_score = float(neighbor.distance)

                        # Apply similarity threshold
                        if similarity_score >= similarity_threshold:
                            datapoint_id = neighbor.id

                            # Parse the datapoint_id to extract metadata
                            # Format: {document_id}_chunk_{chunk_index}
                            parts = datapoint_id.split('_chunk_')
                            document_id = parts[0] if len(parts) > 0 else ""
                            chunk_index = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

                            result = {
                                "document_id": document_id,
                                "chunk_id": datapoint_id,
                                "chunk_index": chunk_index,
                                "similarity_score": similarity_score,
                                "text": "",
                                "filename": "Unknown Document",
                            }
                            results.append(result)

                    # Only process the first query's results since we're sending one
                    break

            logger.info(f"Vector Search found {len(results)} relevant chunks")
            return results[:max_results]

        except Exception as e:
            logger.error(f"Vector Search query failed: {e}")
            raise Exception(f"Vector Search query failed: {e}")

    async def upload_document_to_vector_search(
        self,
        file_bytes: bytes,
        filename: str,
        enable_graph_storage: bool = True
    ) -> VectorStoreUploadResult:
        """Upload a document to Vertex AI Vector Search"""
        start_time = time.time()
        document_id = str(uuid.uuid4())

        try:
            # Extract text from PDF
            text = await extract_text_from_pdf(file_bytes)

            if not text.strip():
                raise Exception(f"No text found in PDF: {filename}")

            # Chunk the text using semantic chunking
            chunk_dicts = self._chunk_text(text)
            logger.info(f"Created {len(chunk_dicts)} semantic chunks for {filename}")

            # Extract text for embeddings
            chunk_texts = [chunk_dict['text'] for chunk_dict in chunk_dicts]

            # Generate embeddings for all chunks
            embeddings = await self._generate_embeddings(chunk_texts)

            # Create datapoints
            datapoints = []
            for i, (chunk_dict, embedding) in enumerate(zip(chunk_dicts, embeddings)):
                chunk_id = f"{document_id}_chunk_{i}"
                metadata = {
                    "filename": filename,
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "chunk_text": chunk_dict['text'],
                    "semantic_type": chunk_dict['semantic_type'],
                    "start_position": chunk_dict['start_position'],
                    "end_position": chunk_dict['end_position'],
                    "upload_timestamp": datetime.utcnow().isoformat(),
                }

                datapoint = self._create_datapoint(chunk_id, embedding, metadata)
                datapoints.append(datapoint)

            # Upload to Vector Search
            await self._upsert_datapoints(datapoints)

            # Store chunk data and metadata in GCS for QA service retrieval
            await self._store_chunks_in_gcs(document_id, filename, chunk_dicts, embeddings, datapoints)

            processing_time = time.time() - start_time

            logger.info(
                f"Successfully uploaded {filename} to Vector Search: "
                f"{len(chunk_dicts)} semantic chunks, {processing_time:.2f}s"
            )

            return VectorStoreUploadResult(
                filename=filename,
                document_id=document_id,
                total_chunks=len(chunk_dicts),
                successful_chunks=len(chunk_dicts),
                failed_chunks=0,
                processing_time=processing_time,
                is_duplicate=False,
                duplicate_of_document_id=None,
                file_hash="",  # Not used for Vector Search
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Failed to upload {filename} to Vector Search: {e}")

            return VectorStoreUploadResult(
                filename=filename,
                document_id=document_id,
                total_chunks=0,
                successful_chunks=0,
                failed_chunks=1,
                processing_time=processing_time,
                is_duplicate=False,
                duplicate_of_document_id=None,
                file_hash="",
            )

    async def delete_document_from_vector_search(self, document_id: str) -> Dict[str, any]:
        """Delete all chunks of a document from Vector Search index"""
        if not self.index_id:
            raise Exception("Vector Search index ID not configured")

        try:
            # Get the index
            index = aiplatform.MatchingEngineIndex(self.index_id)

            datapoint_ids_to_remove = []
            for i in range(1000):  # Max chunks to try
                chunk_id = f"{document_id}_chunk_{i}"
                datapoint_ids_to_remove.append(chunk_id)

            logger.info(f"Attempting to remove datapoints for document {document_id}")

            # Remove datapoints from index
            index.remove_datapoints(datapoint_ids=datapoint_ids_to_remove)

            logger.info(f"Successfully removed datapoints for document {document_id}")
            return {"status": "success", "document_id": document_id, "chunks_removed": "all"}

        except Exception as e:
            logger.error(f"Failed to remove datapoints for document {document_id}: {e}")
            raise

    async def delete_document_from_gcs(self, document_id: str) -> Dict[str, any]:
        """Delete document data from GCS bucket"""
        try:
            from google.cloud import storage

            storage_client = storage.Client(project=self.project_id)
            bucket = storage_client.bucket(settings.gcs_staging_bucket)

            # Delete all blobs with the document_id prefix
            prefix = f"vector_search/{document_id}/"
            blobs = list(bucket.list_blobs(prefix=prefix))

            deleted_files = []
            for blob in blobs:
                blob.delete()
                deleted_files.append(blob.name)
                logger.info(f"Deleted GCS blob: {blob.name}")

            logger.info(f"Successfully deleted {len(deleted_files)} files for document {document_id}")
            return {
                "status": "success",
                "document_id": document_id,
                "files_deleted": len(deleted_files),
                "deleted_files": deleted_files
            }

        except Exception as e:
            logger.error(f"Failed to delete GCS data for document {document_id}: {e}")
            raise

    async def delete_all_documents_from_vector_search(self) -> Dict[str, any]:
        """Delete all documents from Vector Search index (cleanup entire index)"""
        if not self.index_id:
            raise Exception("Vector Search index ID not configured")

        try:
            # Get the index
            aiplatform.MatchingEngineIndex(self.index_id)

            # Note: Vector Search doesn't provide a direct way to list all datapoints
            # This is a more aggressive approach that would require recreating the index
            # For now, we'll log a warning about this limitation

            logger.warning(
                "Complete index cleanup not implemented - "
                "Vector Search doesn't support listing all datapoints"
            )
            logger.info("To completely clean the index, consider recreating it through the Console or CLI")

            return {
                "status": "warning",
                "message": "Complete index cleanup not supported - use delete by document_id instead"
            }

        except Exception as e:
            logger.error(f"Failed to clean up Vector Search index: {e}")
            raise

    async def delete_all_documents_from_gcs(self) -> Dict[str, any]:
        """Delete all document data from GCS bucket"""
        try:
            from google.cloud import storage

            storage_client = storage.Client(project=self.project_id)
            bucket = storage_client.bucket(settings.gcs_staging_bucket)

            # Delete all blobs with the vector_search prefix
            prefix = "vector_search/"
            blobs = list(bucket.list_blobs(prefix=prefix))

            deleted_files = []
            for blob in blobs:
                blob.delete()
                deleted_files.append(blob.name)
                logger.info(f"Deleted GCS blob: {blob.name}")

            logger.info(f"Successfully deleted all {len(deleted_files)} vector search files from GCS")
            return {
                "status": "success",
                "files_deleted": len(deleted_files),
                "deleted_files": deleted_files
            }

        except Exception as e:
            logger.error(f"Failed to delete all GCS vector search data: {e}")
            raise

# Global service instance
vertex_vector_search_service = VertexVectorSearchService()
