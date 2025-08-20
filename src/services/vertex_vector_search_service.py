import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types import IndexDatapoint
from google import genai
from google.genai.types import EmbedContentConfig, HttpOptions

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
        """Initialize the embedding model using the newer Gemini API"""
        try:
            # Force the correct project in environment
            import os
            os.environ["GOOGLE_CLOUD_PROJECT"] = self.project_id
            
            logger.info(f"Initializing embedding client with project: {self.project_id}")
            # Use the newer Gemini embedding API
            self.embedding_model = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location,
                http_options=HttpOptions(api_version="v1")
            )
            logger.info(f"✅ Initialized embedding client for project: {self.project_id}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding model: {e}")
            self.embedding_model = None
    
    def _chunk_text(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
        """Split text into chunks for embedding"""
        chunk_size = chunk_size or settings.chunk_size
        chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at word boundaries
            if end < len(text):
                last_space = chunk.rfind(' ')
                if last_space > chunk_size * 0.8:  # Only break if reasonably close to end
                    chunk = chunk[:last_space]
                    end = start + last_space
            
            chunks.append(chunk.strip())
            start = end - chunk_overlap
            
            if start >= len(text):
                break
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using Gemini API"""
        if not self.embedding_model:
            raise Exception("Embedding model not initialized")
        
        try:
            # Use text-embedding-005 model for 768 dimensions (matches your index)
            response = self.embedding_model.models.embed_content(
                model="text-embedding-005",
                contents=texts,
                config=EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=768  # Match your Vector Search index
                )
            )
            
            # Extract embeddings from response
            embeddings = [embedding.values for embedding in response.embeddings]
            
            # Track tokens for embedding generation
            # Estimate tokens (rough approximation for monitoring)
            estimated_tokens = sum(len(text.split()) for text in texts) * 1.3  # Rough estimate
            simple_counter.total_tokens += int(estimated_tokens)
            simple_counter.request_count += 1
            
            logger.info(f"Generated {len(embeddings)} embeddings with 768 dimensions")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    async def _upsert_datapoints(self, datapoints: List[IndexDatapoint]) -> bool:
        """Upload datapoints to Vertex AI Vector Search index"""
        if not self.index_id:
            raise Exception("Vector Search index ID not configured")
        
        try:
            # Get the index
            index = aiplatform.MatchingEngineIndex(self.index_id)
            
            # Upsert datapoints
            logger.info(f"Upserting {len(datapoints)} datapoints to index {self.index_id}")
            
            # Use the index's upsert_datapoints method
            response = index.upsert_datapoints(datapoints=datapoints)
            
            logger.info(f"Successfully upserted datapoints to index")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert datapoints: {e}")
            raise
    
    def _create_datapoint(self, datapoint_id: str, embedding: List[float], metadata: Dict) -> IndexDatapoint:
        """Create an IndexDatapoint for Vector Search"""
        return IndexDatapoint(
            datapoint_id=datapoint_id,
            feature_vector=embedding,
            restricts=[],  # Add restricts if needed for filtering
            crowding_tag=None  # Add crowding tag if needed
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
                index = aiplatform.MatchingEngineIndex(self.index_id)
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
    
    async def upload_document_to_vector_search(
        self, 
        file_bytes: bytes, 
        filename: str
    ) -> VectorStoreUploadResult:
        """Upload a document to Vertex AI Vector Search"""
        start_time = time.time()
        document_id = str(uuid.uuid4())
        
        try:
            # Extract text from PDF
            text = await extract_text_from_pdf(file_bytes)
            
            if not text.strip():
                raise Exception(f"No text found in PDF: {filename}")
            
            # Chunk the text
            chunks = self._chunk_text(text)
            logger.info(f"Created {len(chunks)} chunks for {filename}")
            
            # Generate embeddings for all chunks
            embeddings = await self._generate_embeddings(chunks)
            
            # Create datapoints
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
                
                datapoint = self._create_datapoint(chunk_id, embedding, metadata)
                datapoints.append(datapoint)
            
            # Upload to Vector Search
            await self._upsert_datapoints(datapoints)
            
            processing_time = time.time() - start_time
            
            logger.info(
                f"Successfully uploaded {filename} to Vector Search: "
                f"{len(chunks)} chunks, {processing_time:.2f}s"
            )
            
            return VectorStoreUploadResult(
                filename=filename,
                document_id=document_id,
                total_chunks=len(chunks),
                successful_chunks=len(chunks),
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


# Global service instance
vertex_vector_search_service = VertexVectorSearchService()