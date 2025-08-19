# main.py
"""
Enterprise-Grade Legal Document Assistant API
Handles PDF parsing, entity extraction, and vector indexing using Vertex AI
"""

import os
import sys
import json
import logging
import hashlib
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
import traceback

import fitz  # PyMuPDF
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator, ConfigDict
from pydantic_settings import BaseSettings

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, HarmCategory, HarmBlockThreshold

from llama_index.core import Document, Settings, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.vertexaivectorsearch import VertexAIVectorStore

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration Settings ---
class AppSettings(BaseSettings):
    """Application configuration with validation"""
    google_cloud_project: str = Field(default="legal-assistant-poc", env="GOOGLE_CLOUD_PROJECT")
    location: str = Field(default="us-central1", env="LOCATION")
    vector_index_id: Optional[str] = Field(default="6646980997487263744", env="VECTOR_INDEX_ID")
    vector_endpoint_id: Optional[str] = Field(default="5613703950168489984", env="VECTOR_ENDPOINT_ID")
    # gcs_staging_bucket: Optional[str] = Field(default="legal-assistant-staging-bucket-datatonicpoc", env="GCS_STAGING_BUCKET")

    # Model settings
    model_name: str = Field(default="gemini-2.5-flash-lite", env="MODEL_NAME")
    embedding_model: str = Field(default="infly/inf-retriever-v1-1.5b", env="EMBEDDING_MODEL")
    
    # Processing settings
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    max_text_length: int = Field(default=100000, env="MAX_TEXT_LENGTH")
    
    # API settings
    api_title: str = "Legal Document Assistant API"
    api_version: str = "1.1.0"
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    allowed_hosts: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(default=60, env="RATE_LIMIT_PERIOD")
    
    model_config = ConfigDict(env_file=".env", case_sensitive=False)

# Initialize settings
settings = AppSettings()

# --- Global Variables ---
model: Optional[GenerativeModel] = None
vector_store: Optional[VertexAIVectorStore] = None
is_initialized = False

# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown"""
    global model, vector_store, is_initialized
    
    logger.info("Starting up Legal Document Assistant API...")
    
    try:
        # Initialize Vertex AI
        vertexai.init(project=settings.google_cloud_project, location=settings.location)
        logger.info(f"✅ Vertex AI initialized (project={settings.google_cloud_project}, location={settings.location})")
        
        # Initialize Gemini model with safety settings
        model = GenerativeModel(
            settings.model_name,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            }
        )
        logger.info(f"✅ Model {settings.model_name} initialized")
        
        # Initialize LlamaIndex settings
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=settings.embedding_model,
            cache_folder="./embedding_cache"
        )
        Settings.chunk_size = settings.chunk_size
        Settings.chunk_overlap = settings.chunk_overlap
        logger.info(f"✅ Embedding model {settings.embedding_model} initialized")
        
        # Initialize Vector Store if credentials are provided
        if settings.vector_index_id and settings.vector_endpoint_id:
            vector_store = VertexAIVectorStore(
                project_id=settings.google_cloud_project,
                index_id=settings.vector_index_id,
                endpoint_id=settings.vector_endpoint_id,
                region=settings.location,
                # gcs_staging_dir=f"gs://{settings.gcs_staging_bucket}/staging",
                index_management=True
            )
            logger.info("✅ Vector store initialized")
            logger.info(
                f"Initializing VertexAIVectorStore with: "
                f"project_id={settings.google_cloud_project}, "
                f"index_id={settings.vector_index_id}, "
                f"endpoint_id={settings.vector_endpoint_id}, "
                f"region={settings.location}, "
                # f"gcs_staging_dir={settings.gcs_staging_bucket}"
            )

        else:
            logger.warning("⚠️ Vector store not initialized - missing VECTOR_INDEX_ID or VECTOR_ENDPOINT_ID")
        
        is_initialized = True
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        logger.error(traceback.format_exc())
        is_initialized = False
    
    yield  # Application runs here
    
    # Cleanup on shutdown
    logger.info("Shutting down Legal Document Assistant API...")
    # Add any cleanup code here if needed

# --- FastAPI Application ---
app = FastAPI(
    title=settings.api_title,
    description="Enterprise API for parsing legal documents and extracting structured information using AI",
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- Middleware ---
# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted host middleware for security
if settings.allowed_hosts != ["*"]:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts
    )

# --- Exception Handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please try again later."}
    )

# --- Pydantic Models ---
class TextBlock(BaseModel):
    """Represents a single block of text extracted from a PDF page"""
    x0: float = Field(..., description="X-coordinate of top-left corner")
    y0: float = Field(..., description="Y-coordinate of top-left corner")
    x1: float = Field(..., description="X-coordinate of bottom-right corner")
    y1: float = Field(..., description="Y-coordinate of bottom-right corner")
    text: str = Field(..., min_length=1)
    block_type: int
    block_no: int

class PageContent(BaseModel):
    """Represents all extracted content from a single PDF page"""
    page_number: int = Field(..., ge=0)
    blocks: List[TextBlock]

class ParsedDocument(BaseModel):
    """Structured representation of entire parsed PDF document"""
    filename: str
    total_pages: int = Field(..., ge=1)
    content: List[PageContent]
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    file_hash: Optional[str] = None

class LineItem(BaseModel):
    """Represents a single line item of work performed"""
    date: Optional[str] = Field(None, description="Date work was performed")
    timekeeper_name: Optional[str] = Field(None, description="Name of person who performed work")
    timekeeper_role: Optional[str] = Field(None, description="Role (e.g., Partner, Associate)")
    description: Optional[str] = Field(None, description="Description of work performed")
    hours_worked: Optional[float] = Field(None, ge=0, description="Hours worked")
    total_spent: Optional[float] = Field(None, ge=0, description="Total cost for line item")
    
    @validator('hours_worked', 'total_spent')
    def validate_positive(cls, v):
        if v is not None and v < 0:
            raise ValueError('Value must be non-negative')
        return v

class ExtractedInvoice(BaseModel):
    """Detailed schema for structured data extracted from invoice"""
    vendor_name: Optional[str] = Field(None, description="Law firm or vendor name")
    invoice_number: Optional[str] = Field(None, description="Unique invoice identifier")
    invoice_date: Optional[str] = Field(None, description="Invoice issue date")
    professional_fees: Optional[float] = Field(None, ge=0, description="Subtotal for professional fees")
    discounts: Optional[float] = Field(None, ge=0, description="Total discounts applied")
    tax_amount: Optional[float] = Field(None, ge=0, description="Total tax amount")
    total_amount: Optional[float] = Field(None, ge=0, description="Final total amount due")
    line_items: List[LineItem] = Field(default_factory=list, description="Individual work items")
    extraction_confidence: Optional[float] = Field(None, ge=0, le=1, description="Confidence score")
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)

class IndexingResponse(BaseModel):
    """Response model for indexing endpoint"""
    status: str
    message: str
    document_id: str
    nodes_added: int
    processing_time_seconds: float
    file_hash: str

class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str
    vertex_ai_configured: bool
    vector_store_configured: bool
    project: Optional[str]
    location: Optional[str]
    model_name: Optional[str]
    timestamp: datetime

# --- Dependency Injection ---
async def verify_initialization():
    """Dependency to ensure services are initialized"""
    if not is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is still initializing. Please try again in a moment."
        )

async def verify_model():
    """Dependency to ensure model is available"""
    if not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI model is not available. Please check configuration."
        )

async def verify_vector_store():
    """Dependency to ensure vector store is available"""
    if not vector_store:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store is not configured. Please check VECTOR_INDEX_ID and VECTOR_ENDPOINT_ID."
        )

# --- Helper Functions ---
def calculate_file_hash(file_bytes: bytes) -> str:
    """Calculate SHA256 hash of file for deduplication"""
    return hashlib.sha256(file_bytes).hexdigest()

async def extract_text_from_pdf(file_bytes: bytes, max_pages: Optional[int] = None) -> str:
    """
    Extract text from PDF with error handling and page limit
    
    Args:
        file_bytes: PDF file content
        max_pages: Maximum number of pages to process (None for all)
    
    Returns:
        Extracted text content
    """
    try:
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        full_text = ""
        
        pages_to_process = min(len(pdf_document), max_pages) if max_pages else len(pdf_document)
        
        for page_num in range(pages_to_process):
            page = pdf_document[page_num]
            page_text = page.get_text()
            full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        
        pdf_document.close()
        
        # Truncate if too long
        if len(full_text) > settings.max_text_length:
            logger.warning(f"Text truncated from {len(full_text)} to {settings.max_text_length} characters")
            full_text = full_text[:settings.max_text_length]
        
        return full_text.strip()
    
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to extract text from PDF: {str(e)}"
        )

async def validate_pdf_file(file: UploadFile) -> bytes:
    """
    Validate uploaded PDF file
    
    Args:
        file: Uploaded file
    
    Returns:
        File content as bytes
    """
    # Check content type
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. Please upload a PDF file."
        )
    
    # Read file content
    file_bytes = await file.read()
    
    # Check file size
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed size ({settings.max_file_size_mb} MB)"
        )
    
    # Verify it's a valid PDF
    try:
        test_doc = fitz.open(stream=file_bytes, filetype="pdf")
        test_doc.close()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File appears to be corrupted or is not a valid PDF"
        )
    
    return file_bytes

# --- API Endpoints ---

@app.get("/", summary="Root endpoint")
async def read_root():
    """Welcome endpoint with API information"""
    return {
        "message": "Welcome to the Legal Document Assistant API",
        "version": settings.api_version,
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheckResponse, summary="Health check")
async def health_check():
    """Comprehensive health check endpoint"""
    return HealthCheckResponse(
        status="healthy" if is_initialized else "initializing",
        vertex_ai_configured=model is not None,
        vector_store_configured=vector_store is not None,
        project=settings.google_cloud_project,
        location=settings.location,
        model_name=settings.model_name if model else None,
        timestamp=datetime.utcnow()
    )

@app.post("/upload-and-parse/", 
         response_model=ParsedDocument,
         summary="Upload and Parse PDF",
         dependencies=[Depends(verify_initialization)])
async def upload_and_parse_pdf(
    file: UploadFile = File(..., description="PDF file to parse"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Parse PDF and extract structured text with layout information
    
    - Validates PDF file
    - Extracts text blocks with coordinates
    - Returns structured document representation
    """
    logger.info(f"Parsing PDF: {file.filename}")
    
    # Validate and read file
    file_bytes = await validate_pdf_file(file)
    file_hash = calculate_file_hash(file_bytes)
    
    try:
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        document_content = []
        
        for page_num, page in enumerate(pdf_document):
            blocks = page.get_text("blocks")
            structured_blocks = [
                TextBlock(
                    x0=b[0], y0=b[1], x1=b[2], y1=b[3],
                    text=b[4].strip(),
                    block_type=b[5],
                    block_no=b[6]
                )
                for b in blocks if b[4].strip()
            ]
            document_content.append(
                PageContent(page_number=page_num, blocks=structured_blocks)
            )
        
        total_pages = len(pdf_document)
        pdf_document.close()
        
        logger.info(f"Successfully parsed {total_pages} pages from {file.filename}")
        
        return ParsedDocument(
            filename=file.filename,
            total_pages=total_pages,
            content=document_content,
            file_hash=file_hash
        )
        
    except Exception as e:
        logger.error(f"Error parsing PDF {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error parsing PDF: {str(e)}"
        )

@app.post("/extract-entities/",
         response_model=ExtractedInvoice,
         summary="Extract Invoice Entities",
         dependencies=[Depends(verify_initialization), Depends(verify_model)])
async def extract_entities_from_pdf(
    file: UploadFile = File(..., description="Invoice PDF to analyze")
):
    """
    Extract structured invoice data using Gemini AI
    
    - Extracts vendor, dates, amounts, line items
    - Uses advanced prompting for accurate extraction
    - Returns structured invoice data
    """
    logger.info(f"Extracting entities from: {file.filename}")
    
    # Validate and extract text
    file_bytes = await validate_pdf_file(file)
    full_text = await extract_text_from_pdf(file_bytes)
    
    if not full_text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No text found in PDF"
        )
    
    # Enhanced prompt with better instructions
    prompt = f"""You are an expert legal invoice parser with years of experience analyzing law firm invoices.
    
Your task is to carefully extract ALL the following information from the invoice text:

1. **vendor_name**: The law firm or vendor name (look for letterhead, "From:", or company names)
2. **invoice_number**: The invoice number (look for "Invoice #", "Bill No.", "Matter No.")
3. **invoice_date**: The invoice date (look for "Date:", "Invoice Date:", "Bill Date:")
4. **professional_fees**: The subtotal of professional services BEFORE tax and discounts
5. **discounts**: Total discounts applied (as positive number)
6. **tax_amount**: Total tax charged
7. **total_amount**: The final total amount due (after all adjustments)
8. **line_items**: Extract ALL individual billing entries with:
   - date: When work was performed
   - timekeeper_name: Person who did the work
   - timekeeper_role: Their title (Partner, Associate, Paralegal, etc.)
   - description: What work was done
   - hours_worked: Hours billed
   - total_spent: Amount for that line item

IMPORTANT:
- Extract ALL line items, even if there are many
- All monetary values should be numbers without currency symbols
- If a field is not found, use null
- Ensure the response is valid JSON only

Document Text:
---
{full_text[:30000]}  # Increased limit for better extraction
---

Respond with JSON only:"""

    try:
        # Configure generation for structured output
        generation_config = GenerationConfig(
            temperature=0.1,  # Low for consistency
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            response_mime_type="application/json",
        )
        
        # Generate response
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
        )
        
        # Parse and validate response
        try:
            json_text = response.text.strip()
            
            # Clean up common JSON formatting issues
            for prefix in ["```json", "```", "json"]:
                if json_text.startswith(prefix):
                    json_text = json_text[len(prefix):]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            json_text = json_text.strip()
            
            # Parse JSON
            extracted_data = ExtractedInvoice.model_validate_json(json_text)
            
            # Add confidence score based on completeness
            fields_found = sum([
                extracted_data.vendor_name is not None,
                extracted_data.invoice_number is not None,
                extracted_data.invoice_date is not None,
                extracted_data.total_amount is not None,
                len(extracted_data.line_items) > 0
            ])
            extracted_data.extraction_confidence = fields_found / 5.0
            
            logger.info(f"Successfully extracted {len(extracted_data.line_items)} line items from {file.filename}")
            return extracted_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Raw response: {response.text[:500]}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="AI model returned invalid JSON. Please try again."
            )
            
    except Exception as e:
        logger.error(f"Error during entity extraction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Entity extraction failed: {str(e)}"
        )

@app.post("/upload-and-index/",
         response_model=IndexingResponse,
         summary="Index PDF in Vector Store",
         dependencies=[Depends(verify_initialization), Depends(verify_vector_store)])
async def upload_and_index_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF to index"),
    document_id: str = Form(..., description="Unique document identifier", min_length=1, max_length=100),
    metadata: Optional[str] = Form(None, description="Additional metadata as JSON string")
):
    """
    Index PDF content in Vertex AI Vector Search
    
    - Chunks document intelligently
    - Generates embeddings using legal-specific model
    - Stores in vector database for retrieval
    - Supports custom metadata
    """
    import time
    start_time = time.time()
    
    logger.info(f"Indexing document: {document_id} from file: {file.filename}")
    
    # Validate document_id format (alphanumeric, hyphens, underscores)
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', document_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document ID must contain only alphanumeric characters, hyphens, and underscores"
        )
    
    # Validate and extract text
    file_bytes = await validate_pdf_file(file)
    file_hash = calculate_file_hash(file_bytes)
    document_text = await extract_text_from_pdf(file_bytes)
    
    if not document_text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No text found in PDF"
        )
    
    # Parse metadata if provided
    doc_metadata = {"filename": file.filename, "file_hash": file_hash}
    if metadata:
        try:
            additional_metadata = json.loads(metadata)
            if isinstance(additional_metadata, dict):
                doc_metadata.update(additional_metadata)
        except json.JSONDecodeError:
            logger.warning(f"Invalid metadata JSON provided for {document_id}")
    
    try:
        # Create LlamaIndex Document
        llama_document = Document(
            text=document_text,
            doc_id=document_id,
            metadata=doc_metadata,
            excluded_embed_metadata_keys=["file_hash"],  # Exclude from embeddings
            excluded_llm_metadata_keys=["file_hash"],  # Exclude from LLM context
        )
        
        # Configure storage context with vector store
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index and add document
        # This automatically chunks, embeds, and stores in vector database
        index = VectorStoreIndex.from_documents(
            [llama_document],
            storage_context=storage_context,
            show_progress=True,
            insert_batch_size=100,  # Optimize batch size
        )
        
        # Get number of nodes created
        nodes_added = len(index.docstore.docs) if hasattr(index, 'docstore') else 0
        
        processing_time = time.time() - start_time
        
        logger.info(f"Successfully indexed {document_id}: {nodes_added} nodes in {processing_time:.2f}s")
        
        # Optional: Schedule background cleanup or notification
        # background_tasks.add_task(notify_indexing_complete, document_id)
        
        return IndexingResponse(
            status="success",
            message=f"Document indexed successfully in {processing_time:.2f} seconds",
            document_id=document_id,
            nodes_added=nodes_added,
            processing_time_seconds=processing_time,
            file_hash=file_hash
        )
        
    except Exception as e:
        logger.error(f"Indexing failed for {document_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Provide more specific error messages
        if "already exists" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Document with ID '{document_id}' already exists in the index"
            )
        elif "quota" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Vector index quota exceeded. Please try again later."
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Indexing failed: {str(e)}"
            )

@app.delete("/documents/{document_id}",
           summary="Delete Document from Index",
           dependencies=[Depends(verify_initialization), Depends(verify_vector_store)])
async def delete_document(document_id: str):
    """
    Delete a document from the vector index
    
    Args:
        document_id: ID of document to delete
    """
    logger.info(f"Deleting document: {document_id}")
    
    try:
        # Note: This depends on the vector store implementation
        # You may need to implement custom deletion logic
        vector_store.delete(document_id)
        
        logger.info(f"Successfully deleted document: {document_id}")
        return {"status": "success", "message": f"Document {document_id} deleted"}
        
    except Exception as e:
        logger.error(f"Failed to delete {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deletion failed: {str(e)}"
        )

# --- Main Entry Point ---
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to False in production
        log_level="info",
        access_log=True,
        workers=1  # Increase for production based on CPU cores
    )