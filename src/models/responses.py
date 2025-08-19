from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

from ..utils.helpers import parse_currency_string


class TextBlock(BaseModel):
    x0: float = Field(..., description="Left X coordinate")
    y0: float = Field(..., description="Top Y coordinate")
    x1: float = Field(..., description="Right X coordinate")
    y1: float = Field(..., description="Bottom Y coordinate")
    text: str = Field(..., description="Extracted text")
    block_type: int = Field(..., description="Block type from fitz")
    block_no: int = Field(..., description="Block number")


class PageContent(BaseModel):
    page_number: int = Field(..., ge=0)
    blocks: List[TextBlock]


class ParsedDocument(BaseModel):
    filename: str
    total_pages: int = Field(..., ge=1)
    content: List[PageContent]
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    file_hash: str


class BulkParseResponse(BaseModel):
    total_files: int
    successful_files: int
    failed_files: int
    results: List[ParsedDocument]
    errors: List[Dict[str, str]]


class LineItem(BaseModel):
    date: Optional[str] = None
    timekeeper_name: Optional[str] = None
    timekeeper_role: Optional[str] = None
    description: Optional[str] = None
    hours_worked: Optional[float] = None
    total_spent: Optional[float] = None

    @field_validator("hours_worked", "total_spent", mode="before")
    @classmethod
    def parse_numeric_fields(cls, v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        return parse_currency_string(v)


class ExtractedInvoice(BaseModel):
    vendor_name: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    professional_fees: Optional[float] = None
    discounts: Optional[float] = None
    tax_amount: Optional[float] = None
    total_amount: Optional[float] = None
    line_items: List[LineItem] = Field(default_factory=list)
    extraction_confidence: Optional[float] = None
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("professional_fees", "discounts", "tax_amount", "total_amount", mode="before")
    @classmethod
    def parse_currency_fields(cls, v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        return parse_currency_string(v)


class BulkEntityResponse(BaseModel):
    total_files: int
    successful_extractions: int
    failed_extractions: int
    results: List[ExtractedInvoice]
    errors: List[Dict[str, str]]


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: Optional[int] = None
    chunk_index: int
    text: str
    chunk_size: int


class VectorStoreUploadResult(BaseModel):
    filename: str
    document_id: str
    total_chunks: int
    successful_chunks: int
    failed_chunks: int
    processing_time: float
    is_duplicate: bool = False
    duplicate_of_document_id: Optional[str] = None
    file_hash: Optional[str] = None


class BulkVectorStoreResponse(BaseModel):
    total_files: int
    successful_files: int
    failed_files: int
    duplicate_files: int
    total_chunks_processed: int
    results: List[VectorStoreUploadResult]
    errors: List[Dict[str, str]]


class SourceReference(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    chunk_index: int
    text_excerpt: str = Field(..., description="Relevant text excerpt from the chunk")
    similarity_score: float = Field(..., description="Semantic similarity score (0-1)")
    page_number: Optional[int] = None


class QAResponse(BaseModel):
    question: str
    answer: str
    confidence_score: Optional[float] = Field(None, description="AI confidence in the answer (0-1)")
    sources: List[SourceReference] = Field(default_factory=list, description="Supporting source references")
    total_documents_searched: int = Field(..., description="Total documents searched")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ClearVectorStoreResponse(BaseModel):
    message: str
    documents_deleted: int
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)