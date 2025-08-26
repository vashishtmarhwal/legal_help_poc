from pydantic import BaseModel, Field


class QARequest(BaseModel):
    question: str = Field(..., description="The question to answer", min_length=1, max_length=1000)
    max_sources: int = Field(default=5, description="Maximum number of source references to return", ge=1, le=20)
    similarity_threshold: float = Field(
        default=0.3, 
        description="Minimum similarity score for source inclusion", 
        ge=0.0, 
        le=1.0
    )
    include_context: bool = Field(default=True, description="Whether to include context in the response")
