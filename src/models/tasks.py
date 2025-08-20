"""
Models for task status tracking and responses
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task status enumeration"""
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"
    REVOKED = "REVOKED"


class TaskType(str, Enum):
    """Task type enumeration"""
    VECTOR_SEARCH_UPLOAD = "vector_search_upload"
    ENTITY_EXTRACTION = "entity_extraction"
    PDF_PARSING = "pdf_parsing"
    DOCUMENT_QA = "document_qa"
    DOCUMENT_DELETE = "document_delete"
    BULK_CLEANUP = "bulk_cleanup"


class TaskProgress(BaseModel):
    """Task progress information"""
    current: int = Field(..., description="Current step number")
    total: int = Field(..., description="Total number of steps")
    description: str = Field(..., description="Current step description")
    percentage: float = Field(..., description="Progress percentage (0-100)")


class TaskResult(BaseModel):
    """Task result information"""
    task_id: str = Field(..., description="Unique task identifier")
    task_type: TaskType = Field(..., description="Type of task")
    status: TaskStatus = Field(..., description="Current task status")
    progress: Optional[TaskProgress] = Field(None, description="Progress information")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result data")
    error: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(..., description="Task creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Task start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Task completion timestamp")
    processing_time: Optional[float] = Field(None, description="Total processing time in seconds")


class TaskResponse(BaseModel):
    """Response model for task creation"""
    task_id: str = Field(..., description="Unique task identifier")
    task_type: TaskType = Field(..., description="Type of task")
    status: TaskStatus = Field(..., description="Initial task status")
    message: str = Field(..., description="Confirmation message")
    status_url: str = Field(..., description="URL to check task status")


class BulkTaskResponse(BaseModel):
    """Response model for bulk task creation"""
    tasks: List[TaskResponse] = Field(..., description="List of created tasks")
    total_tasks: int = Field(..., description="Total number of tasks created")
    message: str = Field(..., description="Confirmation message")


class TaskListResponse(BaseModel):
    """Response model for task listing"""
    tasks: List[TaskResult] = Field(..., description="List of tasks")
    total: int = Field(..., description="Total number of tasks")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of tasks per page")


class TaskStatsResponse(BaseModel):
    """Response model for task statistics"""
    total_tasks: int = Field(..., description="Total number of tasks")
    pending_tasks: int = Field(..., description="Number of pending tasks")
    running_tasks: int = Field(..., description="Number of running tasks")
    completed_tasks: int = Field(..., description="Number of completed tasks")
    failed_tasks: int = Field(..., description="Number of failed tasks")
    queues: Dict[str, int] = Field(..., description="Tasks per queue")