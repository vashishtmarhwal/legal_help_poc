"""
Task management and status endpoints
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from celery import current_app as celery_app
from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, status, Query

from ...models.tasks import TaskResult, TaskResponse, TaskStatus, TaskType, TaskListResponse, TaskStatsResponse
from ...services.auth_service import verify_admin_key
from ...tasks.celery_app import celery_app as app

logger = logging.getLogger(__name__)
router = APIRouter()


def get_celery_task_result(task_id: str) -> AsyncResult:
    """Get Celery task result by ID"""
    return AsyncResult(task_id, app=app)


def convert_celery_state_to_task_status(state: str) -> TaskStatus:
    """Convert Celery task state to TaskStatus enum"""
    state_mapping = {
        'PENDING': TaskStatus.PENDING,
        'STARTED': TaskStatus.STARTED,
        'PROGRESS': TaskStatus.STARTED,  # Map PROGRESS to STARTED for simplicity
        'SUCCESS': TaskStatus.SUCCESS,
        'FAILURE': TaskStatus.FAILURE,
        'RETRY': TaskStatus.RETRY,
        'REVOKED': TaskStatus.REVOKED,
    }
    return state_mapping.get(state, TaskStatus.PENDING)


def get_task_type_from_task_name(task_name: str) -> TaskType:
    """Determine task type from Celery task name"""
    if 'upload_document_to_vector_search' in task_name:
        return TaskType.VECTOR_SEARCH_UPLOAD
    elif 'extract_entities' in task_name:
        return TaskType.ENTITY_EXTRACTION
    elif 'parse_pdf' in task_name:
        return TaskType.PDF_PARSING
    elif 'document_qa' in task_name:
        return TaskType.DOCUMENT_QA
    elif 'delete_document' in task_name:
        return TaskType.DOCUMENT_DELETE
    elif 'cleanup_all' in task_name:
        return TaskType.BULK_CLEANUP
    else:
        return TaskType.VECTOR_SEARCH_UPLOAD  # Default fallback


@router.get("/tasks/{task_id}", response_model=TaskResult)
async def get_task_status(task_id: str):
    """
    Get the status and result of a specific task
    
    Args:
        task_id: The task ID to check
        
    Returns:
        Task status and result information
    """
    try:
        result = get_celery_task_result(task_id)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        task_status = convert_celery_state_to_task_status(result.state)
        task_type = get_task_type_from_task_name(result.name or "")
        
        # Get task info and timestamps
        task_info = result.info if result.info else {}
        
        # Handle different result states
        if result.state == 'PENDING':
            task_result = TaskResult(
                task_id=task_id,
                task_type=task_type,
                status=task_status,
                created_at=datetime.utcnow(),  # We don't have exact creation time
                result=None,
                error=None
            )
        elif result.state == 'PROGRESS' or result.state == 'STARTED':
            progress_info = task_info.get('progress')
            task_result = TaskResult(
                task_id=task_id,
                task_type=task_type,
                status=task_status,
                progress=progress_info,
                created_at=datetime.utcnow(),  # Approximation
                started_at=datetime.utcnow(),  # Approximation
                result=None,
                error=None
            )
        elif result.state == 'SUCCESS':
            completed_at = datetime.utcnow()  # Approximation
            processing_time = task_info.get('processing_time') if isinstance(task_info, dict) else None
            
            task_result = TaskResult(
                task_id=task_id,
                task_type=task_type,
                status=task_status,
                result=task_info,
                created_at=datetime.utcnow(),  # Approximation
                started_at=datetime.utcnow(),   # Approximation
                completed_at=completed_at,
                processing_time=processing_time,
                error=None
            )
        elif result.state == 'FAILURE':
            error_message = str(result.info) if result.info else "Unknown error"
            task_result = TaskResult(
                task_id=task_id,
                task_type=task_type,
                status=task_status,
                result=None,
                error=error_message,
                created_at=datetime.utcnow(),  # Approximation
                completed_at=datetime.utcnow()   # Approximation
            )
        else:
            # Handle other states (RETRY, REVOKED, etc.)
            task_result = TaskResult(
                task_id=task_id,
                task_type=task_type,
                status=task_status,
                result=task_info if isinstance(task_info, dict) else None,
                error=str(task_info) if result.state in ['RETRY', 'REVOKED'] else None,
                created_at=datetime.utcnow()  # Approximation
            )
        
        return task_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status for {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve task status: {e}"
        )


@router.get("/tasks/{task_id}/result")
async def get_task_result(task_id: str):
    """
    Get the result of a completed task
    
    Args:
        task_id: The task ID to get results for
        
    Returns:
        Task result data
    """
    try:
        result = get_celery_task_result(task_id)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        if result.state == 'PENDING':
            raise HTTPException(
                status_code=status.HTTP_202_ACCEPTED,
                detail="Task is still pending"
            )
        elif result.state in ['PROGRESS', 'STARTED']:
            raise HTTPException(
                status_code=status.HTTP_202_ACCEPTED,
                detail="Task is still running"
            )
        elif result.state == 'SUCCESS':
            return {
                "task_id": task_id,
                "status": "success",
                "result": result.result
            }
        elif result.state == 'FAILURE':
            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(result.info)
            }
        else:
            return {
                "task_id": task_id,
                "status": result.state.lower(),
                "result": result.info if result.info else None
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task result for {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve task result: {e}"
        )


@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str, _: str = Depends(verify_admin_key)):
    """
    Cancel a running task (admin only)
    
    Args:
        task_id: The task ID to cancel
        
    Returns:
        Cancellation confirmation
    """
    try:
        result = get_celery_task_result(task_id)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        if result.state in ['SUCCESS', 'FAILURE', 'REVOKED']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Task {task_id} cannot be cancelled (state: {result.state})"
            )
        
        # Revoke the task
        app.control.revoke(task_id, terminate=True)
        
        logger.info(f"Task {task_id} has been cancelled")
        
        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": f"Task {task_id} has been cancelled"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel task: {e}"
        )


@router.get("/tasks", response_model=TaskListResponse)
async def list_tasks(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Number of tasks per page"),
    status_filter: Optional[TaskStatus] = Query(None, description="Filter by task status"),
    _: str = Depends(verify_admin_key)
):
    """
    List tasks with pagination (admin only)
    
    Args:
        page: Page number (starting from 1)
        page_size: Number of tasks per page
        status_filter: Optional filter by task status
        
    Returns:
        Paginated list of tasks
    """
    try:
        # Get active tasks from Celery
        inspect = app.control.inspect()
        
        # Get all task information
        active_tasks = inspect.active() or {}
        scheduled_tasks = inspect.scheduled() or {}
        reserved_tasks = inspect.reserved() or {}
        
        # Combine all tasks
        all_tasks = []
        
        # Process active tasks
        for worker, tasks in active_tasks.items():
            for task_info in tasks:
                task_id = task_info['id']
                task_name = task_info['name']
                task_type = get_task_type_from_task_name(task_name)
                
                all_tasks.append(TaskResult(
                    task_id=task_id,
                    task_type=task_type,
                    status=TaskStatus.STARTED,
                    created_at=datetime.utcnow(),
                    started_at=datetime.utcnow()
                ))
        
        # Process scheduled tasks
        for worker, tasks in scheduled_tasks.items():
            for task_info in tasks:
                task_id = task_info['request']['id']
                task_name = task_info['request']['name']
                task_type = get_task_type_from_task_name(task_name)
                
                all_tasks.append(TaskResult(
                    task_id=task_id,
                    task_type=task_type,
                    status=TaskStatus.PENDING,
                    created_at=datetime.utcnow()
                ))
        
        # Apply status filter if provided
        if status_filter:
            all_tasks = [task for task in all_tasks if task.status == status_filter]
        
        # Apply pagination
        total_tasks = len(all_tasks)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_tasks = all_tasks[start_idx:end_idx]
        
        return TaskListResponse(
            tasks=paginated_tasks,
            total=total_tasks,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve task list: {e}"
        )


@router.get("/tasks/stats", response_model=TaskStatsResponse)
async def get_task_stats(_: str = Depends(verify_admin_key)):
    """
    Get task statistics (admin only)
    
    Returns:
        Task statistics including counts by status and queue
    """
    try:
        inspect = app.control.inspect()
        
        # Get task information
        active_tasks = inspect.active() or {}
        scheduled_tasks = inspect.scheduled() or {}
        reserved_tasks = inspect.reserved() or {}
        
        # Count tasks
        total_active = sum(len(tasks) for tasks in active_tasks.values())
        total_scheduled = sum(len(tasks) for tasks in scheduled_tasks.values())
        total_reserved = sum(len(tasks) for tasks in reserved_tasks.values())
        
        # Get queue information
        queues = {}
        for worker, tasks in active_tasks.items():
            for task in tasks:
                queue_name = task.get('delivery_info', {}).get('routing_key', 'default')
                queues[queue_name] = queues.get(queue_name, 0) + 1
        
        return TaskStatsResponse(
            total_tasks=total_active + total_scheduled + total_reserved,
            pending_tasks=total_scheduled,
            running_tasks=total_active,
            completed_tasks=0,  # We don't track completed tasks in memory
            failed_tasks=0,     # We don't track failed tasks in memory
            queues=queues
        )
        
    except Exception as e:
        logger.error(f"Failed to get task statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve task statistics: {e}"
        )