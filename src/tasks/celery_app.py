"""
Celery application configuration for background task processing
"""

import logging
from celery import Celery

from ..config import settings

logger = logging.getLogger(__name__)

celery_app = Celery(
    "legal_assistant",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "src.tasks.document_tasks",
        "src.tasks.processing_tasks",
    ]
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    result_expires=settings.task_result_expires,
    task_track_started=True,
    task_reject_on_worker_lost=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_routes={
        'src.tasks.document_tasks.*': {'queue': 'document_processing'},
        'src.tasks.processing_tasks.*': {'queue': 'general_processing'},
    },
    task_default_queue='default',
    task_create_missing_queues=True,
)

if __name__ == '__main__':
    celery_app.start()