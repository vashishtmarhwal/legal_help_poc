import logging
from datetime import datetime
from typing import Optional

from ..config import settings
from .storage.base import StorageBackend, UsageRecord
from .storage.file_storage import FileStorage
from .storage.gcs_storage import GCSStorage

logger = logging.getLogger(__name__)


class MetricsCollector:
    def __init__(self, storage_backend: Optional[StorageBackend] = None):
        if storage_backend:
            self.storage = storage_backend
        else:
            self.storage = self._get_default_storage()

    def _get_default_storage(self) -> StorageBackend:
        backend_type = settings.monitoring_storage_backend.lower()
        
        if backend_type == "gcs":
            return GCSStorage()
        elif backend_type == "file":
            return FileStorage(settings.monitoring_file_path)
        elif backend_type == "auto":
            try:
                if settings.gcs_staging_bucket:
                    return GCSStorage()
            except Exception as e:
                logger.warning(f"Failed to initialize GCS storage, falling back to file storage: {e}")
            return FileStorage(settings.monitoring_file_path)
        else:
            logger.warning(f"Unknown storage backend '{backend_type}', using file storage")
            return FileStorage(settings.monitoring_file_path)

    async def store_usage_record(self, record: UsageRecord) -> None:
        try:
            await self.storage.store_usage_record(record)
        except Exception as e:
            logger.error(f"Failed to store usage record: {e}")
            raise

    async def get_daily_usage(
        self,
        date: datetime,
        model_name: Optional[str] = None,
        endpoint: Optional[str] = None,
    ):
        try:
            return await self.storage.get_daily_aggregation(date, model_name, endpoint)
        except Exception as e:
            logger.error(f"Failed to get daily usage: {e}")
            return {}

    async def get_monthly_usage(
        self,
        year: int,
        month: int,
        model_name: Optional[str] = None,
        endpoint: Optional[str] = None,
    ):
        try:
            return await self.storage.get_monthly_aggregation(year, month, model_name, endpoint)
        except Exception as e:
            logger.error(f"Failed to get monthly usage: {e}")
            return {}

    async def get_usage_records(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        model_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        try:
            return await self.storage.get_usage_records(
                start_date, end_date, model_name, endpoint, limit
            )
        except Exception as e:
            logger.error(f"Failed to get usage records: {e}")
            return []