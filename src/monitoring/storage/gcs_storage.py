import json
import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

from google.cloud import storage

from ...config import settings
from .base import StorageBackend, UsageRecord

logger = logging.getLogger(__name__)


class GCSStorage(StorageBackend):
    def __init__(self, bucket_name: Optional[str] = None):
        self.bucket_name = bucket_name or settings.gcs_staging_bucket
        if not self.bucket_name:
            raise ValueError("GCS bucket name is required for GCS storage")
        
        self.client = storage.Client(project=settings.google_cloud_project)
        self.bucket = self.client.bucket(self.bucket_name)
        self.prefix = "monitoring/"

    def _get_daily_blob_path(self, date: datetime) -> str:
        date_str = date.strftime("%Y-%m-%d")
        return f"{self.prefix}daily/usage_{date_str}.json"

    def _get_monthly_blob_path(self, year: int, month: int) -> str:
        return f"{self.prefix}monthly/usage_{year}_{month:02d}_monthly.json"

    async def store_usage_record(self, record: UsageRecord) -> None:
        try:
            blob_path = self._get_daily_blob_path(record.timestamp)
            blob = self.bucket.blob(blob_path)
            
            records = []
            if blob.exists():
                existing_data = blob.download_as_text()
                records = json.loads(existing_data)
            
            records.append(record.to_dict())
            
            blob.upload_from_string(json.dumps(records, indent=2))
            logger.debug(f"Stored usage record to GCS: {blob_path}")
            
        except Exception as e:
            logger.error(f"Failed to store usage record to GCS: {e}")
            raise

    async def get_usage_records(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        model_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[UsageRecord]:
        records = []
        
        try:
            if start_date and end_date:
                current_date = start_date
                while current_date <= end_date:
                    blob_path = self._get_daily_blob_path(current_date)
                    blob = self.bucket.blob(blob_path)
                    
                    if blob.exists():
                        data = blob.download_as_text()
                        daily_records = json.loads(data)
                        
                        for record_data in daily_records:
                            record = UsageRecord.from_dict(record_data)
                            
                            if model_name and record.model_name != model_name:
                                continue
                            if endpoint and record.endpoint != endpoint:
                                continue
                            
                            records.append(record)
                    
                    current_date = current_date.replace(day=current_date.day + 1)
            
            if limit:
                records = records[:limit]
            
            return records
            
        except Exception as e:
            logger.error(f"Failed to get usage records from GCS: {e}")
            return []

    async def get_daily_aggregation(
        self,
        date: datetime,
        model_name: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> Dict:
        try:
            blob_path = self._get_daily_blob_path(date)
            blob = self.bucket.blob(blob_path)
            
            if not blob.exists():
                return {
                    "date": date.strftime("%Y-%m-%d"),
                    "total_requests": 0,
                    "total_tokens": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_cost": 0.0,
                    "by_model": {},
                    "by_endpoint": {},
                }
            
            data = blob.download_as_text()
            records_data = json.loads(data)
            
            filtered_records = []
            for record_data in records_data:
                record = UsageRecord.from_dict(record_data)
                if model_name and record.model_name != model_name:
                    continue
                if endpoint and record.endpoint != endpoint:
                    continue
                filtered_records.append(record)
            
            total_requests = len(filtered_records)
            total_tokens = sum(r.total_tokens for r in filtered_records)
            total_input_tokens = sum(r.input_tokens for r in filtered_records)
            total_output_tokens = sum(r.output_tokens for r in filtered_records)
            total_cost = sum(r.estimated_cost for r in filtered_records)
            
            by_model = defaultdict(lambda: {"requests": 0, "tokens": 0, "cost": 0.0})
            by_endpoint = defaultdict(lambda: {"requests": 0, "tokens": 0, "cost": 0.0})
            
            for record in filtered_records:
                by_model[record.model_name]["requests"] += 1
                by_model[record.model_name]["tokens"] += record.total_tokens
                by_model[record.model_name]["cost"] += record.estimated_cost
                
                by_endpoint[record.endpoint]["requests"] += 1
                by_endpoint[record.endpoint]["tokens"] += record.total_tokens
                by_endpoint[record.endpoint]["cost"] += record.estimated_cost
            
            return {
                "date": date.strftime("%Y-%m-%d"),
                "total_requests": total_requests,
                "total_tokens": total_tokens,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_cost": total_cost,
                "by_model": dict(by_model),
                "by_endpoint": dict(by_endpoint),
            }
            
        except Exception as e:
            logger.error(f"Failed to get daily aggregation from GCS: {e}")
            return {}

    async def get_monthly_aggregation(
        self,
        year: int,
        month: int,
        model_name: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> Dict:
        try:
            monthly_blob_path = self._get_monthly_blob_path(year, month)
            monthly_blob = self.bucket.blob(monthly_blob_path)
            
            if monthly_blob.exists() and not model_name and not endpoint:
                data = monthly_blob.download_as_text()
                return json.loads(data)
            
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year + 1, 1, 1)
            else:
                end_date = datetime(year, month + 1, 1)
            
            records = await self.get_usage_records(start_date, end_date, model_name, endpoint)
            
            if not records:
                return {
                    "year": year,
                    "month": month,
                    "total_requests": 0,
                    "total_tokens": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_cost": 0.0,
                    "by_model": {},
                    "by_endpoint": {},
                    "daily_breakdown": {},
                }
            
            total_requests = len(records)
            total_tokens = sum(r.total_tokens for r in records)
            total_input_tokens = sum(r.input_tokens for r in records)
            total_output_tokens = sum(r.output_tokens for r in records)
            total_cost = sum(r.estimated_cost for r in records)
            
            by_model = defaultdict(lambda: {"requests": 0, "tokens": 0, "cost": 0.0})
            by_endpoint = defaultdict(lambda: {"requests": 0, "tokens": 0, "cost": 0.0})
            daily_breakdown = defaultdict(lambda: {"requests": 0, "tokens": 0, "cost": 0.0})
            
            for record in records:
                by_model[record.model_name]["requests"] += 1
                by_model[record.model_name]["tokens"] += record.total_tokens
                by_model[record.model_name]["cost"] += record.estimated_cost
                
                by_endpoint[record.endpoint]["requests"] += 1
                by_endpoint[record.endpoint]["tokens"] += record.total_tokens
                by_endpoint[record.endpoint]["cost"] += record.estimated_cost
                
                day_key = record.timestamp.strftime("%Y-%m-%d")
                daily_breakdown[day_key]["requests"] += 1
                daily_breakdown[day_key]["tokens"] += record.total_tokens
                daily_breakdown[day_key]["cost"] += record.estimated_cost
            
            result = {
                "year": year,
                "month": month,
                "total_requests": total_requests,
                "total_tokens": total_tokens,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_cost": total_cost,
                "by_model": dict(by_model),
                "by_endpoint": dict(by_endpoint),
                "daily_breakdown": dict(daily_breakdown),
            }
            
            if not model_name and not endpoint:
                monthly_blob.upload_from_string(json.dumps(result, indent=2))
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get monthly aggregation from GCS: {e}")
            return {}