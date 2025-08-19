import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

from .base import StorageBackend, UsageRecord


class FileStorage(StorageBackend):
    def __init__(self, storage_path: str = "monitoring_data"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

    def _get_daily_file_path(self, date: datetime) -> str:
        date_str = date.strftime("%Y-%m-%d")
        return os.path.join(self.storage_path, f"usage_{date_str}.json")

    def _get_monthly_file_path(self, year: int, month: int) -> str:
        return os.path.join(self.storage_path, f"usage_{year}_{month:02d}_monthly.json")

    async def store_usage_record(self, record: UsageRecord) -> None:
        file_path = self._get_daily_file_path(record.timestamp)
        
        records = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                records = json.load(f)
        
        records.append(record.to_dict())
        
        with open(file_path, 'w') as f:
            json.dump(records, f, indent=2)

    async def get_usage_records(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        model_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[UsageRecord]:
        records = []
        
        if start_date and end_date:
            current_date = start_date
            while current_date <= end_date:
                file_path = self._get_daily_file_path(current_date)
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        daily_records = json.load(f)
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

    async def get_daily_aggregation(
        self,
        date: datetime,
        model_name: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> Dict:
        file_path = self._get_daily_file_path(date)
        
        if not os.path.exists(file_path):
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
        
        with open(file_path, 'r') as f:
            records_data = json.load(f)
        
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

    async def get_monthly_aggregation(
        self,
        year: int,
        month: int,
        model_name: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> Dict:
        monthly_file = self._get_monthly_file_path(year, month)
        
        if os.path.exists(monthly_file):
            with open(monthly_file, 'r') as f:
                cached_data = json.load(f)
                if not model_name and not endpoint:
                    return cached_data
        
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
            with open(monthly_file, 'w') as f:
                json.dump(result, f, indent=2)
        
        return result