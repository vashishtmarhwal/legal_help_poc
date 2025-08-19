from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional


class UsageRecord:
    def __init__(
        self,
        timestamp: datetime,
        model_name: str,
        endpoint: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        estimated_cost: float,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        self.timestamp = timestamp
        self.model_name = model_name
        self.endpoint = endpoint
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens
        self.estimated_cost = estimated_cost
        self.request_id = request_id
        self.user_id = user_id
        self.metadata = metadata or {}

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "model_name": self.model_name,
            "endpoint": self.endpoint,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost": self.estimated_cost,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "UsageRecord":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            model_name=data["model_name"],
            endpoint=data["endpoint"],
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            total_tokens=data["total_tokens"],
            estimated_cost=data["estimated_cost"],
            request_id=data.get("request_id"),
            user_id=data.get("user_id"),
            metadata=data.get("metadata", {}),
        )


class StorageBackend(ABC):
    @abstractmethod
    async def store_usage_record(self, record: UsageRecord) -> None:
        pass

    @abstractmethod
    async def get_usage_records(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        model_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[UsageRecord]:
        pass

    @abstractmethod
    async def get_daily_aggregation(
        self,
        date: datetime,
        model_name: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> Dict:
        pass

    @abstractmethod
    async def get_monthly_aggregation(
        self,
        year: int,
        month: int,
        model_name: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> Dict:
        pass