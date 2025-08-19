import logging
import time
import uuid
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional

from vertexai.generative_models import GenerativeModel

from ..config import settings
from .cost_calculator import CostCalculator
from .metrics_collector import MetricsCollector
from .storage.base import UsageRecord

logger = logging.getLogger(__name__)


class TokenTracker:
    def __init__(self):
        self.cost_calculator = CostCalculator()
        self.metrics_collector = MetricsCollector()

    async def count_tokens_for_request(self, model: GenerativeModel, content: str) -> int:
        try:
            response = model.count_tokens(content)
            return response.total_tokens
        except Exception as e:
            logger.warning(f"Failed to count tokens: {e}")
            return 0

    async def extract_usage_from_response(self, response) -> Dict[str, int]:
        try:
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                return {
                    "input_tokens": response.usage_metadata.prompt_token_count,
                    "output_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                }
        except Exception as e:
            logger.warning(f"Failed to extract usage metadata: {e}")
        
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    async def track_ai_usage(
        self,
        model_name: str,
        endpoint: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        processing_time: float,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        try:
            estimated_cost = self.cost_calculator.calculate_cost(
                model_name=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            record = UsageRecord(
                timestamp=datetime.utcnow(),
                model_name=model_name,
                endpoint=endpoint,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                estimated_cost=estimated_cost,
                request_id=request_id,
                user_id=user_id,
                metadata={
                    **(metadata or {}),
                    "processing_time": processing_time,
                },
            )

            await self.metrics_collector.store_usage_record(record)
            
            logger.info(
                f"Token usage tracked - Model: {model_name}, Endpoint: {endpoint}, "
                f"Tokens: {total_tokens} ({input_tokens} in, {output_tokens} out), "
                f"Cost: ${estimated_cost:.6f}, Time: {processing_time:.2f}s"
            )

        except Exception as e:
            logger.error(f"Failed to track token usage: {e}")


def track_tokens(
    endpoint: str,
    model_param: str = "model",
    content_param: Optional[str] = None,
    user_id_param: Optional[str] = None,
):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            if not settings.enable_token_tracking:
                return await func(*args, **kwargs)
                
            start_time = time.time()
            request_id = str(uuid.uuid4())
            tracker = TokenTracker()
            
            model = kwargs.get(model_param)
            content = kwargs.get(content_param, "") if content_param else ""
            user_id = kwargs.get(user_id_param)
            
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0
            
            try:
                if model and content:
                    input_tokens = await tracker.count_tokens_for_request(model, content)
                
                result = await func(*args, **kwargs)
                
                if hasattr(result, 'usage_metadata') or hasattr(result, 'text'):
                    usage_data = await tracker.extract_usage_from_response(result)
                    input_tokens = usage_data["input_tokens"] or input_tokens
                    output_tokens = usage_data["output_tokens"]
                    total_tokens = usage_data["total_tokens"]
                
                processing_time = time.time() - start_time
                
                model_name = getattr(model, 'model_name', settings.model_name) if model else settings.model_name
                
                await tracker.track_ai_usage(
                    model_name=model_name,
                    endpoint=endpoint,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    processing_time=processing_time,
                    request_id=request_id,
                    user_id=user_id,
                    metadata={
                        "function": func.__name__,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()),
                    },
                )
                
                return result
                
            except Exception as e:
                processing_time = time.time() - start_time
                
                if model:
                    model_name = getattr(model, 'model_name', settings.model_name)
                    await tracker.track_ai_usage(
                        model_name=model_name,
                        endpoint=endpoint,
                        input_tokens=input_tokens,
                        output_tokens=0,
                        total_tokens=input_tokens,
                        processing_time=processing_time,
                        request_id=request_id,
                        user_id=user_id,
                        metadata={
                            "function": func.__name__,
                            "error": str(e),
                            "success": False,
                        },
                    )
                
                raise e

        return wrapper
    return decorator