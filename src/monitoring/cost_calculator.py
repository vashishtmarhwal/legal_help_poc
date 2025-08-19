import logging
from typing import Dict

from ..config.pricing import get_model_pricing

logger = logging.getLogger(__name__)


class CostCalculator:
    def __init__(self):
        self._pricing_cache: Dict[str, Dict] = {}

    def get_model_pricing_info(self, model_name: str) -> Dict[str, float]:
        if model_name not in self._pricing_cache:
            self._pricing_cache[model_name] = get_model_pricing(model_name)
        return self._pricing_cache[model_name]

    def calculate_cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        try:
            pricing = self.get_model_pricing_info(model_name)
            
            input_cost_per_1k = pricing.get("input_tokens_per_1k", 0.001)
            output_cost_per_1k = pricing.get("output_tokens_per_1k", 0.002)
            
            input_cost = (input_tokens / 1000) * input_cost_per_1k
            output_cost = (output_tokens / 1000) * output_cost_per_1k
            
            total_cost = input_cost + output_cost
            
            logger.debug(
                f"Cost calculation for {model_name}: "
                f"{input_tokens} input tokens (${input_cost:.6f}) + "
                f"{output_tokens} output tokens (${output_cost:.6f}) = ${total_cost:.6f}"
            )
            
            return total_cost
            
        except Exception as e:
            logger.error(f"Failed to calculate cost for {model_name}: {e}")
            return 0.0

    def calculate_batch_cost(self, usage_records: list) -> Dict[str, float]:
        model_costs = {}
        total_cost = 0.0
        
        for record in usage_records:
            model_name = record.model_name if hasattr(record, 'model_name') else record.get('model_name', 'unknown')
            cost = record.estimated_cost if hasattr(record, 'estimated_cost') else record.get('estimated_cost', 0.0)
            
            if model_name not in model_costs:
                model_costs[model_name] = 0.0
            
            model_costs[model_name] += cost
            total_cost += cost
        
        return {
            "by_model": model_costs,
            "total": total_cost,
        }

    def estimate_monthly_cost(
        self,
        model_name: str,
        daily_input_tokens: int,
        daily_output_tokens: int,
        days_in_month: int = 30,
    ) -> Dict[str, float]:
        daily_cost = self.calculate_cost(model_name, daily_input_tokens, daily_output_tokens)
        monthly_cost = daily_cost * days_in_month
        
        return {
            "daily_cost": daily_cost,
            "monthly_cost": monthly_cost,
            "daily_input_tokens": daily_input_tokens,
            "daily_output_tokens": daily_output_tokens,
            "monthly_input_tokens": daily_input_tokens * days_in_month,
            "monthly_output_tokens": daily_output_tokens * days_in_month,
        }