import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status

from ...monitoring.cost_calculator import CostCalculator
from ...monitoring.metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)
router = APIRouter()

metrics_collector = MetricsCollector()
cost_calculator = CostCalculator()


@router.get("/monitoring/usage/daily")
async def get_daily_usage(
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    endpoint: Optional[str] = Query(None, description="Filter by endpoint"),
):
    try:
        parsed_date = datetime.strptime(date, "%Y-%m-%d")
        usage_data = await metrics_collector.get_daily_usage(
            date=parsed_date,
            model_name=model_name,
            endpoint=endpoint,
        )
        return usage_data
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid date format. Use YYYY-MM-DD",
        )
    except Exception as e:
        logger.error(f"Failed to get daily usage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve daily usage data",
        )


@router.get("/monitoring/usage/monthly")
async def get_monthly_usage(
    year: int = Query(..., description="Year"),
    month: int = Query(..., description="Month (1-12)"),
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    endpoint: Optional[str] = Query(None, description="Filter by endpoint"),
):
    if month < 1 or month > 12:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Month must be between 1 and 12",
        )
    
    try:
        usage_data = await metrics_collector.get_monthly_usage(
            year=year,
            month=month,
            model_name=model_name,
            endpoint=endpoint,
        )
        return usage_data
    except Exception as e:
        logger.error(f"Failed to get monthly usage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve monthly usage data",
        )


@router.get("/monitoring/costs/summary")
async def get_cost_summary(
    start_date: Optional[str] = Query(None, description="Start date in YYYY-MM-DD format"),
    end_date: Optional[str] = Query(None, description="End date in YYYY-MM-DD format"),
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    endpoint: Optional[str] = Query(None, description="Filter by endpoint"),
):
    try:
        parsed_start_date = None
        parsed_end_date = None
        
        if start_date:
            parsed_start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            parsed_end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        records = await metrics_collector.get_usage_records(
            start_date=parsed_start_date,
            end_date=parsed_end_date,
            model_name=model_name,
            endpoint=endpoint,
        )
        
        cost_breakdown = cost_calculator.calculate_batch_cost(records)
        
        return {
            "period": {
                "start_date": start_date,
                "end_date": end_date,
            },
            "filters": {
                "model_name": model_name,
                "endpoint": endpoint,
            },
            "total_records": len(records),
            "cost_breakdown": cost_breakdown,
        }
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid date format. Use YYYY-MM-DD",
        )
    except Exception as e:
        logger.error(f"Failed to get cost summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve cost summary",
        )


@router.get("/monitoring/health/tokens")
async def get_token_health():
    try:
        today = datetime.now()
        yesterday = datetime(today.year, today.month, today.day - 1) if today.day > 1 else datetime(today.year, today.month - 1, 30)
        
        today_usage = await metrics_collector.get_daily_usage(today)
        yesterday_usage = await metrics_collector.get_daily_usage(yesterday)
        
        current_month_usage = await metrics_collector.get_monthly_usage(
            year=today.year,
            month=today.month,
        )
        
        return {
            "status": "healthy",
            "timestamp": today.isoformat(),
            "today": {
                "total_requests": today_usage.get("total_requests", 0),
                "total_tokens": today_usage.get("total_tokens", 0),
                "total_cost": today_usage.get("total_cost", 0.0),
            },
            "yesterday": {
                "total_requests": yesterday_usage.get("total_requests", 0),
                "total_tokens": yesterday_usage.get("total_tokens", 0),
                "total_cost": yesterday_usage.get("total_cost", 0.0),
            },
            "current_month": {
                "total_requests": current_month_usage.get("total_requests", 0),
                "total_tokens": current_month_usage.get("total_tokens", 0),
                "total_cost": current_month_usage.get("total_cost", 0.0),
            },
        }
        
    except Exception as e:
        logger.error(f"Failed to get token health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve token health metrics",
        )


@router.get("/monitoring/estimates/monthly")
async def get_monthly_cost_estimate(
    model_name: str = Query(..., description="Model name for cost estimation"),
    daily_input_tokens: int = Query(..., description="Average daily input tokens"),
    daily_output_tokens: int = Query(..., description="Average daily output tokens"),
    days_in_month: int = Query(30, description="Number of days in month"),
):
    try:
        estimate = cost_calculator.estimate_monthly_cost(
            model_name=model_name,
            daily_input_tokens=daily_input_tokens,
            daily_output_tokens=daily_output_tokens,
            days_in_month=days_in_month,
        )
        
        return {
            "model_name": model_name,
            "estimation_parameters": {
                "daily_input_tokens": daily_input_tokens,
                "daily_output_tokens": daily_output_tokens,
                "days_in_month": days_in_month,
            },
            "estimates": estimate,
        }
        
    except Exception as e:
        logger.error(f"Failed to calculate monthly estimate: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate monthly cost estimate",
        )