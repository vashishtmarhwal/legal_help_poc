from typing import Dict


VERTEX_AI_PRICING = {
    "gemini-2.0-flash-lite": {
        "input_tokens_per_1k": 0.00015,
        "output_tokens_per_1k": 0.0006,
        "region": "us-central1",
    },
    "gemini-1.5-flash": {
        "input_tokens_per_1k": 0.00015,
        "output_tokens_per_1k": 0.0006,
        "region": "us-central1", 
    },
    "gemini-1.5-pro": {
        "input_tokens_per_1k": 0.00125,
        "output_tokens_per_1k": 0.005,
        "region": "us-central1",
    },
    "gemini-pro": {
        "input_tokens_per_1k": 0.0005,
        "output_tokens_per_1k": 0.0015,
        "region": "us-central1",
    },
    "text-embedding-005": {
        "input_tokens_per_1k": 0.00002,
        "output_tokens_per_1k": 0.0,
        "region": "us-central1",
    },
}


def get_model_pricing(model_name: str) -> Dict[str, float]:
    return VERTEX_AI_PRICING.get(model_name, {
        "input_tokens_per_1k": 0.001,
        "output_tokens_per_1k": 0.002,
        "region": "us-central1",
    })