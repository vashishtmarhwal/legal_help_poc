import logging
from typing import Dict, Optional
from google import genai
from google.genai.types import HttpOptions

logger = logging.getLogger(__name__)


class SimpleTokenCounter:
    """Simple in-memory token counter that resets on API restart"""
    
    def __init__(self):
        self.total_tokens = 0
        self.request_count = 0
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Gemini client for token counting"""
        try:
            # Try to initialize with project credentials from environment
            import os
            project = os.getenv('GOOGLE_CLOUD_PROJECT', 'legal-datatonic-poc')
            location = os.getenv('LOCATION', 'us-central1')
            
            self.client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
                http_options=HttpOptions(api_version="v1")
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self.client = None
    
    async def count_tokens(self, content: str, model_name: str = "gemini-2.5-flash") -> int:
        """Count tokens for given content using Vertex AI Gemini SDK"""
        if not self.client:
            logger.warning("Gemini client not initialized")
            return 0
        
        try:
            response = self.client.models.count_tokens(
                model=model_name,
                contents=content,
            )
            token_count = response.total_tokens
            
            # Update counters
            self.total_tokens += token_count
            self.request_count += 1
            
            logger.info(f"Counted {token_count} tokens for request. Total: {self.total_tokens}")
            return token_count
            
        except Exception as e:
            logger.error(f"Failed to count tokens: {e}")
            return 0
    
    def get_current_stats(self) -> Dict[str, int]:
        """Get current session statistics"""
        return {
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
            "average_tokens_per_request": self.total_tokens // self.request_count if self.request_count > 0 else 0
        }
    
    def reset(self):
        """Reset all counters"""
        self.total_tokens = 0
        self.request_count = 0
        logger.info("Token counters reset")


# Global instance
simple_counter = SimpleTokenCounter()