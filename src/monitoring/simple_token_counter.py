import logging
from typing import Dict
import google.generativeai as genai

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
            # Environment variables are available but not needed for basic genai configuration
            
            # Configure genai for Vertex AI
            genai.configure()
            self.client = genai
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self.client = None
    def get_current_stats(self) -> Dict[str, int]:
        """Get current session statistics"""
        return {
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
            "average_tokens_per_request": self.total_tokens // self.request_count if self.request_count > 0 else 0
        }

# Global instance
simple_counter = SimpleTokenCounter()
