#rate_limited_llm.py
import time
import random
import logging
from typing import Optional, List, Any
from pydantic import Field, PrivateAttr
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from google.api_core import exceptions
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import LLMResult

logger = logging.getLogger(__name__)

class RateLimitedGoogleLLM(ChatGoogleGenerativeAI):
    """Rate-limited version of ChatGoogleGenerativeAI that's compatible with LangChain"""
    
    requests_per_minute: int = Field(default=10, description="Maximum requests per minute")
    _min_delay: float = PrivateAttr()
    _last_request_time: float = PrivateAttr()
    
    def __init__(
        self,
        google_api_key: str,
        requests_per_minute: int = 10,
        **kwargs
    ):
        super().__init__(
            model="gemini-1.5-pro",
            google_api_key=google_api_key,
            temperature=0.7,
            max_output_tokens=2048,
            top_p=0.95,
            **kwargs
        )
        self.requests_per_minute = requests_per_minute
        self._min_delay = 60.0 / requests_per_minute
        self._last_request_time = 0
    
    def _wait_for_rate_limit(self):
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        if time_since_last_request < self._min_delay:
            jitter = random.uniform(0, 0.1)
            sleep_time = self._min_delay - time_since_last_request + jitter
            logger.debug(f"Waiting for {sleep_time:.2f} seconds to meet rate limit.")
            time.sleep(sleep_time)
        self._last_request_time = time.time()
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((exceptions.ResourceExhausted, exceptions.ServiceUnavailable)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Override _generate to add rate limiting"""
        self._wait_for_rate_limit()
        try:
            return super()._generate(prompts, stop, run_manager, **kwargs)
        except Exception as e:
            logger.error(f"Error in LLM generation: {e}")
            raise

def get_rate_limited_llm(google_api_key: str) -> RateLimitedGoogleLLM:
    """Factory function to create rate-limited LLM instance"""
    try:
        llm = RateLimitedGoogleLLM(
            google_api_key=google_api_key,
            requests_per_minute=10  # Reduced from 30 to avoid rate limits
        )
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize rate-limited LLM: {e}")
        return None
