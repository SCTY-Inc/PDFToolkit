"""Lazy-loaded API clients with retry logic."""

import os
from functools import lru_cache

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# Retry decorator for API calls - retry on network/API errors only
api_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    reraise=True,
)


@lru_cache(maxsize=1)
def get_openai_client():
    """Get or create OpenAI client (singleton)."""
    from openai import OpenAI
    return OpenAI()


@lru_cache(maxsize=1)
def get_together_client():
    """Get or create Together client (singleton)."""
    from together import Together
    return Together(api_key=os.getenv("TOGETHER_API_KEY"))


@lru_cache(maxsize=1)
def get_ollama_client():
    """Get or create Ollama client (singleton)."""
    from ollama import Client
    return Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
