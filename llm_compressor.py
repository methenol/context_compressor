"""
LLM Compressor Module

Handles compression of content using OpenAI-compatible LLM APIs.
"""

import logging
import time
import json
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger('context_compressor.llm_compressor')

class LLMCompressor:
    """
    Compresses content using OpenAI-compatible LLM APIs.

    Attributes:
        base_url (str): Base URL for the OpenAI-compatible API
        api_key (str): API key for authentication
        model (str): Model to use for compression
        compression_level (int): Level of compression (1-10)
        max_tokens (int): Maximum tokens per API call
        max_retries (int): Maximum number of retries for API calls
        retry_delay (int): Delay between retries in seconds
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        compression_level: int = 5,
        max_tokens: int = 4000,
        max_retries: int = 3,
        retry_delay: int = 5,
        api_timeout: int = 60
    ):
        """
        Initialize the LLMCompressor.

        Args:
            base_url (str): Base URL for the OpenAI-compatible API
            api_key (str): API key for authentication
            model (str): Model to use for compression
            compression_level (int, optional): Level of compression (1-10). Defaults to 5.
            max_tokens (int, optional): Maximum tokens per API call. Defaults to 4000.
            max_retries (int, optional): Maximum number of retries for API calls. Defaults to 3.
            retry_delay (int, optional): Delay between retries in seconds. Defaults to 5.
            api_timeout (int, optional): Timeout for API requests in seconds. Defaults to 60.
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.compression_level = compression_level
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.api_timeout = api_timeout

        logger.debug(f"Initialized LLMCompressor with model: {model}, compression level: {compression_level}")

    def compress(self, content: str, compression_level: Optional[int] = None) -> str:
        """
        Compress the given content using the LLM.

        Args:
            content (str): Content to compress
            compression_level (int, optional): Override the default compression level

        Returns:
            str: Compressed content
        """
        level = compression_level if compression_level is not None else self.compression_level

        # Create the prompt based on compression level
        prompt = self._create_compression_prompt(content, level)

        # Call the API
        response = self._call_api(prompt)

        # Extract the compressed content from the response
        compressed_content = response.get('choices', [{}])[0].get('message', {}).get('content', '')

        if not compressed_content:
            raise ValueError("Failed to get compressed content from API response")

        logger.debug(f"Compressed content from {len(content)} to {len(compressed_content)} characters")
        return compressed_content

    def _create_compression_prompt(self, content: str, level: int) -> list:
        """
        Create a prompt for the LLM based on the compression level.

        Args:
            content (str): Content to compress
            level (int): Compression level (1-10)

        Returns:
            list: List of messages for the API call
        """
        # Adjust instructions based on compression level
        if level <= 3:
            instruction = (
                "Compress the following content while preserving almost all information. "
                "Focus on removing redundancy and verbose language, but keep all key details. "
                "The result should be a slightly shorter version that maintains nearly all the original meaning."
            )
        elif level <= 6:
            instruction = (
                "Compress the following content while preserving the core information and context. "
                "Remove redundancy, verbose language, and less important details. "
                "The result should be a moderately compressed version that maintains the essential meaning and key points."
            )
        else:
            instruction = (
                "Aggressively compress the following content while preserving only the most essential information. "
                "Remove all redundancy, verbose language, and any details that aren't critical. "
                "The result should be a highly compressed version that captures only the most important points and context."
            )

        # Add specific guidance based on the content type
        instruction += (
            "\n\nMaintain code snippets, important technical details, and key terminology. "
            "Preserve the overall structure and flow of information. "
            "Keep all original titles and headings exactly as they appear without adding any prefixes like 'Compressed'. "
            "Do not add any new information or commentary not present in the original."
        )

        return [
            {"role": "system", "content": instruction},
            {"role": "user", "content": content}
        ]

    def _call_api(self, messages: list) -> Dict[str, Any]:
        """
        Call the OpenAI-compatible API with retry logic.

        Args:
            messages (list): List of messages for the API call

        Returns:
            Dict[str, Any]: API response
        """
        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.3,  # Lower temperature for more deterministic results
        }

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Calling API (attempt {attempt + 1}/{self.max_retries})")
                response = requests.post(endpoint, headers=headers, json=data, timeout=self.api_timeout)

                if response.status_code == 200:
                    return response.json()

                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', self.retry_delay))
                    logger.warning(f"Rate limited. Retrying after {retry_after} seconds")
                    time.sleep(retry_after)
                    continue

                # Handle other errors
                error_msg = f"API error: {response.status_code} - {response.text}"
                logger.error(error_msg)

                # If it's the last attempt, raise an exception
                if attempt == self.max_retries - 1:
                    raise ValueError(error_msg)

                # Otherwise, wait and retry
                time.sleep(self.retry_delay)

            except requests.RequestException as e:
                logger.error(f"Request error: {str(e)}")

                # If it's the last attempt, raise an exception
                if attempt == self.max_retries - 1:
                    raise

                # Otherwise, wait and retry
                time.sleep(self.retry_delay)

        # This should not be reached due to the exception in the loop
        raise ValueError("Failed to call API after multiple attempts")
