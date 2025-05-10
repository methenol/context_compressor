"""
LLM Validator Module

Validates the fidelity of compressed content compared to the original.
"""

import logging
import time
import json
import requests
from typing import Dict, Any, Tuple

logger = logging.getLogger('context_compressor.llm_validator')

class LLMValidator:
    """
    Validates compressed content against the original using LLM.

    Attributes:
        base_url (str): Base URL for the OpenAI-compatible API
        api_key (str): API key for authentication
        model (str): Model to use for validation
        max_retries (int): Maximum number of retries for API calls
        retry_delay (int): Delay between retries in seconds
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_retries: int = 3,
        retry_delay: int = 5,
        api_timeout: int = 60
    ):
        """
        Initialize the LLMValidator.

        Args:
            base_url (str): Base URL for the OpenAI-compatible API
            api_key (str): API key for authentication
            model (str): Model to use for validation
            max_retries (int, optional): Maximum number of retries for API calls. Defaults to 3.
            retry_delay (int, optional): Delay between retries in seconds. Defaults to 5.
            api_timeout (int, optional): Timeout for API requests in seconds. Defaults to 60.
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.api_timeout = api_timeout

        logger.debug(f"Initialized LLMValidator with model: {model}")

    def validate(self, original_content: str, compressed_content: str) -> Tuple[bool, str]:
        """
        Validate the compressed content against the original.

        Args:
            original_content (str): Original content
            compressed_content (str): Compressed content

        Returns:
            Tuple[bool, str]: (is_valid, validation_message)
        """
        # Create the validation prompt
        prompt = self._create_validation_prompt(original_content, compressed_content)

        # Call the API
        response = self._call_api(prompt)

        # Extract the validation result from the response
        validation_text = response.get('choices', [{}])[0].get('message', {}).get('content', '')

        if not validation_text:
            raise ValueError("Failed to get validation result from API response")

        # Parse the validation result
        is_valid, message = self._parse_validation_result(validation_text)

        logger.debug(f"Validation result: {is_valid}, message: {message}")
        return is_valid, message

    def _create_validation_prompt(self, original_content: str, compressed_content: str) -> list:
        """
        Create a prompt for validating the compressed content.

        Args:
            original_content (str): Original content
            compressed_content (str): Compressed content

        Returns:
            list: List of messages for the API call
        """
        system_message = (
            "You are a validation system that compares original content with its compressed version. "
            "Your task is to determine if the compressed version maintains the essential information "
            "and context from the original. Focus on whether key concepts, important details, and "
            "critical context are preserved, not on exact wording."
        )

        user_message = (
            "I need you to validate a compressed version of some content. "
            "Please analyze both versions and determine if the compressed version "
            "maintains the essential information and context.\n\n"
            f"Original content:\n```\n{original_content}\n```\n\n"
            f"Compressed content:\n```\n{compressed_content}\n```\n\n"
            "Respond with a JSON object with the following structure:\n"
            "{\n"
            '  "is_valid": true/false,\n'
            '  "message": "Your explanation of why it is valid or not",\n'
            '  "missing_information": ["List of important information missing in the compressed version"]\n'
            "}"
        )

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

    def _parse_validation_result(self, validation_text: str) -> Tuple[bool, str]:
        """
        Parse the validation result from the API response.

        Args:
            validation_text (str): Validation text from the API

        Returns:
            Tuple[bool, str]: (is_valid, validation_message)
        """
        try:
            # Try to parse as JSON
            result = json.loads(validation_text)
            is_valid = result.get('is_valid', False)
            message = result.get('message', '')

            # Include missing information in the message if available
            missing_info = result.get('missing_information', [])
            if missing_info and not is_valid:
                missing_info_str = '\n'.join([f"- {item}" for item in missing_info])
                message = f"{message}\n\nMissing information:\n{missing_info_str}"

            return is_valid, message

        except json.JSONDecodeError:
            # If not valid JSON, try to extract the result from the text
            logger.warning("Failed to parse validation result as JSON, attempting text extraction")

            # Look for indicators of validity in the text
            lower_text = validation_text.lower()
            if "valid" in lower_text and not any(x in lower_text for x in ["not valid", "invalid"]):
                return True, "Validation passed based on text analysis"
            else:
                return False, "Validation failed based on text analysis"

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
            "temperature": 0.2,  # Lower temperature for more deterministic results
        }

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Calling validation API (attempt {attempt + 1}/{self.max_retries})")
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
        raise ValueError("Failed to call validation API after multiple attempts")
