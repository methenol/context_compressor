"""
LLM Compressor Module

Handles compression of content using OpenAI-compatible LLM APIs.
"""

import logging
import time
import json
import requests
import re
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

    def _detect_content_type(self, content: str) -> str:
        """
        Detect the type of content to apply appropriate compression strategies.

        Args:
            content (str): Content to analyze

        Returns:
            str: Content type ('code', 'documentation', 'config', 'mixed')
        """
        # Count different indicators
        code_indicators = 0
        doc_indicators = 0
        config_indicators = 0
        
        lines = content.split('\n')
        total_lines = len(lines)
        
        # Patterns for code detection
        code_patterns = [
            r'^\s*(def|class|function|var|let|const|import|from|#include)\s+',
            r'^\s*[\w\d_]+\s*[=:]\s*[\[\{]',  # assignments to lists/dicts
            r'^\s*[\w\d_]+\(.*\)\s*[:{]',     # function calls/definitions
            r'^\s*(if|else|elif|for|while|try|except|catch)\s*[\(\:]',
            r'^\s*[})\];]\s*$',               # closing brackets/braces
            r'```\w+',                        # code blocks in markdown
        ]
        
        # Patterns for documentation
        doc_patterns = [
            r'^#{1,6}\s+',                    # markdown headers
            r'^\s*[-*+]\s+',                  # markdown lists
            r'^\d+\.\s+',                     # numbered lists
            r'^>\s+',                         # blockquotes
            r'\[.*\]\(.*\)',                  # markdown links
            r'\*\*.*\*\*',                    # bold text
            r'^## ',                          # headers
        ]
        
        # Patterns for configuration files
        config_patterns = [
            r'^\s*[\w\d_]+\s*[:=]\s*',        # key-value pairs
            r'^\s*\[.*\]\s*$',                # ini sections
            r'^\s*<.*>\s*$',                  # xml tags
            r'^\s*{\s*".*":\s*',              # json objects
        ]
        
        for line in lines:
            # Check for code patterns
            for pattern in code_patterns:
                if re.search(pattern, line):
                    code_indicators += 1
                    break
            
            # Check for documentation patterns  
            for pattern in doc_patterns:
                if re.search(pattern, line):
                    doc_indicators += 1
                    break
                    
            # Check for config patterns
            for pattern in config_patterns:
                if re.search(pattern, line):
                    config_indicators += 1
                    break
        
        # Calculate percentages
        if total_lines == 0:
            return 'mixed'
            
        code_pct = code_indicators / total_lines
        doc_pct = doc_indicators / total_lines
        config_pct = config_indicators / total_lines
        
        # Determine primary content type
        if code_pct > 0.3:
            return 'code'
        elif doc_pct > 0.3:
            return 'documentation'
        elif config_pct > 0.3:
            return 'config'
        else:
            return 'mixed'

    def _extract_protected_elements(self, content: str) -> Dict[str, set]:
        """
        Extract only the most critical elements that must be protected during compression.

        Args:
            content (str): Content to analyze

        Returns:
            Dict[str, set]: Dictionary of protected elements by type
        """
        protected = {
            'function_names': set(),
            'api_endpoints': set(),
        }
        
        # Extract function names - only clear function definitions
        func_patterns = [
            r'def\s+(\w+)\s*\(',
            r'function\s+(\w+)\s*\(',
        ]
        for pattern in func_patterns:
            for match in re.finditer(pattern, content):
                protected['function_names'].add(match.group(1))
        
        # Extract API endpoints and URLs - only HTTP(S) URLs
        url_pattern = r'https?://[^\s"\']+'
        for match in re.finditer(url_pattern, content):
            url = match.group(0).rstrip('",\'')  # Remove trailing quotes/commas
            protected['api_endpoints'].add(url)
        
        return protected
    def _create_compression_prompt(self, content: str, level: int) -> list:
        """
        Create a prompt for the LLM based on the compression level and content type.

        Args:
            content (str): Content to compress
            level (int): Compression level (1-10)

        Returns:
            list: List of messages for the API call
        """
        # Detect content type and extract protected elements
        content_type = self._detect_content_type(content)
        protected_elements = self._extract_protected_elements(content)
        
        # Base instructions based on compression level - focused on aggressive compression
        if level <= 3:
            base_instruction = (
                "Aggressively compress the following content to achieve maximum token reduction. "
                "Remove all redundancy, verbose language, filler words, and unnecessary explanations. "
                "Convert paragraphs to concise bullet points where possible. Use abbreviations and acronyms for repeated terms. "
                "The result should be significantly shorter while preserving essential information and technical accuracy."
            )
        elif level <= 6:
            base_instruction = (
                "Extremely aggressively compress the following content to achieve substantial token reduction. "
                "Remove all redundancy, verbose explanations, example details, and non-essential context. "
                "Convert all paragraphs to terse bullet points. Use abbreviations extensively. Eliminate filler words completely. "
                "The result should be drastically shorter while maintaining core technical information and critical details."
            )
        else:
            base_instruction = (
                "Maximally compress the following content to achieve the highest possible token reduction. "
                "Remove ALL redundancy, verbose language, examples, explanations, and non-critical details. "
                "Convert everything to ultra-concise bullet points or abbreviated phrases. Use acronyms for all repeated terms. "
                "The result should be extremely condensed - aim for 50-70% size reduction while preserving only the most critical technical information."
            )

        # Content-type specific instructions - focused on targeted compression
        if content_type == 'code':
            content_specific = (
                "\n\nCODE CONTENT - TARGETED COMPRESSION:\n"
                "- Keep code blocks (``` or indented), function names, variable names, API endpoints, file paths EXACTLY unchanged\n"
                "- Aggressively compress ALL explanatory text, comments, and documentation around code\n"
                "- Convert verbose explanations to terse bullet points or single sentences\n"
                "- Remove example descriptions - keep only the actual code examples\n"
                "- Eliminate redundant explanations of what code does - let code speak for itself"
            )
        elif content_type == 'documentation':
            content_specific = (
                "\n\nDOCUMENTATION - AGGRESSIVE COMPRESSION:\n"
                "- Keep code examples, commands, file paths, URLs, technical terms EXACTLY unchanged\n"
                "- Convert all paragraphs to concise bullet points or numbered lists\n"
                "- Remove verbose explanations, background context, and detailed examples\n"
                "- Use abbreviations: documentation->docs, configuration->config, application->app, etc.\n"
                "- Eliminate filler phrases, transition sentences, and redundant information\n"
                "- Keep only essential instructions and critical information"
            )
        elif content_type == 'config':
            content_specific = (
                "\n\nCONFIGURATION - SELECTIVE COMPRESSION:\n"
                "- Keep ALL configuration syntax, keys, values, paths, URLs EXACTLY unchanged\n"
                "- Aggressively compress comments, descriptions, and explanatory text\n"
                "- Convert setup instructions to minimal bullet points\n"
                "- Remove example scenarios - keep only actual configuration examples"
            )
        else:  # mixed content
            content_specific = (
                "\n\nMIXED CONTENT - MAXIMUM COMPRESSION:\n"
                "- Keep code blocks, function names, URLs, file paths, commands EXACTLY unchanged\n"
                "- Convert ALL prose to bullet points or abbreviated phrases\n"
                "- Use heavy abbreviation: information->info, example->ex, configuration->cfg\n"
                "- Remove background context, detailed explanations, and verbose descriptions\n"
                "- Eliminate redundant information and filler content aggressively"
            )

        # Add protection for detected elements - minimal and focused
        protection_notes = []
        if protected_elements['function_names']:
            functions = list(protected_elements['function_names'])[:5]  # Limit to most important
            protection_notes.append(f"Preserve exactly: {', '.join(functions)}")
        if protected_elements['api_endpoints']:
            urls = list(protected_elements['api_endpoints'])[:3]  # Limit to most important  
            protection_notes.append(f"URLs unchanged: {', '.join(urls)}")
            
        if protection_notes:
            content_specific += f"\n\nCRITICAL ELEMENTS: " + " | ".join(protection_notes)

        # Final instructions - focused on maximum compression
        final_instructions = (
            "\n\nCOMPRESSION TECHNIQUES:\n"
            "- Convert paragraphs → bullet points\n"
            "- Use abbreviations extensively (config, docs, info, ex, etc.)\n"
            "- Eliminate redundant phrases and filler words\n"
            "- Combine related points into single concise statements\n"
            "- Remove transition sentences and verbose explanations\n"
            "- Keep only actionable information and critical technical details\n"
            "TARGET: Achieve 50-70% size reduction while maintaining technical accuracy"
        )

        instruction = base_instruction + content_specific + final_instructions

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
            "temperature": 0.1,  # Very low temperature for consistent aggressive compression
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
