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
        Extract elements that should be protected during compression.

        Args:
            content (str): Content to analyze

        Returns:
            Dict[str, set]: Dictionary of protected elements by type
        """
        protected = {
            'code_blocks': set(),
            'function_names': set(),
            'variable_names': set(),
            'api_endpoints': set(),
            'file_paths': set(),
            'technical_terms': set()
        }
        
        # Extract code blocks (markdown)
        code_block_pattern = r'```[\w]*\n(.*?)\n```'
        for match in re.finditer(code_block_pattern, content, re.DOTALL):
            protected['code_blocks'].add(match.group(1).strip())
            
        # Extract inline code
        inline_code_pattern = r'`([^`]+)`'
        for match in re.finditer(inline_code_pattern, content):
            protected['technical_terms'].add(match.group(1))
        
        # Extract function names
        func_patterns = [
            r'def\s+(\w+)\s*\(',
            r'function\s+(\w+)\s*\(',
            r'(\w+)\s*\([^)]*\)\s*{',
            r'(\w+)\s*=>\s*'
        ]
        for pattern in func_patterns:
            for match in re.finditer(pattern, content):
                protected['function_names'].add(match.group(1))
        
        # Extract variable names from common patterns
        var_patterns = [
            r'(\w+)\s*[=:]\s*',
            r'let\s+(\w+)',
            r'const\s+(\w+)',
            r'var\s+(\w+)'
        ]
        for pattern in var_patterns:
            for match in re.finditer(pattern, content):
                name = match.group(1)
                if len(name) > 2 and not name.lower() in ['the', 'and', 'for', 'this', 'that']:
                    protected['variable_names'].add(name)
        
        # Extract API endpoints and URLs
        url_pattern = r'https?://[^\s"]+'
        for match in re.finditer(url_pattern, content):
            url = match.group(0).rstrip('",\'')  # Remove trailing quotes/commas
            protected['api_endpoints'].add(url)
            
        # Extract file paths
        path_patterns = [
            r'/[\w/.-]+\.\w+',
            r'[A-Z]:\\[\w\\.-]+',
            r'\.\/[\w/.-]+',
            r'[\w-]+\.(py|js|html|css|json|xml|yml|yaml|md|txt)'
        ]
        for pattern in path_patterns:
            for match in re.finditer(pattern, content):
                protected['file_paths'].add(match.group(0))
        
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
        
        # Base instructions based on compression level
        if level <= 3:
            base_instruction = (
                "Compress the following content while preserving almost all information. "
                "Focus on removing redundancy and verbose language, but keep all key details. "
                "The result should be a slightly shorter version that maintains nearly all the original meaning."
            )
        elif level <= 6:
            base_instruction = (
                "Compress the following content while preserving the core information and context. "
                "Remove redundancy, verbose language, and less important details. "
                "The result should be a moderately compressed version that maintains the essential meaning and key points."
            )
        else:
            base_instruction = (
                "Aggressively compress the following content while preserving only the most essential information. "
                "Remove all redundancy, verbose language, and any details that aren't critical. "
                "The result should be a highly compressed version that captures only the most important points and context."
            )

        # Content-type specific instructions
        if content_type == 'code':
            content_specific = (
                "\n\nSPECIAL INSTRUCTIONS FOR CODE CONTENT:\n"
                "- NEVER modify or shorten code examples, function names, variable names, or code snippets\n"
                "- Preserve ALL code blocks exactly as they appear (within ``` markers or indented)\n"
                "- Keep all technical terminology, API names, and method names unchanged\n"
                "- Maintain exact spacing and formatting within code sections\n"
                "- Preserve import statements, class definitions, and function signatures completely\n"
                "- Keep file paths, URLs, and configuration values exactly as written\n"
                "- Only compress explanatory text and comments, never the actual code"
            )
        elif content_type == 'documentation':
            content_specific = (
                "\n\nSPECIAL INSTRUCTIONS FOR DOCUMENTATION CONTENT:\n"
                "- Preserve all code examples, command snippets, and inline code (in backticks) exactly\n"
                "- Keep technical terms, API names, function names, and variable names unchanged\n"
                "- Maintain all file paths, URLs, and configuration examples precisely\n"
                "- Preserve the structure of lists, headers, and important formatting\n"
                "- Keep installation commands, code samples, and examples intact\n"
                "- Only compress verbose explanations while keeping essential instructions clear"
            )
        elif content_type == 'config':
            content_specific = (
                "\n\nSPECIAL INSTRUCTIONS FOR CONFIGURATION CONTENT:\n"
                "- NEVER change configuration keys, values, or syntax\n"
                "- Preserve all file paths, URLs, and environment variables exactly\n"
                "- Keep property names, settings, and parameters unchanged\n"
                "- Maintain exact formatting for JSON, YAML, XML, or other structured data\n"
                "- Only compress comments and documentation, never the actual configuration"
            )
        else:  # mixed content
            content_specific = (
                "\n\nSPECIAL INSTRUCTIONS FOR MIXED CONTENT:\n"
                "- Identify and preserve ALL code examples, snippets, and technical references exactly\n"
                "- Never modify function names, variable names, API endpoints, or file paths\n"
                "- Keep all content within code blocks (``` or backticks) completely unchanged\n"
                "- Preserve technical terminology, command names, and configuration values\n"
                "- Maintain the logical structure and flow of technical information\n"
                "- Only compress verbose explanatory text, not technical specifications"
            )

        # Add protection for detected elements
        protection_notes = []
        if protected_elements['function_names']:
            protection_notes.append(f"Function names to preserve exactly: {', '.join(list(protected_elements['function_names'])[:10])}")
        if protected_elements['api_endpoints']:
            protection_notes.append(f"URLs/endpoints to preserve exactly: {', '.join(list(protected_elements['api_endpoints'])[:5])}")
        if protected_elements['file_paths']:
            protection_notes.append(f"File paths to preserve exactly: {', '.join(list(protected_elements['file_paths'])[:5])}")
            
        if protection_notes:
            content_specific += f"\n\nCRITICAL ELEMENTS DETECTED (preserve exactly):\n" + "\n".join(protection_notes)

        # Final instructions
        final_instructions = (
            "\n\nFINAL REQUIREMENTS:\n"
            "- Preserve all original titles and headings exactly as they appear without adding prefixes\n"
            "- Do not add any new information or commentary not present in the original\n"
            "- Maintain the overall structure and logical flow of information\n"
            "- When in doubt about preserving something technical, always err on the side of preservation"
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
