"""
LLM Validator Module

Validates the fidelity of compressed content compared to the original.
"""

import logging
import time
import json
import requests
import re
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

    def _extract_critical_elements(self, content: str) -> Dict[str, set]:
        """
        Extract critical elements that must be preserved during compression.

        Args:
            content (str): Content to analyze

        Returns:
            Dict[str, set]: Dictionary of critical elements by type
        """
        critical = {
            'code_blocks': set(),
            'function_names': set(),
            'variable_names': set(),
            'api_endpoints': set(),
            'file_paths': set(),
            'technical_terms': set(),
            'commands': set()
        }
        
        # Extract code blocks (markdown and other formats)
        code_block_patterns = [
            r'```[\w]*\n(.*?)\n```',
            r'<code>(.*?)</code>',
        ]
        for pattern in code_block_patterns:
            for match in re.finditer(pattern, content, re.DOTALL):
                critical['code_blocks'].add(match.group(1).strip())
        
        # Extract indented code (4+ spaces)
        indented_code_pattern = r'^\s{4,}(.*)$'
        for match in re.finditer(indented_code_pattern, content, re.MULTILINE):
            line = match.group(1).strip()
            if line:  # Only add non-empty lines
                critical['code_blocks'].add(line)
            
        # Extract inline code and technical terms
        inline_patterns = [
            r'`([^`]+)`',
            r'\b[A-Z_][A-Z0-9_]{2,}\b',  # constants
            r'\b\w+\(\)',  # function calls
        ]
        for pattern in inline_patterns:
            for match in re.finditer(pattern, content):
                if '(' in pattern and match.lastindex and match.lastindex >= 1:
                    critical['technical_terms'].add(match.group(1))
                else:
                    critical['technical_terms'].add(match.group(0))
        
        # Extract function and method names
        func_patterns = [
            r'def\s+(\w+)\s*\(',
            r'function\s+(\w+)\s*\(',
            r'(\w+)\s*\([^)]*\)\s*{',
            r'(\w+)\s*=>\s*',
            r'(\w+)\s*\(',  # simple function calls
        ]
        for pattern in func_patterns:
            for match in re.finditer(pattern, content):
                func_name = match.group(1)
                # Filter out common words that aren't function names
                if func_name not in ['if', 'for', 'while', 'with', 'print', 'return']:
                    critical['function_names'].add(func_name)
        
        # Extract variable names
        var_patterns = [
            r'(\w+)\s*[=:]\s*',
            r'let\s+(\w+)',
            r'const\s+(\w+)',
            r'var\s+(\w+)',
            r'\$(\w+)',  # shell variables
        ]
        for pattern in var_patterns:
            for match in re.finditer(pattern, content):
                name = match.group(1)
                if len(name) > 2 and name not in ['the', 'and', 'for', 'this', 'that', 'with', 'from']:
                    critical['variable_names'].add(name)
        
        # Extract URLs and API endpoints
        url_patterns = [
            r'https?://[^\s"]+',
            r'ftp://[^\s"]+',
            r'[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/[^\s"]*)?'
        ]
        for pattern in url_patterns:
            for match in re.finditer(pattern, content):
                url = match.group(0).rstrip('",\'')  # Remove trailing quotes/commas
                critical['api_endpoints'].add(url)
            
        # Extract file paths
        path_patterns = [
            r'/[\w/.-]+\.\w+',
            r'[A-Z]:\\[\w\\.-]+',
            r'\.\/[\w/.-]+',
            r'[\w-]+\.(py|js|html|css|json|xml|yml|yaml|md|txt|conf|cfg)',
            r'~\/[\w/.-]+',
        ]
        for pattern in path_patterns:
            for match in re.finditer(pattern, content):
                critical['file_paths'].add(match.group(0))
                
        # Extract commands
        command_patterns = [
            r'^\s*\$\s+(.+)$',  # shell commands
            r'npm\s+\w+',
            r'pip\s+install',
            r'git\s+\w+',
            r'docker\s+\w+',
        ]
        for pattern in command_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                critical['commands'].add(match.group(1) if '$' in pattern else match.group(0))
        
        return critical
    def _create_validation_prompt(self, original_content: str, compressed_content: str) -> list:
        """
        Create a prompt for validating the compressed content with focus on technical preservation.

        Args:
            original_content (str): Original content
            compressed_content (str): Compressed content

        Returns:
            list: List of messages for the API call
        """
        # Extract critical elements from original content
        critical_elements = self._extract_critical_elements(original_content)
        
        system_message = (
            "You are a technical validation system that compares original content with its compressed version. "
            "Your primary task is to ensure that ALL critical technical information is preserved, including:\n"
            "- Code examples and snippets (must be EXACTLY preserved)\n"
            "- Function names, variable names, and method names\n"
            "- API endpoints, URLs, and file paths\n"
            "- Technical terminology and command syntax\n"
            "- Configuration values and parameters\n"
            "- Any character-sensitive technical content\n\n"
            "Focus on technical accuracy and completeness, not just general meaning preservation."
        )

        # Build specific validation points based on detected elements
        validation_points = []
        if critical_elements['code_blocks']:
            validation_points.append("- All code blocks are preserved exactly (including spacing and syntax)")
        if critical_elements['function_names']:
            validation_points.append("- All function/method names are preserved unchanged")
        if critical_elements['variable_names']:
            validation_points.append("- All variable names and identifiers are preserved")
        if critical_elements['api_endpoints']:
            validation_points.append("- All URLs and API endpoints are preserved exactly")
        if critical_elements['file_paths']:
            validation_points.append("- All file paths and directory references are preserved")
        if critical_elements['commands']:
            validation_points.append("- All commands and CLI instructions are preserved exactly")

        validation_criteria = ""
        if validation_points:
            validation_criteria = (
                f"\n\nCRITICAL VALIDATION POINTS for this content:\n" + 
                "\n".join(validation_points) +
                "\n\nPay special attention to these elements during validation."
            )

        user_message = (
            "Please validate this compressed content against the original. "
            "Check that essential information and ALL technical details are preserved. "
            f"{validation_criteria}\n\n"
            f"Original content:\n```\n{original_content}\n```\n\n"
            f"Compressed content:\n```\n{compressed_content}\n```\n\n"
            "Analyze the compressed version and determine if it maintains:\n"
            "1. All code examples, function names, and technical terminology exactly\n"
            "2. All URLs, file paths, and configuration values precisely\n"
            "3. The essential technical meaning and context\n"
            "4. Proper formatting for any code or structured content\n\n"
            "Respond with a JSON object with the following structure:\n"
            "{\n"
            '  "is_valid": true/false,\n'
            '  "message": "Your detailed explanation of why it is valid or not",\n'
            '  "missing_information": ["List any important technical information missing"],\n'
            '  "technical_errors": ["List any code/technical elements that were incorrectly modified"]\n'
            "}"
        )

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

    def _parse_validation_result(self, validation_text: str) -> Tuple[bool, str]:
        """
        Parse the validation result from the API response with improved reliability.

        Args:
            validation_text (str): Validation text from the API

        Returns:
            Tuple[bool, str]: (is_valid, validation_message)
        """
        # Clean the text first
        validation_text = validation_text.strip()
        
        # Try to extract JSON from the response (handle cases where JSON is embedded in text)
        json_patterns = [
            r'\{[^}]*"is_valid"[^}]*\}',  # Simple JSON extraction
            r'```json\s*(\{.*?\})\s*```',  # JSON in code blocks
            r'```\s*(\{.*?\})\s*```',      # JSON in unmarked code blocks
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, validation_text, re.DOTALL)
            if match:
                try:
                    json_str = match.group(1) if pattern.startswith(r'```') else match.group(0)
                    result = json.loads(json_str)
                    is_valid = result.get('is_valid', False)
                    message = result.get('message', '')
                    
                    # Include technical errors if present
                    technical_errors = result.get('technical_errors', [])
                    missing_info = result.get('missing_information', [])
                    
                    if technical_errors and not is_valid:
                        message += f"\n\nTechnical errors found:\n" + '\n'.join([f"- {error}" for error in technical_errors])
                    
                    if missing_info and not is_valid:
                        message += f"\n\nMissing information:\n" + '\n'.join([f"- {item}" for item in missing_info])
                    
                    return is_valid, message
                    
                except json.JSONDecodeError:
                    continue
        
        # If JSON parsing fails, try structured text analysis
        logger.warning("Failed to parse validation result as JSON, attempting enhanced text extraction")
        
        # Look for explicit validity statements
        validity_patterns = [
            (r'is[_\s]*valid["\s]*:["\s]*true', True),
            (r'is[_\s]*valid["\s]*:["\s]*false', False),
            (r'validation[:\s]*passed', True),
            (r'validation[:\s]*failed', False),
            (r'valid[:\s]*true', True),
            (r'valid[:\s]*false', False),
        ]
        
        lower_text = validation_text.lower()
        validity_found = False
        for pattern, validity in validity_patterns:
            if re.search(pattern, lower_text):
                validity_found = True
                return validity, f"Validation {'passed' if validity else 'failed'} based on pattern analysis"
        
        # Only do indicator analysis if no explicit validity pattern found
        if not validity_found:
            # Fallback: look for positive/negative indicators
            positive_indicators = ['valid', 'preserved', 'maintained', 'correct', 'accurate', 'complete']
            negative_indicators = ['invalid', 'missing', 'lost', 'incorrect', 'incomplete', 'error', 'wrong', 'failed']
            
            positive_count = sum(1 for indicator in positive_indicators if indicator in lower_text)
            negative_count = sum(1 for indicator in negative_indicators if indicator in lower_text)
            
            # Check for specific technical preservation mentions
            technical_preservation = [
                'code preserved', 'functions preserved', 'variables preserved',
                'urls preserved', 'paths preserved', 'technical details preserved'
            ]
            technical_issues = [
                'code missing', 'functions missing', 'variables missing',
                'urls missing', 'paths missing', 'technical details missing'
            ]
            
            tech_positive = sum(1 for phrase in technical_preservation if phrase in lower_text)
            tech_negative = sum(1 for phrase in technical_issues if phrase in lower_text)
            
            # Weight technical indicators more heavily
            total_positive = positive_count + (tech_positive * 2)
            total_negative = negative_count + (tech_negative * 3)  # Technical issues are more critical
            
            if total_positive > total_negative:
                return True, "Validation passed based on enhanced text analysis"
            else:
                return False, "Validation failed based on enhanced text analysis - potential technical issues detected"

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
