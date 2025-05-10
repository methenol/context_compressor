"""
Utilities Module

Provides utility functions for the Context Compressor.
"""

import logging
import os
import sys
from typing import Dict, Any, Optional

logger = logging.getLogger('context_compressor.utils')

def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration.

    Args:
        log_level (str, optional): Logging level. Defaults to 'INFO'.
        log_file (str, optional): Path to log file. If None, logs to console only.
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Configure handlers
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        # Create directory for log file if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Add file handler
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    logger.debug(f"Logging configured with level: {log_level}")

def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text.
    This is a rough estimate based on the assumption that 1 token ≈ 4 characters.

    Args:
        text (str): Text to estimate tokens for

    Returns:
        int: Estimated number of tokens
    """
    return len(text) // 4

def chunk_text(text: str, max_chunk_tokens: int = 3000) -> list:
    """
    Split text into chunks that fit within token limits.

    Args:
        text (str): Text to split
        max_chunk_tokens (int, optional): Maximum tokens per chunk. Defaults to 3000.

    Returns:
        list: List of text chunks
    """
    # Estimate max characters per chunk
    max_chars = max_chunk_tokens * 4

    # If text is already small enough, return as is
    if len(text) <= max_chars:
        return [text]

    chunks = []

    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    current_chunk = ""

    for paragraph in paragraphs:
        # If adding this paragraph would exceed the limit, save current chunk and start a new one
        if len(current_chunk) + len(paragraph) + 2 > max_chars and current_chunk:
            chunks.append(current_chunk)
            current_chunk = ""

        # If the paragraph itself is too large, split it further
        if len(paragraph) > max_chars:
            # If we have content in the current chunk, save it first
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            # Split by newlines
            lines = paragraph.split('\n')

            for line in lines:
                if len(current_chunk) + len(line) + 1 > max_chars and current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""

                # If the line itself is too large, split it into smaller chunks
                if len(line) > max_chars:
                    # If we have content in the current chunk, save it first
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""

                    # Split the line into fixed-size chunks
                    for i in range(0, len(line), max_chars):
                        line_chunk = line[i:i+max_chars]
                        if i + max_chars < len(line):
                            chunks.append(line_chunk)
                        else:
                            current_chunk = line_chunk
                else:
                    if current_chunk:
                        current_chunk += '\n' + line
                    else:
                        current_chunk = line
        else:
            if current_chunk:
                current_chunk += '\n\n' + paragraph
            else:
                current_chunk = paragraph

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)

    logger.debug(f"Split text into {len(chunks)} chunks")
    return chunks
