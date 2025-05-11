#!/usr/bin/env python3
"""
Context Compressor

A tool for recursively processing and compressing documentation used in agentic coding systems.
The primary goal is to reduce the token count of each file while preserving the core information
and contextual integrity.

Usage:
    python context_compressor.py --input_dir <input_directory> --output_dir <output_directory> [options]

Options:
    --input_dir, -i     Input directory containing files to compress
    --output_dir, -o    Output directory for compressed files
    --base_url          Base URL for OpenAI-compatible API (overrides BASE_URL env var)
    --model             Model to use for compression (overrides MODEL env var)
    --compression       Compression level (1-10, overrides COMPRESSION_LEVEL env var)
    --api_key           OpenAI API key (overrides OPENAI_API_KEY env var)
    --extensions        File extensions to process (comma-separated, overrides FILE_EXTENSIONS env var)
    --max_tokens        Maximum tokens per API call (overrides MAX_TOKENS env var)
    -e                  Python-style list of file paths to exclude from compression (will be copied directly)
    --verbose, -v       Enable verbose logging
    --help, -h          Show this help message

Environment Variables:
    OPENAI_API_KEY      API key for OpenAI or compatible service (required)
    BASE_URL            Base URL for OpenAI-compatible API (default: https://api.openai.com/v1)
                        Note: Should end with /v1 as we use the /v1/chat/completions endpoint
    MODEL               Model to use for compression (default: gpt-3.5-turbo)
    COMPRESSION_LEVEL   Compression level (1-10, default: 5)
    COMPRESSION_RETRIES Number of retries at same compression level before reducing (default: 3)
    FILE_EXTENSIONS     File extensions to process (comma-separated, default: .md,.txt,.py)
    MAX_TOKENS          Maximum tokens per API call (default: 4000)
    API_TIMEOUT         API request timeout in seconds (default: 60)
"""

import argparse
import ast
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Import modules
from file_processor import FileProcessor
from llm_compressor import LLMCompressor
from llm_validator import LLMValidator
from directory_handler import DirectoryHandler
from utils import setup_logging

# Load environment variables from .env file
# Get the absolute path to the .env file
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

# Get log level from environment variable and set up logging
log_level = os.environ.get('LOG_LEVEL', 'INFO')
setup_logging(log_level=log_level)

# Create logger for this module
logger = logging.getLogger('context_compressor')

def parse_arguments():
    """Parse command line arguments."""
    # Get defaults from environment variables
    default_base_url = os.environ.get('BASE_URL', 'https://api.openai.com/v1')
    default_model = os.environ.get('MODEL', 'gpt-3.5-turbo')

    # Parse numeric values safely
    try:
        default_compression = int(os.environ.get('COMPRESSION_LEVEL', '5'))
    except ValueError:
        # In case there are comments or whitespace in the value
        compression_level_str = os.environ.get('COMPRESSION_LEVEL', '5')
        default_compression = int(compression_level_str.split('#')[0].strip())

    # Get file extensions
    default_extensions = os.environ.get('FILE_EXTENSIONS', '.md,.txt,.py')
    # Remove any comments
    if '#' in default_extensions:
        default_extensions = default_extensions.split('#')[0].strip()

    # Add empty extension to process files without extensions
    if 'no_ext' in default_extensions.lower() or 'noext' in default_extensions.lower():
        default_extensions = default_extensions.replace('no_ext', '').replace('noext', '')
        if ',' not in default_extensions:
            default_extensions = ''
        default_extensions += ',.'  # Add dot to represent files without extensions

    # Parse max tokens
    try:
        default_max_tokens = int(os.environ.get('MAX_TOKENS', '4000'))
    except ValueError:
        # In case there are comments or whitespace in the value
        max_tokens_str = os.environ.get('MAX_TOKENS', '4000')
        default_max_tokens = int(max_tokens_str.split('#')[0].strip())

    # Parse API timeout
    try:
        default_api_timeout = int(os.environ.get('API_TIMEOUT', '60'))
    except ValueError:
        # In case there are comments or whitespace in the value
        api_timeout_str = os.environ.get('API_TIMEOUT', '60')
        default_api_timeout = int(api_timeout_str.split('#')[0].strip())

    # Parse compression retries
    try:
        default_compression_retries = int(os.environ.get('COMPRESSION_RETRIES', '3'))
    except ValueError:
        # In case there are comments or whitespace in the value
        compression_retries_str = os.environ.get('COMPRESSION_RETRIES', '3')
        default_compression_retries = int(compression_retries_str.split('#')[0].strip())

    parser = argparse.ArgumentParser(
        description='Compress documentation files while preserving core information.'
    )

    parser.add_argument('--input_dir', '-i', required=True,
                        help='Input directory containing files to compress')
    parser.add_argument('--output_dir', '-o', required=True,
                        help='Output directory for compressed files')
    parser.add_argument('--base_url', default=default_base_url,
                        help=f'Base URL for OpenAI-compatible API (default: {default_base_url})')
    parser.add_argument('--model', default=default_model,
                        help=f'Model to use for compression (default: {default_model})')
    parser.add_argument('--compression', type=int, default=default_compression,
                        choices=range(1, 11),
                        help=f'Compression level (1-10, default: {default_compression})')
    parser.add_argument('--api_key',
                        help='OpenAI API key (can also be set via OPENAI_API_KEY env var)')
    parser.add_argument('--extensions', default=default_extensions,
                        help=f'File extensions to process (comma-separated, default: {default_extensions})')
    parser.add_argument('--max_tokens', type=int, default=default_max_tokens,
                        help=f'Maximum tokens per API call (default: {default_max_tokens})')
    parser.add_argument('--api_timeout', type=int, default=default_api_timeout,
                        help=f'API request timeout in seconds (default: {default_api_timeout})')
    parser.add_argument('--compression_retries', type=int, default=default_compression_retries,
                        help=f'Number of retries at same compression level before reducing (default: {default_compression_retries})')
    parser.add_argument('-e',
                        help='List of file paths to exclude from compression (will be copied directly)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    return parser.parse_args()

def main():
    """Main entry point for the context compressor."""
    args = parse_arguments()

    # Set log level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        # Set root logger to DEBUG as well to ensure all module loggers respect this setting
        logging.getLogger().setLevel(logging.DEBUG)

    # Log configuration
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Using model: {args.model}")
    logger.info(f"Compression level: {args.compression}")

    # Get API key from args or environment
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        logger.error("API key not provided. Use --api_key or set OPENAI_API_KEY environment variable.")
        sys.exit(1)

    # Parse extensions
    extensions = [ext.strip() for ext in args.extensions.split(',')]

    # Parse excluded paths
    exclude_paths = []
    if args.e:
        try:
            # Try to parse as a Python list
            exclude_paths = ast.literal_eval(args.e)
            if not isinstance(exclude_paths, list):
                exclude_paths = [exclude_paths]  # Convert to list if it's a single string
        except (SyntaxError, ValueError):
            # If parsing fails, treat it as a single path
            exclude_paths = [args.e]
        logger.info(f"Excluding paths: {exclude_paths}")

    try:
        # Initialize components
        file_processor = FileProcessor(extensions=extensions, exclude_paths=exclude_paths)
        llm_compressor = LLMCompressor(
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            compression_level=args.compression,
            max_tokens=args.max_tokens,
            max_retries=args.compression_retries,
            api_timeout=args.api_timeout
        )
        llm_validator = LLMValidator(
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            api_timeout=args.api_timeout
        )
        directory_handler = DirectoryHandler(
            input_dir=args.input_dir,
            output_dir=args.output_dir
        )

        # Create output directory structure
        directory_handler.create_output_structure()

        # Process files
        files = file_processor.discover_files(args.input_dir)
        logger.info(f"Found {len(files)} files to process")

        # Copy excluded files directly
        if file_processor.excluded_files:
            logger.info(f"Found {len(file_processor.excluded_files)} excluded files to copy directly")
            for excluded_file in file_processor.excluded_files:
                try:
                    # Read file content
                    content = file_processor.read_file(excluded_file)

                    # Get relative path for output
                    rel_path = os.path.relpath(excluded_file, args.input_dir)
                    logger.info(f"Copying excluded file: {rel_path}")

                    # Write to output directory
                    output_path = directory_handler.get_output_path(excluded_file)
                    directory_handler.write_file(output_path, content)

                    logger.info(f"Successfully copied excluded file: {rel_path}")
                except Exception as e:
                    logger.error(f"Error copying excluded file {excluded_file}: {str(e)}")

        # Process each file
        for file_path in files:
            try:
                # Read file content
                content = file_processor.read_file(file_path)

                # Get relative path for output
                rel_path = os.path.relpath(file_path, args.input_dir)
                logger.info(f"Processing: {rel_path}")

                # Compress content
                compressed_content = llm_compressor.compress(content)

                # Validate compression
                is_valid, validation_message = llm_validator.validate(
                    original_content=content,
                    compressed_content=compressed_content
                )

                # If validation fails, try again with the same compression level multiple times before reducing
                if not is_valid:
                    logger.warning(f"Validation failed for {rel_path}: {validation_message}")
                    compression_succeeded = False

                    # First retry at the same compression level multiple times
                    for retry_attempt in range(args.compression_retries):
                        if compression_succeeded:
                            break

                        logger.info(f"Retrying at the same compression level {args.compression} (attempt {retry_attempt + 1}/{args.compression_retries})")
                        try:
                            # Try again with the same compression level
                            compressed_content = llm_compressor.compress(
                                content,
                                compression_level=args.compression
                            )

                            # Validate again
                            is_valid, validation_message = llm_validator.validate(
                                original_content=content,
                                compressed_content=compressed_content
                            )

                            if is_valid:
                                logger.info(f"Recompression succeeded on retry {retry_attempt + 1} with same level {args.compression}")
                                compression_succeeded = True
                                break
                            else:
                                logger.warning(f"Validation still failed on retry {retry_attempt + 1} with same level {args.compression}: {validation_message}")
                        except Exception as e:
                            logger.warning(f"Error during retry {retry_attempt + 1} with same level {args.compression}: {str(e)}")

                    # If all retries at same level failed, try with progressively lower compression levels
                    if not compression_succeeded and not is_valid:
                        # Try with progressively lower compression levels
                        compression_levels = [max(1, args.compression - 2), max(1, args.compression - 4), 1]

                        for level in compression_levels:
                            if compression_succeeded:
                                break

                            logger.info(f"Attempting recompression with lower compression level {level}")
                            try:
                                # Try with a lower compression level
                                compressed_content = llm_compressor.compress(
                                    content,
                                    compression_level=level
                                )

                                # Validate again
                                is_valid, validation_message = llm_validator.validate(
                                    original_content=content,
                                    compressed_content=compressed_content
                                )

                                if is_valid:
                                    logger.info(f"Recompression succeeded with level {level}")
                                    compression_succeeded = True
                                    break
                                else:
                                    logger.warning(f"Validation still failed with level {level}: {validation_message}")
                            except Exception as e:
                                logger.warning(f"Error during recompression with level {level}: {str(e)}")

                    # If all compression attempts failed, copy the original file
                    if not compression_succeeded:
                        logger.warning(f"All recompression attempts failed for {rel_path}, copying original file")
                        compressed_content = content
                        is_valid = True  # Force continuation

                # Write compressed content to output directory
                output_path = directory_handler.get_output_path(file_path)
                directory_handler.write_file(output_path, compressed_content)

                # Calculate percentage savings
                original_size = len(content)
                compressed_size = len(compressed_content)

                # If compression resulted in a larger file, use the original instead
                if compressed_size > original_size:
                    logger.warning(f"Compression increased file size for {rel_path}, using original file instead")
                    compressed_content = content
                    compressed_size = original_size
                    # Rewrite the file with the original content
                    directory_handler.write_file(output_path, compressed_content)

                if original_size > 0:  # Avoid division by zero
                    savings_percentage = ((original_size - compressed_size) / original_size) * 100
                    if compressed_size < original_size:
                        logger.info(f"Successfully compressed: {rel_path} ({original_size:,} → {compressed_size:,} chars, {savings_percentage:.1f}% savings)")
                    else:
                        logger.info(f"File copied without compression: {rel_path} (compression would have increased size)")
                else:
                    logger.info(f"File processed: {rel_path} (empty file)")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")

                # Copy the original file to the output directory
                try:
                    # Get relative path if not already defined
                    if 'rel_path' not in locals():
                        rel_path = os.path.relpath(file_path, args.input_dir)

                    logger.warning(f"Copying original file due to processing error: {rel_path}")
                    output_path = directory_handler.get_output_path(file_path)

                    # Read content if not already defined
                    if 'content' not in locals():
                        content = file_processor.read_file(file_path)

                    directory_handler.write_file(output_path, content)
                    logger.info(f"Successfully copied original file: {rel_path}")
                except Exception as copy_error:
                    logger.error(f"Failed to copy original file {file_path}: {str(copy_error)}")

        logger.info("Compression process completed")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
