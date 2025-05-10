#!/usr/bin/env python3
"""
Example script for using the Context Compressor.

This script demonstrates how to use the Context Compressor to compress documentation.
It reads configuration from environment variables.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
# Get the absolute path to the .env file in the project root
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

# Add parent directory to path to import context_compressor modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_processor import FileProcessor
from llm_compressor import LLMCompressor
from llm_validator import LLMValidator
from directory_handler import DirectoryHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('example')

def main():
    """Main function to demonstrate Context Compressor usage."""
    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Use the input directory inside the example directory
    input_dir = os.path.join(current_dir, 'input')
    # Create output directory separate from the input directory
    output_dir = os.path.join(current_dir, 'output')

    # Get configuration from environment variables
    api_key = os.environ.get('OPENAI_API_KEY')
    base_url = os.environ.get('BASE_URL', 'https://api.openai.com/v1')
    model = os.environ.get('MODEL', 'gpt-3.5-turbo')

    # Parse numeric values safely
    try:
        compression_level = int(os.environ.get('COMPRESSION_LEVEL', '5'))
    except ValueError:
        # In case there are comments or whitespace in the value
        compression_level_str = os.environ.get('COMPRESSION_LEVEL', '5')
        compression_level = int(compression_level_str.split('#')[0].strip())

    try:
        max_tokens = int(os.environ.get('MAX_TOKENS', '4000'))
    except ValueError:
        # In case there are comments or whitespace in the value
        max_tokens_str = os.environ.get('MAX_TOKENS', '4000')
        max_tokens = int(max_tokens_str.split('#')[0].strip())

    try:
        api_timeout = int(os.environ.get('API_TIMEOUT', '60'))
    except ValueError:
        # In case there are comments or whitespace in the value
        api_timeout_str = os.environ.get('API_TIMEOUT', '60')
        api_timeout = int(api_timeout_str.split('#')[0].strip())

    extensions_str = os.environ.get('FILE_EXTENSIONS', '.md,.txt')
    # Remove any comments
    if '#' in extensions_str:
        extensions_str = extensions_str.split('#')[0]

    # Add empty extension to process files without extensions
    if 'no_ext' in extensions_str.lower() or 'noext' in extensions_str.lower():
        extensions_str = extensions_str.replace('no_ext', '').replace('noext', '')
        if ',' not in extensions_str:
            extensions_str = ''
        extensions_str += ',.'  # Add dot to represent files without extensions

    extensions = [ext.strip() for ext in extensions_str.split(',')]

    if not api_key:
        logger.error("API key not provided. Set OPENAI_API_KEY environment variable.")
        sys.exit(1)

    # Log configuration
    logger.info(f"Using base URL: {base_url}")
    logger.info(f"Using model: {model}")
    logger.info(f"Compression level: {compression_level}")
    logger.info(f"File extensions: {extensions}")

    # Initialize components
    file_processor = FileProcessor(extensions=extensions)
    llm_compressor = LLMCompressor(
        base_url=base_url,
        api_key=api_key,
        model=model,
        compression_level=compression_level,
        max_tokens=max_tokens,
        api_timeout=api_timeout
    )
    llm_validator = LLMValidator(
        base_url=base_url,
        api_key=api_key,
        model=model,
        api_timeout=api_timeout
    )
    directory_handler = DirectoryHandler(
        input_dir=input_dir,
        output_dir=output_dir
    )

    try:
        # Create output directory structure
        directory_handler.create_output_structure()

        # Process all files in the input directory
        files = file_processor.discover_files(input_dir)
        if not files:
            logger.error(f"No files found in input directory: {input_dir}")
            sys.exit(1)
        logger.info(f"Found {len(files)} files to process")

        # Process each file
        for file_path in files:
            try:
                # Read file content
                content = file_processor.read_file(file_path)

                # Get relative path for output
                rel_path = os.path.relpath(file_path, input_dir)
                logger.info(f"Processing: {rel_path}")

                # Compress content
                compressed_content = llm_compressor.compress(content)

                # Validate compression
                is_valid, validation_message = llm_validator.validate(
                    original_content=content,
                    compressed_content=compressed_content
                )

                # If validation fails, try again with progressively lower compression levels
                if not is_valid:
                    logger.warning(f"Validation failed for {rel_path}: {validation_message}")
                    compression_succeeded = False

                    # First retry at the same compression level
                    logger.info(f"Retrying at the same compression level {compression_level}")
                    try:
                        # Try again with the same compression level
                        compressed_content = llm_compressor.compress(
                            content,
                            compression_level=compression_level
                        )

                        # Validate again
                        is_valid, validation_message = llm_validator.validate(
                            original_content=content,
                            compressed_content=compressed_content
                        )

                        if is_valid:
                            logger.info(f"Recompression succeeded on retry with same level {compression_level}")
                            compression_succeeded = True
                        else:
                            logger.warning(f"Validation still failed on retry with same level {compression_level}: {validation_message}")
                    except Exception as e:
                        logger.warning(f"Error during retry with same level {compression_level}: {str(e)}")

                    # If retry at same level failed, try with progressively lower compression levels
                    if not compression_succeeded and not is_valid:
                        # Try with progressively lower compression levels
                        compression_levels = [3, 2, 1]

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

                logger.info(f"Successfully compressed: {rel_path}")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                # Print full exception details for debugging
                import traceback
                logger.error(f"Exception details: {traceback.format_exc()}")

                # Copy the original file to the output directory
                try:
                    # Get relative path if not already defined
                    if 'rel_path' not in locals():
                        rel_path = os.path.relpath(file_path, input_dir)

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
