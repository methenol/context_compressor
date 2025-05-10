"""
Directory Handler Module

Handles directory structure mirroring and file writing operations.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger('context_compressor.directory_handler')

class DirectoryHandler:
    """
    Handles directory structure mirroring and file operations.

    Attributes:
        input_dir (str): Input directory path
        output_dir (str): Output directory path
    """

    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the DirectoryHandler.

        Args:
            input_dir (str): Input directory path
            output_dir (str): Output directory path
        """
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.abspath(output_dir)

        logger.debug(f"Initialized DirectoryHandler with input_dir: {self.input_dir}, output_dir: {self.output_dir}")

    def create_output_structure(self) -> None:
        """
        Create the output directory structure.
        """
        # Create the root output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")

        # We'll create subdirectories as needed when writing files
        # This avoids creating unnecessary directories

    def get_output_path(self, input_file_path: str) -> str:
        """
        Get the corresponding output path for an input file.

        Args:
            input_file_path (str): Path to the input file

        Returns:
            str: Path to the output file
        """
        # Get the relative path from input_dir
        rel_path = os.path.relpath(input_file_path, self.input_dir)

        # Construct the output path
        output_path = os.path.join(self.output_dir, rel_path)

        return output_path

    def write_file(self, file_path: str, content: str) -> None:
        """
        Write content to a file, creating parent directories if needed.

        Args:
            file_path (str): Path to the output file
            content (str): Content to write
        """
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Write the content to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)

        logger.debug(f"Wrote {len(content)} characters to {file_path}")

    def get_relative_path(self, file_path: str) -> str:
        """
        Get the relative path from the input directory.

        Args:
            file_path (str): Absolute file path

        Returns:
            str: Relative path from input directory
        """
        return os.path.relpath(file_path, self.input_dir)
