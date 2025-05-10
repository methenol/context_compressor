"""
File Processor Module

Handles file discovery, filtering, and reading operations for the Context Compressor.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger('context_compressor.file_processor')

class FileProcessor:
    """
    Handles file discovery and processing operations.

    Attributes:
        extensions (List[str]): List of file extensions to process
        max_file_size (int): Maximum file size in bytes to process
        exclude_paths (List[str]): List of file paths to exclude from processing
        excluded_files (List[str]): List of files that were excluded during processing
    """

    def __init__(self, extensions: List[str], max_file_size: int = 1024 * 1024, exclude_paths: List[str] = None):
        """
        Initialize the FileProcessor.

        Args:
            extensions (List[str]): List of file extensions to process
            max_file_size (int, optional): Maximum file size in bytes. Defaults to 1MB.
            exclude_paths (List[str], optional): List of file paths to exclude from processing.
        """
        self.extensions = extensions
        self.max_file_size = max_file_size
        self.exclude_paths = exclude_paths or []
        self.excluded_files = []
        logger.debug(f"Initialized FileProcessor with extensions: {extensions}")
        if self.exclude_paths:
            logger.debug(f"Excluding paths: {self.exclude_paths}")

    def discover_files(self, directory: str) -> List[str]:
        """
        Recursively discover files in the given directory.

        Args:
            directory (str): Directory to scan for files

        Returns:
            List[str]: List of file paths that match the criteria
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Not a directory: {directory}")

        discovered_files = []

        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)

                # Check if file matches criteria
                if self._should_process_file(file_path):
                    discovered_files.append(file_path)
                    logger.debug(f"Discovered file: {file_path}")

        logger.info(f"Discovered {len(discovered_files)} files in {directory}")
        return discovered_files

    def read_file(self, file_path: str) -> str:
        """
        Read the content of a file.

        Args:
            file_path (str): Path to the file

        Returns:
            str: Content of the file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                logger.debug(f"Read {len(content)} characters from {file_path}")
                return content
        except UnicodeDecodeError:
            logger.warning(f"Could not decode {file_path} as UTF-8, trying with errors='ignore'")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                logger.debug(f"Read {len(content)} characters from {file_path} with errors ignored")
                return content

    def _should_process_file(self, file_path: str) -> bool:
        """
        Determine if a file should be processed based on extension, size, and exclusion list.

        Args:
            file_path (str): Path to the file

        Returns:
            bool: True if the file should be processed, False otherwise
        """
        # Check if file is in the exclusion list
        for exclude_path in self.exclude_paths:
            if file_path.startswith(exclude_path) or file_path == exclude_path:
                logger.debug(f"Skipping {file_path}: matches excluded path {exclude_path}")
                self.excluded_files.append(file_path)
                return False

        # Check file extension
        file_ext = os.path.splitext(file_path)[1].lower()

        # Special handling for files without extensions
        if file_ext == '':
            # Check if we should process files without extensions
            if '' in self.extensions or '.' in self.extensions:
                logger.debug(f"Processing file without extension: {file_path}")
                # Continue with other checks
            else:
                # Try to determine file type by content (optional)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        first_line = f.readline().strip()
                        # Check for common file headers
                        if first_line.startswith('#!') and '/bin/' in first_line:
                            logger.debug(f"Detected script file without extension: {file_path}")
                            return True
                        elif first_line.startswith('<?xml') or first_line.startswith('<html'):
                            logger.debug(f"Detected XML/HTML file without extension: {file_path}")
                            return True
                except Exception:
                    pass

                logger.debug(f"Skipping {file_path}: no extension and no extension matching '' in {self.extensions}")
                return False
        elif not any(file_ext == ext.lower() for ext in self.extensions):
            logger.debug(f"Skipping {file_path}: extension {file_ext} not in {self.extensions}")
            return False

        # Check file size
        try:
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                logger.warning(f"Skipping {file_path}: size {file_size} exceeds limit {self.max_file_size}")
                return False

            # Skip empty files
            if file_size == 0:
                logger.debug(f"Skipping {file_path}: empty file")
                return False

        except OSError as e:
            logger.error(f"Error checking file size for {file_path}: {str(e)}")
            return False

        return True
