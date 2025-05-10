#!/usr/bin/env python3
"""
Test script for the Context Compressor.

This script tests the functionality of the Context Compressor modules.
"""

import unittest
import os
import shutil
import tempfile
from pathlib import Path

from file_processor import FileProcessor
from directory_handler import DirectoryHandler
from utils import estimate_tokens, chunk_text

class TestFileProcessor(unittest.TestCase):
    """Test the FileProcessor class."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

        # Create test files
        self.test_files = {
            'test1.txt': 'This is a test file.',
            'test2.md': '# Markdown Test\n\nThis is a markdown file.',
            'test3.py': '# Python Test\n\ndef test_function():\n    return "test"',
            'test4.json': '{"test": "This is a JSON file."}',
            'subdir/test5.txt': 'This is a test file in a subdirectory.'
        }

        for file_path, content in self.test_files.items():
            full_path = os.path.join(self.temp_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_discover_files(self):
        """Test file discovery."""
        processor = FileProcessor(extensions=['.txt', '.md', '.py'])
        files = processor.discover_files(self.temp_dir)

        # Should find 4 files (3 with matching extensions, 1 in subdirectory)
        self.assertEqual(len(files), 4)

        # Check if all expected files are found
        file_names = [os.path.basename(f) for f in files]
        self.assertIn('test1.txt', file_names)
        self.assertIn('test2.md', file_names)
        self.assertIn('test3.py', file_names)
        self.assertIn('test5.txt', file_names)

        # JSON file should not be included
        self.assertNotIn('test4.json', file_names)

    def test_read_file(self):
        """Test file reading."""
        processor = FileProcessor(extensions=['.txt'])
        file_path = os.path.join(self.temp_dir, 'test1.txt')
        content = processor.read_file(file_path)

        self.assertEqual(content, 'This is a test file.')

class TestDirectoryHandler(unittest.TestCase):
    """Test the DirectoryHandler class."""

    def setUp(self):
        """Set up test environment."""
        self.input_dir = tempfile.mkdtemp()
        self.output_dir = tempfile.mkdtemp()

        # Create test directory structure
        os.makedirs(os.path.join(self.input_dir, 'subdir1'))
        os.makedirs(os.path.join(self.input_dir, 'subdir2/subsubdir'))

        # Create test files
        with open(os.path.join(self.input_dir, 'test1.txt'), 'w') as f:
            f.write('Test file 1')

        with open(os.path.join(self.input_dir, 'subdir1/test2.txt'), 'w') as f:
            f.write('Test file 2')

        with open(os.path.join(self.input_dir, 'subdir2/subsubdir/test3.txt'), 'w') as f:
            f.write('Test file 3')

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.input_dir)
        shutil.rmtree(self.output_dir)

    def test_create_output_structure(self):
        """Test output directory structure creation."""
        handler = DirectoryHandler(self.input_dir, self.output_dir)
        handler.create_output_structure()

        # Check if directories are created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'subdir1')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'subdir2/subsubdir')))

    def test_get_output_path(self):
        """Test output path generation."""
        handler = DirectoryHandler(self.input_dir, self.output_dir)

        input_path = os.path.join(self.input_dir, 'subdir1/test2.txt')
        expected_output_path = os.path.join(self.output_dir, 'subdir1/test2.txt')

        self.assertEqual(handler.get_output_path(input_path), expected_output_path)

    def test_write_file(self):
        """Test file writing."""
        handler = DirectoryHandler(self.input_dir, self.output_dir)

        output_path = os.path.join(self.output_dir, 'subdir1/test_output.txt')
        content = 'Test output content'

        handler.write_file(output_path, content)

        # Check if file is created with correct content
        self.assertTrue(os.path.exists(output_path))
        with open(output_path, 'r') as f:
            self.assertEqual(f.read(), content)

class TestUtils(unittest.TestCase):
    """Test utility functions."""

    def test_estimate_tokens(self):
        """Test token estimation."""
        text = "This is a test string with approximately 12 tokens."
        estimated_tokens = estimate_tokens(text)

        # Rough estimate: 50 characters / 4 ≈ 12-13 tokens
        self.assertTrue(10 <= estimated_tokens <= 15)

    def test_chunk_text_small(self):
        """Test text chunking with small text."""
        text = "This is a small text that should not be chunked."

        # Use a larger token limit to ensure it fits in one chunk
        chunks = chunk_text(text, max_chunk_tokens=20)

        # Text is small enough to fit in one chunk
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_chunk_text_large(self):
        """Test text chunking with large text."""
        # Create a large text with multiple paragraphs
        paragraphs = ["Paragraph " + str(i) * 100 for i in range(10)]
        text = "\n\n".join(paragraphs)

        chunks = chunk_text(text, max_chunk_tokens=50)  # About 200 characters per chunk

        # Should be split into multiple chunks
        self.assertTrue(len(chunks) > 1)

if __name__ == '__main__':
    unittest.main()
