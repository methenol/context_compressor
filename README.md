# Context Compressor

A Python-based tool for recursively processing and compressing documentation used in agentic coding systems. The primary goal is to reduce the token count of each file while preserving the core information and contextual integrity.

## Overview

This tool is designed for developers and teams leveraging LLMs in constrained token environments, particularly those building or maintaining autonomous coding agents. It helps optimize LLM interactions and enhance processing efficiency without losing essential documentation fidelity.

## Core Features

- **Recursive File Discovery**
  - Scans the input directory and all subdirectories for files
  - Ensures complete coverage of documentation folders
  - Uses standard filesystem operations for traversal and filtering

- **Token Compression with LLM**
  - Sends file content to a configurable OpenAI-compatible LLM endpoint for token compression
  - Configurable options include `base_url`, `model`, and `compression level`
  - Reduces token usage while retaining informational value
  - Utilizes the `/v1/chat/completions` endpoint with prompt engineering focused on compression

- **Compression Validation**
  - Performs a second LLM call to compare original and compressed content for fidelity
  - Ensures that important content/context has been retained
  - Reattempts compression if validation fails

- **Mirrored Output Directory**
  - Saves compressed files to an output directory with the same structure as the input
  - Ensures usability and maintainability of the compressed documentation
  - Uses path manipulation and directory recreation techniques

## Installation

```bash
# Clone the repository
git clone https://github.com/methenol/context-compressor.git
cd context-compressor

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python context_compressor.py --input_dir <input_directory> --output_dir <output_directory> [options]
```

### Options

- `--input_dir`, `-i`: Input directory containing files to compress (required)
- `--output_dir`, `-o`: Output directory for compressed files (required)
- `--base_url`: Base URL for OpenAI-compatible API (default: https://api.openai.com)
- `--model`: Model to use for compression (default: gpt-3.5-turbo)
- `--compression`: Compression level (1-10, default: 5)
- `--api_key`: OpenAI API key (can also be set via OPENAI_API_KEY env var)
- `--extensions`: File extensions to process (comma-separated, default: .md,.txt,.py)
- `--max_tokens`: Maximum tokens per API call (default: 4000)
- `-e`: Python-style list of file paths to exclude from compression (will be copied directly)
- `--verbose`, `-v`: Enable verbose logging

### Example

```bash
# Basic usage with environment variable for API key
export OPENAI_API_KEY=your_api_key_here
python context_compressor.py -i ./docs -o ./compressed_docs

# Specify compression level and model
python context_compressor.py -i ./docs -o ./compressed_docs --compression 7 --model gpt-4.1-mini

# Process only markdown files with verbose logging
python context_compressor.py -i ./docs -o ./compressed_docs --extensions .md --verbose

# Exclude specific files or directories from compression (they will be copied directly)
python context_compressor.py -i ./docs -o ./compressed_docs -e "['./docs/config.json', './docs/scripts/', './docs/.gitignore']"
```

## System Components

- **File Processor Module**: Handles file discovery and reading operations
- **LLM Compression Module**: Manages compression using OpenAI-compatible APIs
- **LLM Validation Module**: Validates compressed content against the original
- **Directory Handler Module**: Handles directory structure mirroring and file writing

## Requirements

- Python 3.8+
- Network access for API calls
- API key for OpenAI or compatible providers

## License

MIT
