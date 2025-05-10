# Context Compressor Example

This directory contains an example of how to use the Context Compressor tool.

## Directory Structure

- `input/`: Contains sample documentation files to be compressed
- `output/`: Where compressed files will be saved (created when running the example)
- `run_example.py`: Example script that demonstrates how to use the Context Compressor

## Running the Example

To run the example:

```bash
# Make sure you're in the example directory
cd example

# Run the example script
python run_example.py
```

This will:
1. Read the sample documentation from the `input/` directory
2. Compress it using the Context Compressor
3. Save the compressed version to the `output/` directory

## Configuration

The example uses the configuration from the `.env` file in the project root. You can modify these settings to change how the compression works:

- `BASE_URL`: Base URL for the OpenAI-compatible API
- `MODEL`: Model to use for compression
- `COMPRESSION_LEVEL`: Compression level (1-10)
- `FILE_EXTENSIONS`: File extensions to process
- `MAX_TOKENS`: Maximum tokens per API call

## API Key

Make sure you have set your OpenAI API key in the `.env` file in the project root:

```
OPENAI_API_KEY=your_api_key_here
```
