# Middleware System

A middleware layer for OpenAI-compatible APIs with flexible prompt processing capabilities.
The middleware intercepts requests between clients and a vLLM backend, processing prompts through a configurable pipeline before forwarding them for inference.
It enables vLLM position-independent caching (PIC), allowing KV-Cache chunks to be reused.

## Architecture

The system comprises four Python modules and a configuration file:

### 1. **`config_models.py`** - Pydantic Configuration Models
   - `ServerConfig` - Server host and port configuration
   - `BackendConfig` - Backend vLLM server connection configuration
   - `ModelConfig` - Model ID, tokenizer, and delimiter settings
   - `ProcessingConfig` - Pipeline step settings (tokenization, padding, splitting, span mode, warmup, rag_wb_use_case)
   - `LoggingConfig` - Metrics and response logging configuration
   - `MiddlewareConfig` - Root configuration with YAML loading and saving capabilities

### 2. **`processors.py`** - Prompt Processing Logic
   - **Helper Functions:**
     - `tokenize()` - Tokenizes text using the HuggingFace tokenizer
     - `pad_tokens()` - Pads token sequences to block size multiples
     - `split_prompt()` - Splits prompts by delimiters into system/documents/query components
     - `insert_span_tokens()` - Inserts span tokens between prompt components
   - **Main Class:**
     - `PromptProcessor` - Orchestrates the complete processing pipeline with configurable steps

### 3. **`logger.py`** - Logging System
   - `MetricsLogger` - Captures performance metrics (TTFT, latency, token counts)
   - `ResponseLogger` - Persists complete request/response data to JSON files
   - `MiddlewareLogger` - Unified logger interface with request ID generation

### 4. **`middleware.py`** - FastAPI Application
   - **Request Models:** `ChatMessage`, `ChatCompletionRequest` (Pydantic models)
   - **Helper Functions:**
     - `process_stream()` - Handles streaming responses and measures TTFT
     - `calculate_usage_info()` - Computes token usage statistics
     - `build_response()` - Constructs OpenAI-compatible response objects
   - **Main Function:** `create_app()` - Initializes the FastAPI application with all endpoints
   - **Endpoints:** `/v1/models`, `/v1/chat/completions`, `/v1/metrics`

### 5. **`middleware_config.yaml`** - Configuration File
   - Defines server, backend, model, processing, and logging settings
   - All processing steps can be independently enabled or disabled

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable dependency management.

### Install uv

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### Install Dependencies

```bash
# Install all dependencies (recommended)
uv pip install -e .

# Or install with test dependencies
uv pip install -e ".[test]"

# Or install with all optional dependencies
uv pip install -e ".[all]"

# Or install with performance optimizations
uv pip install -e ".[performance]"
```

## Configuration


**Note:** The `rag_wb_use_case` flag controls prompt extraction behavior:
- `true` (default): Uses `messages[1]["content"]` and adds `assistant_placeholder` (RAG Workbench format)
- `false`: Concatenates all message contents without adding `assistant_placeholder` (standard format)
Review and modify the `middleware_config.yaml` file to configure the middleware settings.

## Usage

### Starting the Server

```bash
# Using uv with uvicorn
uv run uvicorn middleware:app --host 0.0.0.0 --port 9000

# Or run the script directly with uv
uv run python middleware.py
```

### Making Requests

The middleware provides OpenAI-compatible endpoints:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="NousResearch/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    max_tokens=128,
    temperature=0.0
)

print(response.choices[0].message.content)
```

### API Endpoints

- `GET /v1/models` - Lists available models
- `POST /v1/chat/completions` - Handles chat completion requests
- `GET /v1/metrics` - Retrieves performance metrics
- `DELETE /v1/metrics` - Clears accumulated metrics

### Viewing Logs

Logs are stored in the configured output directory (default: `./middleware_logs/`):

```bash
# View metrics
cat middleware_logs/metrics_20260308_160000.json

# View specific response
cat middleware_logs/response_req_000001.json
```

## Testing

```bash
# Install with test dependencies (if not already installed)
uv pip install -e ".[test]"

# Run all tests
uv run pytest tests/
```
