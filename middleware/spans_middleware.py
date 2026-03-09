import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, Request
from openai import OpenAI
from transformers import AutoTokenizer

# --- Logging setup ---

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Configuration ---

model_defs_path = Path(__file__).parent / "model_defs.yaml"
model_defs = yaml.safe_load(model_defs_path.read_text())


@dataclass
class ModelConfig:
    """Model-specific configuration parameters."""

    model_id: str = ""
    special_delim: str = ""
    system_prompt_delim: str = ""
    conv_start_placeholder: str = ""
    assistant_placeholder: str = ""
    block_attention_placeholder: str | None = None


class SpansMiddleware:
    """Encapsulates all global state and operations."""

    def __init__(self):
        self.vllm_client = OpenAI(base_url="http://0.0.0.0:8000/v1", api_key="vllm")
        self.tokenizer = None
        self.model_config = ModelConfig()
        self.ttft_measurements: dict[str, float] = {}
        self._request_counter = 0
        self._env_setup_done = False

    def _setup_environment(self) -> None:
        """Configure environment variables for VLLM spans using the initialized tokenizer."""
        if self._env_setup_done or self.tokenizer is None:
            return

        spans_plus = str(self.tokenize("+")[0])
        spans_cross = str(self.tokenize("@")[0])
        spans_pad = str(self.tokenize("<")[0])

        os.environ.update(
            {
                "VLLM_USE_V1": "1",
                "TOKENIZERS_PARALLELISM": "False",
                "VLLM_V1_SPANS_ENABLED": "True",
                "VLLM_V1_SPANS_TOKEN": spans_plus,
                "VLLM_V1_SPANS_TOKEN_PLUS": spans_plus,
                "VLLM_V1_SPANS_TOKEN_RECOMPUTE": spans_cross,
                "VLLM_V1_SPANS_TOKEN_CROSS": spans_cross,
                "VLLM_V1_SPANS_TOKEN_PAD": spans_pad,
            }
        )

        self._env_setup_done = True
        logger.info(
            "Environment configured with tokens: +=%s, @=%s, <=%s",
            spans_plus,
            spans_cross,
            spans_pad,
        )

    def get_next_id(self) -> str:
        """Generate sequential request IDs."""
        request_id = str(self._request_counter)
        self._request_counter += 1
        return request_id

    def init_model(self, model_id: str) -> None:
        """Initialize tokenizer and load model configuration."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=os.getenv("LOCAL_MODELS_HOME")
        )

        # Setup environment variables after tokenizer is ready
        self._setup_environment()

        model_config_dict = next(
            (cfg for cfg in model_defs.values() if cfg.get("model_id") == model_id),
            None,
        )

        if model_config_dict is None:
            raise ValueError(f"No model definition found for model_id: {model_id}")

        self.model_config = ModelConfig(
            **{
                k: v
                for k, v in model_config_dict.items()
                if k in ModelConfig.__dataclass_fields__
            }
        )

    def tokenize(self, text: str) -> list[int]:
        """Tokenize text without special tokens."""
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def pad_tokens(self, tokens: list[int]) -> list[int]:
        """Pad token sequence to multiple of 16."""
        pad_token = int(os.environ["VLLM_V1_SPANS_TOKEN_PAD"])
        pad_len = (16 - len(tokens)) % 16
        return [*tokens[:-1], *([pad_token] * pad_len), tokens[-1]]

    def prepare_spans_prompt(
        self, messages: list[dict[str, str]], span_mode: str
    ) -> tuple[list[int], list[int], list[int], list[list[int]]]:
        """Prepare prompt with span tokens based on mode."""
        span_tok = int(os.environ["VLLM_V1_SPANS_TOKEN"])
        span_recomp_tok = int(os.environ["VLLM_V1_SPANS_TOKEN_RECOMPUTE"])

        mode_map = {
            "naive": (span_tok, span_tok),
            "full": (span_recomp_tok, span_recomp_tok),
            "spans": (span_tok, span_recomp_tok),
        }

        if span_mode not in mode_map:
            raise ValueError(f"Unknown span_mode: {span_mode}")

        regular_sep, last_sep = mode_map[span_mode]

        prompt_text = messages[1]["content"]

        # Parse prompt sections
        system_text, conv_text = prompt_text.split(
            self.model_config.system_prompt_delim, 1
        )
        docs_text, query_text = conv_text.split(
            self.model_config.conv_start_placeholder, 1
        )

        system_text += self.model_config.system_prompt_delim
        query_text = self.model_config.conv_start_placeholder + query_text

        # Process documents
        docs = [
            d for d in docs_text.split(self.model_config.special_delim) if d.strip()
        ]

        # Build prompt components
        prefix = self.tokenize(system_text)
        if self.model_config.block_attention_placeholder is not None:
            prefix = self.tokenize("[Block-Attention]") + prefix
        prefix = self.pad_tokens(prefix)

        tokenized_docs = [
            self.pad_tokens([regular_sep, *self.tokenize(doc)]) for doc in docs
        ]

        postfix = [
            last_sep,
            *self.tokenize(query_text + self.model_config.assistant_placeholder),
        ]

        full_prompt = [*prefix, *sum(tokenized_docs, []), *postfix]

        return full_prompt, prefix, postfix, tokenized_docs


# --- FastAPI Application ---


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI()
    middleware = SpansMiddleware()
    app.state.span_mode = os.getenv("SPAN_MODE", "spans")
    app.state.middleware = middleware

    @app.on_event("startup")
    async def on_startup() -> None:
        """Configure access logging on startup."""
        access_logger = logging.getLogger("uvicorn.access")
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        access_logger.addHandler(handler)

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        """List available models."""
        model_ids = [cfg.get("model_id") for cfg in model_defs.values()]
        return {
            "object": "list",
            "data": [
                {"id": model_id, "object": "model", "owned_by": "vllm"}
                for model_id in model_ids
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_to_completions(req: Request) -> dict[str, Any]:
        """Handle chat completion requests with spans optimization."""
        body = await req.json()
        model = body.get("model")
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 128)
        temperature = body.get("temperature", 0.0)

        mw = app.state.middleware

        # Initialize model if needed
        if mw.tokenizer is None or mw.model_config.model_id != model:
            mw.init_model(model)

        # Prepare prompt
        full_prompt, prefix, postfix, tokenized_docs = mw.prepare_spans_prompt(
            messages, app.state.span_mode
        )

        # Optional warmup
        if os.getenv("WARMUP_QUERY") == "DOC":
            warmup_prompts = [prefix, *tokenized_docs, postfix]
            if warmup_prompts:
                mw.vllm_client.completions.create(
                    model=model,
                    prompt=warmup_prompts,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

        # Main inference
        request_id = mw.get_next_id()
        start_time = time.time()

        stream = mw.vllm_client.completions.create(
            model=model,
            prompt=full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        # Process stream
        full_text = ""
        first_chunk = None
        finish_reason = None
        usage_info = {}
        ttft_measured = False
        ttft_seconds = 0.0

        for chunk in stream:
            if not ttft_measured:
                ttft_seconds = time.time() - start_time
                logger.info("Time to first token: %.3f seconds", ttft_seconds)
                mw.ttft_measurements[request_id] = ttft_seconds
                ttft_measured = True

            first_chunk = first_chunk or chunk

            if chunk.choices:
                choice = chunk.choices[0]
                if choice.text:
                    full_text += choice.text
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

            if hasattr(chunk, "usage") and chunk.usage:
                usage_info = chunk.usage

        # Calculate usage if not provided
        if not usage_info:
            output_ids = mw.tokenize(full_text)
            usage_info = {
                "prompt_tokens": len(full_prompt),
                "completion_tokens": len(output_ids),
                "total_tokens": len(full_prompt) + len(output_ids),
            }
            if ttft_measured:
                usage_info["ttft"] = ttft_seconds

        return {
            "id": getattr(first_chunk, "id", None),
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full_text},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage_info,
        }

    return app


app = create_app()
