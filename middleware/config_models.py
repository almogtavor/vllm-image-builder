"""
Pydantic models for middleware configuration.
"""

from typing import Optional

import yaml
from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=9000, description="Server port")


class BackendConfig(BaseModel):
    """Backend server configuration."""
    base_url: str = Field(default="", description="Backend server base URL (overridden by VLLM_ADDRESS env var)")
    api_key: str = Field(default="my-apikey", description="API key")
    timeout: int = Field(default=300, description="Request timeout in seconds")
    model: Optional[str] = Field(default=None, description="Model to send to the backend (defaults to model.model_id)")
    # "vllm" (fork: /render + /completions token mode + /parse) or "sglang"
    # (local tokenize + native /generate input_ids + native parser). Overridden by BACKEND_KIND env.
    kind: str = Field(default="vllm", description="Backend server kind: vllm | sglang")


class ModelConfig(BaseModel):
    """Model configuration."""
    model_id: str = Field(default="", description="Model identifier (overridden by MODEL_NAME env var)")


class PaddingConfig(BaseModel):
    """Padding configuration. The render path pads each per-message chunk to a
    KV-block multiple so the next chunk starts on a block boundary. The block
    size itself is read from vLLM at runtime (/metrics), not configured here."""
    pad_token_id: int = Field(default=27, description="Token ID for padding symbol")


class WarmupConfig(BaseModel):
    """Warmup configuration. When enabled, each per-message span is prefilled
    (max_tokens=1) before the real request to prime PIC span reuse."""
    enabled: bool = Field(default=True, description="Warm per-message spans before generating")
    cache_size: int = Field(
        default=2048,
        description="Max already-warmed spans tracked per worker for dedup; 0 disables dedup",
    )


class ProcessingConfig(BaseModel):
    """Processing pipeline configuration."""
    padding: PaddingConfig
    warmup: WarmupConfig = Field(default_factory=WarmupConfig)
    span_policy: str = Field(
        default="all_messages",
        description="Span-selection policy name — decides which messages are PIC spans (see span_policy.py)",
    )
    min_span_tokens: int = Field(
        default=0,
        description="Only declare a unit a span if its real (unpadded) token length "
        "exceeds this (0 = no minimum). The 'atleast1000' variant sets 1000 so only "
        "large messages become spans. Overridden by MIDDLEWARE_MIN_SPAN_TOKENS.",
    )


class MetricsLoggingConfig(BaseModel):
    """Metrics logging configuration."""
    enabled: bool = Field(default=True, description="Enable metrics logging")
    format: str = Field(default="json", description="Metrics format")
    file_pattern: str = Field(default="metrics_{timestamp}.json", description="Metrics file pattern")


class ResponseLoggingConfig(BaseModel):
    """Response logging configuration."""
    enabled: bool = Field(default=True, description="Enable response logging")
    format: str = Field(default="json", description="Response format")
    file_pattern: str = Field(default="response_{request_id}.json", description="Response file pattern")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    output_dir: str = Field(
        default="/tmp/middleware_logs",
        description="Output directory for logs",
    )
    metrics: MetricsLoggingConfig
    responses: ResponseLoggingConfig


class MiddlewareConfig(BaseModel):
    """Complete middleware configuration."""
    server: ServerConfig
    backend: BackendConfig
    model: ModelConfig
    processing: ProcessingConfig
    logging: LoggingConfig
    
    @classmethod
    def from_yaml(cls, path: str) -> "MiddlewareConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = self.model_dump(mode='python', exclude_none=True)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def model_dump_dict(self) -> dict:
        """Return configuration as a dictionary (for backward compatibility)."""
        return self.model_dump(mode='python')
