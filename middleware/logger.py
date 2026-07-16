"""
Logging system for middleware metrics and responses.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from config_models import MiddlewareConfig


class MetricsLogger:
    """Logs performance metrics (latency, tokens)."""
    
    def __init__(self, config: MiddlewareConfig):
        """Initialize metrics logger with Pydantic config."""
        self.config = config.logging.metrics
        self.enabled = self.config.enabled
        self.output_dir = Path(config.logging.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.request_start_times: Dict[str, float] = {}
    
    def log_request_start(self, request_id: str) -> None:
        """Record request start time."""
        if not self.enabled:
            return
        self.request_start_times[request_id] = time.time()

    def log_request_complete(self, request_id: str, data: dict) -> None:
        """Record complete request metrics."""
        if not self.enabled:
            return
        
        if request_id not in self.metrics:
            self.metrics[request_id] = {}
        
        # Calculate total latency
        if request_id in self.request_start_times:
            total_latency = time.time() - self.request_start_times[request_id]
            self.metrics[request_id]['total_latency_seconds'] = total_latency
            del self.request_start_times[request_id]
        
        # Add other metrics
        self.metrics[request_id].update({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'request_id': request_id,
            **data
        })
        
        # Save to file
        self._save_metrics()
    
    def _save_metrics(self) -> None:
        """Save metrics to JSON file."""
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        file_pattern = self.config.file_pattern
        filename = file_pattern.replace('{timestamp}', timestamp)
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_metrics(self) -> dict:
        """Get all metrics."""
        return self.metrics.copy()
    
    def clear_metrics(self) -> int:
        """Clear all metrics and return count."""
        count = len(self.metrics)
        self.metrics.clear()
        self.request_start_times.clear()
        return count


class ResponseLogger:
    """Logs full request/response data."""
    
    def __init__(self, config: MiddlewareConfig):
        """Initialize response logger with Pydantic config."""
        self.config = config.logging.responses
        self.enabled = self.config.enabled
        self.output_dir = Path(config.logging.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def log_response(self, request_id: str, data: dict) -> None:
        """Save response data to file."""
        if not self.enabled:
            return
        
        file_pattern = self.config.file_pattern
        filename = file_pattern.replace('{request_id}', request_id)
        filepath = self.output_dir / filename
        
        response_data = {
            'request_id': request_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **data
        }
        
        with open(filepath, 'w') as f:
            json.dump(response_data, f, indent=2)


class MiddlewareLogger:
    """Combined logger for metrics and responses."""
    
    def __init__(self, config: MiddlewareConfig):
        """Initialize middleware logger with Pydantic config."""
        self.metrics = MetricsLogger(config)
        self.responses = ResponseLogger(config)
        self._request_counter = 0
    
    def get_next_request_id(self) -> str:
        """Generate sequential request ID."""
        request_id = f"req_{self._request_counter:06d}"
        self._request_counter += 1
        return request_id
    
    def log_request_start(self, request_id: str) -> None:
        """Log request start."""
        self.metrics.log_request_start(request_id)

    def log_complete(self, request_id: str, request_data: dict,
                     response_data: dict, metrics_data: dict) -> None:
        """Log complete request/response."""
        # Log metrics
        self.metrics.log_request_complete(request_id, metrics_data)
        
        # Log full response
        self.responses.log_response(request_id, {
            'request': request_data,
            'response': response_data,
            'metrics': metrics_data
        })
