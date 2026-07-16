"""
Unit tests for logger modules.
"""

import json
import pytest
import tempfile
import time
from pathlib import Path

from logger import MetricsLogger, ResponseLogger, MiddlewareLogger
from config_models import MiddlewareConfig


class TestMetricsLogger:
    """Test MetricsLogger functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create test Pydantic configuration."""
        config_dict = {
            'server': {'host': '0.0.0.0', 'port': 8000},
            'backend': {'base_url': 'http://localhost:9000/v1', 'api_key': 'test-key'},
            'model': {
                'model_id': 'test-model',
                'system_prompt_delim': '<|eot_id|>',
                'conv_start_placeholder': '[conv]',
                'assistant_placeholder': '<|assistant|>\n'
            },
            'processing': {
                'tokenization': {'enabled': True, 'add_special_tokens': False},
                'padding': {'enabled': True, 'block_size': 16, 'pad_token_symbol': '<'},
                'delimiter_splitting': {'enabled': True, 'special_delim': ' # # '},
                'span_mode': {'enabled': True, 'mode': 'spans', 'plus_token_id': 10, 'recompute_token_id': 31},
                'warmup': {'enabled': False}
            },
            'logging': {
                'output_dir': temp_dir,
                'metrics': {
                    'enabled': True,
                    'file_pattern': 'metrics_{timestamp}.json'
                },
                'responses': {
                    'enabled': True,
                    'file_pattern': 'response_{request_id}.json'
                }
            }
        }
        return MiddlewareConfig(**config_dict)
    
    def test_log_request_complete(self, config):
        """Test logging complete request."""
        logger = MetricsLogger(config)
        
        request_id = "req_000001"
        logger.log_request_start(request_id)
        time.sleep(0.01)
        
        data = {'prompt_tokens': 100, 'completion_tokens': 50}
        logger.log_request_complete(request_id, data)
        
        assert request_id in logger.metrics
        assert 'total_latency_seconds' in logger.metrics[request_id]
        assert logger.metrics[request_id]['prompt_tokens'] == 100
    
    def test_disabled_logger(self, temp_dir):
        """Test disabled logger does nothing."""
        config_dict = {
            'server': {'host': '0.0.0.0', 'port': 8000},
            'backend': {'base_url': 'http://localhost:9000/v1', 'api_key': 'test-key'},
            'model': {
                'model_id': 'test-model',
                'system_prompt_delim': '<|eot_id|>',
                'conv_start_placeholder': '[conv]',
                'assistant_placeholder': '<|assistant|>\n'
            },
            'processing': {
                'tokenization': {'enabled': True, 'add_special_tokens': False},
                'padding': {'enabled': True, 'block_size': 16, 'pad_token_symbol': '<'},
                'delimiter_splitting': {'enabled': True, 'special_delim': ' # # '},
                'span_mode': {'enabled': True, 'mode': 'spans', 'plus_token_id': 10, 'recompute_token_id': 31},
                'warmup': {'enabled': False}
            },
            'logging': {
                'output_dir': temp_dir,
                'metrics': {'enabled': False},
                'responses': {'enabled': True, 'file_pattern': 'response_{request_id}.json'}
            }
        }
        config = MiddlewareConfig(**config_dict)
        logger = MetricsLogger(config)
        
        logger.log_request_start("req_001")

        assert len(logger.metrics) == 0


class TestResponseLogger:
    """Test ResponseLogger functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create test Pydantic configuration."""
        config_dict = {
            'server': {'host': '0.0.0.0', 'port': 8000},
            'backend': {'base_url': 'http://localhost:9000/v1', 'api_key': 'test-key'},
            'model': {
                'model_id': 'test-model',
                'system_prompt_delim': '<|eot_id|>',
                'conv_start_placeholder': '[conv]',
                'assistant_placeholder': '<|assistant|>\n'
            },
            'processing': {
                'tokenization': {'enabled': True, 'add_special_tokens': False},
                'padding': {'enabled': True, 'block_size': 16, 'pad_token_symbol': '<'},
                'delimiter_splitting': {'enabled': True, 'special_delim': ' # # '},
                'span_mode': {'enabled': True, 'mode': 'spans', 'plus_token_id': 10, 'recompute_token_id': 31},
                'warmup': {'enabled': False}
            },
            'logging': {
                'output_dir': temp_dir,
                'metrics': {
                    'enabled': True,
                    'file_pattern': 'metrics_{timestamp}.json'
                },
                'responses': {
                    'enabled': True,
                    'file_pattern': 'response_{request_id}.json'
                }
            }
        }
        return MiddlewareConfig(**config_dict)
    
    def test_log_response(self, config):
        """Test logging response data."""
        logger = ResponseLogger(config)
        
        request_id = "req_000001"
        data = {
            'request': {'model': 'test-model'},
            'response': {'content': 'Test response'},
            'metrics': {'prompt_tokens': 100}
        }
        
        logger.log_response(request_id, data)
        
        filepath = Path(config.logging.output_dir) / f'response_{request_id}.json'
        assert filepath.exists()


class TestMiddlewareLogger:
    """Test MiddlewareLogger integration."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create test Pydantic configuration."""
        config_dict = {
            'server': {'host': '0.0.0.0', 'port': 8000},
            'backend': {'base_url': 'http://localhost:9000/v1', 'api_key': 'test-key'},
            'model': {
                'model_id': 'test-model',
                'system_prompt_delim': '<|eot_id|>',
                'conv_start_placeholder': '[conv]',
                'assistant_placeholder': '<|assistant|>\n'
            },
            'processing': {
                'tokenization': {'enabled': True, 'add_special_tokens': False},
                'padding': {'enabled': True, 'block_size': 16, 'pad_token_symbol': '<'},
                'delimiter_splitting': {'enabled': True, 'special_delim': ' # # '},
                'span_mode': {'enabled': True, 'mode': 'spans', 'plus_token_id': 10, 'recompute_token_id': 31},
                'warmup': {'enabled': False}
            },
            'logging': {
                'output_dir': temp_dir,
                'metrics': {
                    'enabled': True,
                    'file_pattern': 'metrics_{timestamp}.json'
                },
                'responses': {
                    'enabled': True,
                    'file_pattern': 'response_{request_id}.json'
                }
            }
        }
        return MiddlewareConfig(**config_dict)
    
    def test_get_next_request_id(self, config):
        """Test request ID generation."""
        logger = MiddlewareLogger(config)
        
        id1 = logger.get_next_request_id()
        id2 = logger.get_next_request_id()
        
        assert id1 == "req_000000"
        assert id2 == "req_000001"
    
    def test_log_complete(self, config):
        """Test logging complete request."""
        logger = MiddlewareLogger(config)
        
        request_id = "req_000001"
        logger.log_request_start(request_id)
        
        logger.log_complete(
            request_id,
            request_data={'model': 'test-model'},
            response_data={'content': 'Test response'},
            metrics_data={'prompt_tokens': 100}
        )
        
        assert request_id in logger.metrics.metrics
        
        filepath = Path(config.logging.output_dir) / f'response_{request_id}.json'
        assert filepath.exists()
