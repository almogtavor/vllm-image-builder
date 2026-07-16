"""Pytest configuration and shared fixtures."""

import sys
import tempfile
from pathlib import Path

import pytest

# Add the middleware package dir to the path so tests can import its modules.
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
