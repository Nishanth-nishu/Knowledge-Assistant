"""
Pytest configuration and fixtures.
"""

import os
import pytest
from typing import Generator

# Set test environment
os.environ["LLM_API_KEY"] = "test-key"
os.environ["NEO4J_PASSWORD"] = "test-password"
os.environ["JWT_SECRET_KEY"] = "test-secret"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings():
    """Provide mock settings for testing."""
    from src.config import Settings
    return Settings()


@pytest.fixture
def sample_messages():
    """Sample chat messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "id": "doc1",
            "content": "Paris is the capital of France.",
            "metadata": {"source": "geography.txt"},
        },
        {
            "id": "doc2",
            "content": "The Eiffel Tower is located in Paris.",
            "metadata": {"source": "landmarks.txt"},
        },
    ]
