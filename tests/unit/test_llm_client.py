"""
Tests for the LLM client module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.models.llm_client import (
    LLMProvider,
    Message,
    LLMResponse,
    LLMClientConfig,
    OpenAICompatibleClient,
    LLMClientFactory,
    get_llm_client,
)


class TestLLMClientConfig:
    """Tests for LLMClientConfig."""
    
    def test_from_settings(self):
        """Test creating config from settings."""
        from src.config import LLMSettings
        
        settings = LLMSettings(
            provider="openai",
            api_base="https://api.openai.com/v1",
            api_key="test-key",
            model="gpt-4",
            temperature=0.5,
        )
        
        config = LLMClientConfig.from_settings(settings)
        
        assert config.provider == LLMProvider.OPENAI
        assert config.api_base == "https://api.openai.com/v1"
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
    
    def test_default_values(self):
        """Test default configuration values."""
        config = LLMClientConfig(
            provider=LLMProvider.OPENAI,
            api_base="https://api.openai.com/v1",
            api_key="test-key",
            model="gpt-4",
        )
        
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.timeout == 60
        assert config.max_retries == 3


class TestMessage:
    """Tests for Message class."""
    
    def test_to_dict_basic(self):
        """Test basic message conversion."""
        msg = Message(role="user", content="Hello")
        
        result = msg.to_dict()
        
        assert result == {"role": "user", "content": "Hello"}
    
    def test_to_dict_with_name(self):
        """Test message with name field."""
        msg = Message(role="function", content="result", name="my_function")
        
        result = msg.to_dict()
        
        assert result["name"] == "my_function"


class TestLLMResponse:
    """Tests for LLMResponse class."""
    
    def test_token_properties(self):
        """Test token count properties."""
        response = LLMResponse(
            content="Hello",
            model="gpt-4",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        )
        
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 5
        assert response.total_tokens == 15


class TestOpenAICompatibleClient:
    """Tests for OpenAICompatibleClient."""
    
    def test_create_client_openai(self):
        """Test creating OpenAI client."""
        config = LLMClientConfig(
            provider=LLMProvider.OPENAI,
            api_base="https://api.openai.com/v1",
            api_key="test-key",
            model="gpt-4",
        )
        
        client = OpenAICompatibleClient(config)
        
        assert client.config.provider == LLMProvider.OPENAI
    
    def test_create_client_ollama(self):
        """Test creating Ollama client."""
        config = LLMClientConfig(
            provider=LLMProvider.OLLAMA,
            api_base="http://localhost:11434/v1",
            api_key="",
            model="llama3.1",
        )
        
        client = OpenAICompatibleClient(config)
        
        assert client.config.provider == LLMProvider.OLLAMA
    
    def test_normalize_messages(self):
        """Test message normalization."""
        config = LLMClientConfig(
            provider=LLMProvider.OPENAI,
            api_base="https://api.openai.com/v1",
            api_key="test-key",
            model="gpt-4",
        )
        client = OpenAICompatibleClient(config)
        
        messages = [
            Message(role="user", content="Hello"),
            {"role": "assistant", "content": "Hi there"},
        ]
        
        normalized = client._normalize_messages(messages)
        
        assert len(normalized) == 2
        assert all(isinstance(m, dict) for m in normalized)
    
    def test_count_tokens(self):
        """Test token counting."""
        config = LLMClientConfig(
            provider=LLMProvider.OPENAI,
            api_base="https://api.openai.com/v1",
            api_key="test-key",
            model="gpt-4",
        )
        client = OpenAICompatibleClient(config)
        
        count = client.count_tokens("Hello, world!")
        
        assert count > 0
        assert isinstance(count, int)


class TestLLMClientFactory:
    """Tests for LLMClientFactory."""
    
    def test_create_client(self):
        """Test factory creates client."""
        from src.config import LLMSettings
        
        settings = LLMSettings(
            provider="openai",
            api_base="https://api.openai.com/v1",
            api_key="test-key",
            model="gpt-4",
        )
        
        client = LLMClientFactory.create(settings=settings)
        
        assert isinstance(client, OpenAICompatibleClient)
    
    def test_get_or_create_singleton(self):
        """Test singleton behavior."""
        from src.config import LLMSettings
        
        settings = LLMSettings(
            provider="openai",
            api_base="https://api.openai.com/v1",
            api_key="test-key",
            model="gpt-4",
        )
        
        # Clear existing instances
        LLMClientFactory._instances.clear()
        
        client1 = LLMClientFactory.get_or_create("test", settings=settings)
        client2 = LLMClientFactory.get_or_create("test", settings=settings)
        
        assert client1 is client2


@pytest.mark.asyncio
class TestOpenAICompatibleClientAsync:
    """Async tests for OpenAICompatibleClient."""
    
    async def test_chat_mock(self):
        """Test chat with mocked response."""
        config = LLMClientConfig(
            provider=LLMProvider.OPENAI,
            api_base="https://api.openai.com/v1",
            api_key="test-key",
            model="gpt-4",
        )
        client = OpenAICompatibleClient(config)
        
        # Mock the OpenAI client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.choices[0].message.function_call = None
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        
        with patch.object(client, '_async_client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            client._async_client = mock_client
            
            response = await client.chat([{"role": "user", "content": "Hi"}])
            
            assert response.content == "Hello!"
            assert response.total_tokens == 15
