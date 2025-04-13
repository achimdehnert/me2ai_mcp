"""Tests for LLM providers."""
import pytest
from me2ai.llms.openai_provider import OpenAIProvider
from me2ai.llms.groq_provider import GroqProvider
from me2ai.llms.anthropic_provider import AnthropicProvider

def test_openai_provider_init():
    """Test OpenAI provider initialization."""
    provider = OpenAIProvider()
    assert provider.model == "gpt-4o-mini"

def test_groq_provider_init():
    """Test Groq provider initialization."""
    provider = GroqProvider()
    assert provider.model == "mixtral-8x7b-32768"

def test_anthropic_provider_init():
    """Test Anthropic provider initialization."""
    provider = AnthropicProvider()
    assert provider.model == "claude-3-opus-20240229"

@pytest.mark.integration
def test_openai_provider_generate():
    """Test OpenAI provider response generation."""
    provider = OpenAIProvider()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello!"}
    ]
    response = provider.generate_response(messages)
    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.integration
def test_groq_provider_generate():
    """Test Groq provider response generation."""
    provider = GroqProvider()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello!"}
    ]
    response = provider.generate_response(messages)
    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.integration
def test_anthropic_provider_generate():
    """Test Anthropic provider response generation."""
    provider = AnthropicProvider()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello!"}
    ]
    response = provider.generate_response(messages)
    assert isinstance(response, str)
    assert len(response) > 0

def test_provider_error_handling(mock_llm):
    """Test error handling in providers."""
    # Simulate an API error
    mock_llm.responses = {"error": "API Error"}
    response = mock_llm.generate_response([
        {"role": "user", "content": "error"}
    ])
    assert "Error" not in response  # Mock provider should handle errors gracefully
