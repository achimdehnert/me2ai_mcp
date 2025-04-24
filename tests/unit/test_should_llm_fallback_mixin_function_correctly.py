"""
Tests for the LLMFallbackMixin class in the ME2AI MCP package.

These tests validate the robustness of the multi-LLM fallback pattern
and ensure compliance with the user-defined coding standards.
"""
import os
import json
import pytest
from unittest.mock import patch, MagicMock

from me2ai_mcp import LLMFallbackMixin, LLMProvider


class TestLLMFallbackMixin:
    """Test suite for the LLMFallbackMixin class."""
    
    @pytest.fixture
    def fallback_instance(self):
        """Create a simple instance of the LLMFallbackMixin for testing."""
        class TestAgent(LLMFallbackMixin):
            """Test agent class using the LLMFallbackMixin."""
            pass
        
        return TestAgent()
    
    def test_should_initialize_correctly(self, fallback_instance):
        """Test that the mixin initializes correctly."""
        assert hasattr(fallback_instance, "logger")
        assert fallback_instance.logger is not None
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"})
    @patch("me2ai_mcp.llm_fallback.OpenAI")
    def test_should_prefer_openai_when_available(self, mock_openai, fallback_instance):
        """Test that the mixin prefers OpenAI when it's available."""
        # Setup the mock
        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm
        
        # Get the LLM
        result = fallback_instance.get_available_llm()
        
        # Check the result
        assert result is not None
        provider, llm = result
        assert provider == LLMProvider.OPENAI
        assert llm == mock_llm
        mock_openai.assert_called_once()
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "fake-key"})
    @patch("me2ai_mcp.llm_fallback.Anthropic")
    def test_should_use_anthropic_when_openai_unavailable(self, mock_anthropic, fallback_instance):
        """Test that the mixin uses Anthropic when OpenAI is unavailable."""
        # Setup the mock
        mock_llm = MagicMock()
        mock_anthropic.return_value = mock_llm
        
        # Get the LLM
        result = fallback_instance.get_available_llm()
        
        # Check the result
        assert result is not None
        provider, llm = result
        assert provider == LLMProvider.ANTHROPIC
        assert llm == mock_llm
        mock_anthropic.assert_called_once()
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""})
    def test_should_return_none_when_no_llms_available(self, fallback_instance):
        """Test that the mixin returns None when no LLMs are available."""
        # Get the LLM
        result = fallback_instance.get_available_llm()
        
        # Check the result
        assert result is None
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"})
    @patch("me2ai_mcp.llm_fallback.OpenAI")
    def test_should_run_llm_chain_with_openai(self, mock_openai, fallback_instance):
        """Test that run_llm_chain uses OpenAI when available."""
        # Setup the mock
        mock_llm = MagicMock()
        mock_llm.return_value = "This is a test response from OpenAI."
        mock_openai.return_value = mock_llm
        
        # Run the LLM chain
        result = fallback_instance.run_llm_chain("Test prompt {text}", "test content")
        
        # Check the result
        assert result == "This is a test response from OpenAI."
        mock_llm.assert_called_once()
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"})
    @patch("me2ai_mcp.llm_fallback.OpenAI")
    def test_should_format_prompt_correctly(self, mock_openai, fallback_instance):
        """Test that run_llm_chain formats the prompt correctly."""
        # Setup the mock
        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm
        
        # Create a test prompt
        prompt = "Summarize the following text: {text}"
        content = "This is test content."
        
        # Run the LLM chain
        fallback_instance.run_llm_chain(prompt, content)
        
        # Check the formatted prompt
        call_args = mock_llm.call_args[0][0]
        assert "Summarize the following text: This is test content." == call_args
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""})
    def test_should_extract_entities_with_heuristic_fallback(self, fallback_instance):
        """Test that extract_entities_with_fallback uses heuristic method when no LLMs are available."""
        # Sample text with named entities
        text = "John Smith is the CEO of Acme Corporation based in New York City. Contact him at john@acme.com or visit https://acme.com."
        
        # Extract entities
        entities = fallback_instance.extract_entities_with_fallback(text)
        
        # Check the result
        assert len(entities) > 0
        
        # Verify some expected entities were found
        entity_names = [e["name"] for e in entities]
        assert "John Smith" in entity_names
        assert "Acme Corporation" in entity_names
        assert "New York City" in entity_names
        assert "john@acme.com" in entity_names
        assert "https://acme.com" in entity_names
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""})
    def test_should_summarize_text_with_extractive_fallback(self, fallback_instance):
        """Test that summarize_text_with_fallback uses extractive method when no LLMs are available."""
        # Sample text to summarize
        text = """
        Artificial Intelligence (AI) is transforming industries across the globe. Machine learning, 
        a subset of AI, enables computers to learn from data and improve over time. Deep learning, 
        a specialized form of machine learning, uses neural networks with many layers to analyze 
        complex patterns. Natural Language Processing allows computers to understand and generate 
        human language, while Computer Vision enables machines to interpret and make decisions based 
        on visual input. The ethical implications of AI are important to consider, including issues 
        of bias, privacy, and job displacement.
        """
        
        # Generate summary
        result = fallback_instance.summarize_text_with_fallback(text, max_length=150)
        
        # Check the result
        assert "summary" in result
        assert len(result["summary"]) <= 150
        assert "method" in result
        assert result["method"] == "extractive"
        
        # Verify the summary contains key information
        assert "Artificial Intelligence" in result["summary"]
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""})
    def test_should_generate_embeddings_with_numpy_fallback(self, fallback_instance):
        """Test that generate_embeddings_with_fallback uses numpy method when no embedding models are available."""
        # Mock the imports to force the numpy fallback
        with patch("me2ai_mcp.llm_fallback.SentenceTransformer", side_effect=ImportError):
            # Sample texts to embed
            texts = [
                "This is the first document.",
                "This is the second document."
            ]
            
            # Generate embeddings
            embeddings = fallback_instance.generate_embeddings_with_fallback(texts)
            
            # Check the result
            assert embeddings is not None
            assert len(embeddings) == 2
            assert len(embeddings[0]) == 384  # Default embedding dimension
            
            # Verify the embeddings are normalized (unit vectors)
            import numpy as np
            for embedding in embeddings:
                norm = np.linalg.norm(embedding)
                assert abs(norm - 1.0) < 1e-6  # Should be approximately 1.0
