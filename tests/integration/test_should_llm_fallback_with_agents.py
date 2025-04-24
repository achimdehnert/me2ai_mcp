"""
Integration tests for the LLMFallbackMixin with ME2AI MCP agents.

These tests validate how the LLMFallbackMixin integrates with different agent types
and handles the OpenAI → Anthropic → heuristic fallback chain in real-world scenarios.
"""
import os
import pytest
from unittest.mock import patch, MagicMock

from me2ai_mcp import (
    LLMFallbackMixin, 
    LLMProvider, 
    SpecializedAgent, 
    CollaborativeAgent
)


class TestLLMAwareAgent(LLMFallbackMixin, SpecializedAgent):
    """Test agent class implementing the LLMFallbackMixin."""
    
    def __init__(self, agent_id: str, name: str):
        """Initialize the test agent."""
        # Define tools
        self.tool_names = ["process_content", "analyze_text"]
        # Initialize both parent classes
        SpecializedAgent.__init__(self, agent_id, name, "Test LLM-aware agent")
        LLMFallbackMixin.__init__(self)
    
    def process_content(self, text: str, max_length: int = 200):
        """Process content using LLM capabilities."""
        summary = self.summarize_text_with_fallback(text, max_length)
        entities = self.extract_entities_with_fallback(text)
        
        return {
            "summary": summary.get("summary", ""),
            "entities": entities,
            "summary_method": summary.get("method", "unknown")
        }
    
    def analyze_text(self, text: str):
        """Analyze text sentiment and extract key phrases."""
        prompt = """
        Analyze the sentiment and key phrases in this text.
        Return in JSON format with these fields:
        {
            "sentiment": "positive/negative/neutral",
            "score": 0.0 to 1.0,
            "key_phrases": ["phrase1", "phrase2", ...]
        }
        
        Text: {text}
        
        JSON result:
        """
        
        result = self.run_llm_chain(prompt, text)
        
        if not result:
            # Fallback to simple analysis
            sentiment = "neutral"
            score = 0.5
            # Extract simple key phrases (capitalized words or phrases)
            import re
            key_phrases = re.findall(r'\b[A-Z][a-zA-Z]+(?: [A-Za-z]+){0,3}\b', text)
            
            return {
                "sentiment": sentiment,
                "score": score,
                "key_phrases": key_phrases[:5],
                "method": "heuristic"
            }
        
        # Try to parse JSON
        try:
            import json
            data = json.loads(result)
            data["method"] = "llm"
            return data
        except:
            # If JSON parsing fails, return the raw result
            return {
                "raw_result": result,
                "method": "llm_raw"
            }


class TestLLMAwareCollaborativeAgent(LLMFallbackMixin, CollaborativeAgent):
    """Test collaborative agent implementing the LLMFallbackMixin."""
    
    def __init__(self, agent_id: str, name: str):
        """Initialize the test collaborative agent."""
        # Initialize both parent classes
        CollaborativeAgent.__init__(self, agent_id, name, "Test collaborative LLM agent")
        LLMFallbackMixin.__init__(self)
    
    def process_query(self, query: str, context: str = ""):
        """Process a query using LLM capabilities and collaboration context."""
        # Combine query and context
        full_text = f"{query}\n\nContext: {context}" if context else query
        
        # Generate a response using LLM
        prompt = """
        Please answer the following query using the provided context if available.
        Be concise and specific in your response.
        
        Query: {text}
        
        Response:
        """
        
        result = self.run_llm_chain(prompt, full_text)
        
        if not result:
            # Fallback to simple response
            return {
                "response": "I'm unable to process this query right now. Please try again later.",
                "method": "heuristic"
            }
        
        return {
            "response": result,
            "method": "llm"
        }
    
    def can_handle_task(self, task_description: str) -> bool:
        """Determine if this agent can handle the described task."""
        # Use LLM to evaluate capabilities if available
        prompt = """
        Evaluate if the agent with these capabilities:
        - Text summarization
        - Entity extraction
        - Semantic analysis
        
        Can handle the following task. Return YES or NO only.
        
        Task: {text}
        
        Response (YES/NO):
        """
        
        result = self.run_llm_chain(prompt, task_description)
        
        if result and "YES" in result.upper():
            return True
        
        # Fallback to keyword matching
        keywords = ["summarize", "extract", "analyze", "text", "content", "document"]
        return any(keyword in task_description.lower() for keyword in keywords)


class TestAgentIntegration:
    """Integration test suite for LLMFallbackMixin with agents."""
    
    @pytest.fixture
    def llm_agent(self):
        """Create a test LLM-aware agent."""
        return TestLLMAwareAgent("test_agent", "Test LLM Agent")
    
    @pytest.fixture
    def collab_agent(self):
        """Create a test LLM-aware collaborative agent."""
        return TestLLMAwareCollaborativeAgent("test_collab", "Test Collaborative LLM Agent")
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"})
    @patch("me2ai_mcp.llm_fallback.OpenAI")
    def test_should_specialized_agent_use_llm(self, mock_openai, llm_agent):
        """Test that a specialized agent can use the LLM for text processing."""
        # Setup mock
        mock_llm = MagicMock()
        mock_llm.return_value = '{"sentiment": "positive", "score": 0.9, "key_phrases": ["Great Product", "Excellent Service"]}'
        mock_openai.return_value = mock_llm
        
        # Test text
        text = "This is an excellent product with great features. The service was outstanding."
        
        # Process content
        result = llm_agent.analyze_text(text)
        
        # Verify LLM was used
        assert result["method"] == "llm"
        assert result["sentiment"] == "positive"
        assert "Great Product" in result["key_phrases"]
        mock_llm.assert_called_once()
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""})
    def test_should_specialized_agent_use_fallback_when_no_llm(self, llm_agent):
        """Test that a specialized agent falls back to heuristics when no LLM is available."""
        # Test text
        text = "This is a test for the Fallback Logic with Capitalized Phrases."
        
        # Process content
        result = llm_agent.analyze_text(text)
        
        # Verify fallback was used
        assert result["method"] == "heuristic"
        assert "Fallback Logic" in result["key_phrases"]
        assert "Capitalized Phrases" in result["key_phrases"]
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"})
    @patch("me2ai_mcp.llm_fallback.OpenAI")
    def test_should_collaborative_agent_use_llm(self, mock_openai, collab_agent):
        """Test that a collaborative agent can use the LLM for processing queries."""
        # Setup mock
        mock_llm = MagicMock()
        mock_llm.return_value = "To summarize a document, you can use the TextSummarizer tool."
        mock_openai.return_value = mock_llm
        
        # Test query
        query = "How do I summarize a document?"
        context = "Previous tools used: DocumentLoader, TextAnalyzer"
        
        # Process query
        result = collab_agent.process_query(query, context)
        
        # Verify LLM was used
        assert result["method"] == "llm"
        assert "TextSummarizer" in result["response"]
        mock_llm.assert_called_once()
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"})
    @patch("me2ai_mcp.llm_fallback.OpenAI")
    def test_should_agent_check_capabilities_with_llm(self, mock_openai, collab_agent):
        """Test that an agent can evaluate its capabilities using an LLM."""
        # Setup mock
        mock_llm = MagicMock()
        mock_llm.return_value = "YES"
        mock_openai.return_value = mock_llm
        
        # Test task descriptions
        task1 = "I need to extract key entities from this document"
        task2 = "Can you generate a 3D model of a building?"
        
        # Check capabilities
        result1 = collab_agent.can_handle_task(task1)
        
        # Reset mock return value for second task
        mock_llm.return_value = "NO"
        result2 = collab_agent.can_handle_task(task2)
        
        # Verify LLM was used correctly
        assert result1 is True
        assert result2 is False
        assert mock_llm.call_count == 2
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"})
    @patch("me2ai_mcp.llm_fallback.OpenAI")
    def test_should_handle_complete_workflow(self, mock_openai, llm_agent):
        """Test a complete workflow using multiple LLM-based methods."""
        # Setup mock for different responses
        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm
        
        # Mock different responses based on input
        def mock_response(prompt):
            if "Analyze the sentiment" in prompt:
                return '{"sentiment": "positive", "score": 0.8, "key_phrases": ["Machine Learning", "AI Technology"]}'
            elif "summary" in prompt.lower():
                return "This text discusses advances in AI and Machine Learning."
            else:
                return '[{"name": "Machine Learning", "type": "TECHNOLOGY", "relevance": 0.9}, {"name": "AI", "type": "TECHNOLOGY", "relevance": 0.9}]'
        
        mock_llm.side_effect = mock_response
        
        # Test text
        text = """
        Machine Learning and AI technologies are advancing rapidly. These technologies 
        are transforming industries like healthcare, finance, and transportation.
        Deep learning models have shown impressive results in image recognition and 
        natural language processing tasks.
        """
        
        # Process complete workflow
        analysis = llm_agent.analyze_text(text)
        content_result = llm_agent.process_content(text)
        
        # Verify all parts worked correctly
        assert analysis["sentiment"] == "positive"
        assert "Machine Learning" in analysis["key_phrases"]
        
        assert "AI" in content_result["summary"]
        assert content_result["summary_method"] == "llm"
        
        assert len(content_result["entities"]) >= 1
        assert any(e["name"] == "Machine Learning" for e in content_result["entities"])
