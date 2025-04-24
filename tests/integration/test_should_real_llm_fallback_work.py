"""
Real LLM fallback tests for the ME2AI MCP package.

These tests use actual LLM API keys from environment variables to test the
fallback chain with real LLM providers. Tests are skipped if API keys aren't present.
Tests use actual API calls and will consume quota.
"""
import os
import json
import pytest
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables from any .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from me2ai_mcp import LLMFallbackMixin, LLMProvider


class RealLLMTester(LLMFallbackMixin):
    """Test class that uses real LLM API calls to validate the fallback chain."""
    
    def __init__(self):
        """Initialize the tester with logging enabled."""
        super().__init__()
        # Track which provider was actually used
        self.last_provider_used = None
    
    def get_available_llm(self):
        """Override to track which provider is being used."""
        result = super().get_available_llm()
        if result:
            self.last_provider_used = result[0]
        return result
    
    def summarize_article(self, article_text: str) -> Dict[str, Any]:
        """Summarize an article using a real LLM."""
        result = self.summarize_text_with_fallback(article_text, max_length=150)
        # Add which provider was used
        if result and self.last_provider_used:
            result["provider"] = self.last_provider_used.name
        return result
    
    def extract_entities_from_text(self, text: str) -> Dict[str, Any]:
        """Extract entities from text using a real LLM."""
        entities = self.extract_entities_with_fallback(text)
        return {
            "entities": entities,
            "count": len(entities),
            "provider": self.last_provider_used.name if self.last_provider_used else "HEURISTIC"
        }
    
    def generate_questions(self, text: str, count: int = 3) -> Dict[str, Any]:
        """Generate questions from text using a real LLM."""
        prompt = f"""
        Generate {count} questions based on the following text.
        Return the questions as a JSON array of strings.
        Each question should be thought-provoking and related to the content.
        
        Text: {text}
        
        Questions (JSON array):
        """
        
        result = self.run_llm_chain(prompt, "")
        
        if result:
            try:
                questions = json.loads(result)
                return {
                    "questions": questions[:count],
                    "count": min(len(questions), count),
                    "provider": self.last_provider_used.name if self.last_provider_used else "UNKNOWN",
                    "success": True
                }
            except json.JSONDecodeError:
                # Fall back to parsing line by line
                lines = result.strip().split("\n")
                questions = [line.strip().strip('"').strip("'") for line in lines if "?" in line]
                return {
                    "questions": questions[:count],
                    "count": len(questions[:count]),
                    "provider": self.last_provider_used.name if self.last_provider_used else "UNKNOWN",
                    "success": True,
                    "parse_method": "line_by_line"
                }
        
        # Heuristic fallback for question generation
        # Generate simple questions based on the content
        import re
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        
        # Find potential topics from the text (capitalized words or phrases)
        topics = re.findall(r'\b[A-Z][a-zA-Z]+(?: [A-Za-z]+){0,3}\b', text)
        
        questions = []
        templates = [
            "What is the significance of {topic}?",
            "How does {topic} impact the subject matter?",
            "Why is {topic} mentioned in the text?",
            "What can we learn about {topic} from this text?",
            "How would you explain {topic} based on this text?"
        ]
        
        # Generate questions using templates and topics
        for i, topic in enumerate(topics[:count]):
            if i < len(templates):
                questions.append(templates[i].format(topic=topic))
        
        # If not enough topics, use sentences
        if len(questions) < count and sentences:
            for i, sentence in enumerate(sentences[:count-len(questions)]):
                if len(sentence.split()) > 5:  # Only use substantial sentences
                    # Convert statement to question
                    words = sentence.strip().split()
                    if len(words) > 3:
                        question = f"Why does the text state that {' '.join(words[:5]).lower()}...?"
                        questions.append(question)
        
        return {
            "questions": questions[:count],
            "count": len(questions[:count]),
            "provider": "HEURISTIC",
            "success": True,
            "parse_method": "heuristic"
        }


@pytest.mark.skipif(
    not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")),
    reason="No LLM API keys found in environment"
)
class TestRealLLMFallback:
    """Test suite for real LLM fallback using actual API keys."""
    
    @pytest.fixture
    def llm_tester(self):
        """Create a real LLM tester."""
        return RealLLMTester()
    
    @pytest.fixture
    def test_article(self):
        """Provide a test article for processing."""
        return """
        Artificial Intelligence (AI) is transforming industries across the globe. 
        Machine Learning, a subset of AI, enables systems to learn from data and improve 
        over time without explicit programming. Deep Learning, which uses neural networks 
        with many layers, has led to breakthroughs in image recognition, natural language 
        processing, and game playing. Companies like OpenAI, Anthropic, and Google DeepMind 
        are pushing the boundaries of what AI can accomplish. However, these advances also 
        raise important ethical questions about privacy, bias, job displacement, and the 
        long-term implications of increasingly autonomous systems.
        """
    
    def test_should_use_real_llm_for_summarization(self, llm_tester, test_article):
        """Test that the mixin can use a real LLM for summarization."""
        # Check that at least one provider is available
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("No LLM API keys available")
        
        # Summarize the article
        result = llm_tester.summarize_article(test_article)
        
        # Log which provider was used
        logger.info(f"Summarization used provider: {result.get('provider', 'UNKNOWN')}")
        logger.info(f"Summary: {result.get('summary', '')[:100]}...")
        
        # Verify the result
        assert "summary" in result
        assert len(result["summary"]) > 0
        assert "method" in result
        assert result["method"] == "llm"
        assert "provider" in result
        assert result["provider"] in ["OPENAI", "ANTHROPIC"]
    
    def test_should_use_real_llm_for_entity_extraction(self, llm_tester, test_article):
        """Test that the mixin can use a real LLM for entity extraction."""
        # Check that at least one provider is available
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("No LLM API keys available")
        
        # Extract entities
        result = llm_tester.extract_entities_from_text(test_article)
        
        # Log which provider was used
        logger.info(f"Entity extraction used provider: {result.get('provider', 'UNKNOWN')}")
        
        # Log the first few entities
        for entity in result["entities"][:3]:
            logger.info(f"Entity: {entity.get('name')} ({entity.get('type')})")
        
        # Verify the result
        assert "entities" in result
        assert len(result["entities"]) > 0
        assert "provider" in result
        assert result["provider"] in ["OPENAI", "ANTHROPIC", "HEURISTIC"]
        
        # Check for specific entities that should be found
        entity_names = [e.get("name") for e in result["entities"]]
        assert any("AI" in name or "Artificial Intelligence" in name for name in entity_names)
        assert any("OpenAI" in name for name in entity_names)
    
    def test_should_use_real_llm_for_question_generation(self, llm_tester, test_article):
        """Test that the mixin can use a real LLM for generating questions."""
        # Check that at least one provider is available
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("No LLM API keys available")
        
        # Generate questions
        result = llm_tester.generate_questions(test_article)
        
        # Log which provider was used
        logger.info(f"Question generation used provider: {result.get('provider', 'UNKNOWN')}")
        
        # Log the questions
        for i, question in enumerate(result["questions"], 1):
            logger.info(f"Question {i}: {question}")
        
        # Verify the result
        assert "questions" in result
        assert len(result["questions"]) > 0
        assert "provider" in result
        assert result["success"] is True
    
    @pytest.mark.parametrize("env_var", ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"])
    def test_should_fallback_to_alternative_when_primary_removed(self, llm_tester, test_article, env_var):
        """
        Test fallback by temporarily removing the primary LLM API key.
        
        This test works best if both OpenAI and Anthropic keys are present.
        It will:
        1. Store the original key
        2. Remove it temporarily
        3. Test if the system falls back to the alternative LLM
        4. Restore the original key
        """
        # Skip if the API key isn't present
        if not os.getenv(env_var):
            pytest.skip(f"{env_var} not present in environment")
        
        # Check if an alternative provider is available for fallback
        has_openai = bool(os.getenv("OPENAI_API_KEY"))
        has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
        
        if not (has_openai and has_anthropic):
            pytest.skip("Need both OpenAI and Anthropic keys to test real fallback")
        
        # Store original key
        original_key = os.environ.get(env_var)
        
        try:
            # Temporarily remove the key
            del os.environ[env_var]
            logger.info(f"Temporarily removed {env_var} to test fallback")
            
            # Try summarization
            result = llm_tester.summarize_article(test_article)
            
            # Verify fallback worked
            assert "summary" in result
            assert len(result["summary"]) > 0
            assert "method" in result
            assert result["method"] == "llm"  # Should still use LLM, just the fallback one
            assert "provider" in result
            
            # If we removed OpenAI, should fall back to Anthropic
            if env_var == "OPENAI_API_KEY":
                assert result["provider"] == "ANTHROPIC"
            # If we removed Anthropic, should use OpenAI
            elif env_var == "ANTHROPIC_API_KEY":
                assert result["provider"] == "OPENAI"
            
            logger.info(f"Successfully fell back to {result['provider']} when {env_var} was removed")
            
        finally:
            # Restore the original key
            if original_key:
                os.environ[env_var] = original_key
                logger.info(f"Restored {env_var}")
    
    def test_should_fall_back_to_heuristic_when_no_llms_available(self, llm_tester, test_article):
        """Test that the system falls back to heuristics when no LLMs are available."""
        # Store original keys
        openai_key = os.environ.get("OPENAI_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        
        try:
            # Remove all LLM API keys
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
            if "ANTHROPIC_API_KEY" in os.environ:
                del os.environ["ANTHROPIC_API_KEY"]
            
            logger.info("Temporarily removed all LLM API keys")
            
            # Test all methods with heuristic fallback
            summary_result = llm_tester.summarize_article(test_article)
            entity_result = llm_tester.extract_entities_from_text(test_article)
            question_result = llm_tester.generate_questions(test_article)
            
            # Verify all fell back to heuristics
            assert summary_result["method"] == "extractive"
            assert entity_result["provider"] == "HEURISTIC"
            assert question_result["provider"] == "HEURISTIC"
            
            # Verify we still got usable results
            assert len(summary_result["summary"]) > 0
            assert len(entity_result["entities"]) > 0
            assert len(question_result["questions"]) > 0
            
            logger.info("Successfully fell back to heuristic methods when no LLMs available")
            
        finally:
            # Restore the original keys
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key
            if anthropic_key:
                os.environ["ANTHROPIC_API_KEY"] = anthropic_key
            
            logger.info("Restored original API keys")
