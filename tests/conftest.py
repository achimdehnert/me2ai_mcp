"""Test fixtures."""
import pytest
from typing import List, Dict, Any
from me2ai.llms.base import LLMProvider
from me2ai.memory import ConversationMemory
from me2ai.agents.individual import (
    GermanProfessor,
    DatingExpert,
    SEOExpert,
    Researcher
)

class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self):
        """Initialize mock provider."""
        self.default_response = "Mock response"
        self.calls = []
        
    async def generate(self, messages: List[Dict[str, Any]]) -> str:
        """Mock generate method."""
        self.calls.append({"messages": messages})
        return self.default_response
        
    async def generate_response(self, messages: List[Dict[str, Any]]) -> str:
        """Mock generate_response method."""
        return await self.generate(messages)

class MockMemory(ConversationMemory):
    """Mock conversation memory for testing."""
    
    def __init__(self):
        """Initialize mock memory."""
        super().__init__()
        self._messages = []
        
    def add_message(self, role: str, content: str):
        """Add message to memory."""
        self._messages.append({"role": role, "content": content})
        
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self._messages
        
    def clear(self):
        """Clear memory."""
        self._messages = []

@pytest.fixture
def mock_llm():
    """Create mock LLM provider."""
    return MockLLMProvider()

@pytest.fixture
def mock_memory():
    """Create mock conversation memory."""
    return MockMemory()

@pytest.fixture
def german_professor(mock_llm, mock_memory):
    """Create German professor agent."""
    return GermanProfessor(mock_llm, mock_memory)

@pytest.fixture
def dating_expert(mock_llm, mock_memory):
    """Create dating expert agent."""
    return DatingExpert(mock_llm, mock_memory)

@pytest.fixture
def seo_expert(mock_llm, mock_memory):
    """Create SEO expert agent."""
    return SEOExpert(mock_llm, mock_memory)

@pytest.fixture
def researcher(mock_llm, mock_memory):
    """Create researcher agent."""
    return Researcher(mock_llm, mock_memory)
