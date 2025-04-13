"""Tests for the routing agent."""
import pytest
from unittest.mock import Mock, AsyncMock
from me2ai.agents.management.routing.router import RouterAgent
from me2ai.agents.base import BaseAgent
from me2ai.llms.base import LLMProvider

@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    llm = Mock(spec=LLMProvider)
    llm.generate_response = AsyncMock(return_value="Test response")
    return llm

@pytest.fixture
def mock_memory():
    """Create a mock conversation memory."""
    memory = Mock()
    memory.chat_history.messages = []
    memory.clear = Mock()
    return memory

@pytest.fixture
def mock_agents():
    """Create mock expert agents."""
    german_professor = Mock(spec=BaseAgent)
    german_professor.role = "German Professor"
    german_professor.respond = AsyncMock(return_value="German response")
    
    dating_expert = Mock(spec=BaseAgent)
    dating_expert.role = "Dating Expert"
    dating_expert.respond = AsyncMock(return_value="Dating advice")
    
    seo_expert = Mock(spec=BaseAgent)
    seo_expert.role = "SEO Expert"
    seo_expert.respond = AsyncMock(return_value="SEO tips")
    
    return {
        'german_professor': german_professor,
        'dating_expert': dating_expert,
        'seo_expert': seo_expert
    }

@pytest.mark.asyncio
async def test_router_initialization(mock_llm, mock_memory, mock_agents):
    """Test router initialization."""
    router = RouterAgent(mock_agents, mock_llm, mock_memory)
    assert router.role == "Router"
    assert router.agents == mock_agents
    assert len(router.routing_rules) == 3

@pytest.mark.asyncio
async def test_router_query_parsing(mock_llm, mock_memory, mock_agents):
    """Test query parsing and routing."""
    router = RouterAgent(mock_agents, mock_llm, mock_memory)
    
    # Test routing to German professor
    expert, reason = await router.get_agent("How do I learn German grammar?")
    assert expert == mock_agents['german_professor']
    assert "german grammar" in reason.lower()
    
    # Test routing to dating expert
    expert, reason = await router.get_agent("What are good first date ideas?")
    assert expert == mock_agents['dating_expert']
    assert "first date" in reason.lower()
    
    # Test routing to SEO expert
    expert, reason = await router.get_agent("How can I improve my website's SEO?")
    assert expert == mock_agents['seo_expert']
    assert "seo" in reason.lower()
    
    # Test unmatched query
    expert, reason = await router.get_agent("What's the weather like?")
    assert expert == router
    assert "no specific expert" in reason.lower()

@pytest.mark.asyncio
async def test_router_agent_selection(mock_llm, mock_memory, mock_agents):
    """Test agent selection logic."""
    router = RouterAgent(mock_agents, mock_llm, mock_memory)
    
    # Test direct routing
    expert, reason = await router.get_agent("german grammar")
    assert expert == mock_agents['german_professor']
    
    # Test case insensitive routing
    expert, reason = await router.get_agent("GERMAN GRAMMAR")
    assert expert == mock_agents['german_professor']
    
    # Test partial keyword match
    expert, reason = await router.get_agent("I need dating advice")
    assert expert == mock_agents['dating_expert']
    
    # Test no match
    expert, reason = await router.get_agent("unrelated query")
    assert expert == router

@pytest.mark.asyncio
async def test_router_real_responses(mock_llm, mock_memory, mock_agents):
    """Test router with real responses."""
    router = RouterAgent(mock_agents, mock_llm, mock_memory)
    
    # Test German query
    response = await router.respond("How do I learn German grammar?")
    assert response == "German response"
    
    # Test dating query
    response = await router.respond("What are good first date ideas?")
    assert response == "Dating advice"
    
    # Test unmatched query
    response = await router.respond("What's the weather like?")
    assert "not sure which expert" in response.lower()
