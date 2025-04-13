"""Tests for expert agents."""
import pytest
import os
from me2ai.agents.factory import create_expert_agent
from me2ai.agents.individual import (
    GermanProfessor,
    DatingExpert,
    SEOExpert,
    Researcher
)

@pytest.mark.asyncio
async def test_german_professor_agent(german_professor, mock_llm):
    """Test German professor agent responses."""
    # Test basic response
    response = await german_professor.respond("Tell me about German universities")
    assert response == mock_llm.default_response
    assert len(mock_llm.calls) == 1
    
    # Check that system prompt was included
    messages = mock_llm.calls[0]["messages"]
    assert any("deutscher Professor" in m["content"] for m in messages)
    
    # Test German language handling
    response = await german_professor.respond("Was ist der Unterschied zwischen Universität und Hochschule?")
    assert response == mock_llm.default_response
    
    # Test academic guidance
    response = await german_professor.respond("How do I write a dissertation?")
    assert response == mock_llm.default_response

@pytest.mark.asyncio
async def test_dating_expert_agent(dating_expert, mock_llm):
    """Test dating expert agent responses."""
    # Test basic advice
    response = await dating_expert.respond("How do I start dating?")
    assert response == mock_llm.default_response
    assert len(mock_llm.calls) == 1
    
    # Check that system prompt was included
    messages = mock_llm.calls[0]["messages"]
    assert any("dating and relationship expert" in m["content"].lower() for m in messages)
    
    # Test relationship advice
    response = await dating_expert.respond("How do I know if they're the one?")
    assert response == mock_llm.default_response
    
    # Test online dating advice
    response = await dating_expert.respond("Tips for creating a dating profile?")
    assert response == mock_llm.default_response
    
    # Test conflict resolution
    response = await dating_expert.respond("How to handle relationship conflicts?")
    assert response == mock_llm.default_response

@pytest.mark.asyncio
async def test_create_expert_agent(mock_llm, mock_memory):
    """Test expert agent factory function."""
    # Test creating different types of agents
    german_prof = create_expert_agent("german_professor", mock_llm, mock_memory)
    assert isinstance(german_prof, GermanProfessor)
    assert german_prof.memory == mock_memory
    
    dating_expert = create_expert_agent("dating_expert", mock_llm, mock_memory)
    assert isinstance(dating_expert, DatingExpert)
    assert dating_expert.memory == mock_memory
    
    seo_expert = create_expert_agent("seo_expert", mock_llm, mock_memory)
    assert isinstance(seo_expert, SEOExpert)
    assert seo_expert.memory == mock_memory
    
    researcher = create_expert_agent("researcher", mock_llm, mock_memory)
    assert isinstance(researcher, Researcher)
    assert researcher.memory == mock_memory

@pytest.mark.asyncio
async def test_create_expert_agent_invalid_type(mock_llm, mock_memory):
    """Test error handling for invalid agent type."""
    with pytest.raises(ValueError) as exc_info:
        create_expert_agent("invalid_type", mock_llm, mock_memory)
    assert "Unknown agent type: invalid_type" in str(exc_info.value)

@pytest.mark.asyncio
async def test_agent_memory_usage(german_professor, mock_llm):
    """Test that agents properly use memory."""
    # Test conversation history
    await german_professor.respond("Hallo")
    await german_professor.respond("Wie geht's?")
    
    # Check that memory was used in the second call
    assert len(mock_llm.calls) == 2
    second_call = mock_llm.calls[1]["messages"]
    assert any("Hallo" in m["content"] for m in second_call)
    
    # Test memory clearing
    german_professor.memory.clear()
    await german_professor.respond("Was ist los?")
    
    third_call = mock_llm.calls[2]["messages"]
    assert not any("Hallo" in m["content"] for m in third_call)
    assert not any("Wie geht's" in m["content"] for m in third_call)

@pytest.mark.asyncio
async def test_real_agent_responses():
    """Test agents with real LLM responses."""
    # Skip if no API key
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OpenAI API key not found")
        
    from llms.openai_provider import OpenAIProvider
    llm = OpenAIProvider()
    agent = GermanProfessor(llm)
    
    # Test basic German response
    response = await agent.respond("Wie funktioniert das deutsche Universitätssystem?")
    assert isinstance(response, str)
    assert len(response) > 0
    assert any(word in response.lower() for word in ["universität", "studium", "bachelor", "master"])
