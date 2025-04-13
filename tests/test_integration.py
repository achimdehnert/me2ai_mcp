"""Integration tests for the entire system."""
import os
import pytest
from me2ai.cli import AgentCLI
from me2ai.agents.expert_agents import create_expert_agent
from me2ai.llms.openai_provider import OpenAIProvider
from me2ai.llms.groq_provider import GroqProvider
from me2ai.llms.anthropic_provider import AnthropicProvider
from me2ai.memory import ConversationMemory

@pytest.mark.integration
def test_full_system_integration():
    """Test the entire system working together."""
    # Initialize CLI with real providers
    cli = AgentCLI()
    
    # Test each agent type with a relevant query
    test_cases = [
        ("moderator", "I need help with my career"),
        ("life_coach", "How can I improve my work-life balance?"),
        ("german_professor", "Erklären Sie mir das deutsche Universitätssystem"),
        ("dating_expert", "What makes a good first date?"),
        ("seo_expert", "How do I optimize my website for mobile devices?")
    ]
    
    for agent_type, query in test_cases:
        # Switch to agent
        cli.do_switch(agent_type)
        assert cli.current_agent == agent_type
        
        # Get response
        agent = cli.agents[agent_type]
        response = agent.respond(query)
        
        # Verify response
        assert isinstance(response, str)
        assert len(response) > 100  # Should be a substantial response
        assert response != "Error"  # Should not have errors
        
        # Verify memory is working
        memory_vars = agent.memory.load_memory_variables({})
        assert len(memory_vars["chat_history"]) > 0
        
        # Clear memory for next test
        agent.memory.clear()

@pytest.mark.integration
def test_llm_provider_switching():
    """Test switching between different LLM providers."""
    memory = ConversationMemory()
    providers = []
    
    # Create available providers
    if os.getenv("OPENAI_API_KEY"):
        providers.append(OpenAIProvider())
    if os.getenv("GROQ_API_KEY"):
        providers.append(GroqProvider())
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append(AnthropicProvider())
    
    assert len(providers) > 0, "No LLM providers available"
    
    # Test each provider with each agent type
    agent_types = ['german_professor', 'dating_expert', 'seo_expert']
    
    for provider in providers:
        for agent_type in agent_types:
            agent = create_expert_agent(agent_type, provider, memory)
            
            # Test response generation
            response = agent.respond("Tell me about your expertise")
            assert isinstance(response, str)
            assert len(response) > 0
            
            # Clear memory for next test
            memory.clear()

@pytest.mark.integration
def test_memory_persistence():
    """Test that memory persists across conversations."""
    cli = AgentCLI()
    agent = cli.agents['german_professor']
    
    # Send a series of related messages
    messages = [
        "Guten Tag, ich möchte über deutsche Universitäten sprechen",
        "Welche sind die besten?",
        "Und wie ist das Bewerbungsverfahren?"
    ]
    
    responses = []
    for msg in messages:
        response = agent.respond(msg)
        responses.append(response)
    
    # Verify memory contains the conversation
    memory_vars = agent.memory.load_memory_variables({})
    chat_history = memory_vars["chat_history"]
    
    # Check that all messages and responses are in the history
    history_text = str(chat_history)
    assert all(msg in history_text for msg in messages)
    assert all(resp in history_text for resp in responses)
    
    # Verify summaries are being created
    assert len(memory_vars["summaries"]) > 0

@pytest.mark.integration
def test_error_recovery():
    """Test system's ability to handle and recover from errors."""
    cli = AgentCLI()
    
    # Test with invalid API keys
    with pytest.raises(Exception):
        provider = OpenAIProvider()
        provider.client.api_key = "invalid_key"
        agent = create_expert_agent('seo_expert', provider, ConversationMemory())
        response = agent.respond("Test message")
    
    # System should continue working with valid provider
    cli.do_switch('seo_expert')
    response = cli.agents['seo_expert'].respond("How do I improve my SEO?")
    assert isinstance(response, str)
    assert len(response) > 0
