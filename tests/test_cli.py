"""Tests for the CLI interface."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio
from me2ai.cli import AgentCLI
from me2ai.agents.base import BaseAgent
from me2ai.memory import ConversationMemory

@pytest.fixture
def mock_cli(mocker):
    """Create a mock CLI instance."""
    # Mock LLM provider
    mock_llm = Mock()
    mock_llm.respond = AsyncMock(return_value="Mocked response")
    mocker.patch('me2ai.llms.openai_provider.OpenAIProvider', return_value=mock_llm)
    
    # Mock router agent
    mock_router = Mock(spec=BaseAgent)
    mock_router.role = "router"
    mock_router.respond = AsyncMock(return_value="Routing response")
    mock_router.get_agent = AsyncMock(return_value=(mock_router, "test reason"))
    mock_router.memory = ConversationMemory()
    
    # Mock expert agent
    mock_expert = Mock(spec=BaseAgent)
    mock_expert.role = "german_professor"
    mock_expert.respond = AsyncMock(return_value="Expert response")
    mock_expert.memory = ConversationMemory()
    
    # Mock other agents
    mock_agent = Mock(spec=BaseAgent)
    mock_agent.role = "Mock Agent"
    mock_agent.respond = AsyncMock(return_value="Mocked response")
    mock_agent.memory = ConversationMemory()
    
    # Create CLI instance
    cli = AgentCLI()
    cli.agents = {
        "router": mock_router,
        "german_professor": mock_expert,
        "dating_expert": mock_agent,
        "seo_expert": mock_agent,
        "relationship_team": mock_agent,
        "language_team": mock_agent,
        "business_team": mock_agent
    }
    cli.current_agent = "router"
    
    return cli

def test_cli_initialization(mock_cli):
    """Test CLI initialization."""
    assert mock_cli.current_agent == 'router'
    assert len(mock_cli.agents) == 7  # router, german_professor, dating_expert, seo_expert, relationship_team, language_team, business_team
    assert all(isinstance(agent, Mock) for agent in mock_cli.agents.values())

def test_cli_talk_command(mock_cli, capsys):
    """Test the talk command."""
    # Test basic message
    mock_cli.do_talk("Hello")
    captured = capsys.readouterr()
    assert "Processing" in captured.out
    assert "router: Routing response" in captured.out
    
    # Test empty message
    mock_cli.do_talk("")
    captured = capsys.readouterr()
    assert "Please provide a message" in captured.out

def test_cli_switch_command(mock_cli, capsys):
    """Test the switch command."""
    # Test valid switch
    mock_cli.do_switch("german_professor")
    assert mock_cli.current_agent == "german_professor"
    captured = capsys.readouterr()
    assert "Switched to german_professor" in captured.out
    
    # Test invalid agent
    mock_cli.do_switch("invalid_agent")
    assert mock_cli.current_agent == "german_professor"  # Should not change
    captured = capsys.readouterr()
    assert "Unknown agent" in captured.out
    
    # Test empty input
    mock_cli.do_switch("")
    captured = capsys.readouterr()
    assert "Please specify an agent name" in captured.out

def test_cli_list_command(mock_cli, capsys):
    """Test the list command."""
    mock_cli.do_list("")
    captured = capsys.readouterr()
    assert "Available agents" in captured.out
    assert "router" in captured.out
    assert "german_professor" in captured.out

def test_cli_clear_command(mock_cli, capsys):
    """Test the clear command."""
    # Add some messages first
    mock_cli.do_talk("Hello")
    
    # Clear and verify
    mock_cli.do_clear("")
    captured = capsys.readouterr()
    assert "Conversation history cleared" in captured.out

def test_cli_quit_command(mock_cli, capsys):
    """Test the quit command."""
    result = mock_cli.do_quit("")
    assert result is True  # Should return True to exit
    captured = capsys.readouterr()
    assert "Goodbye" in captured.out

def test_cli_eof_command(mock_cli, capsys):
    """Test EOF (Ctrl+D) command."""
    result = mock_cli.do_EOF("")
    assert result is True  # Should return True to exit
    captured = capsys.readouterr()
    assert "Goodbye" in captured.out

def test_cli_auto_command(mock_cli, capsys):
    """Test auto-routing command."""
    # Test basic message
    mock_cli.do_auto("How do I learn German grammar?")
    captured = capsys.readouterr()
    assert "Routing your question" in captured.out
    assert "Selected expert" in captured.out
    assert "router: Routing response" in captured.out
    
    # Test empty message
    mock_cli.do_auto("")
    captured = capsys.readouterr()
    assert "Please provide a message" in captured.out

def test_cli_conversation_flow(mock_cli, capsys):
    """Test a full conversation flow."""
    # Start with router
    assert mock_cli.current_agent == 'router'
    
    # Switch to german professor
    mock_cli.do_switch("german_professor")
    assert mock_cli.current_agent == "german_professor"
    
    # Send a message
    mock_cli.do_talk("How do I learn German grammar?")
    captured = capsys.readouterr()
    assert "Processing" in captured.out
    
    # Switch to dating expert
    mock_cli.do_switch("dating_expert")
    assert mock_cli.current_agent == "dating_expert"
    
    # Send another message
    mock_cli.do_talk("What are good first date ideas?")
    captured = capsys.readouterr()
    assert "Processing" in captured.out
    
    # Clear history
    mock_cli.do_clear("")
    captured = capsys.readouterr()
    assert "Conversation history cleared" in captured.out

def test_cli_real_responses(mock_cli):
    """Test CLI with real API responses."""
    # This test uses real API responses
    pass

def test_cli_talk_command_async(mock_cli, capsys):
    """Test the talk command with async operations."""
    # Test basic message
    mock_cli.do_talk("Hello")
    captured = capsys.readouterr()
    assert "Processing" in captured.out
    assert "router: Routing response" in captured.out
    
    # Test empty message
    mock_cli.do_talk("")
    captured = capsys.readouterr()
    assert "Please provide a message" in captured.out

def test_cli_switch_command_async(mock_cli, capsys):
    """Test the switch command with async operations."""
    # Test valid switch
    mock_cli.do_switch("german_professor")
    assert mock_cli.current_agent == "german_professor"
    captured = capsys.readouterr()
    assert "Switched to german_professor" in captured.out
    
    # Test invalid agent
    mock_cli.do_switch("invalid_agent")
    assert mock_cli.current_agent == "german_professor"  # Should not change
    captured = capsys.readouterr()
    assert "Unknown agent" in captured.out
    
    # Test empty input
    mock_cli.do_switch("")
    captured = capsys.readouterr()
    assert "Please specify an agent name" in captured.out

def test_cli_list_command_async(mock_cli, capsys):
    """Test the list command with async operations."""
    mock_cli.do_list("")
    captured = capsys.readouterr()
    assert "Available agents" in captured.out
    assert "router" in captured.out
    assert "german_professor" in captured.out

def test_cli_clear_command_async(mock_cli, capsys):
    """Test the clear command with async operations."""
    # Add some messages first
    mock_cli.do_talk("Hello")
    
    # Clear and verify
    mock_cli.do_clear("")
    captured = capsys.readouterr()
    assert "Conversation history cleared" in captured.out

def test_cli_quit_command_async(mock_cli, capsys):
    """Test the quit command with async operations."""
    result = mock_cli.do_quit("")
    assert result is True  # Should return True to exit
    captured = capsys.readouterr()
    assert "Goodbye" in captured.out

def test_cli_eof_command_async(mock_cli, capsys):
    """Test EOF (Ctrl+D) command with async operations."""
    result = mock_cli.do_EOF("")
    assert result is True  # Should return True to exit
    captured = capsys.readouterr()
    assert "Goodbye" in captured.out

def test_cli_auto_command_async(mock_cli, capsys):
    """Test auto-routing command with async operations."""
    # Test basic message
    mock_cli.do_auto("How do I learn German grammar?")
    captured = capsys.readouterr()
    assert "Routing your question" in captured.out
    assert "Selected expert" in captured.out
    assert "router: Routing response" in captured.out
    
    # Test empty message
    mock_cli.do_auto("")
    captured = capsys.readouterr()
    assert "Please provide a message" in captured.out

def test_cli_conversation_flow_async(mock_cli, capsys):
    """Test a full conversation flow with async operations."""
    # Start with router
    assert mock_cli.current_agent == 'router'
    
    # Switch to german professor
    mock_cli.do_switch("german_professor")
    assert mock_cli.current_agent == "german_professor"
    
    # Send a message
    mock_cli.do_talk("How do I learn German grammar?")
    captured = capsys.readouterr()
    assert "Processing" in captured.out
    
    # Switch to dating expert
    mock_cli.do_switch("dating_expert")
    assert mock_cli.current_agent == "dating_expert"
    
    # Send another message
    mock_cli.do_talk("What are good first date ideas?")
    captured = capsys.readouterr()
    assert "Processing" in captured.out
    
    # Clear history
    mock_cli.do_clear("")
    captured = capsys.readouterr()
    assert "Conversation history cleared" in captured.out

@pytest.mark.integration
def test_cli_real_responses_async(mock_cli):
    """Test CLI with real API responses and async operations."""
    # Only run if API keys are configured
    if not mock_cli.llm_provider:
        pytest.skip("No LLM provider configured")
    
    # Test auto-routing
    mock_cli.do_auto("How do I learn German grammar?")
    assert mock_cli.agents["german_professor"] is not None
    
    # Test direct expert
    mock_cli.do_switch("dating_expert")
    mock_cli.do_talk("What are good first date ideas?")
    assert mock_cli.agents["dating_expert"] is not None
