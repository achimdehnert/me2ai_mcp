"""Tests for memory management."""
import pytest
from datetime import datetime, timedelta
from me2ai.memory import ConversationMemory, ConversationSummary

def test_memory_initialization():
    """Test memory initialization."""
    memory = ConversationMemory()
    assert memory.chat_history is not None
    assert memory.max_tokens == 4000
    assert memory.summaries == []
    assert memory.current_topic == ""

def test_memory_save_and_load():
    """Test saving and loading memory variables."""
    memory = ConversationMemory()
    
    # Save some context
    memory.save_context(
        {"input": "Hello"},
        {"output": "Hi there!"}
    )
    
    # Load and verify
    vars = memory.load_memory_variables({})
    messages = vars["chat_history"]
    assert len(messages) == 2  # Input and output
    assert "Hello" in str(messages)
    assert "Hi there!" in str(messages)

def test_memory_clear():
    """Test clearing memory."""
    memory = ConversationMemory()
    
    # Add some data
    memory.save_context(
        {"input": "Hello"},
        {"output": "Hi"}
    )
    memory.summaries.append(ConversationSummary(
        start_time=datetime.now(),
        end_time=datetime.now(),
        topic="Greeting",
        key_points=["Said hello"],
        sentiment="positive",
        action_items=[]
    ))
    
    # Clear and verify
    memory.clear()
    assert len(memory.chat_history.messages) == 0
    assert len(memory.summaries) == 0
    assert memory.current_topic == ""

def test_memory_topic_tracking():
    """Test topic tracking in memory."""
    memory = ConversationMemory()
    
    # Add messages about different topics
    topics = [
        "Tell me about careers",
        "How about relationships?",
        "Let's discuss hobbies"
    ]
    
    for topic in topics:
        memory.save_context(
            {"input": topic},
            {"output": f"Here's info about {topic}"}
        )
        assert memory.current_topic.startswith(topic.split()[0])

def test_memory_summarization():
    """Test conversation summarization."""
    memory = ConversationMemory()
    
    # Add enough messages to trigger summarization
    for i in range(12):
        memory.save_context(
            {"input": f"Message {i}"},
            {"output": f"Response {i}"}
        )
    
    # Check that summaries were created
    assert len(memory.summaries) > 0
    assert len(memory.chat_history.messages) < 24  # Should have been pruned

def test_memory_token_limit():
    """Test memory token limit handling."""
    memory = ConversationMemory(max_tokens=5)
    
    # Add more messages than the token limit
    for i in range(10):
        memory.save_context(
            {"input": f"Message {i}"},
            {"output": f"Response {i}"}
        )
    
    # Load and verify we only get recent messages
    vars = memory.load_memory_variables({})
    messages = vars["chat_history"]
    assert len(messages) <= 10  # Should be limited
    
    # Check that we have summaries
    assert len(vars["summaries"]) > 0

def test_conversation_summary():
    """Test conversation summary model."""
    now = datetime.now()
    summary = ConversationSummary(
        start_time=now,
        end_time=now + timedelta(minutes=30),
        topic="Test Conversation",
        key_points=["Point 1", "Point 2"],
        sentiment="positive",
        action_items=["Action 1", "Action 2"]
    )
    
    assert summary.topic == "Test Conversation"
    assert len(summary.key_points) == 2
    assert summary.sentiment == "positive"
    assert len(summary.action_items) == 2
    assert summary.end_time > summary.start_time
