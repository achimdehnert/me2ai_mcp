"""Test basic imports."""

def test_imports():
    """Test that basic imports work."""
    from agents.base import BaseAgent
    assert BaseAgent is not None
