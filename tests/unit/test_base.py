"""Unit tests for ME2AI MCP base functionality."""

import pytest
from typing import Dict, Any
from unittest.mock import MagicMock, patch

from me2ai_mcp.base import ME2AIMCPServer


@pytest.fixture
def fixture_server():
    """Create a test server instance."""
    return ME2AIMCPServer(
        server_name="test_server",
        description="Test Server",
        version="0.0.5"
    )


def test_should_initialize_server_with_defaults():
    """Test server initialization with default values."""
    server = ME2AIMCPServer()
    
    assert server.server_name == "default", "Default server name should be 'default'"
    assert server.description == "ME2AI MCP Server", "Default description should be set"
    assert server.version == "0.0.5", "Version should be set to package version"


def test_should_initialize_server_with_custom_values():
    """Test server initialization with custom values."""
    server = ME2AIMCPServer(
        server_name="custom_server",
        description="Custom Description",
        version="1.0.0"
    )
    
    assert server.server_name == "custom_server", "Server name should be customizable"
    assert server.description == "Custom Description", "Description should be customizable"
    assert server.version == "1.0.0", "Version should be customizable"


def test_should_register_tool_successfully():
    """Test tool registration functionality."""
    server = ME2AIMCPServer()
    
    @server.register_tool
    def test_tool(param1: str, param2: int = 0) -> Dict[str, Any]:
        """Test tool function."""
        return {"param1": param1, "param2": param2}
    
    assert "test_tool" in server.tools, "Tool should be registered"
    assert server.tools["test_tool"].func == test_tool, "Tool function should be accessible"
    assert server.tools["test_tool"].name == "test_tool", "Tool name should be set"


def test_should_handle_tool_execution_errors():
    """Test error handling during tool execution."""
    server = ME2AIMCPServer()
    
    @server.register_tool
    def failing_tool() -> Dict[str, Any]:
        """Tool that raises an exception."""
        raise ValueError("Test error")
    
    with patch("me2ai_mcp.base.logging") as mock_logging:
        result = server.execute_tool("failing_tool", {})
        
        assert "error" in result, "Error key should be present in result"
        assert "Test error" in result["error"], "Error message should be included"
        mock_logging.error.assert_called_once(), "Error should be logged"


def test_should_return_tool_list():
    """Test getting a list of available tools."""
    server = ME2AIMCPServer()
    
    @server.register_tool
    def tool1() -> Dict[str, Any]:
        """First test tool."""
        return {}
        
    @server.register_tool
    def tool2() -> Dict[str, Any]:
        """Second test tool."""
        return {}
    
    tools = server.get_tools()
    
    assert len(tools) == 2, "Should return all registered tools"
    assert "tool1" in [t["name"] for t in tools], "First tool should be included"
    assert "tool2" in [t["name"] for t in tools], "Second tool should be included"
