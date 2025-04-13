"""
Tests for the ME2AI MCP base classes.
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock
import json
from pathlib import Path

from me2ai_mcp.base import ME2AIMCPServer, BaseTool, register_tool


class TestME2AIMCPServer:
    """Tests for the ME2AIMCPServer class."""

    def test_init(self):
        """Test server initialization."""
        server = ME2AIMCPServer(
            server_name="test-server",
            description="Test server",
            version="0.1.0"
        )
        
        assert server.server_name == "test-server"
        assert server.description == "Test server"
        assert server.version == "0.1.0"
        assert "server_info" in server.tools
        
    def test_stats_tracking(self):
        """Test server stats tracking."""
        server = ME2AIMCPServer("test-server")
        
        assert server.stats["requests"] == 0
        assert server.stats["errors"] == 0
        assert server.stats["tool_calls"] == {}
        
    @pytest.mark.asyncio
    async def test_server_info_tool(self):
        """Test the built-in server_info tool."""
        server = ME2AIMCPServer(
            server_name="test-server",
            description="Test server",
            version="0.1.0"
        )
        
        # Get the server_info tool function
        server_info_tool = server.tools["server_info"]
        
        # Call the tool
        result = await server_info_tool()
        
        assert result["success"] is True
        assert result["server"]["server_name"] == "test-server"
        assert result["server"]["description"] == "Test server"
        assert result["server"]["version"] == "0.1.0"
        assert "server_info" in result["tools"]
        
        # Check stats update
        assert server.stats["requests"] == 1
        assert server.stats["tool_calls"]["server_info"] == 1
        
    @pytest.mark.asyncio
    async def test_from_config(self):
        """Test creating a server from a configuration file."""
        # Create a temp config file
        config = {
            "server_name": "config-server",
            "description": "Config-loaded server",
            "version": "0.2.0",
            "debug": True
        }
        
        with patch("builtins.open", MagicMock()):
            with patch("json.load", MagicMock(return_value=config)):
                with patch("pathlib.Path.exists", MagicMock(return_value=True)):
                    server = ME2AIMCPServer.from_config("test_config.json")
                    
                    assert server.server_name == "config-server"
                    assert server.description == "Config-loaded server"
                    assert server.version == "0.2.0"


class TestRegisterTool:
    """Tests for the register_tool decorator."""
    
    @pytest.mark.asyncio
    async def test_register_tool_success(self):
        """Test successful tool execution with register_tool decorator."""
        server = ME2AIMCPServer("test-server")
        
        # Add a test tool
        @register_tool
        async def test_tool(self, param: str):
            return {"result": f"Processed {param}"}
        
        # Bind the tool to the server
        server.tools["test_tool"] = test_tool.__get__(server, server.__class__)
        
        # Call the tool
        result = await server.tools["test_tool"]("test_param")
        
        assert result["success"] is True
        assert result["result"] == "Processed test_param"
        
        # Check stats update
        assert server.stats["requests"] == 1
        assert server.stats["tool_calls"]["test_tool"] == 1
        
    @pytest.mark.asyncio
    async def test_register_tool_error_handling(self):
        """Test error handling in register_tool decorator."""
        server = ME2AIMCPServer("test-server")
        
        # Add a test tool that raises an exception
        @register_tool
        async def error_tool(self, param: str):
            raise ValueError("Test error")
        
        # Bind the tool to the server
        server.tools["error_tool"] = error_tool.__get__(server, server.__class__)
        
        # Call the tool
        result = await server.tools["error_tool"]("test_param")
        
        assert result["success"] is False
        assert "error" in result
        assert "Test error" in result["error"]
        assert result["exception_type"] == "ValueError"
        
        # Check stats update
        assert server.stats["requests"] == 1
        assert server.stats["errors"] == 1
        assert server.stats["tool_calls"]["error_tool"] == 1


class TestBaseTool:
    """Tests for the BaseTool class."""
    
    def test_init(self):
        """Test BaseTool initialization."""
        tool = BaseTool(
            name="test-tool",
            description="Test tool"
        )
        
        assert tool.name == "test-tool"
        assert tool.description == "Test tool"
        assert tool.enabled is True
        assert tool.stats["calls"] == 0
        assert tool.stats["errors"] == 0
        assert tool.stats["last_call"] is None
        
    @pytest.mark.asyncio
    async def test_execute_not_implemented(self):
        """Test that BaseTool.execute raises NotImplementedError."""
        tool = BaseTool(name="test-tool")
        
        with pytest.raises(NotImplementedError):
            await tool.execute({})
