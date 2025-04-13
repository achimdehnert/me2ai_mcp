"""Integration tests for ME2AI MCP tools functionality."""

import pytest
from typing import Dict, Any
import os
from unittest.mock import patch, MagicMock

from me2ai_mcp.base import ME2AIMCPServer
from me2ai_mcp.auth import AuthManager


class TestToolsIntegration:
    """Test integration between different tool components."""
    
    @pytest.fixture
    def server(self):
        """Create a test server with multiple tools."""
        server = ME2AIMCPServer(server_name="integration_test")
        
        @server.register_tool
        def tool_a(param: str) -> Dict[str, Any]:
            """Test tool A."""
            return {"result": f"Tool A: {param}"}
            
        @server.register_tool
        def tool_b(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Test tool B that uses results from other tools."""
            # This tool might process results from tool_a
            if "tool_a_result" in input_data:
                return {"result": f"Tool B processed: {input_data['tool_a_result']}"}
            return {"result": "Tool B: direct execution"}
        
        return server
    
    def test_should_execute_tools_in_sequence(self, server):
        """Test running multiple tools in sequence with result passing."""
        # Run first tool
        tool_a_result = server.execute_tool("tool_a", {"param": "test_input"})
        assert "result" in tool_a_result, "Tool A should return results"
        
        # Use results in second tool
        tool_b_result = server.execute_tool("tool_b", {
            "input_data": {"tool_a_result": tool_a_result["result"]}
        })
        assert "result" in tool_b_result, "Tool B should return results"
        assert "Tool B processed" in tool_b_result["result"], "Tool B should process Tool A results"
    
    def test_should_integrate_with_auth_system(self):
        """Test integration between tools and authentication system."""
        server = ME2AIMCPServer()
        
        # Create a tool that requires authentication
        @server.register_tool
        def secure_tool(token: str = None) -> Dict[str, Any]:
            """Tool that requires authentication."""
            if not token or token != "valid_token":
                return {"error": "Authentication failed"}
            return {"result": "Secure operation completed"}
        
        # Mock the environment variable
        with patch.dict(os.environ, {"API_TOKEN": "valid_token"}):
            # Create auth manager
            auth = AuthManager()
            token = auth.get_token("API_TOKEN")
            
            # Execute tool with authentication
            result = server.execute_tool("secure_tool", {"token": token})
            assert "result" in result, "Tool should execute successfully with valid authentication"
            assert "error" not in result, "No error should be present when authentication succeeds"
            
        # Test with invalid token
        result = server.execute_tool("secure_tool", {"token": "invalid_token"})
        assert "error" in result, "Tool should fail with invalid authentication"


class TestCrossComponentIntegration:
    """Test integration between different ME2AI MCP components."""
    
    def test_should_integrate_server_with_tools_and_auth(self):
        """Test complete integration between server, tools, and authentication."""
        server = ME2AIMCPServer()
        
        # Register a tool with authentication requirement
        @server.register_tool
        def authenticated_tool(auth_token: str = None) -> Dict[str, Any]:
            """Tool requiring authentication."""
            if not AuthManager.validate_token(auth_token, "valid_token"):
                return {"error": "Authentication required"}
            return {"status": "success", "data": "Protected resource"}
        
        # Mock the validation function
        with patch('me2ai_mcp.auth.AuthManager.validate_token', 
                  return_value=True) as mock_validate:
            # Test authenticated access
            result = server.execute_tool("authenticated_tool", {"auth_token": "test_token"})
            assert "status" in result and result["status"] == "success", \
                "Tool should succeed with valid authentication"
            mock_validate.assert_called_once_with("test_token", "valid_token"), \
                "Token validation should be called"
