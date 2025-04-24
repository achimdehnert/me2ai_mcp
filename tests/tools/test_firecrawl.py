"""
Tests for FireCrawl integration in ME2AI MCP.

This module contains tests for the FireCrawl web scraping tools
that provide advanced browser-based content extraction capabilities.
"""

import os
import sys
import pytest
import unittest.mock as mock
from typing import Dict, Any

import me2ai_mcp.tools
from me2ai_mcp.tools.firecrawl import FireCrawlTool, WebContentTool, create_firecrawl_tool


class TestFireCrawlTool:
    """Tests for the FireCrawlTool class."""
    
    def test_should_initialize_firecrawl_tool(self):
        """Test that the FireCrawlTool initializes with default parameters."""
        tool = FireCrawlTool(auto_start_server=False)
        assert tool.name == "web_content"
        assert tool.description is not None
        assert tool.server_host == "localhost"
        assert tool.server_port == 8787
        assert tool.server_url == f"http://{tool.server_host}:{tool.server_port}"
        
    @mock.patch("me2ai_mcp.tools.firecrawl.subprocess.Popen")
    @mock.patch("me2ai_mcp.tools.firecrawl.requests.get")
    def test_should_check_server_running(self, mock_requests_get, mock_popen):
        """Test server health check logic."""
        # Mock successful health check response
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_requests_get.return_value = mock_response
        
        # Initialize tool
        tool = FireCrawlTool(auto_start_server=False)
        
        # Call _ensure_server_running
        result = tool._ensure_server_running()
        
        # Verify results
        assert result is True
        mock_requests_get.assert_called_once_with(
            f"{tool.server_url}/health", 
            timeout=2
        )
        mock_popen.assert_not_called()
    
    @mock.patch("me2ai_mcp.tools.firecrawl.subprocess.Popen")
    @mock.patch("me2ai_mcp.tools.firecrawl.requests.get")
    @mock.patch("me2ai_mcp.tools.firecrawl.os.path.exists")
    def test_should_start_server_when_not_running(self, mock_exists, mock_requests_get, mock_popen):
        """Test server startup when not already running."""
        # Mock failed health check response to trigger server start
        mock_requests_get.side_effect = Exception("Connection refused")
        
        # Mock server script existence
        mock_exists.return_value = True
        
        # Initialize tool with mocked firecrawl_path
        tool = FireCrawlTool(auto_start_server=False, firecrawl_path="/mock/path")
        
        # Mock successful post-startup health check
        def second_get(*args, **kwargs):
            response = mock.MagicMock()
            response.status_code = 200
            return response
            
        # Setup mocks for server start sequence
        mock_popen.return_value = mock.MagicMock()
        with mock.patch("me2ai_mcp.tools.firecrawl.time.sleep"):
            with mock.patch("me2ai_mcp.tools.firecrawl.requests.get", side_effect=[
                Exception("Connection refused"),  # First call fails
                second_get()  # Second call succeeds
            ]):
                # Call _ensure_server_running
                tool._ensure_server_running()
        
        # Verify server process was started
        mock_popen.assert_called_once()
    
    @pytest.mark.asyncio
    @mock.patch("me2ai_mcp.tools.firecrawl.requests.post")
    async def test_should_execute_firecrawl_request(self, mock_requests_post):
        """Test execute method with successful scraping."""
        # Mock successful response
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "url": "https://example.com",
            "title": "Example Domain",
            "html": "<html><body><h1>Example Domain</h1></body></html>",
            "text": "Example Domain",
            "links": ["https://example.org"]
        }
        mock_requests_post.return_value = mock_response
        
        # Create tool with server auto-start disabled
        tool = FireCrawlTool(auto_start_server=False)
        
        # Mock server check to avoid actual server start
        with mock.patch.object(tool, "_ensure_server_running", return_value=True):
            # Execute the tool
            result = await tool.execute({"url": "https://example.com"})
        
        # Verify request was made with correct parameters
        mock_requests_post.assert_called_once()
        assert mock_requests_post.call_args[0][0] == f"{tool.server_url}/scrape"
        
        # Verify the result contains expected keys
        assert result["success"] is True
        assert "url" in result
        assert "title" in result
        assert "html" in result
        assert "text" in result
        assert "links" in result
    
    @pytest.mark.asyncio
    @mock.patch("me2ai_mcp.tools.firecrawl.requests.post")
    async def test_should_handle_error_response(self, mock_requests_post):
        """Test error handling for failed requests."""
        # Mock error response
        mock_response = mock.MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {
            "error": "Internal server error"
        }
        mock_requests_post.return_value = mock_response
        
        # Create tool with server auto-start disabled
        tool = FireCrawlTool(auto_start_server=False)
        
        # Mock server check to avoid actual server start
        with mock.patch.object(tool, "_ensure_server_running", return_value=True):
            # Execute the tool
            result = await tool.execute({"url": "https://example.com"})
        
        # Verify error is properly returned
        assert result["success"] is False
        assert "error" in result
        assert result["status_code"] == 500
    
    def test_should_create_firecrawl_tool_with_factory(self):
        """Test factory function for creating FireCrawl tools."""
        # Use the factory function
        tool = create_firecrawl_tool(server_port=8888, auto_start=False)
        
        # Verify the tool is properly configured
        assert isinstance(tool, FireCrawlTool)
        assert tool.server_port == 8888
        assert tool.auto_start_server is False
    
    def test_should_use_web_content_tool_as_alias(self):
        """Test that WebContentTool is an alias for FireCrawlTool."""
        # Create both tool variants
        firecrawl_tool = FireCrawlTool(auto_start_server=False)
        web_content_tool = WebContentTool(auto_start_server=False)
        
        # Verify they have the expected class relationship
        assert isinstance(web_content_tool, FireCrawlTool)
        assert web_content_tool.name == "extract_web_content"
        assert firecrawl_tool.name == "web_content"


class TestFireCrawlLangChainIntegration:
    """Tests for FireCrawl integration with LangChain."""
    
    @mock.patch("me2ai_mcp.tools.firecrawl.FireCrawlTool")
    def test_should_create_langchain_tools(self, mock_firecrawl_tool):
        """Test that LangChainToolFactory creates FireCrawl tools."""
        # Import here to avoid module-level import errors if LangChain is not installed
        from me2ai_mcp.integrations.langchain import LangChainToolFactory
        
        # Create tools
        tools = LangChainToolFactory.create_firecrawl_tools()
        
        # Verify tools were created
        assert len(tools) > 0
        
    @mock.patch("me2ai_mcp.tools.firecrawl.FireCrawlTool")
    def test_should_include_firecrawl_in_web_tools(self, mock_firecrawl_tool):
        """Test that web tools include FireCrawl tools."""
        # Import here to avoid module-level import errors if LangChain is not installed
        from me2ai_mcp.integrations.langchain import LangChainToolFactory
        
        # Create web tools (should include FireCrawl)
        with mock.patch("me2ai_mcp.integrations.langchain.LangChainToolFactory.create_firecrawl_tools") as mock_create_firecrawl:
            # Setup mock to return a list with one tool
            mock_tool = mock.MagicMock()
            mock_create_firecrawl.return_value = [mock_tool]
            
            # Call the method under test
            tools = LangChainToolFactory.create_web_tools()
            
            # Verify FireCrawl tools were included
            mock_create_firecrawl.assert_called_once()
            assert mock_tool in tools
