"""
Integration tests for the ME2AI MCP framework.
Tests multiple components working together.
"""
import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, Mock
import os
import tempfile
from pathlib import Path

from me2ai_mcp.base import ME2AIMCPServer, BaseTool, register_tool
from me2ai_mcp.auth import AuthManager, APIKeyAuth
from me2ai_mcp.tools.web import WebFetchTool, HTMLParserTool


class TestServerIntegration:
    """Integration tests for the ME2AI MCP server with tools."""

    def test_server_initialization_and_tool_registration(self):
        """Test server initialization with automatic tool registration."""
        # Create a server with a name and description
        server = ME2AIMCPServer(
            server_name="test-server",
            description="Test server for integration tests",
            version="1.0.0"
        )
        
        # Define some tools for testing
        @register_tool(server)
        class TestTool1(BaseTool):
            """Test tool 1."""
            
            name = "test_tool_1"
            description = "A test tool"
            
            async def execute(self, parameters):
                return {"success": True, "result": "Tool 1 executed"}
        
        @register_tool(server)
        class TestTool2(BaseTool):
            """Test tool 2."""
            
            name = "test_tool_2"
            description = "Another test tool"
            
            async def execute(self, parameters):
                return {"success": True, "result": "Tool 2 executed"}
        
        # Verify tools were registered
        assert "test_tool_1" in server.tools
        assert "test_tool_2" in server.tools
        
        # Verify server metadata
        assert server.server_name == "test-server"
        assert server.description == "Test server for integration tests"
        assert server.version == "1.0.0"
        assert server.started == False
        
        # Register a tool manually and verify
        server.register_tool(TestTool1)
        assert len([t for t in server.tools.values() if isinstance(t, TestTool1)]) == 2

    @pytest.mark.asyncio
    async def test_server_handle_request_with_tool_execution(self):
        """Test server handling a complete request including tool execution."""
        # Create a server
        server = ME2AIMCPServer(server_name="test-server")
        
        # Define a test tool
        @register_tool(server)
        class TestTool(BaseTool):
            """Test tool."""
            
            name = "test_tool"
            description = "A test tool"
            
            async def execute(self, parameters):
                return {"success": True, "result": parameters.get("input", "default")}
        
        # Create a mock request
        mock_request = Mock()
        mock_request.json = Mock(return_value={
            "tool": "test_tool",
            "parameters": {"input": "test-value"}
        })
        
        # Handle the request
        with patch.object(server, 'send_response') as mock_send_response:
            await server.handle_request(mock_request)
            
            # Verify response
            args, kwargs = mock_send_response.call_args
            response = args[0]
            assert response["tool"] == "test_tool"
            assert response["success"] is True
            assert response["result"] == "test-value"
            
            # Check that statistics were updated
            assert server.stats["requests_total"] == 1
            assert server.stats["requests_success"] == 1
            assert server.stats["requests_error"] == 0

    @pytest.mark.asyncio
    async def test_server_handle_request_with_unknown_tool(self):
        """Test server handling a request with an unknown tool."""
        # Create a server
        server = ME2AIMCPServer(server_name="test-server")
        
        # Create a mock request with unknown tool
        mock_request = Mock()
        mock_request.json = Mock(return_value={
            "tool": "unknown_tool",
            "parameters": {}
        })
        
        # Handle the request
        with patch.object(server, 'send_response') as mock_send_response:
            await server.handle_request(mock_request)
            
            # Verify error response
            args, kwargs = mock_send_response.call_args
            response = args[0]
            assert response["tool"] == "unknown_tool"
            assert response["success"] is False
            assert "unknown tool" in response["error"].lower()
            
            # Check that statistics were updated
            assert server.stats["requests_total"] == 1
            assert server.stats["requests_success"] == 0
            assert server.stats["requests_error"] == 1

    @pytest.mark.asyncio
    async def test_server_handle_request_with_authentication(self):
        """Test server handling a request with authentication."""
        # Create a server with authentication
        server = ME2AIMCPServer(server_name="test-server")
        
        # Add authentication provider
        auth_manager = AuthManager()
        auth_manager.add_provider(APIKeyAuth("test-api-key"))
        server.auth_manager = auth_manager
        
        # Define a test tool
        @register_tool(server)
        class TestTool(BaseTool):
            """Test tool."""
            
            name = "test_tool"
            description = "A test tool"
            
            async def execute(self, parameters):
                return {"success": True, "result": "authenticated"}
        
        # Create a mock request with valid authentication
        mock_request_valid = Mock()
        mock_request_valid.json = Mock(return_value={
            "tool": "test_tool",
            "parameters": {}
        })
        mock_request_valid.headers = {"Authorization": "Bearer test-api-key"}
        
        # Create a mock request with invalid authentication
        mock_request_invalid = Mock()
        mock_request_invalid.json = Mock(return_value={
            "tool": "test_tool",
            "parameters": {}
        })
        mock_request_invalid.headers = {"Authorization": "Bearer wrong-key"}
        
        # Handle the valid request
        with patch.object(server, 'send_response') as mock_send_response:
            await server.handle_request(mock_request_valid)
            
            # Verify successful response
            args, kwargs = mock_send_response.call_args
            response = args[0]
            assert response["tool"] == "test_tool"
            assert response["success"] is True
            
            # Check that statistics were updated
            assert server.stats["requests_total"] == 1
            assert server.stats["requests_success"] == 1
        
        # Handle the invalid request
        with patch.object(server, 'send_response') as mock_send_response:
            await server.handle_request(mock_request_invalid)
            
            # Verify authentication error
            args, kwargs = mock_send_response.call_args
            response = args[0]
            assert response["success"] is False
            assert "authentication" in response["error"].lower()
            
            # Check that statistics were updated
            assert server.stats["requests_total"] == 2
            assert server.stats["requests_success"] == 1
            assert server.stats["requests_error"] == 1

    @pytest.mark.asyncio
    async def test_server_handle_request_with_tool_error(self):
        """Test server handling a request where the tool raises an exception."""
        # Create a server
        server = ME2AIMCPServer(server_name="test-server")
        
        # Define a test tool that raises an exception
        @register_tool(server)
        class ErrorTool(BaseTool):
            """Tool that raises an error."""
            
            name = "error_tool"
            description = "A tool that raises an error"
            
            async def execute(self, parameters):
                raise ValueError("Test error")
        
        # Create a mock request
        mock_request = Mock()
        mock_request.json = Mock(return_value={
            "tool": "error_tool",
            "parameters": {}
        })
        
        # Handle the request
        with patch.object(server, 'send_response') as mock_send_response:
            await server.handle_request(mock_request)
            
            # Verify error response
            args, kwargs = mock_send_response.call_args
            response = args[0]
            assert response["tool"] == "error_tool"
            assert response["success"] is False
            assert "test error" in response["error"].lower()
            assert response["exception_type"] == "ValueError"
            
            # Check that statistics were updated
            assert server.stats["requests_total"] == 1
            assert server.stats["requests_success"] == 0
            assert server.stats["requests_error"] == 1


class TestToolIntegration:
    """Integration tests for tools working together."""

    @pytest.mark.asyncio
    async def test_web_tools_chain(self):
        """Test chaining web fetch and HTML parser tools."""
        fetch_tool = WebFetchTool()
        parser_tool = HTMLParserTool()
        
        # Mock HTML content
        html_content = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Test Heading</h1>
                <p>This is a test paragraph.</p>
                <div class="product">
                    <h2>Product 1</h2>
                    <span class="price">$10.99</span>
                </div>
                <div class="product">
                    <h2>Product 2</h2>
                    <span class="price">$20.99</span>
                </div>
            </body>
        </html>
        """
        
        # Mock fetch response
        mock_fetch_response = Mock()
        mock_fetch_response.text = html_content
        mock_fetch_response.headers = {"Content-Type": "text/html"}
        mock_fetch_response.raise_for_status = Mock()
        
        with patch('requests.get', return_value=mock_fetch_response):
            with patch('me2ai_mcp.tools.web.BS4_AVAILABLE', True):
                # First fetch the web content
                fetch_result = await fetch_tool.execute({
                    "url": "https://example.com"
                })
                
                # Verify fetch success
                assert fetch_result["success"] is True
                assert fetch_result["content_type"] == "text/html"
                
                # Take the HTML from the fetch and use it in the parser
                html = fetch_result["content"]
                
                # Configure BeautifulSoup mock for HTML parser
                with patch('me2ai_mcp.tools.web.BeautifulSoup') as mock_bs:
                    # Mock the BeautifulSoup instance
                    mock_soup = MagicMock()
                    mock_bs.return_value = mock_soup
                    
                    # Mock the soup get_text method
                    mock_soup.get_text.return_value = "Test Heading This is a test paragraph. Product 1 $10.99 Product 2 $20.99"
                    
                    # Mock the soup select method
                    product1 = MagicMock()
                    product1.get_text.return_value = "Product 1 $10.99"
                    product1.select_one.side_effect = lambda selector: MagicMock(get_text=lambda: "$10.99") if selector == ".price" else None
                    
                    product2 = MagicMock()
                    product2.get_text.return_value = "Product 2 $20.99"
                    product2.select_one.side_effect = lambda selector: MagicMock(get_text=lambda: "$20.99") if selector == ".price" else None
                    
                    mock_soup.select.side_effect = lambda selector: [product1, product2] if selector == ".product" else []
                    
                    # Then parse the HTML
                    parser_result = await parser_tool.execute({
                        "html": html,
                        "selectors": {
                            "products": {
                                "selector": ".product",
                                "multiple": True
                            },
                            "first_product_price": {
                                "selector": ".product:first-child .price"
                            }
                        },
                        "extract_text": True
                    })
                    
                    # Verify parser success
                    assert parser_result["success"] is True
                    assert "extracted" in parser_result
                    assert "products" in parser_result["extracted"]
                    assert len(parser_result["extracted"]["products"]) == 2
                    assert "text" in parser_result
                    assert "Test Heading" in parser_result["text"]

    @pytest.mark.asyncio
    async def test_tool_error_handling_and_recovery(self):
        """Test error handling and recovery in a chain of tool calls."""
        # Create a server for the tools
        server = ME2AIMCPServer(server_name="test-server")
        
        # Define tools with error handling
        @register_tool(server)
        class FirstTool(BaseTool):
            """First tool in the chain."""
            
            name = "first_tool"
            description = "First tool"
            
            async def execute(self, parameters):
                # Simulate success
                return {"success": True, "result": "first_data"}
        
        @register_tool(server)
        class SecondTool(BaseTool):
            """Second tool that may fail."""
            
            name = "second_tool"
            description = "Second tool"
            
            async def execute(self, parameters):
                # Check if we should simulate failure
                if parameters.get("fail", False):
                    raise ValueError("Simulated failure")
                
                # Get input from first tool
                first_data = parameters.get("input", "")
                return {"success": True, "result": f"{first_data}_processed"}
        
        @register_tool(server)
        class FallbackTool(BaseTool):
            """Fallback tool used when second tool fails."""
            
            name = "fallback_tool"
            description = "Fallback tool"
            
            async def execute(self, parameters):
                # Get original input
                first_data = parameters.get("input", "")
                return {"success": True, "result": f"{first_data}_fallback"}
        
        # Get tool instances
        first_tool = server.tools["first_tool"]
        second_tool = server.tools["second_tool"]
        fallback_tool = server.tools["fallback_tool"]
        
        # Execute successful chain
        first_result = await first_tool.execute({})
        assert first_result["success"] is True
        
        second_result_success = await second_tool.execute({
            "input": first_result["result"],
            "fail": False
        })
        assert second_result_success["success"] is True
        assert second_result_success["result"] == "first_data_processed"
        
        # Execute chain with failure and fallback
        first_result = await first_tool.execute({})
        assert first_result["success"] is True
        
        try:
            second_result_failure = await second_tool.execute({
                "input": first_result["result"],
                "fail": True
            })
            # Should not reach here due to exception
            assert False, "Expected exception not raised"
        except ValueError:
            # Handle the error and use fallback
            fallback_result = await fallback_tool.execute({
                "input": first_result["result"]
            })
            assert fallback_result["success"] is True
            assert fallback_result["result"] == "first_data_fallback"


class TestFileSystemIntegration:
    """Integration tests for filesystem operations."""

    def test_file_operations_with_temp_directory(self):
        """Test file operations using a temporary directory."""
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a test file
            test_file_path = temp_path / "test_file.txt"
            test_content = "This is test content."
            test_file_path.write_text(test_content)
            
            # Create a server to host the tools
            server = ME2AIMCPServer(server_name="filesystem-test-server")
            
            # Define file reader tool
            @register_tool(server)
            class FileReaderTool(BaseTool):
                """Tool to read file contents."""
                
                name = "file_reader"
                description = "Reads file contents"
                
                async def execute(self, parameters):
                    filepath = parameters.get("path", "")
                    try:
                        with open(filepath, 'r') as file:
                            content = file.read()
                        return {"success": True, "content": content}
                    except Exception as e:
                        return {"success": False, "error": str(e)}
            
            # Define file writer tool
            @register_tool(server)
            class FileWriterTool(BaseTool):
                """Tool to write file contents."""
                
                name = "file_writer"
                description = "Writes file contents"
                
                async def execute(self, parameters):
                    filepath = parameters.get("path", "")
                    content = parameters.get("content", "")
                    try:
                        with open(filepath, 'w') as file:
                            file.write(content)
                        return {"success": True, "path": filepath}
                    except Exception as e:
                        return {"success": False, "error": str(e)}
            
            # Get tool instances
            reader_tool = server.tools["file_reader"]
            writer_tool = server.tools["file_writer"]
            
            # Test reading the file
            read_result = asyncio.run(reader_tool.execute({
                "path": str(test_file_path)
            }))
            assert read_result["success"] is True
            assert read_result["content"] == test_content
            
            # Test writing to a new file
            new_file_path = temp_path / "new_file.txt"
            new_content = "This is new content."
            
            write_result = asyncio.run(writer_tool.execute({
                "path": str(new_file_path),
                "content": new_content
            }))
            assert write_result["success"] is True
            
            # Verify the content was written correctly
            assert new_file_path.exists()
            assert new_file_path.read_text() == new_content
            
            # Test updating an existing file
            updated_content = "This is updated content."
            
            update_result = asyncio.run(writer_tool.execute({
                "path": str(test_file_path),
                "content": updated_content
            }))
            assert update_result["success"] is True
            
            # Verify the content was updated
            assert test_file_path.read_text() == updated_content
