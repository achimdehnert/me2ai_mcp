"""
Advanced tests for the ME2AI MCP web tools focusing on edge cases and error handling.
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock, Mock
import json
import requests
from io import BytesIO

from me2ai_mcp.tools.web import WebFetchTool, HTMLParserTool, URLUtilsTool


class TestWebFetchToolAdvanced:
    """Advanced tests for the WebFetchTool class."""

    @pytest.mark.asyncio
    async def test_web_fetch_large_content(self):
        """Test web fetch with content larger than limit."""
        # Create a tool with a small max content length
        tool = WebFetchTool(max_content_length=100)
        
        # Mock the response with content larger than the limit
        mock_response = Mock()
        mock_response.headers = {
            "Content-Type": "text/html", 
            "Content-Length": "1000"  # Larger than our limit
        }
        mock_response.raise_for_status = Mock()
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            result = await tool.execute({"url": "https://example.com"})
            
            # Should fail due to content size
            assert result["success"] is False
            assert "Content too large" in result["error"]

    @pytest.mark.asyncio
    async def test_web_fetch_streaming_response(self):
        """Test web fetch with a streaming response."""
        tool = WebFetchTool()
        
        # Create a mock response that simulates streaming
        mock_response = Mock()
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.raise_for_status = Mock()
        
        # No Content-Length header (common in chunked/streaming responses)
        # Set text property to be a large string when accessed
        mock_response.text = "X" * 10000
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            result = await tool.execute({"url": "https://example.com"})
            
            # Should succeed and process the content
            assert result["success"] is True
            assert result["content_type"] == "text/html"
            assert result["content_length"] == 10000

    @pytest.mark.asyncio
    async def test_web_fetch_timeout(self):
        """Test web fetch that times out."""
        tool = WebFetchTool(timeout=1)  # Short timeout
        
        # Mock requests.get to raise a timeout error
        with patch('requests.get', side_effect=requests.Timeout("Connection timed out")) as mock_get:
            result = await tool.execute({"url": "https://example.com"})
            
            # Should fail with timeout error
            assert result["success"] is False
            assert "timed out" in result["error"].lower()
            assert result["exception_type"] == "Timeout"

    @pytest.mark.asyncio
    async def test_web_fetch_connection_error(self):
        """Test web fetch with connection error."""
        tool = WebFetchTool()
        
        # Mock requests.get to raise a connection error
        with patch('requests.get', side_effect=requests.ConnectionError("Connection refused")) as mock_get:
            result = await tool.execute({"url": "https://example.com"})
            
            # Should fail with connection error
            assert result["success"] is False
            assert "connection" in result["error"].lower()
            assert result["exception_type"] == "ConnectionError"

    @pytest.mark.asyncio
    async def test_web_fetch_unsupported_content_type(self):
        """Test web fetch with unsupported content type."""
        tool = WebFetchTool()
        
        # Mock response with unsupported content type
        mock_response = Mock()
        mock_response.headers = {"Content-Type": "application/octet-stream"}
        mock_response.raise_for_status = Mock()
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            result = await tool.execute({"url": "https://example.com/binary-file"})
            
            # Should fail due to unsupported content type
            assert result["success"] is False
            assert "Unsupported content type" in result["error"]

    @pytest.mark.asyncio
    async def test_web_fetch_http_error(self):
        """Test web fetch with HTTP error from server."""
        tool = WebFetchTool()
        
        # Mock requests.get to raise an HTTP error
        mock_response = Mock()
        mock_response.raise_for_status = Mock(side_effect=requests.HTTPError("404 Client Error"))
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            result = await tool.execute({"url": "https://example.com/not-found"})
            
            # Should fail with HTTP error
            assert result["success"] is False
            assert "404" in result["error"]
            assert result["exception_type"] == "HTTPError"

    @pytest.mark.asyncio
    async def test_web_fetch_custom_headers(self):
        """Test web fetch with custom headers."""
        tool = WebFetchTool()
        
        # Mock successful response
        mock_response = Mock()
        mock_response.text = "<html><body>Content</body></html>"
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.raise_for_status = Mock()
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            # Execute with custom headers
            result = await tool.execute({
                "url": "https://example.com",
                "headers": {
                    "X-Custom-Header": "CustomValue",
                    "User-Agent": "Custom User Agent"
                }
            })
            
            # Verify headers were passed correctly
            args, kwargs = mock_get.call_args
            headers = kwargs["headers"]
            assert headers["X-Custom-Header"] == "CustomValue"
            assert headers["User-Agent"] == "Custom User Agent"
            
            # Should succeed
            assert result["success"] is True


class TestHTMLParserToolAdvanced:
    """Advanced tests for the HTMLParserTool class."""

    @pytest.mark.asyncio
    async def test_html_parser_with_complex_selectors(self):
        """Test HTML parser with complex CSS selectors."""
        tool = HTMLParserTool()
        
        # Complex HTML with nested elements
        html = """
        <html>
            <body>
                <div class="container">
                    <section id="products">
                        <div class="product" data-id="1">
                            <h2>Product 1</h2>
                            <span class="price">$10.99</span>
                        </div>
                        <div class="product" data-id="2">
                            <h2>Product 2</h2>
                            <span class="price">$20.99</span>
                        </div>
                    </section>
                </div>
            </body>
        </html>
        """
        
        with patch('me2ai_mcp.tools.web.BS4_AVAILABLE', True):
            with patch('me2ai_mcp.tools.web.BeautifulSoup') as mock_bs:
                # Configure mock BeautifulSoup
                mock_soup = MagicMock()
                mock_bs.return_value = mock_soup
                
                # Mock the product elements
                product1 = MagicMock()
                product1.get_text.return_value = "Product 1 $10.99"
                product1.get.side_effect = lambda attr: "1" if attr == "data-id" else None
                
                product2 = MagicMock()
                product2.get_text.return_value = "Product 2 $20.99"
                product2.get.side_effect = lambda attr: "2" if attr == "data-id" else None
                
                mock_soup.select.return_value = [product1, product2]
                
                # Execute with complex selectors
                result = await tool.execute({
                    "html": html,
                    "selectors": {
                        "products": {
                            "selector": ".product",
                            "multiple": True
                        },
                        "first_product_id": {
                            "selector": ".product",
                            "attribute": "data-id"
                        }
                    },
                    "extract_metadata": False,
                    "extract_text": False
                })
                
                # Verify result
                assert result["success"] is True
                assert "extracted" in result
                assert "products" in result["extracted"]
                assert len(result["extracted"]["products"]) == 2
                assert "first_product_id" in result["extracted"]
                assert result["extracted"]["first_product_id"] == "1"

    @pytest.mark.asyncio
    async def test_html_parser_with_invalid_html(self):
        """Test HTML parser with invalid HTML."""
        tool = HTMLParserTool()
        
        # Invalid HTML with unclosed tags
        html = "<html><body><div>Unclosed div<span>Unclosed span</body></html>"
        
        with patch('me2ai_mcp.tools.web.BS4_AVAILABLE', True):
            with patch('me2ai_mcp.tools.web.BeautifulSoup') as mock_bs:
                # Configure mock BeautifulSoup
                mock_soup = MagicMock()
                mock_bs.return_value = mock_soup
                
                # Mock get_text to return expected content
                mock_soup.get_text.return_value = "Unclosed div Unclosed span"
                
                # Execute parser
                result = await tool.execute({
                    "html": html,
                    "extract_text": True,
                    "extract_metadata": False
                })
                
                # Should still succeed with beautifulsoup handling the invalid HTML
                assert result["success"] is True
                assert "text" in result
                assert "Unclosed div Unclosed span" in result["text"]

    @pytest.mark.asyncio
    async def test_html_parser_with_missing_beautifulsoup(self):
        """Test HTML parser when BeautifulSoup is not available."""
        tool = HTMLParserTool()
        
        with patch('me2ai_mcp.tools.web.BS4_AVAILABLE', False):
            result = await tool.execute({
                "html": "<html><body>Test</body></html>"
            })
            
            # Should fail gracefully
            assert result["success"] is False
            assert "BeautifulSoup is not available" in result["error"]

    @pytest.mark.asyncio
    async def test_html_parser_extract_headings_hierarchy(self):
        """Test HTML parser extracting heading hierarchy."""
        tool = HTMLParserTool()
        
        # HTML with headings
        html = """
        <html>
            <body>
                <h1>Main Title</h1>
                <p>Some text</p>
                <h2>Subtitle 1</h2>
                <p>More text</p>
                <h3>Subheading</h3>
                <h2>Subtitle 2</h2>
            </body>
        </html>
        """
        
        with patch('me2ai_mcp.tools.web.BS4_AVAILABLE', True):
            with patch('me2ai_mcp.tools.web.BeautifulSoup') as mock_bs:
                # Configure mock BeautifulSoup
                mock_soup = MagicMock()
                mock_bs.return_value = mock_soup
                
                # Mock find_all for headings
                h1 = MagicMock()
                h1.get_text.return_value = "Main Title"
                
                h2_1 = MagicMock()
                h2_1.get_text.return_value = "Subtitle 1"
                
                h3 = MagicMock()
                h3.get_text.return_value = "Subheading"
                
                h2_2 = MagicMock()
                h2_2.get_text.return_value = "Subtitle 2"
                
                def find_all_side_effect(tag):
                    if tag == "h1":
                        return [h1]
                    elif tag == "h2":
                        return [h2_1, h2_2]
                    elif tag == "h3":
                        return [h3]
                    else:
                        return []
                
                mock_soup.find_all.side_effect = find_all_side_effect
                
                # Mock extraction of text and removal of scripts
                mock_soup.get_text.return_value = "Main Title Some text Subtitle 1 More text Subheading Subtitle 2"
                
                # Execute parser with heading extraction
                result = await tool.execute({
                    "html": html,
                    "extract_text": True,
                    "extract_metadata": False
                })
                
                # Should succeed with headings
                assert result["success"] is True
                assert "headings" in result
                assert len(result["headings"]) > 0
                
                # Check if headings match expectations
                headings_text = [h["text"] for h in result["headings"]]
                assert "Main Title" in headings_text
                assert "Subtitle 1" in headings_text
                assert "Subheading" in headings_text
                assert "Subtitle 2" in headings_text


class TestURLUtilsToolAdvanced:
    """Advanced tests for the URLUtilsTool class."""

    @pytest.mark.asyncio
    async def test_url_utils_parsing_complex_url(self):
        """Test URL parsing with complex URL containing all components."""
        tool = URLUtilsTool()
        
        # Complex URL with all components
        url = "https://user:pass@example.com:8080/path/to/page.html?query=value&page=2#section"
        
        result = await tool.execute({
            "operation": "parse",
            "url": url
        })
        
        # Should parse correctly
        assert result["success"] is True
        assert result["parsed"]["scheme"] == "https"
        assert result["parsed"]["netloc"] == "user:pass@example.com:8080"
        assert result["parsed"]["path"] == "/path/to/page.html"
        assert result["parsed"]["username"] == "user"
        assert result["parsed"]["password"] == "pass"
        assert result["parsed"]["hostname"] == "example.com"
        assert result["parsed"]["port"] == 8080
        assert result["parsed"]["query"] == "query=value&page=2"
        assert result["parsed"]["fragment"] == "section"
        
        # Check query parameters
        assert result["query_params"]["query"] == "value"
        assert result["query_params"]["page"] == "2"

    @pytest.mark.asyncio
    async def test_url_utils_joining_with_relative_paths(self):
        """Test URL joining with relative paths."""
        tool = URLUtilsTool()
        
        test_cases = [
            {
                "base_url": "https://example.com/dir/",
                "path": "../page.html",
                "expected": "https://example.com/page.html"
            },
            {
                "base_url": "https://example.com/dir/file.html",
                "path": "subdir/page.html",
                "expected": "https://example.com/dir/subdir/page.html"
            },
            {
                "base_url": "https://example.com/dir/",
                "path": "/absolute/path.html",
                "expected": "https://example.com/absolute/path.html"
            }
        ]
        
        for case in test_cases:
            result = await tool.execute({
                "operation": "join",
                "base_url": case["base_url"],
                "path": case["path"]
            })
            
            # Should join correctly
            assert result["success"] is True
            assert result["joined_url"] == case["expected"]

    @pytest.mark.asyncio
    async def test_url_utils_normalizing_complex_url(self):
        """Test URL normalization with complex path."""
        tool = URLUtilsTool()
        
        # URL with complex path that needs normalization
        url = "https://example.com/a/b/../c/./d/../../e/f/./g"
        
        result = await tool.execute({
            "operation": "normalize",
            "url": url
        })
        
        # Should normalize correctly
        assert result["success"] is True
        assert result["normalized_url"] == "https://example.com/a/e/f/g"

    @pytest.mark.asyncio
    async def test_url_utils_with_invalid_url(self):
        """Test URL utils with invalid URL."""
        tool = URLUtilsTool()
        
        # Invalid URL
        url = "not a valid url"
        
        # Parsing should still work, just with limited components
        result = await tool.execute({
            "operation": "parse",
            "url": url
        })
        
        assert result["success"] is True
        assert result["parsed"]["scheme"] == ""
        assert result["parsed"]["netloc"] == ""
        assert result["parsed"]["path"] == "not a valid url"

    @pytest.mark.asyncio
    async def test_url_utils_with_missing_parameters(self):
        """Test URL utils with missing required parameters."""
        tool = URLUtilsTool()
        
        # Test parse without URL
        result = await tool.execute({
            "operation": "parse"
        })
        
        assert result["success"] is False
        assert "URL parameter is required" in result["error"]
        
        # Test join without base_url
        result = await tool.execute({
            "operation": "join",
            "path": "page.html"
        })
        
        assert result["success"] is False
        assert "base_url and path parameters are required" in result["error"]
