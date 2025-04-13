"""
Tests for the ME2AI MCP web tools.
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock, Mock
import json
import requests

from me2ai_mcp.tools.web import WebFetchTool, HTMLParserTool, URLUtilsTool


class TestWebFetchTool:
    """Tests for the WebFetchTool class."""

    def test_init(self):
        """Test tool initialization."""
        tool = WebFetchTool(
            name="custom-fetch",
            description="Custom fetch tool",
            user_agent="Test Agent/1.0",
            timeout=60
        )
        
        assert tool.name == "custom-fetch"
        assert tool.description == "Custom fetch tool"
        assert tool.user_agent == "Test Agent/1.0"
        assert tool.timeout == 60
        
    @pytest.mark.asyncio
    async def test_missing_url(self):
        """Test fetch with missing URL."""
        tool = WebFetchTool()
        
        result = await tool.execute({})
        
        assert result["success"] is False
        assert "URL parameter is required" in result["error"]
        
    @pytest.mark.asyncio
    async def test_invalid_url_scheme(self):
        """Test fetch with invalid URL scheme."""
        tool = WebFetchTool()
        
        result = await tool.execute({"url": "ftp://example.com"})
        
        assert result["success"] is False
        assert "Invalid URL scheme" in result["error"]
        
    @pytest.mark.asyncio
    async def test_fetch_success(self):
        """Test successful fetch."""
        tool = WebFetchTool()
        
        # Mock the requests.get method
        mock_response = Mock()
        mock_response.text = "<html><head><title>Test Page</title></head><body>Test content</body></html>"
        mock_response.status_code = 200
        mock_response.headers = {
            "Content-Type": "text/html", 
            "Content-Length": str(len(mock_response.text))
        }
        mock_response.raise_for_status = Mock()
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            result = await tool.execute({"url": "https://example.com"})
            
            # Check request was made correctly
            mock_get.assert_called_once()
            args, kwargs = mock_get.call_args
            assert args[0] == "https://example.com"
            assert "User-Agent" in kwargs["headers"]
            
            # Check result
            assert result["success"] is True
            assert result["url"] == "https://example.com"
            assert result["status_code"] == 200
            assert result["content_type"] == "text/html"
            assert result["content"] == mock_response.text
            assert "Test Page" in result["title"]
            
    @pytest.mark.asyncio
    async def test_fetch_error_handling(self):
        """Test error handling during fetch."""
        tool = WebFetchTool()
        
        # Test request exception
        with patch('requests.get', side_effect=requests.RequestException("Connection error")) as mock_get:
            result = await tool.execute({"url": "https://example.com"})
            
            assert result["success"] is False
            assert "Request error" in result["error"]
            assert result["exception_type"] == "RequestException"
            
        # Test content too large
        mock_response = Mock()
        mock_response.text = "X" * 1000
        mock_response.status_code = 200
        mock_response.headers = {
            "Content-Type": "text/html", 
            "Content-Length": str(1024 * 1024 * 10)  # 10MB (larger than max)
        }
        mock_response.raise_for_status = Mock()
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            result = await tool.execute({"url": "https://example.com"})
            
            assert result["success"] is False
            assert "Content too large" in result["error"]


class TestHTMLParserTool:
    """Tests for the HTMLParserTool class."""
    
    @pytest.mark.asyncio
    async def test_missing_html(self):
        """Test parse with missing HTML."""
        tool = HTMLParserTool()
        
        result = await tool.execute({})
        
        assert result["success"] is False
        assert "HTML parameter is required" in result["error"]
        
    @pytest.mark.asyncio
    async def test_parse_metadata(self):
        """Test extracting metadata from HTML."""
        tool = HTMLParserTool()
        
        html = """
        <html>
            <head>
                <title>Test Page</title>
                <meta name="description" content="Test description">
                <meta property="og:title" content="Open Graph Title">
            </head>
            <body>
                <h1>Heading 1</h1>
                <p>Test paragraph</p>
            </body>
        </html>
        """
        
        with patch('me2ai_mcp.tools.web.BS4_AVAILABLE', True):
            with patch('me2ai_mcp.tools.web.BeautifulSoup') as mock_bs:
                # Configure mock BeautifulSoup
                mock_soup = MagicMock()
                mock_bs.return_value = mock_soup
                
                # Mock find method for title
                mock_title = MagicMock()
                mock_title.string = "Test Page"
                mock_soup.find.return_value = mock_title
                
                # Mock find_all for meta tags
                meta1 = MagicMock()
                meta1.get.side_effect = lambda x: "description" if x == "name" else ("Test description" if x == "content" else None)
                
                meta2 = MagicMock()
                meta2.get.side_effect = lambda x: "og:title" if x == "property" else ("Open Graph Title" if x == "content" else None)
                
                mock_soup.find_all.return_value = [meta1, meta2]
                
                # Mock get_text
                mock_soup.get_text.return_value = "Heading 1 Test paragraph"
                
                result = await tool.execute({
                    "html": html,
                    "extract_metadata": True,
                    "extract_text": True
                })
                
                assert result["success"] is True
                assert "metadata" in result
                assert "title" in result["metadata"]
                assert result["metadata"]["title"] == "Test Page"
                assert "meta_tags" in result["metadata"]
                assert "text" in result
                
    @pytest.mark.asyncio
    async def test_parse_with_selectors(self):
        """Test parsing HTML with custom selectors."""
        tool = HTMLParserTool()
        
        html = "<html><body><div class='item'>Item 1</div><div class='item'>Item 2</div></body></html>"
        
        with patch('me2ai_mcp.tools.web.BS4_AVAILABLE', True):
            with patch('me2ai_mcp.tools.web.BeautifulSoup') as mock_bs:
                # Configure mock BeautifulSoup
                mock_soup = MagicMock()
                mock_bs.return_value = mock_soup
                
                # Mock select_one
                mock_soup.select_one.return_value.get_text.return_value = "Item 1"
                
                # Mock select
                mock_element1 = MagicMock()
                mock_element1.get_text.return_value = "Item 1"
                
                mock_element2 = MagicMock()
                mock_element2.get_text.return_value = "Item 2"
                
                mock_soup.select.return_value = [mock_element1, mock_element2]
                
                result = await tool.execute({
                    "html": html,
                    "selectors": {
                        "first_item": ".item",
                        "all_items": {
                            "selector": ".item",
                            "multiple": True
                        }
                    },
                    "extract_metadata": False,
                    "extract_text": False
                })
                
                assert result["success"] is True
                assert "extracted" in result
                assert "first_item" in result["extracted"]
                assert result["extracted"]["first_item"] == "Item 1"
                assert "all_items" in result["extracted"]
                assert len(result["extracted"]["all_items"]) == 2


class TestURLUtilsTool:
    """Tests for the URLUtilsTool class."""
    
    @pytest.mark.asyncio
    async def test_parse_url(self):
        """Test URL parsing operation."""
        tool = URLUtilsTool()
        
        result = await tool.execute({
            "operation": "parse",
            "url": "https://example.com:8080/path?query=value#fragment"
        })
        
        assert result["success"] is True
        assert result["url"] == "https://example.com:8080/path?query=value#fragment"
        assert result["parsed"]["scheme"] == "https"
        assert result["parsed"]["netloc"] == "example.com:8080"
        assert result["parsed"]["path"] == "/path"
        assert result["parsed"]["query"] == "query=value"
        assert result["parsed"]["fragment"] == "fragment"
        assert result["parsed"]["hostname"] == "example.com"
        assert result["parsed"]["port"] == 8080
        assert result["query_params"] == {"query": "value"}
        
    @pytest.mark.asyncio
    async def test_join_url(self):
        """Test URL joining operation."""
        tool = URLUtilsTool()
        
        result = await tool.execute({
            "operation": "join",
            "base_url": "https://example.com/base/",
            "path": "../path/file.html"
        })
        
        assert result["success"] is True
        assert result["base_url"] == "https://example.com/base/"
        assert result["path"] == "../path/file.html"
        assert result["joined_url"] == "https://example.com/path/file.html"
        
    @pytest.mark.asyncio
    async def test_normalize_url(self):
        """Test URL normalization operation."""
        tool = URLUtilsTool()
        
        result = await tool.execute({
            "operation": "normalize",
            "url": "https://example.com/path/../another/./file.html"
        })
        
        assert result["success"] is True
        assert result["original_url"] == "https://example.com/path/../another/./file.html"
        # The actual normalized URL will depend on urllib.parse implementation
        assert "normalized_url" in result
        
    @pytest.mark.asyncio
    async def test_invalid_operation(self):
        """Test invalid operation handling."""
        tool = URLUtilsTool()
        
        result = await tool.execute({
            "operation": "invalid_op",
            "url": "https://example.com/"
        })
        
        assert result["success"] is False
        assert "Unknown operation" in result["error"]
