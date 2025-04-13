"""Fetch MCP Server for ME2AI using the enhanced ME2AI MCP framework.

This server provides web content fetching capabilities for agents, with improved
error handling, logging, and statistics tracking using the ME2AI MCP framework.
"""
import os
import sys
import logging
from typing import Dict, Any, Optional
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("fetch-mcp")

try:
    # Import ME2AI MCP libraries
    from me2ai_mcp import ME2AIMCPServer, register_tool
    from me2ai_mcp.tools.web import WebFetchTool, HTMLParserTool, URLUtilsTool
    
    # Additional dependencies
    import requests
    from bs4 import BeautifulSoup
    import markdown
    from urllib.parse import urlparse
    
    logger.info("Successfully imported required libraries")
except ImportError as e:
    logger.error(f"Failed to import required libraries: {e}")
    logger.error("Please install required packages: pip install -e . requests beautifulsoup4 markdown")
    sys.exit(1)


class FetchMCPServer(ME2AIMCPServer):
    """Enhanced MCP server for web content fetching and processing using ME2AI MCP framework."""
    
    def __init__(self):
        """Initialize the Fetch MCP server with the ME2AI MCP framework."""
        super().__init__(
            server_name="fetch-mcp",
            description="Web content fetching and processing service",
            version="1.0.0"
        )
        
        # Initialize tool instances
        self.web_fetch_tool = WebFetchTool(
            user_agent="Mozilla/5.0 ME2AI Web Fetcher/1.0",
            timeout=15
        )
        self.html_parser_tool = HTMLParserTool()
        self.url_utils_tool = URLUtilsTool()
        
        logger.info("Enhanced Fetch MCP server initialized")
    
    @register_tool
    async def fetch_webpage(self, url: str, max_length: int = 10000) -> Dict[str, Any]:
        """Fetch and convert web content to a format optimized for language models.
        
        Args:
            url: URL of the webpage to fetch
            max_length: Maximum length of content to return (characters)
            
        Returns:
            Dictionary containing processed web content
        """
        logger.info(f"Fetching webpage: {url}")
        
        # First validate URL format
        url_validation = await self.url_utils_tool.execute({
            "operation": "parse",
            "url": url
        })
        
        if not url_validation["success"]:
            return url_validation  # Return error from URL tool
            
        parsed_url = url_validation["parsed"]
        if not parsed_url["scheme"] or not parsed_url["netloc"]:
            return {"success": False, "error": "Invalid URL format"}
        
        # Use the WebFetchTool for the actual fetching
        fetch_result = await self.web_fetch_tool.execute({
            "url": url,
            "headers": {
                "User-Agent": "Mozilla/5.0 ME2AI Web Fetcher/1.0"
            }
        })
        
        if not fetch_result["success"]:
            return fetch_result  # Return error from fetch tool
        
        # Process the content with HTMLParserTool
        parse_result = await self.html_parser_tool.execute({
            "html": fetch_result["content"],
            "extract_metadata": True,
            "extract_text": True
        })
        
        if not parse_result["success"]:
            return parse_result  # Return error from parser tool
        
        # Extract main components
        title = parse_result.get("metadata", {}).get("title", "No title found")
        raw_text = parse_result.get("text", "")
        headings = parse_result.get("headings", [])
        
        # Truncate content if necessary
        if raw_text and len(raw_text) > max_length:
            raw_text = raw_text[:max_length] + "... [truncated]"
        
        # Prepare the response
        content_type = fetch_result.get("content_type", "unknown")
        is_html = "text/html" in content_type.lower()
        
        # Combine everything into a structured response
        return {
            "success": True,
            "url": url,
            "title": title,
            "content_type": content_type,
            "is_html": is_html,
            "content_length": len(raw_text),
            "raw_text": raw_text,
            "headings": [h["text"] for h in headings],
            "metadata": parse_result.get("metadata", {}),
            "stats": {
                "original_size": fetch_result.get("content_length", 0),
                "status_code": fetch_result.get("status_code", 0)
            }
        }
    
    @register_tool
    async def extract_elements(self, url: str, css_selector: str) -> Dict[str, Any]:
        """Extract specific elements from a webpage using CSS selectors.
        
        Args:
            url: URL of the webpage to fetch
            css_selector: CSS selector to extract specific elements
            
        Returns:
            Dictionary containing extracted elements
        """
        logger.info(f"Extracting elements from {url} using selector: {css_selector}")
        
        # First fetch the webpage
        fetch_result = await self.web_fetch_tool.execute({
            "url": url
        })
        
        if not fetch_result["success"]:
            return fetch_result  # Return error from fetch tool
        
        # Use HTMLParserTool with the specific selector
        parse_result = await self.html_parser_tool.execute({
            "html": fetch_result["content"],
            "extract_metadata": True,
            "selectors": {
                "extracted_elements": {
                    "selector": css_selector,
                    "multiple": True
                }
            }
        })
        
        if not parse_result["success"]:
            return parse_result  # Return error from parser tool
        
        # Get the extracted elements
        extracted = parse_result.get("extracted", {}).get("extracted_elements", [])
        
        # Return the results
        return {
            "success": True,
            "url": url,
            "css_selector": css_selector,
            "elements_count": len(extracted) if isinstance(extracted, list) else 1,
            "elements": extracted,
            "title": parse_result.get("metadata", {}).get("title", "No title found")
        }
    
    @register_tool
    async def summarize_webpage(self, url: str) -> Dict[str, Any]:
        """Fetch a webpage and extract the main content in a summarized format.
        
        Args:
            url: URL of the webpage to summarize
            
        Returns:
            Dictionary containing summarized web content
        """
        logger.info(f"Summarizing webpage: {url}")
        
        # First fetch the webpage with standard processing
        fetch_result = await self.fetch_webpage(url, max_length=50000)
        
        if not fetch_result["success"]:
            return fetch_result  # Return error from fetch
        
        # Extract the key components for summary
        title = fetch_result.get("title", "")
        headings = fetch_result.get("headings", [])
        
        # Use the first 25% of the content for the summary
        raw_text = fetch_result.get("raw_text", "")
        text_preview = raw_text[:min(len(raw_text) // 4, 2000)]
        
        # Create a concise structure for the summary
        return {
            "success": True,
            "url": url,
            "title": title,
            "headings": headings[:10],  # Limit to top 10 headings
            "preview": text_preview,
            "total_content_length": len(raw_text),
            "estimated_reading_time": len(raw_text) // 1500  # Rough estimate: 1500 chars per minute
        }


async def main():
    """Run the Fetch MCP server."""
    server = FetchMCPServer()
    await server.start()


def shutdown(server):
    """Shut down the server gracefully."""
    logger.info("Shutting down Fetch MCP server...")
    # Here we would add any cleanup operations if needed


if __name__ == "__main__":
    try:
        # Import signal module here to avoid issues on Windows
        import signal
        
        # Run the server
        loop = asyncio.get_event_loop()
        server = loop.run_until_complete(main())
        
        # Handle graceful shutdown on SIGINT/SIGTERM
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: shutdown(server))
        
        # Keep the server running
        loop.run_forever()
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
