"""Fetch MCP Server for ME2AI.

This script implements a Model Context Protocol (MCP) server for web content fetching,
allowing agents to retrieve and process web content in a format optimized for LLMs.
"""
import os
import sys
import logging
import asyncio
import argparse
from typing import Dict, List, Any, Optional, Union

# Import ME2AI MCP package
from me2ai.mcp import ME2AIMCPServer, register_tool
from me2ai.mcp.tools.web import WebFetchTool, HTMLParserTool, URLUtilsTool
from me2ai.mcp.utils import extract_text, sanitize_input, format_response

# Configure optional dependencies
try:
    import requests
    from bs4 import BeautifulSoup
    import markdown
    from urllib.parse import urlparse
    
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logging.warning("BeautifulSoup4 or other dependencies not available. Some features may be limited.")

class FetchMCPServer(ME2AIMCPServer):
    """MCP server for web content fetching and processing using ME2AI MCP framework."""
    
    def __init__(self, debug: bool = False):
        """Initialize the Fetch MCP server.
        
        Args:
            debug: Whether to enable debug logging
        """
        super().__init__(
            server_name="fetch",
            description="Web content fetching and processing for ME2AI agents",
            version="0.1.0",
            debug=debug
        )
        
        # Initialize tools
        self.web_fetch_tool = WebFetchTool(
            name="fetch_webpage_tool",
            user_agent="ME2AI Web Fetcher/1.0",
            timeout=30,
            max_content_length=1024 * 1024  # 1MB
        )
        
        self.html_parser_tool = HTMLParserTool(
            name="html_parser_tool"
        )
        
        self.url_utils_tool = URLUtilsTool(
            name="url_utils_tool"
        )
        
        # Register tools
        self._register_tools()
        self.logger.info("✓ Fetch MCP server initialized with ME2AI framework")
    
    def _register_tools(self):
        """Register web fetching tools with the MCP server."""
        self.logger.info("Registering web fetching tools...")
    
    @register_tool
    async def fetch_webpage(self, url: str, max_length: int = 10000) -> Dict[str, Any]:
        """Fetch and convert web content to a format optimized for language models.
        
        Args:
            url: URL of the webpage to fetch
            max_length: Maximum length of content to return (characters)
            
        Returns:
            Dictionary containing fetch results
        """
        # Validate and sanitize inputs
        url = sanitize_input(url)
        
        # Use the WebFetchTool from our ME2AI MCP package
        fetch_result = await self.web_fetch_tool.execute({
            "url": url,
            "timeout": 30  # Override timeout for this request
        })
        
        if not fetch_result.get("success", False):
            # Just return the error from the underlying tool
            return fetch_result
            
        # Extract data from result
        content = fetch_result.get("content", "")
        content_type = fetch_result.get("content_type", "")
        title = fetch_result.get("title", "No title")
        
        # Process content based on type
        if "text/html" in content_type and BS4_AVAILABLE:
            try:
                # Use BeautifulSoup to extract readable content
                soup = BeautifulSoup(content, "html.parser")
                
                # Remove unnecessary elements
                for tag in soup(["script", "style", "nav", "footer", "iframe"]):
                    tag.decompose()
                
                # Get cleaned text
                text = soup.get_text(separator="\n", strip=True)
                
                # Format and truncate content
                content = text[:max_length] if len(text) > max_length else text
                
            except Exception as e:
                self.logger.error(f"Error processing HTML: {str(e)}")
                return {
                    "success": False,
                    "error": f"Error processing HTML content: {str(e)}"
                }
        else:
            # For non-HTML content, just truncate
            content = content[:max_length] if len(content) > max_length else content
        
        # Return formatted result
        return {
            "success": True,
            "url": url,
            "title": title,
            "content": content,
            "content_length": len(content),
            "content_type": content_type
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
        # Validate and sanitize inputs
        url = sanitize_input(url)
        css_selector = sanitize_input(css_selector)
        
        # First fetch the webpage
        fetch_result = await self.web_fetch_tool.execute({
            "url": url
        })
        
        if not fetch_result.get("success", False):
            # Return error from fetch operation
            return fetch_result
            
        content = fetch_result.get("content", "")
        
        # Now use the HTML parser to extract elements
        parse_result = await self.html_parser_tool.execute({
            "html": content,
            "selectors": {
                "selected_elements": {
                    "selector": css_selector,
                    "multiple": True
                }
            },
            "extract_metadata": True
        })
        
        if not parse_result.get("success", False):
            # Return error from parse operation
            return parse_result
            
        # Get the extracted elements
        extracted = parse_result.get("extracted", {})
        elements = extracted.get("selected_elements", [])
        
        # Get metadata
        metadata = parse_result.get("metadata", {})
        
        return {
            "success": True,
            "url": url,
            "css_selector": css_selector,
            "elements": elements,
            "elements_count": len(elements),
            "title": metadata.get("title", "No title")
        }
    
    @register_tool
    async def summarize_webpage(self, url: str) -> Dict[str, Any]:
        """Fetch a webpage and extract the main content in a summarized format.
        
        Args:
            url: URL of the webpage to fetch
            
        Returns:
            Dictionary containing summarized webpage content
        """
        # Validate and sanitize inputs
        url = sanitize_input(url)
        
        # Fetch the webpage
        fetch_result = await self.web_fetch_tool.execute({
            "url": url
        })
        
        if not fetch_result.get("success", False):
            # Return error from fetch operation
            return fetch_result
            
        content = fetch_result.get("content", "")
        content_type = fetch_result.get("content_type", "")
        title = fetch_result.get("title", "No title")
        
        if "text/html" not in content_type or not BS4_AVAILABLE:
            return {
                "success": False,
                "error": "Cannot summarize non-HTML content or BeautifulSoup is not available"
            }
            
        try:
            # Parse HTML
            soup = BeautifulSoup(content, "html.parser")
            
            # Extract metadata
            metadata = {}
            
            # Extract meta description
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc and "content" in meta_desc.attrs:
                metadata["description"] = meta_desc["content"]
                
            # Extract headings
            headings = []
            for level in range(1, 4):  # H1, H2, H3
                for h in soup.find_all(f"h{level}"):
                    text = h.get_text(strip=True)
                    if text:
                        headings.append({
                            "level": level,
                            "text": text
                        })
            
            # Find main content
            main_content = soup.find("main") or soup.find("article") or soup.find("div", class_="content")
            
            # If no main content container found, look for the largest text block
            if not main_content:
                main_paragraphs = soup.find_all("p")
                if main_paragraphs:
                    main_paragraphs = [p.get_text(strip=True) for p in main_paragraphs if len(p.get_text(strip=True)) > 100]
                else:
                    main_paragraphs = []
            else:
                # Extract paragraphs from main content
                paragraph_elements = main_content.find_all("p")
                main_paragraphs = [p.get_text(strip=True) for p in paragraph_elements if len(p.get_text(strip=True)) > 50]
            
            # Extract links (up to 10)
            links = []
            for link in soup.find_all("a", href=True)[:10]:
                href = link["href"]
                text = link.get_text(strip=True)
                
                # Only include links with text and valid URLs
                if text and href and not href.startswith("#") and not href.startswith("javascript:"):
                    # Make sure href is absolute
                    if not href.startswith(("http://", "https://")):
                        href = urljoin(url, href)
                        
                    links.append({
                        "text": text,
                        "url": href
                    })
            
            return {
                "success": True,
                "url": url,
                "title": title,
                "description": metadata.get("description", ""),
                "headings": headings[:10],  # Limit to 10 headings
                "main_paragraphs": main_paragraphs[:5],  # Limit to 5 paragraphs
                "links": links,
                "summary_generated": True
            }
            
        except Exception as e:
            self.logger.error(f"Error summarizing webpage: {str(e)}")
            return {
                "success": False,
                "error": f"Error summarizing webpage: {str(e)}"
            }


def main():
    """Run the Fetch MCP server."""
    parser = argparse.ArgumentParser(description="Fetch MCP Server")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and start server
    server = FetchMCPServer(debug=args.debug)
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
    except Exception as e:
        logging.error(f"Server error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
