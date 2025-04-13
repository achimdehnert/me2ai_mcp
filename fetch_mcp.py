"""Fetch MCP Server for ME2AI.

This script implements a Model Context Protocol (MCP) server for web content fetching,
allowing agents to retrieve and process web content in a format optimized for LLMs.
"""
import os
import sys
import logging
import asyncio
import argparse
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("fetch-mcp")

try:
    # Import MCP libraries
    from mcp.server import MCPServer
    from mcp.server.schema import ToolDefinition, FunctionDefinition
    import requests
    from bs4 import BeautifulSoup
    import markdown
    from urllib.parse import urlparse
    
    logger.info("Successfully imported required libraries")
except ImportError as e:
    logger.error(f"Failed to import required libraries: {e}")
    logger.error("Please install required packages: pip install mcp requests beautifulsoup4 markdown")
    sys.exit(1)

class FetchMCPServer(MCPServer):
    """MCP server for web content fetching and processing."""
    
    def __init__(self):
        """Initialize the Fetch MCP server."""
        super().__init__()
        self._register_tools()
        logger.info("Fetch MCP server initialized")
    
    def _register_tools(self):
        """Register web fetching tools with the MCP server."""
        # Fetch web content tool
        fetch_tool = ToolDefinition(
            name="fetch_webpage",
            description="Fetch and convert web content to a format optimized for language models",
            function=FunctionDefinition(
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL of the webpage to fetch"
                        },
                        "max_length": {
                            "type": "integer",
                            "description": "Maximum length of content to return (characters)",
                            "default": 10000
                        }
                    },
                    "required": ["url"]
                }
            )
        )
        
        # Extract specific elements tool
        extract_tool = ToolDefinition(
            name="extract_elements",
            description="Extract specific elements from a webpage using CSS selectors",
            function=FunctionDefinition(
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL of the webpage to fetch"
                        },
                        "css_selector": {
                            "type": "string",
                            "description": "CSS selector to extract specific elements (e.g., 'div.content', 'h1', 'article')"
                        }
                    },
                    "required": ["url", "css_selector"]
                }
            )
        )
        
        # Summarize webpage tool
        summarize_tool = ToolDefinition(
            name="summarize_webpage",
            description="Fetch a webpage and extract the main content in a summarized format",
            function=FunctionDefinition(
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL of the webpage to summarize"
                        }
                    },
                    "required": ["url"]
                }
            )
        )
        
        # Register all tools
        self.registry.register_tool(fetch_tool, self.fetch_webpage)
        self.registry.register_tool(extract_tool, self.extract_elements)
        self.registry.register_tool(summarize_tool, self.summarize_webpage)
        
        logger.info(f"Registered {len(self.registry.list_tools())} web fetching tools")
    
    async def fetch_webpage(self, url: str, max_length: int = 10000) -> Dict[str, Any]:
        """Fetch and convert web content to a format optimized for language models.
        
        Args:
            url: URL of the webpage to fetch
            max_length: Maximum length of content to return (characters)
            
        Returns:
            Dictionary containing processed web content
        """
        logger.info(f"Fetching webpage: {url}")
        
        try:
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return {"success": False, "error": "Invalid URL format"}
            
            # Fetch the webpage
            headers = {
                "User-Agent": "Mozilla/5.0 ME2AI Web Fetcher/1.0"
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Process the content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "iframe", "noscript"]):
                script.decompose()
            
            # Get the page title
            title = soup.title.string if soup.title else "No title found"
            
            # Get main content (prioritize main content areas)
            content_candidates = soup.select("main, article, .content, .post, .entry, #content")
            
            if content_candidates:
                # Use the first content area that has substantial text
                for candidate in content_candidates:
                    if len(candidate.get_text(strip=True)) > 100:
                        main_content = candidate
                        break
                else:
                    main_content = content_candidates[0]
            else:
                # If no content areas found, use the body
                main_content = soup.body
            
            # Extract text content
            text_content = main_content.get_text(separator='\n', strip=True)
            
            # Truncate if necessary
            if len(text_content) > max_length:
                text_content = text_content[:max_length] + "... [content truncated]"
            
            # Extract metadata
            meta_description = ""
            meta_tag = soup.find("meta", attrs={"name": "description"})
            if meta_tag and "content" in meta_tag.attrs:
                meta_description = meta_tag["content"]
            
            return {
                "success": True,
                "url": url,
                "title": title,
                "description": meta_description,
                "content": text_content,
                "content_length": len(text_content),
                "content_type": response.headers.get("Content-Type", "")
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {url}: {str(e)}")
            return {"success": False, "error": f"Failed to fetch URL: {str(e)}"}
            
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            return {"success": False, "error": f"Error processing content: {str(e)}"}
    
    async def extract_elements(self, url: str, css_selector: str) -> Dict[str, Any]:
        """Extract specific elements from a webpage using CSS selectors.
        
        Args:
            url: URL of the webpage to fetch
            css_selector: CSS selector to extract specific elements
            
        Returns:
            Dictionary containing extracted elements
        """
        logger.info(f"Extracting elements from {url} using selector: {css_selector}")
        
        try:
            # Fetch the webpage
            headers = {
                "User-Agent": "Mozilla/5.0 ME2AI Web Fetcher/1.0"
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract elements using CSS selector
            elements = soup.select(css_selector)
            
            # Process results
            results = []
            for idx, element in enumerate(elements[:20]):  # Limit to 20 elements
                # Get element text
                text = element.get_text(separator='\n', strip=True)
                
                # Get element attributes
                attrs = {}
                for attr_name, attr_value in element.attrs.items():
                    if isinstance(attr_value, list):
                        attrs[attr_name] = " ".join(attr_value)
                    else:
                        attrs[attr_name] = attr_value
                
                results.append({
                    "index": idx,
                    "tag": element.name,
                    "text": text,
                    "html": str(element),
                    "attributes": attrs
                })
            
            return {
                "success": True,
                "url": url,
                "selector": css_selector,
                "count": len(elements),
                "elements": results,
                "truncated": len(elements) > 20
            }
            
        except Exception as e:
            logger.error(f"Error extracting elements from {url}: {str(e)}")
            return {"success": False, "error": f"Error extracting elements: {str(e)}"}
    
    async def summarize_webpage(self, url: str) -> Dict[str, Any]:
        """Fetch a webpage and extract the main content in a summarized format.
        
        Args:
            url: URL of the webpage to summarize
            
        Returns:
            Dictionary containing summarized web content
        """
        logger.info(f"Summarizing webpage: {url}")
        
        try:
            # Fetch the webpage
            headers = {
                "User-Agent": "Mozilla/5.0 ME2AI Web Fetcher/1.0"
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Process the content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script, style, and other non-content elements
            for tag in soup(["script", "style", "iframe", "noscript", "footer", "nav", "aside"]):
                tag.decompose()
            
            # Get the page title
            title = soup.title.string if soup.title else "No title found"
            
            # Get metadata
            meta_description = ""
            meta_tag = soup.find("meta", attrs={"name": "description"})
            if meta_tag and "content" in meta_tag.attrs:
                meta_description = meta_tag["content"]
            
            # Extract headings
            headings = []
            for heading in soup.find_all(["h1", "h2", "h3"]):
                text = heading.get_text(strip=True)
                if text and len(text) > 3:
                    headings.append({
                        "level": int(heading.name[1]),
                        "text": text
                    })
            
            # Extract important paragraphs (first 3 substantial paragraphs)
            paragraphs = []
            for p in soup.find_all("p"):
                text = p.get_text(strip=True)
                if text and len(text) > 100:  # Only substantial paragraphs
                    paragraphs.append(text)
                    if len(paragraphs) >= 3:
                        break
            
            # Extract links
            important_links = []
            for a in soup.find_all("a", href=True):
                text = a.get_text(strip=True)
                href = a["href"]
                if text and len(text) > 3 and href and not href.startswith("#"):
                    # Convert relative URLs to absolute
                    if not href.startswith(("http://", "https://")):
                        base_url = "{0.scheme}://{0.netloc}".format(urlparse(url))
                        if href.startswith("/"):
                            href = base_url + href
                        else:
                            href = base_url + "/" + href
                    
                    important_links.append({
                        "text": text,
                        "url": href
                    })
                    
                    if len(important_links) >= 5:
                        break
            
            return {
                "success": True,
                "url": url,
                "title": title,
                "description": meta_description,
                "headings": headings[:10],  # Limit to top 10 headings
                "main_paragraphs": paragraphs,
                "important_links": important_links
            }
            
        except Exception as e:
            logger.error(f"Error summarizing {url}: {str(e)}")
            return {"success": False, "error": f"Error summarizing content: {str(e)}"}

async def main():
    """Run the Fetch MCP server."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fetch MCP Server")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Create the server
    server = FetchMCPServer()
    logger.info("Starting Fetch MCP server...")
    
    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(server)))
    
    # Run the server
    await server.run()
    logger.info("Fetch MCP server stopped")

async def shutdown(server):
    """Shut down the server gracefully."""
    logger.info("Shutting down Fetch MCP server...")
    await server.shutdown()

if __name__ == "__main__":
    try:
        # Import signal module here to avoid issues on Windows
        import signal
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        sys.exit(1)
