"""Brave Search MCP Server for ME2AI.

This script implements a Model Context Protocol (MCP) server for web search using Brave Search API,
allowing agents to search the web and retrieve relevant information.
"""
import os
import sys
import logging
import asyncio
import signal
import argparse
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("brave-search-mcp")

try:
    # Import MCP libraries
    from mcp.server import MCPServer
    from mcp.server.schema import ToolDefinition, FunctionDefinition
    import requests
    from datetime import datetime
    
    logger.info("Successfully imported required libraries")
except ImportError as e:
    logger.error(f"Failed to import required libraries: {e}")
    logger.error("Please install required packages: pip install mcp requests python-dotenv")
    sys.exit(1)

class BraveSearchMCPServer(MCPServer):
    """MCP server for web search using Brave Search API."""
    
    def __init__(self):
        """Initialize the Brave Search MCP server."""
        super().__init__()
        self.api_key = None
        self._load_api_key()
        self._register_tools()
        logger.info("Brave Search MCP server initialized")
    
    def _load_api_key(self):
        """Load Brave Search API key from environment variables."""
        # Load environment variables
        load_dotenv()
        
        # Get API key (try both naming conventions)
        self.api_key = os.getenv("BRAVE_API_KEY") or os.getenv("BRAVE_SEARCH_API_KEY")
        
        if not self.api_key:
            logger.warning("No Brave Search API key found in environment variables")
            logger.warning("Please set BRAVE_API_KEY in your .env file")
            logger.warning("You can get an API key from: https://brave.com/search/api/")
            logger.warning("For now, will attempt to use a limited implementation")
        else:
            logger.info("✓ Brave Search API key loaded successfully")
    
    def _register_tools(self):
        """Register search tools with the MCP server."""
        # Web search tool
        search_tool = ToolDefinition(
            name="web_search",
            description="Search the web for information on a topic",
            function=FunctionDefinition(
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "count": {
                            "type": "integer",
                            "description": "Number of results to return (max 20)",
                            "default": 5
                        },
                        "country": {
                            "type": "string",
                            "description": "Country code for localized results (e.g., 'us', 'gb', 'de')",
                            "default": "us"
                        }
                    },
                    "required": ["query"]
                }
            )
        )
        
        # News search tool
        news_tool = ToolDefinition(
            name="news_search",
            description="Search for recent news articles on a topic",
            function=FunctionDefinition(
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for news articles"
                        },
                        "count": {
                            "type": "integer",
                            "description": "Number of results to return (max 10)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            )
        )
        
        # Register all tools
        self.registry.register_tool(search_tool, self.web_search)
        self.registry.register_tool(news_tool, self.news_search)
        
        logger.info(f"Registered {len(self.registry.list_tools())} search tools")
    
    async def web_search(self, query: str, count: int = 5, country: str = "us") -> Dict[str, Any]:
        """Search the web for information on a topic.
        
        Args:
            query: Search query
            count: Number of results to return (max 20)
            country: Country code for localized results
            
        Returns:
            Dictionary containing search results
        """
        logger.info(f"Web search query: {query}")
        
        # Validate parameters
        if not query or len(query.strip()) < 2:
            return {"success": False, "error": "Search query is too short"}
        
        # Limit count to reasonable values
        count = min(max(1, count), 20)
        
        try:
            if not self.api_key:
                return await self._fallback_search(query, count)
            
            # Use Brave Search API
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self.api_key
            }
            
            params = {
                "q": query,
                "count": count,
                "country": country,
                "search_lang": "en"
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Process results
            results = []
            for item in data.get("web", {}).get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("description", ""),
                    "published": item.get("age", "")
                })
            
            return {
                "success": True,
                "query": query,
                "count": len(results),
                "results": results,
                "search_time": data.get("search_time", ""),
                "api": "brave_search"
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {str(e)}")
            # Fall back to alternative search if API fails
            return await self._fallback_search(query, count)
            
        except Exception as e:
            logger.error(f"Error during web search: {str(e)}")
            return {"success": False, "error": f"Search failed: {str(e)}"}
    
    async def news_search(self, query: str, count: int = 5) -> Dict[str, Any]:
        """Search for recent news articles on a topic.
        
        Args:
            query: Search query for news articles
            count: Number of results to return (max 10)
            
        Returns:
            Dictionary containing news search results
        """
        logger.info(f"News search query: {query}")
        
        # Validate parameters
        if not query or len(query.strip()) < 2:
            return {"success": False, "error": "Search query is too short"}
        
        # Limit count to reasonable values
        count = min(max(1, count), 10)
        
        try:
            if not self.api_key:
                return await self._fallback_news_search(query, count)
            
            # Use Brave Search API for news
            url = "https://api.search.brave.com/res/v1/news/search"
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self.api_key
            }
            
            params = {
                "q": query,
                "count": count,
                "search_lang": "en"
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Process results
            results = []
            for item in data.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("description", ""),
                    "published": item.get("age", ""),
                    "source": item.get("source", "")
                })
            
            return {
                "success": True,
                "query": query,
                "count": len(results),
                "results": results,
                "search_time": data.get("search_time", ""),
                "api": "brave_news"
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {str(e)}")
            # Fall back to alternative search if API fails
            return await self._fallback_news_search(query, count)
            
        except Exception as e:
            logger.error(f"Error during news search: {str(e)}")
            return {"success": False, "error": f"News search failed: {str(e)}"}
    
    async def _fallback_search(self, query: str, count: int) -> Dict[str, Any]:
        """Fall back to alternative search method if API is unavailable.
        
        This uses a public search API as fallback. Much more limited but works without API key.
        
        Args:
            query: Search query
            count: Number of results to return
            
        Returns:
            Dictionary containing search results
        """
        logger.info(f"Using fallback search for: {query}")
        
        try:
            # Use a public API as fallback
            url = f"https://ddg-api.herokuapp.com/search?query={query}&limit={count}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Process results
            results = []
            for item in data:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "description": item.get("snippet", ""),
                    "published": ""
                })
            
            return {
                "success": True,
                "query": query,
                "count": len(results),
                "results": results,
                "search_time": "",
                "api": "fallback_search",
                "note": "Using fallback search API. For better results, please set a Brave Search API key."
            }
            
        except Exception as e:
            logger.error(f"Fallback search failed: {str(e)}")
            return {
                "success": False,
                "error": f"Search failed: {str(e)}",
                "note": "To use search functionality properly, please set a Brave Search API key."
            }
    
    async def _fallback_news_search(self, query: str, count: int) -> Dict[str, Any]:
        """Fall back to alternative news search method if API is unavailable.
        
        Args:
            query: Search query
            count: Number of results to return
            
        Returns:
            Dictionary containing news search results
        """
        logger.info(f"Using fallback news search for: {query}")
        
        try:
            # Add "news" to the query to focus on news results
            news_query = f"{query} news latest"
            return await self._fallback_search(news_query, count)
            
        except Exception as e:
            logger.error(f"Fallback news search failed: {str(e)}")
            return {
                "success": False,
                "error": f"News search failed: {str(e)}",
                "note": "To use news search functionality properly, please set a Brave Search API key."
            }

async def main():
    """Run the Brave Search MCP server."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Brave Search MCP Server")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Create the server
    server = BraveSearchMCPServer()
    logger.info("Starting Brave Search MCP server...")
    
    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(server)))
    
    # Run the server
    await server.run()
    logger.info("Brave Search MCP server stopped")

async def shutdown(server):
    """Shut down the server gracefully."""
    logger.info("Shutting down Brave Search MCP server...")
    await server.shutdown()

if __name__ == "__main__":
    try:
        # Import signal module here to avoid issues on Windows
        import signal
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        sys.exit(1)
