"""
Example ME2AI MCP server implementation.

This example demonstrates how to create a custom MCP server using the ME2AI MCP framework.
It shows basic setup, tool implementation, and advanced features like authentication and statistics.
"""
import os
import sys
import logging
import asyncio
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("example-mcp")

# Import ME2AI MCP framework
from me2ai_mcp import ME2AIMCPServer, register_tool
from me2ai_mcp.auth import AuthManager, APIKeyAuth
from me2ai_mcp.tools.web import WebFetchTool
from me2ai_mcp.tools.filesystem import FileReaderTool


class ExampleMCPServer(ME2AIMCPServer):
    """Example custom MCP server using the ME2AI MCP framework."""
    
    def __init__(self):
        """Initialize the Example MCP server."""
        super().__init__(
            server_name="example-mcp",
            description="Example MCP server with custom tools",
            version="1.0.0"
        )
        
        # Set up server-wide tools
        self.web_fetch_tool = WebFetchTool(
            user_agent="Example MCP Web Fetcher/1.0",
            timeout=30
        )
        self.file_reader_tool = FileReaderTool(
            max_file_size=1024 * 1024 * 2  # 2MB
        )
        
        # Set up custom authentication (optional)
        self._setup_auth()
        
        logger.info("Example MCP server initialized")
        
    def _setup_auth(self):
        """Set up authentication for the server."""
        # Example of API key auth
        api_key = os.environ.get("EXAMPLE_API_KEY")
        if api_key:
            self.auth = AuthManager()
            self.auth.add_provider(APIKeyAuth(api_key))
            logger.info("Authentication configured with API key")
        else:
            logger.warning("No API key found, running without authentication")
    
    @register_tool
    async def echo(self, message: str) -> Dict[str, Any]:
        """Simple echo tool that returns the provided message.
        
        Args:
            message: The message to echo back
            
        Returns:
            Dictionary containing the echoed message
        """
        logger.info(f"Echo called with message: {message}")
        return {
            "message": message,
            "timestamp": self.get_timestamp()
        }
    
    @register_tool
    async def fetch_and_summarize(self, url: str, max_length: int = 500) -> Dict[str, Any]:
        """Fetch a webpage and return a simplified summary.
        
        Args:
            url: URL of the webpage to fetch and summarize
            max_length: Maximum length of the summary
            
        Returns:
            Dictionary containing the summary and metadata
        """
        logger.info(f"Fetching and summarizing: {url}")
        
        # Use the built-in web fetch tool
        fetch_result = await self.web_fetch_tool.execute({
            "url": url
        })
        
        if not fetch_result.get("success", False):
            # Return the error from the web fetch tool
            return fetch_result
        
        # Extract title and content
        title = fetch_result.get("title", "No title")
        content = fetch_result.get("content", "")
        
        # Create a simple summary (in a real implementation, this would be more sophisticated)
        if content:
            # Extract first paragraph as a very basic summary
            paragraphs = content.split("\n\n")
            summary = paragraphs[0][:max_length]
            if len(summary) >= max_length:
                summary += "..."
        else:
            summary = "No content available to summarize"
        
        return {
            "success": True,
            "url": url,
            "title": title,
            "summary": summary,
            "full_content_available": bool(content),
            "content_length": len(content) if content else 0,
            "stats": {
                "fetch_time": fetch_result.get("stats", {}).get("execution_time", 0)
            }
        }
    
    @register_tool
    async def read_local_file(self, file_path: str, preview_only: bool = True) -> Dict[str, Any]:
        """Read a local file and return its contents.
        
        Args:
            file_path: Path to the file to read
            preview_only: If true, only return a preview of the file
            
        Returns:
            Dictionary containing the file content and metadata
        """
        logger.info(f"Reading local file: {file_path}")
        
        # Use the built-in file reader tool
        read_result = await self.file_reader_tool.execute({
            "file_path": file_path
        })
        
        if not read_result.get("success", False):
            # Return the error from the file reader tool
            return read_result
        
        content = read_result.get("content", "")
        
        # If preview_only is true, truncate the content
        if preview_only and content and len(content) > 1000:
            content = content[:1000] + "...\n[Content truncated, set preview_only=False to view full content]"
        
        return {
            "success": True,
            "file_path": read_result.get("file_path"),
            "content": content,
            "size": read_result.get("size"),
            "metadata": read_result.get("metadata")
        }
    
    @register_tool
    async def combine_data(self, url: Optional[str] = None, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Combine data from a web URL and a local file.
        
        This tool demonstrates how to combine multiple tools into a higher-level operation.
        
        Args:
            url: URL to fetch data from (optional)
            file_path: Path to a local file (optional)
            
        Returns:
            Dictionary containing combined data
        """
        logger.info(f"Combining data from URL: {url} and file: {file_path}")
        
        results = {}
        
        # Fetch URL if provided
        if url:
            web_result = await self.fetch_and_summarize(url)
            if web_result.get("success", False):
                results["web"] = {
                    "url": url,
                    "title": web_result.get("title"),
                    "summary": web_result.get("summary")
                }
            else:
                results["web"] = {
                    "error": web_result.get("error", "Unknown error fetching URL")
                }
        
        # Read file if provided
        if file_path:
            file_result = await self.read_local_file(file_path, preview_only=True)
            if file_result.get("success", False):
                results["file"] = {
                    "file_path": file_path,
                    "preview": file_result.get("content")[:200] + "..." if file_result.get("content", "") else "",
                    "size": file_result.get("size")
                }
            else:
                results["file"] = {
                    "error": file_result.get("error", "Unknown error reading file")
                }
        
        return {
            "success": True,
            "results": results,
            "sources_count": len(results)
        }


async def main():
    """Run the Example MCP server."""
    server = ExampleMCPServer()
    await server.start()


if __name__ == "__main__":
    try:
        # Run the server
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
