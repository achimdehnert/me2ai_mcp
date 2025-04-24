"""
FireCrawl integration for ME2AI MCP servers.

This module provides advanced web scraping capabilities through the FireCrawl library,
which uses browser rendering for JavaScript-heavy websites and comprehensive content extraction.
"""

from typing import Dict, List, Any, Optional, Union
import logging
import json
import os
import asyncio
import urllib.parse
from dataclasses import dataclass, field
import tempfile
import subprocess
import sys
from pathlib import Path

from ..base import BaseTool

# Configure logging
logger = logging.getLogger("me2ai-mcp-tools-firecrawl")

# Constants
FIRECRAWL_REPO = "https://github.com/mendableai/firecrawl-mcp-server.git"
DEFAULT_PORT = 8787


@dataclass
class FireCrawlTool(BaseTool):
    """
    Advanced web scraping tool using FireCrawl browser rendering.
    
    This tool provides robust web content extraction capabilities for complex websites,
    including those with JavaScript rendering, single-page applications, and dynamic content.
    """
    
    name: str = "web_content"
    description: str = "Extract web content using full browser rendering"
    server_host: str = "localhost"
    server_port: int = DEFAULT_PORT
    server_url: str = field(init=False)
    server_process: Optional[subprocess.Popen] = field(default=None, init=False, repr=False)
    firecrawl_path: Optional[str] = None
    auto_start_server: bool = True
    timeout: int = 60
    max_wait_time: int = 30
    javascript_enabled: bool = True
    recursive: bool = False
    max_depth: int = 2
    max_pages: int = 20
    
    def __post_init__(self):
        """Initialize the server URL and check FireCrawl availability."""
        self.server_url = f"http://{self.server_host}:{self.server_port}"
        
        # Locate FireCrawl if path not provided
        if not self.firecrawl_path:
            self.firecrawl_path = self._find_firecrawl_path()
            
        # Start the server if requested
        if self.auto_start_server:
            self._ensure_server_running()
    
    def _find_firecrawl_path(self) -> Optional[str]:
        """Find the FireCrawl repository path."""
        # Check common locations
        possible_paths = [
            # Relative to the current file
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "firecrawl"),
            # In the parent directory
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "firecrawl"),
            # In the user's home directory
            os.path.join(os.path.expanduser("~"), "firecrawl"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.isdir(path):
                logger.info(f"Found FireCrawl at {path}")
                return path
        
        # Not found, will need to clone
        return None
    
    def _ensure_server_running(self) -> bool:
        """Ensure the FireCrawl server is running, starting it if necessary."""
        try:
            # Check if server is already running
            import requests
            try:
                response = requests.get(f"{self.server_url}/health", timeout=2)
                if response.status_code == 200:
                    logger.info("FireCrawl server is already running")
                    return True
            except requests.RequestException:
                logger.info("FireCrawl server not detected, will start")
                
            # Server not running, need to start it
            if not self.firecrawl_path or not os.path.exists(self.firecrawl_path):
                logger.info("FireCrawl not found, cloning repository")
                self.firecrawl_path = self._clone_firecrawl()
                
            if not self.firecrawl_path:
                logger.error("Failed to locate or clone FireCrawl")
                return False
                
            # Start the server
            return self._start_server()
            
        except Exception as e:
            logger.error(f"Error ensuring FireCrawl server: {str(e)}")
            return False
    
    def _clone_firecrawl(self) -> Optional[str]:
        """Clone the FireCrawl repository."""
        try:
            # Create a temp directory for the clone
            temp_dir = os.path.join(tempfile.gettempdir(), "firecrawl")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Clone the repository
            import subprocess
            result = subprocess.run(
                ["git", "clone", FIRECRAWL_REPO, temp_dir],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to clone FireCrawl: {result.stderr}")
                return None
                
            logger.info(f"Successfully cloned FireCrawl to {temp_dir}")
            return temp_dir
            
        except Exception as e:
            logger.error(f"Error cloning FireCrawl: {str(e)}")
            return None
    
    def _start_server(self) -> bool:
        """Start the FireCrawl server."""
        try:
            # Ensure we're in the FireCrawl directory
            if not os.path.exists(self.firecrawl_path):
                logger.error(f"FireCrawl path does not exist: {self.firecrawl_path}")
                return False
                
            # Install dependencies if needed
            self._install_dependencies()
            
            # Start the server
            server_script = os.path.join(self.firecrawl_path, "server.py")
            if not os.path.exists(server_script):
                server_script = os.path.join(self.firecrawl_path, "app.py")
                
            if not os.path.exists(server_script):
                logger.error(f"FireCrawl server script not found in {self.firecrawl_path}")
                return False
                
            # Build the command
            cmd = [
                sys.executable,
                server_script,
                "--port", str(self.server_port),
                "--host", self.server_host
            ]
            
            # Start the server as a background process
            logger.info(f"Starting FireCrawl server: {' '.join(cmd)}")
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.firecrawl_path
            )
            
            # Wait for the server to start
            import time
            import requests
            start_time = time.time()
            while time.time() - start_time < 30:  # 30 second timeout
                try:
                    response = requests.get(f"{self.server_url}/health", timeout=2)
                    if response.status_code == 200:
                        logger.info("FireCrawl server started successfully")
                        return True
                except requests.RequestException:
                    pass
                    
                time.sleep(1)
                
            logger.error("Timed out waiting for FireCrawl server to start")
            return False
            
        except Exception as e:
            logger.error(f"Error starting FireCrawl server: {str(e)}")
            return False
    
    def _install_dependencies(self) -> None:
        """Install FireCrawl dependencies if needed."""
        try:
            # Check for requirements.txt
            requirements_file = os.path.join(self.firecrawl_path, "requirements.txt")
            if not os.path.exists(requirements_file):
                logger.warning(f"No requirements.txt found at {requirements_file}")
                return
                
            # Install dependencies
            logger.info("Installing FireCrawl dependencies")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", requirements_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            
        except Exception as e:
            logger.error(f"Error installing FireCrawl dependencies: {str(e)}")
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract content from a web page using FireCrawl.
        
        Args:
            params: Dictionary containing:
                - url: URL to scrape
                - javascript_enabled: Whether to enable JavaScript (default: True)
                - wait_time: Time to wait for JavaScript rendering in seconds (default: 5)
                - recursive: Whether to crawl the site recursively (default: False)
                - max_depth: Maximum crawl depth for recursive mode (default: 2)
                - max_pages: Maximum number of pages to crawl (default: 20)
                - headers: Optional custom HTTP headers
                - cookies: Optional cookies to include with the request
                - selectors: Optional CSS selectors to extract specific content
        
        Returns:
            Dictionary containing the extracted content and metadata
        """
        # Get URL parameter
        url = params.get("url")
        if not url:
            return {
                "success": False,
                "error": "URL parameter is required"
            }
            
        # Validate URL
        if not url.startswith(("http://", "https://")):
            return {
                "success": False,
                "error": f"Invalid URL scheme: {url}"
            }
            
        # Ensure server is running
        if not self._ensure_server_running():
            return {
                "success": False,
                "error": "Failed to start FireCrawl server"
            }
            
        # Prepare request parameters
        request_params = {
            "url": url,
            "javascript_enabled": params.get("javascript_enabled", self.javascript_enabled),
            "wait_time": params.get("wait_time", self.max_wait_time),
            "timeout": params.get("timeout", self.timeout),
        }
        
        # Add recursive crawling parameters if needed
        if params.get("recursive", self.recursive):
            request_params["recursive"] = True
            request_params["max_depth"] = params.get("max_depth", self.max_depth)
            request_params["max_pages"] = params.get("max_pages", self.max_pages)
            
        # Add custom headers if provided
        if "headers" in params and isinstance(params["headers"], dict):
            request_params["headers"] = params["headers"]
            
        # Add cookies if provided
        if "cookies" in params and isinstance(params["cookies"], dict):
            request_params["cookies"] = params["cookies"]
            
        # Add selectors if provided
        if "selectors" in params and isinstance(params["selectors"], (dict, list)):
            request_params["selectors"] = params["selectors"]
            
        try:
            # Make request to FireCrawl server
            import requests
            response = requests.post(
                f"{self.server_url}/scrape",
                json=request_params,
                timeout=self.timeout
            )
            
            # Check for errors
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    error_message = error_data.get("error", f"HTTP error: {response.status_code}")
                except:
                    error_message = f"HTTP error: {response.status_code}"
                    
                return {
                    "success": False,
                    "error": error_message,
                    "status_code": response.status_code
                }
                
            # Parse response
            try:
                result = response.json()
                
                # Add success flag
                result["success"] = True
                
                return result
                
            except json.JSONDecodeError:
                return {
                    "success": False,
                    "error": "Invalid JSON response from FireCrawl server",
                    "raw_content": response.text[:1000]  # Include start of raw response for debugging
                }
                
        except requests.RequestException as e:
            return {
                "success": False,
                "error": f"Request error: {str(e)}",
                "exception_type": type(e).__name__
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error scraping webpage: {str(e)}",
                "exception_type": type(e).__name__
            }
    
    def __del__(self):
        """Clean up server process if we started it."""
        if self.server_process:
            try:
                self.server_process.terminate()
                logger.info("FireCrawl server terminated")
            except:
                pass


@dataclass
class WebContentTool(FireCrawlTool):
    """
    Tool for extracting content from a web page.
    
    Alias for FireCrawlTool with a more descriptive name.
    """
    
    name: str = "extract_web_content"
    description: str = "Extract content from a webpage with advanced rendering capabilities"


# Utility functions
def create_firecrawl_tool(
    server_host: str = "localhost",
    server_port: int = DEFAULT_PORT,
    auto_start: bool = True
) -> FireCrawlTool:
    """
    Create a FireCrawl tool with the specified configuration.
    
    Args:
        server_host: Host where the FireCrawl server is running
        server_port: Port where the FireCrawl server is listening
        auto_start: Whether to automatically start the server if not running
        
    Returns:
        Configured FireCrawlTool instance
    """
    return FireCrawlTool(
        server_host=server_host,
        server_port=server_port,
        auto_start_server=auto_start
    )
