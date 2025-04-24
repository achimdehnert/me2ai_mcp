"""
FireCrawl service for ME2AI MCP.

This module provides a microservice implementation of the FireCrawl
web content extraction service for ME2AI MCP.
"""

from typing import Dict, List, Any, Optional, Union
import logging
import asyncio
import json
import os
import tempfile
import subprocess
import sys
from pathlib import Path
import time
import uuid

# Import service components
from .web import WebService
from me2ai_mcp.services.base import ServiceStatus

try:
    import fastapi
    from fastapi import FastAPI, APIRouter, Depends, HTTPException, Request, Response, Body
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available. FireCrawl service will not function.")

# Constants
FIRECRAWL_REPO = "https://github.com/mendableai/firecrawl-mcp-server.git"
DEFAULT_PORT = 8787

# Configure logging
logger = logging.getLogger("me2ai-mcp-firecrawl-service")


class FireCrawlService(WebService):
    """
    Microservice for advanced web scraping using FireCrawl.
    
    This service provides browser-based web content extraction capabilities,
    including JavaScript rendering, for accurate extraction from modern websites.
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        version: str = "0.1.0",
        firecrawl_path: Optional[str] = None,
        auto_setup: bool = True,
        javascript_enabled: bool = True,
        default_wait_time: int = 5,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the FireCrawl service.
        
        Args:
            host: Host to bind the service to
            port: Port to bind the service to
            version: Service version
            firecrawl_path: Path to FireCrawl installation
            auto_setup: Whether to automatically set up FireCrawl
            javascript_enabled: Whether to enable JavaScript by default
            default_wait_time: Default time to wait for JavaScript rendering
            metadata: Additional service metadata
        """
        # Set up metadata
        metadata = metadata or {}
        metadata.update({
            "javascript_enabled": javascript_enabled,
            "default_wait_time": default_wait_time
        })
        
        # Initialize base web service
        super().__init__(
            name="firecrawl", 
            host=host, 
            port=port, 
            version=version,
            metadata=metadata,
            enable_cors=True,
            cors_origins=["*"],
            enable_docs=True
        )
        
        # Set up FireCrawl properties
        self.firecrawl_path = firecrawl_path
        self.auto_setup = auto_setup
        self.javascript_enabled = javascript_enabled
        self.default_wait_time = default_wait_time
        
        # Track child processes
        self.browser_instances = {}
        
        # Register service endpoints
        self._register_service_endpoints()
    
    def _register_service_endpoints(self) -> None:
        """Register FireCrawl service endpoints."""
        if not FASTAPI_AVAILABLE:
            self.logger.error("FastAPI is required for web services")
            return
            
        # Register scraping endpoint
        self.register_route(
            path="/scrape",
            method="POST",
            handler=self.handle_scrape,
            description="Scrape content from a URL using browser rendering"
        )
        
        # Register recursive scraping endpoint
        self.register_route(
            path="/crawl",
            method="POST",
            handler=self.handle_crawl,
            description="Recursively crawl a website starting from a URL"
        )
        
        # Register the status endpoint
        self.register_route(
            path="/status",
            method="GET",
            handler=self.handle_status,
            description="Get the status of the FireCrawl service"
        )
        
        # Register the screenshot endpoint
        self.register_route(
            path="/screenshot",
            method="POST",
            handler=self.handle_screenshot,
            description="Take a screenshot of a web page"
        )
    
    async def start(self) -> bool:
        """
        Start the FireCrawl service.
        
        Returns:
            bool: True if the service started successfully
        """
        # Set up FireCrawl if needed
        if self.auto_setup:
            setup_success = await self._setup_firecrawl()
            if not setup_success:
                self.logger.error("Failed to set up FireCrawl")
                self.status = ServiceStatus.ERROR
                return False
        
        # Start the base web service
        return await super().start()
    
    async def stop(self) -> bool:
        """
        Stop the FireCrawl service.
        
        Returns:
            bool: True if the service stopped successfully
        """
        # Clean up browser instances
        for browser_id, instance in self.browser_instances.items():
            try:
                process = instance.get("process")
                if process:
                    process.terminate()
                    self.logger.info(f"Terminated browser instance {browser_id}")
            except Exception as e:
                self.logger.warning(f"Error terminating browser instance {browser_id}: {str(e)}")
        
        # Stop the base web service
        return await super().stop()
    
    async def _setup_firecrawl(self) -> bool:
        """
        Set up the FireCrawl environment.
        
        Returns:
            bool: True if setup was successful
        """
        try:
            # Locate FireCrawl if path not provided
            if not self.firecrawl_path:
                self.firecrawl_path = self._find_firecrawl_path()
                
            # If still not found, clone the repository
            if not self.firecrawl_path:
                self.logger.info("FireCrawl not found, cloning repository")
                self.firecrawl_path = await self._clone_firecrawl()
                
            if not self.firecrawl_path:
                self.logger.error("Failed to locate or clone FireCrawl")
                return False
                
            # Install dependencies
            await self._install_dependencies()
                
            self.logger.info(f"FireCrawl set up successfully at {self.firecrawl_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up FireCrawl: {str(e)}")
            return False
    
    def _find_firecrawl_path(self) -> Optional[str]:
        """
        Find the FireCrawl repository path.
        
        Returns:
            str or None: Path to FireCrawl repository if found
        """
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
                self.logger.info(f"Found FireCrawl at {path}")
                return path
        
        # Not found
        return None
    
    async def _clone_firecrawl(self) -> Optional[str]:
        """
        Clone the FireCrawl repository.
        
        Returns:
            str or None: Path to cloned repository if successful
        """
        try:
            # Create a temp directory for the clone
            temp_dir = os.path.join(tempfile.gettempdir(), "firecrawl")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Clone the repository
            process = await asyncio.create_subprocess_exec(
                "git", "clone", FIRECRAWL_REPO, temp_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                self.logger.error(f"Failed to clone FireCrawl: {stderr.decode()}")
                return None
                
            self.logger.info(f"Successfully cloned FireCrawl to {temp_dir}")
            return temp_dir
            
        except Exception as e:
            self.logger.error(f"Error cloning FireCrawl: {str(e)}")
            return None
    
    async def _install_dependencies(self) -> bool:
        """
        Install FireCrawl dependencies.
        
        Returns:
            bool: True if dependencies were installed successfully
        """
        try:
            # Check for requirements.txt
            requirements_file = os.path.join(self.firecrawl_path, "requirements.txt")
            if not os.path.exists(requirements_file):
                self.logger.warning(f"No requirements.txt found at {requirements_file}")
                return False
                
            # Install dependencies
            self.logger.info("Installing FireCrawl dependencies")
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "pip", "install", "-r", requirements_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                self.logger.error(f"Failed to install dependencies: {stderr.decode()}")
                return False
                
            self.logger.info("Dependencies installed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error installing dependencies: {str(e)}")
            return False
    
    async def handle_scrape(
        self,
        request: Request,
        params: Dict[str, Any] = Body(...)
    ) -> Dict[str, Any]:
        """
        Handle a scrape request.
        
        Args:
            request: FastAPI request object
            params: Request parameters
            
        Returns:
            Dict[str, Any]: Scrape results
        """
        try:
            # Extract parameters
            url = params.get("url")
            if not url:
                raise HTTPException(status_code=400, detail="URL parameter is required")
                
            # Validate URL
            if not url.startswith(("http://", "https://")):
                raise HTTPException(status_code=400, detail=f"Invalid URL scheme: {url}")
                
            # Get optional parameters
            javascript_enabled = params.get("javascript_enabled", self.javascript_enabled)
            wait_time = params.get("wait_time", self.default_wait_time)
            timeout = params.get("timeout", 60)
            headers = params.get("headers", {})
            
            # Create a unique ID for this scrape
            scrape_id = str(uuid.uuid4())
            
            # Scrape the URL using playwright or puppeteer
            # This would typically call the FireCrawl library
            
            # For now, return a placeholder response
            # In a real implementation, this would use the FireCrawl library
            result = {
                "scrape_id": scrape_id,
                "url": url,
                "title": "Example Website",
                "content": "This is a placeholder for the scraped content.",
                "links": [
                    {"url": "https://example.com/page1", "text": "Page 1"},
                    {"url": "https://example.com/page2", "text": "Page 2"}
                ],
                "metadata": {
                    "javascript_enabled": javascript_enabled,
                    "wait_time": wait_time,
                    "timeout": timeout,
                    "headers": headers
                }
            }
            
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error handling scrape request: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error scraping URL: {str(e)}")
    
    async def handle_crawl(
        self,
        request: Request,
        params: Dict[str, Any] = Body(...)
    ) -> Dict[str, Any]:
        """
        Handle a crawl request.
        
        Args:
            request: FastAPI request object
            params: Request parameters
            
        Returns:
            Dict[str, Any]: Crawl results
        """
        try:
            # Extract parameters
            url = params.get("url")
            if not url:
                raise HTTPException(status_code=400, detail="URL parameter is required")
                
            # Validate URL
            if not url.startswith(("http://", "https://")):
                raise HTTPException(status_code=400, detail=f"Invalid URL scheme: {url}")
                
            # Get optional parameters
            max_depth = params.get("max_depth", 2)
            max_pages = params.get("max_pages", 20)
            javascript_enabled = params.get("javascript_enabled", self.javascript_enabled)
            wait_time = params.get("wait_time", self.default_wait_time)
            
            # Create a unique ID for this crawl
            crawl_id = str(uuid.uuid4())
            
            # For now, return a placeholder response
            # In a real implementation, this would use the FireCrawl library
            result = {
                "crawl_id": crawl_id,
                "url": url,
                "max_depth": max_depth,
                "max_pages": max_pages,
                "pages_crawled": 2,
                "results": [
                    {
                        "url": url,
                        "title": "Example Website",
                        "content": "This is a placeholder for the crawled content.",
                        "links": [
                            {"url": "https://example.com/page1", "text": "Page 1"},
                            {"url": "https://example.com/page2", "text": "Page 2"}
                        ]
                    },
                    {
                        "url": "https://example.com/page1",
                        "title": "Page 1",
                        "content": "This is a placeholder for page 1 content.",
                        "links": []
                    }
                ],
                "metadata": {
                    "javascript_enabled": javascript_enabled,
                    "wait_time": wait_time
                }
            }
            
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error handling crawl request: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error crawling URL: {str(e)}")
    
    async def handle_status(self, request: Request) -> Dict[str, Any]:
        """
        Handle a status request.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Dict[str, Any]: Service status
        """
        try:
            # Get base health check
            health = await self.health_check()
            
            # Add FireCrawl-specific information
            health.update({
                "firecrawl_path": self.firecrawl_path,
                "active_browsers": len(self.browser_instances),
                "javascript_enabled": self.javascript_enabled,
                "default_wait_time": self.default_wait_time
            })
            
            return health
            
        except Exception as e:
            self.logger.error(f"Error handling status request: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")
    
    async def handle_screenshot(
        self,
        request: Request,
        params: Dict[str, Any] = Body(...)
    ) -> Dict[str, Any]:
        """
        Handle a screenshot request.
        
        Args:
            request: FastAPI request object
            params: Request parameters
            
        Returns:
            Dict[str, Any]: Screenshot result
        """
        try:
            # Extract parameters
            url = params.get("url")
            if not url:
                raise HTTPException(status_code=400, detail="URL parameter is required")
                
            # Validate URL
            if not url.startswith(("http://", "https://")):
                raise HTTPException(status_code=400, detail=f"Invalid URL scheme: {url}")
                
            # Get optional parameters
            wait_time = params.get("wait_time", self.default_wait_time)
            full_page = params.get("full_page", False)
            
            # Create a unique ID for this screenshot
            screenshot_id = str(uuid.uuid4())
            
            # For now, return a placeholder response
            # In a real implementation, this would use the FireCrawl library
            result = {
                "screenshot_id": screenshot_id,
                "url": url,
                "timestamp": time.time(),
                "image_format": "base64",
                "image_data": "placeholder_for_base64_encoded_image",
                "metadata": {
                    "wait_time": wait_time,
                    "full_page": full_page
                }
            }
            
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error handling screenshot request: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error taking screenshot: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check for the service.
        
        Returns:
            Dict[str, Any]: Health check information
        """
        # Get base health check
        health = await super().health_check()
        
        # Add FireCrawl-specific checks
        health["firecrawl_available"] = self.firecrawl_path is not None
        
        return health
