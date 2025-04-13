"""
Web-related tools for ME2AI MCP servers.

This module provides common tools for web content fetching, scraping,
and processing that can be used across different MCP servers.
"""
from typing import Dict, List, Any, Optional
import logging
import re
import urllib.parse
from dataclasses import dataclass
import requests
from ..base import BaseTool

# Optional dependencies
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logging.warning("BeautifulSoup not available, some web tools will have limited functionality")

# Configure logging
logger = logging.getLogger("me2ai-mcp-tools-web")


@dataclass
class WebFetchTool(BaseTool):
    """Tool for fetching web content."""
    
    name: str = "fetch_webpage"
    description: str = "Fetch content from a web page"
    user_agent: str = "ME2AI Web Fetcher/1.0"
    timeout: int = 30
    max_content_length: int = 1024 * 1024  # 1MB
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch a webpage and return its content.
        
        Args:
            params: Dictionary containing:
                - url: URL to fetch
                - headers: Optional additional HTTP headers
                - timeout: Optional custom timeout in seconds
        
        Returns:
            Dictionary containing fetch results
        """
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
            
        # Prepare headers
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }
        
        # Add custom headers if provided
        if "headers" in params and isinstance(params["headers"], dict):
            headers.update(params["headers"])
            
        # Get timeout
        timeout = params.get("timeout", self.timeout)
        
        try:
            # Fetch the URL
            response = requests.get(
                url,
                headers=headers,
                timeout=timeout,
                stream=True  # Use streaming to handle large responses
            )
            
            # Check status code
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get("Content-Type", "").lower()
            if not any(ct in content_type for ct in ["text/html", "text/plain", "application/json", "application/xml"]):
                return {
                    "success": False,
                    "error": f"Unsupported content type: {content_type}"
                }
                
            # Check content length
            content_length = int(response.headers.get("Content-Length", 0))
            if content_length > self.max_content_length:
                return {
                    "success": False,
                    "error": f"Content too large: {content_length} bytes (max {self.max_content_length})"
                }
                
            # Get content (with reasonable size limit)
            content = response.text
            
            # Extract basic info
            title = ""
            if BS4_AVAILABLE and "text/html" in content_type:
                soup = BeautifulSoup(content, "html.parser")
                title_tag = soup.find("title")
                if title_tag:
                    title = title_tag.string
                    
            # Return results
            return {
                "success": True,
                "url": url,
                "status_code": response.status_code,
                "content_type": content_type,
                "content_length": len(content),
                "title": title,
                "content": content,
                "headers": dict(response.headers)
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
                "error": f"Error fetching webpage: {str(e)}",
                "exception_type": type(e).__name__
            }


@dataclass
class HTMLParserTool(BaseTool):
    """Tool for parsing and extracting information from HTML content."""
    
    name: str = "parse_html"
    description: str = "Parse and extract structured data from HTML content"
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse HTML and extract structured information.
        
        Args:
            params: Dictionary containing:
                - html: HTML content to parse
                - selectors: Optional dictionary of CSS selectors to extract
                - extract_metadata: Whether to extract metadata (default: True)
                - extract_text: Whether to extract main text (default: True)
        
        Returns:
            Dictionary containing parse results
        """
        if not BS4_AVAILABLE:
            return {
                "success": False,
                "error": "BeautifulSoup is not available"
            }
            
        html = params.get("html")
        if not html:
            return {
                "success": False,
                "error": "HTML parameter is required"
            }
            
        selectors = params.get("selectors", {})
        extract_metadata = params.get("extract_metadata", True)
        extract_text = params.get("extract_text", True)
        
        try:
            # Parse HTML
            soup = BeautifulSoup(html, "html.parser")
            
            result = {
                "success": True,
            }
            
            # Extract metadata if requested
            if extract_metadata:
                metadata = {}
                
                # Title
                title_tag = soup.find("title")
                if title_tag:
                    metadata["title"] = title_tag.string
                    
                # Meta tags
                meta_tags = {}
                for meta in soup.find_all("meta"):
                    name = meta.get("name") or meta.get("property")
                    content = meta.get("content")
                    if name and content:
                        meta_tags[name] = content
                metadata["meta_tags"] = meta_tags
                
                result["metadata"] = metadata
            
            # Extract text if requested
            if extract_text:
                # Extract main content text (remove scripts, styles, etc.)
                for tag in soup(["script", "style", "noscript", "iframe"]):
                    tag.extract()
                    
                text = soup.get_text(separator="\n", strip=True)
                result["text"] = text
                
                # Extract headings
                headings = []
                for level in range(1, 7):
                    for h in soup.find_all(f"h{level}"):
                        headings.append({
                            "level": level,
                            "text": h.get_text(strip=True)
                        })
                result["headings"] = headings
            
            # Extract data using provided selectors
            if selectors:
                extracted = {}
                for name, selector in selectors.items():
                    if isinstance(selector, str):
                        # Single element
                        element = soup.select_one(selector)
                        if element:
                            extracted[name] = element.get_text(strip=True)
                    elif isinstance(selector, dict) and "selector" in selector:
                        # Advanced configuration
                        elements = soup.select(selector["selector"])
                        
                        if "attribute" in selector:
                            # Extract attribute value
                            attr_name = selector["attribute"]
                            values = [el.get(attr_name) for el in elements if el.get(attr_name)]
                        else:
                            # Extract text
                            values = [el.get_text(strip=True) for el in elements]
                            
                        if "multiple" in selector and selector["multiple"]:
                            extracted[name] = values
                        elif values:
                            extracted[name] = values[0]
                            
                result["extracted"] = extracted
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error parsing HTML: {str(e)}",
                "exception_type": type(e).__name__
            }


@dataclass
class URLUtilsTool(BaseTool):
    """Tool for URL manipulation and processing."""
    
    name: str = "url_utils"
    description: str = "Utilities for URL manipulation and processing"
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process and manipulate URLs.
        
        Args:
            params: Dictionary containing:
                - url: URL to process
                - operation: Operation to perform (parse, join, normalize)
                - base_url: Base URL for join operation
                - path: Path to join with base URL
        
        Returns:
            Dictionary containing operation results
        """
        url = params.get("url")
        operation = params.get("operation", "parse")
        
        try:
            if operation == "parse":
                if not url:
                    return {
                        "success": False,
                        "error": "URL parameter is required for parse operation"
                    }
                
                # Parse URL
                parsed = urllib.parse.urlparse(url)
                
                return {
                    "success": True,
                    "url": url,
                    "parsed": {
                        "scheme": parsed.scheme,
                        "netloc": parsed.netloc,
                        "path": parsed.path,
                        "params": parsed.params,
                        "query": parsed.query,
                        "fragment": parsed.fragment,
                        "username": parsed.username,
                        "password": parsed.password,
                        "hostname": parsed.hostname,
                        "port": parsed.port
                    },
                    "query_params": dict(urllib.parse.parse_qsl(parsed.query))
                }
                
            elif operation == "join":
                base_url = params.get("base_url")
                path = params.get("path")
                
                if not base_url or not path:
                    return {
                        "success": False,
                        "error": "base_url and path parameters are required for join operation"
                    }
                
                # Join URLs
                joined_url = urllib.parse.urljoin(base_url, path)
                
                return {
                    "success": True,
                    "base_url": base_url,
                    "path": path,
                    "joined_url": joined_url
                }
                
            elif operation == "normalize":
                if not url:
                    return {
                        "success": False,
                        "error": "URL parameter is required for normalize operation"
                    }
                
                # Normalize URL
                normalized_url = urllib.parse.urljoin(url, urllib.parse.urlparse(url).path)
                
                return {
                    "success": True,
                    "original_url": url,
                    "normalized_url": normalized_url
                }
                
            else:
                return {
                    "success": False,
                    "error": f"Unknown operation: {operation}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Error processing URL: {str(e)}",
                "exception_type": type(e).__name__
            }
