"""Tools for interacting with MCP servers.

This module provides tool classes for interacting with Model Context Protocol
servers, allowing agents to access web content and search functionality.
"""
from typing import Dict, Any, List, Optional
import json
import logging
import requests
from .base import BaseTool

logger = logging.getLogger(__name__)

class FetchWebpageTool(BaseTool):
    """Tool for fetching and processing web content using the Fetch MCP server."""
    
    def __init__(self):
        """Initialize the FetchWebpageTool."""
        super().__init__(
            name="fetch_webpage",
            description="Fetch and convert web content to a format optimized for language models",
            return_direct=True
        )
    
    async def _run(self, url: str, max_length: int = 10000) -> str:
        """Execute the fetch_webpage tool.
        
        Args:
            url: URL of the webpage to fetch
            max_length: Maximum length of content to return (characters)
            
        Returns:
            Formatted string containing the processed web content
        """
        try:
            # Execute the MCP tool via the Windsurf socket
            headers = {"Content-Type": "application/json"}
            payload = {
                "serverName": "fetch",
                "toolName": "fetch_webpage",
                "params": {
                    "url": url,
                    "max_length": max_length
                }
            }
            
            response = requests.post(
                "http://localhost:3000/mcp/execute",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            # Check if the request was successful
            if result.get("success") is True:
                content = result.get("content", "")
                title = result.get("title", "")
                description = result.get("description", "")
                
                # Format the result
                formatted_result = f"## {title}\n\n"
                if description:
                    formatted_result += f"*{description}*\n\n"
                formatted_result += f"Source: {url}\n\n"
                formatted_result += "### Content:\n\n"
                formatted_result += content
                
                return formatted_result
            else:
                error = result.get("error", "Unknown error")
                return f"Error fetching webpage: {error}"
                
        except Exception as e:
            logger.error(f"Error executing fetch_webpage tool: {str(e)}")
            return f"Error: Could not fetch the webpage. {str(e)}"

    def _arun(self, url: str, max_length: int = 10000) -> str:
        """Async version of the execute method."""
        raise NotImplementedError("This tool does not support async execution")


class WebSearchTool(BaseTool):
    """Tool for searching the web using the Brave Search MCP server."""
    
    def __init__(self):
        """Initialize the WebSearchTool."""
        super().__init__(
            name="web_search",
            description="Search the web for information on a topic",
            return_direct=True
        )
    
    async def _run(self, query: str, count: int = 5, country: str = "us") -> str:
        """Execute the web_search tool.
        
        Args:
            query: Search query
            count: Number of results to return (max 20)
            country: Country code for localized results
            
        Returns:
            Formatted string containing search results
        """
        try:
            # Execute the MCP tool via the Windsurf socket
            headers = {"Content-Type": "application/json"}
            payload = {
                "serverName": "brave-search",
                "toolName": "web_search",
                "params": {
                    "query": query,
                    "count": count,
                    "country": country
                }
            }
            
            response = requests.post(
                "http://localhost:3000/mcp/execute",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            # Check if the request was successful
            if result.get("success") is True:
                results = result.get("results", [])
                
                if not results:
                    return f"No results found for query: {query}"
                
                # Format the results
                formatted_result = f"## Web Search Results for: {query}\n\n"
                
                for idx, item in enumerate(results, 1):
                    title = item.get("title", "No title")
                    url = item.get("url", "")
                    description = item.get("description", "No description available")
                    published = item.get("published", "")
                    
                    formatted_result += f"### {idx}. {title}\n"
                    formatted_result += f"{description}\n"
                    formatted_result += f"URL: {url}\n"
                    if published:
                        formatted_result += f"Published: {published}\n"
                    formatted_result += "\n"
                
                note = result.get("note", "")
                if note:
                    formatted_result += f"Note: {note}\n"
                
                return formatted_result
            else:
                error = result.get("error", "Unknown error")
                return f"Error performing web search: {error}"
                
        except Exception as e:
            logger.error(f"Error executing web_search tool: {str(e)}")
            return f"Error: Could not perform web search. {str(e)}"

    def _arun(self, query: str, count: int = 5, country: str = "us") -> str:
        """Async version of the execute method."""
        raise NotImplementedError("This tool does not support async execution")


class NewsSearchTool(BaseTool):
    """Tool for searching news articles using the Brave Search MCP server."""
    
    def __init__(self):
        """Initialize the NewsSearchTool."""
        super().__init__(
            name="news_search",
            description="Search for recent news articles on a topic",
            return_direct=True
        )
    
    async def _run(self, query: str, count: int = 5) -> str:
        """Execute the news_search tool.
        
        Args:
            query: Search query for news articles
            count: Number of results to return (max 10)
            
        Returns:
            Formatted string containing news search results
        """
        try:
            # Execute the MCP tool via the Windsurf socket
            headers = {"Content-Type": "application/json"}
            payload = {
                "serverName": "brave-search",
                "toolName": "news_search",
                "params": {
                    "query": query,
                    "count": count
                }
            }
            
            response = requests.post(
                "http://localhost:3000/mcp/execute",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            # Check if the request was successful
            if result.get("success") is True:
                results = result.get("results", [])
                
                if not results:
                    return f"No news results found for query: {query}"
                
                # Format the results
                formatted_result = f"## News Search Results for: {query}\n\n"
                
                for idx, item in enumerate(results, 1):
                    title = item.get("title", "No title")
                    url = item.get("url", "")
                    description = item.get("description", "No description available")
                    published = item.get("published", "")
                    source = item.get("source", "")
                    
                    formatted_result += f"### {idx}. {title}\n"
                    formatted_result += f"{description}\n"
                    if source:
                        formatted_result += f"Source: {source}\n"
                    formatted_result += f"URL: {url}\n"
                    if published:
                        formatted_result += f"Published: {published}\n"
                    formatted_result += "\n"
                
                note = result.get("note", "")
                if note:
                    formatted_result += f"Note: {note}\n"
                
                return formatted_result
            else:
                error = result.get("error", "Unknown error")
                return f"Error performing news search: {error}"
                
        except Exception as e:
            logger.error(f"Error executing news_search tool: {str(e)}")
            return f"Error: Could not perform news search. {str(e)}"

    def _arun(self, query: str, count: int = 5) -> str:
        """Async version of the execute method."""
        raise NotImplementedError("This tool does not support async execution")


class ExtractElementsTool(BaseTool):
    """Tool for extracting specific elements from a webpage using the Fetch MCP server."""
    
    def __init__(self):
        """Initialize the ExtractElementsTool."""
        super().__init__(
            name="extract_elements",
            description="Extract specific elements from a webpage using CSS selectors",
            return_direct=True
        )
    
    async def _run(self, url: str, css_selector: str) -> str:
        """Execute the extract_elements tool.
        
        Args:
            url: URL of the webpage to fetch
            css_selector: CSS selector to extract specific elements
            
        Returns:
            Formatted string containing the extracted elements
        """
        try:
            # Execute the MCP tool via the Windsurf socket
            headers = {"Content-Type": "application/json"}
            payload = {
                "serverName": "fetch",
                "toolName": "extract_elements",
                "params": {
                    "url": url,
                    "css_selector": css_selector
                }
            }
            
            response = requests.post(
                "http://localhost:3000/mcp/execute",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            # Check if the request was successful
            if result.get("success") is True:
                elements = result.get("elements", [])
                count = result.get("count", 0)
                truncated = result.get("truncated", False)
                
                if not elements:
                    return f"No elements found matching selector '{css_selector}' at {url}"
                
                # Format the results
                formatted_result = f"## Extracted Elements from {url}\n\n"
                formatted_result += f"Selector: `{css_selector}`\n"
                formatted_result += f"Total matches: {count}\n\n"
                
                for idx, element in enumerate(elements, 1):
                    text = element.get("text", "")
                    tag = element.get("tag", "")
                    attrs = element.get("attributes", {})
                    
                    formatted_result += f"### Element {idx} <{tag}>\n"
                    formatted_result += f"```\n{text}\n```\n"
                    if attrs:
                        formatted_result += "Attributes:\n"
                        for attr_name, attr_value in attrs.items():
                            formatted_result += f"- {attr_name}: {attr_value}\n"
                    formatted_result += "\n"
                
                if truncated:
                    formatted_result += "(Results truncated, showing first 20 matches)\n"
                
                return formatted_result
            else:
                error = result.get("error", "Unknown error")
                return f"Error extracting elements: {error}"
                
        except Exception as e:
            logger.error(f"Error executing extract_elements tool: {str(e)}")
            return f"Error: Could not extract elements from the webpage. {str(e)}"

    def _arun(self, url: str, css_selector: str) -> str:
        """Async version of the execute method."""
        raise NotImplementedError("This tool does not support async execution")


class SummarizeWebpageTool(BaseTool):
    """Tool for summarizing a webpage using the Fetch MCP server."""
    
    def __init__(self):
        """Initialize the SummarizeWebpageTool."""
        super().__init__(
            name="summarize_webpage",
            description="Fetch a webpage and extract the main content in a summarized format",
            return_direct=True
        )
    
    async def _run(self, url: str) -> str:
        """Execute the summarize_webpage tool.
        
        Args:
            url: URL of the webpage to summarize
            
        Returns:
            Formatted string containing the summarized web content
        """
        try:
            # Execute the MCP tool via the Windsurf socket
            headers = {"Content-Type": "application/json"}
            payload = {
                "serverName": "fetch",
                "toolName": "summarize_webpage",
                "params": {
                    "url": url
                }
            }
            
            response = requests.post(
                "http://localhost:3000/mcp/execute",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            # Check if the request was successful
            if result.get("success") is True:
                title = result.get("title", "")
                description = result.get("description", "")
                headings = result.get("headings", [])
                paragraphs = result.get("main_paragraphs", [])
                links = result.get("important_links", [])
                
                # Format the results
                formatted_result = f"## Summary of {url}\n\n"
                
                if title:
                    formatted_result += f"### Title\n{title}\n\n"
                
                if description:
                    formatted_result += f"### Description\n{description}\n\n"
                
                if headings:
                    formatted_result += "### Main Headings\n"
                    for heading in headings:
                        level = heading.get("level", 1)
                        text = heading.get("text", "")
                        indent = "  " * (level - 1)
                        formatted_result += f"{indent}- {text}\n"
                    formatted_result += "\n"
                
                if paragraphs:
                    formatted_result += "### Key Content\n"
                    for idx, paragraph in enumerate(paragraphs, 1):
                        formatted_result += f"{idx}. {paragraph}\n\n"
                
                if links:
                    formatted_result += "### Important Links\n"
                    for link in links:
                        text = link.get("text", "")
                        url = link.get("url", "")
                        formatted_result += f"- [{text}]({url})\n"
                
                return formatted_result
            else:
                error = result.get("error", "Unknown error")
                return f"Error summarizing webpage: {error}"
                
        except Exception as e:
            logger.error(f"Error executing summarize_webpage tool: {str(e)}")
            return f"Error: Could not summarize the webpage. {str(e)}"

    def _arun(self, url: str) -> str:
        """Async version of the execute method."""
        raise NotImplementedError("This tool does not support async execution")
