"""
Agent abstractions for ME2AI MCP.

This module provides base classes and utilities for creating
standardized agents that can interact with MCP tools.
"""
from typing import Dict, List, Any, Optional, Union, Callable, Type, Set, Tuple
import logging
import json
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .base import ME2AIMCPServer, BaseTool
from .utils import sanitize_input, format_response


# Configure logging
logger = logging.getLogger("me2ai-mcp-agents")


@dataclass
class ToolCategory:
    """Category for organizing and routing tools."""
    
    name: str
    description: str
    keywords: Set[str] = field(default_factory=set)
    
    def __post_init__(self) -> None:
        """Ensure keywords are a set of lowercase strings."""
        self.keywords = {k.lower() for k in self.keywords}
    
    def matches(self, query: str) -> bool:
        """
        Check if this category matches the query.
        
        Args:
            query: The search query
            
        Returns:
            Whether this category matches
        """
        query_words = set(query.lower().split())
        return bool(query_words.intersection(self.keywords))


class BaseAgent(ABC):
    """
    Abstract base class for all ME2AI MCP agents.
    
    Agents are responsible for processing requests, determining
    appropriate tools to use, and formatting responses.
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str = "",
        server: Optional[ME2AIMCPServer] = None
    ) -> None:
        """
        Initialize a base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name
            description: Description of the agent's capabilities
            server: MCP server to use (optional, can be set later)
        """
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.server = server
        self.logger = logging.getLogger(f"me2ai-agent-{agent_id}")
        self.request_count = 0
        self.error_count = 0
    
    def connect_to_server(self, server: ME2AIMCPServer) -> None:
        """
        Connect this agent to an MCP server.
        
        Args:
            server: MCP server to use
        """
        self.server = server
        self.logger.info(f"Agent {self.agent_id} connected to server {server.server_name}")
    
    @abstractmethod
    def process_request(self, request: str, **kwargs) -> Dict[str, Any]:
        """
        Process a user request.
        
        Args:
            request: User request text
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary
        """
        pass
    
    def _get_available_tools(self) -> Dict[str, Callable]:
        """
        Get all available tools from the connected server.
        
        Returns:
            Dictionary of tool_name -> tool_function
        """
        if not self.server:
            self.logger.error("No server connected")
            return {}
        
        return {
            name: getattr(self.server, name)
            for name in dir(self.server)
            if callable(getattr(self.server, name)) and not name.startswith("_")
        }
    
    def _log_request(self, request: str) -> None:
        """
        Log a request for tracking.
        
        Args:
            request: The user request
        """
        self.request_count += 1
        self.logger.info(
            f"Request #{self.request_count}: {request[:50]}{'...' if len(request) > 50 else ''}"
        )


class RoutingAgent(BaseAgent):
    """
    Agent that automatically routes requests to appropriate tools.
    
    This agent analyzes the content of requests and determines
    which tools are most appropriate based on keywords and content.
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str = "",
        server: Optional[ME2AIMCPServer] = None,
        categories: Optional[List[ToolCategory]] = None
    ) -> None:
        """
        Initialize a routing agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name
            description: Description of the agent's capabilities
            server: MCP server to use (optional, can be set later)
            categories: Tool categories for routing
        """
        super().__init__(agent_id, name, description, server)
        self.categories = categories or []
        self.tool_registry: Dict[str, Tuple[Callable, List[ToolCategory]]] = {}
    
    def register_tool_categories(self) -> None:
        """
        Register all tools from the server with their categories.
        """
        if not self.server:
            self.logger.error("No server connected")
            return
        
        tools = self._get_available_tools()
        for tool_name, tool_func in tools.items():
            if tool_name in self.tool_registry:
                continue
                
            # Skip internal methods
            if tool_name.startswith("_") or tool_name in [
                "list_tools", "get_tool_schema", "register_tool"
            ]:
                continue
            
            # Extract docstring to determine categories
            doc = inspect.getdoc(tool_func) or ""
            matching_categories = []
            
            for category in self.categories:
                # Check if category keywords appear in docstring or function name
                if (
                    any(keyword in doc.lower() for keyword in category.keywords) or
                    any(keyword in tool_name.lower() for keyword in category.keywords)
                ):
                    matching_categories.append(category)
            
            # Register tool with matched categories
            self.tool_registry[tool_name] = (tool_func, matching_categories)
            
            self.logger.info(
                f"Registered tool {tool_name} with "
                f"{len(matching_categories)} categories"
            )
    
    def process_request(self, request: str, **kwargs) -> Dict[str, Any]:
        """
        Process a request by routing to appropriate tools.
        
        Args:
            request: User request text
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary
        """
        self._log_request(request)
        
        if not self.server:
            return format_response(
                None,
                success=False,
                error="Agent not connected to any server"
            )
        
        # Clean input
        clean_request = sanitize_input(request)
        
        # Find matching tools
        matching_tools = self._find_matching_tools(clean_request)
        
        if not matching_tools:
            return format_response(
                None,
                success=False,
                error="No matching tools found for this request"
            )
        
        # Execute tools and collect results
        results = {}
        errors = []
        
        for tool_name, tool_func in matching_tools.items():
            try:
                # For simplicity, we just call with request text
                # In a more advanced implementation, you'd parse parameters
                result = tool_func(clean_request)
                results[tool_name] = result
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Error executing {tool_name}: {str(e)}")
                errors.append(f"{tool_name}: {str(e)}")
        
        if not results and errors:
            return format_response(
                None,
                success=False,
                error=f"All tools failed: {'; '.join(errors)}"
            )
        
        return format_response({
            "results": results,
            "tools_used": list(results.keys()),
            "errors": errors if errors else None
        })
    
    def _find_matching_tools(self, request: str) -> Dict[str, Callable]:
        """
        Find tools that match the given request.
        
        Args:
            request: User request
            
        Returns:
            Dictionary of tool_name -> tool_function
        """
        if not self.tool_registry:
            self.register_tool_categories()
            
        if not self.tool_registry:
            return {}
        
        # Find categories that match the request
        matching_categories = []
        for category in self.categories:
            if category.matches(request):
                matching_categories.append(category)
        
        # Collect tools from matching categories
        matching_tools = {}
        
        for tool_name, (tool_func, tool_categories) in self.tool_registry.items():
            # If tool belongs to any matching category
            if any(category in tool_categories for category in matching_categories):
                matching_tools[tool_name] = tool_func
        
        return matching_tools


class SpecializedAgent(BaseAgent):
    """
    Agent specialized for a specific domain or task.
    
    This agent focuses on a specific set of tools and provides
    more accurate and deeper processing in its domain.
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str = "",
        server: Optional[ME2AIMCPServer] = None,
        tool_names: Optional[List[str]] = None
    ) -> None:
        """
        Initialize a specialized agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name
            description: Description of the agent's capabilities
            server: MCP server to use (optional, can be set later)
            tool_names: Specific tools this agent uses
        """
        super().__init__(agent_id, name, description, server)
        self.tool_names = tool_names or []
        self.tools: Dict[str, Callable] = {}
    
    def connect_to_server(self, server: ME2AIMCPServer) -> None:
        """
        Connect to server and load specified tools.
        
        Args:
            server: MCP server to use
        """
        super().connect_to_server(server)
        self._load_tools()
    
    def _load_tools(self) -> None:
        """Load specified tools from the server."""
        if not self.server:
            self.logger.error("No server connected")
            return
        
        for tool_name in self.tool_names:
            if hasattr(self.server, tool_name) and callable(getattr(self.server, tool_name)):
                self.tools[tool_name] = getattr(self.server, tool_name)
                self.logger.info(f"Loaded tool: {tool_name}")
            else:
                self.logger.warning(f"Tool not found: {tool_name}")
    
    def process_request(self, request: str, **kwargs) -> Dict[str, Any]:
        """
        Process a request using specialized tools.
        
        Args:
            request: User request text
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary
        """
        self._log_request(request)
        
        if not self.server:
            return format_response(
                None,
                success=False,
                error="Agent not connected to any server"
            )
        
        if not self.tools:
            self._load_tools()
            
        if not self.tools:
            return format_response(
                None,
                success=False,
                error="No tools available for this agent"
            )
        
        # Clean input
        clean_request = sanitize_input(request)
        
        # For a specialized agent, we might use a fixed processing sequence
        # Here's a simplified example
        results = {}
        
        for tool_name, tool_func in self.tools.items():
            try:
                # In a real implementation, you would have logic to decide
                # which tools to use and with what parameters
                result = tool_func(clean_request)
                results[tool_name] = result
            except Exception as e:
                self.logger.error(f"Error executing {tool_name}: {str(e)}")
        
        return format_response({
            "agent": self.name,
            "domain": self.description,
            "results": results
        })


# Common tool categories
DEFAULT_CATEGORIES = [
    ToolCategory(
        name="text_processing",
        description="Text analysis and manipulation",
        keywords={"text", "string", "word", "sentence", "paragraph", 
                 "analyze", "format", "translate", "summarize"}
    ),
    ToolCategory(
        name="data_retrieval",
        description="Data fetching and storage",
        keywords={"fetch", "get", "retrieve", "search", "find", 
                 "data", "information", "storage", "database"}
    ),
    ToolCategory(
        name="github",
        description="GitHub operations",
        keywords={"github", "git", "repo", "repository", "commit", 
                 "pull", "push", "branch", "issue"}
    ),
    ToolCategory(
        name="system",
        description="System operations",
        keywords={"system", "server", "status", "config", "configuration", 
                 "setup", "health", "monitor"}
    ),
]
