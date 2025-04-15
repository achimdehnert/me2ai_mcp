"""
ME2AI MCP - Model Context Protocol server extensions for ME2AI.

This package extends the official `mcp` package with custom utilities
and abstractions specific to the ME2AI project.
"""
from .base import ME2AIMCPServer, BaseTool
from .auth import AuthManager, APIKeyAuth, TokenAuth
from .utils import sanitize_input, format_response, extract_text
from .agents import BaseAgent, RoutingAgent, SpecializedAgent, ToolCategory, DEFAULT_CATEGORIES
from .routing import RoutingRule, AgentRegistry, MCPRouter, create_default_rules

__version__ = "0.0.7"

__all__ = [
    # Base server and tools
    "ME2AIMCPServer",
    "BaseTool",
    
    # Authentication
    "AuthManager", 
    "APIKeyAuth",
    "TokenAuth",
    
    # Utilities
    "sanitize_input",
    "format_response",
    "extract_text",
    
    # Agent abstractions
    "BaseAgent",
    "RoutingAgent",
    "SpecializedAgent",
    "ToolCategory",
    "DEFAULT_CATEGORIES",
    
    # Routing layer
    "RoutingRule",
    "AgentRegistry",
    "MCPRouter",
    "create_default_rules"
]
