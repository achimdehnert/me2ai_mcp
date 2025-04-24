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

# Import new components
from .tools_registry import ToolRegistry, register_tool, global_registry, discover_tools_in_module, discover_tools_in_package
from .collaborative_agent import CollaborativeAgent, CollaborationManager, CollaborationContext, global_collaboration_manager
from .dynamic_routing import AdaptiveRouter, PerformanceMetrics
from .marketplace import ToolMarketplace, ToolRepository, ToolMetadata, global_marketplace
from .llm_fallback import LLMFallbackMixin, LLMProvider
from .version import __version__

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
    "create_default_rules",
    
    # LLM fallback functionality
    "LLMFallbackMixin",
    "LLMProvider",
    
    # Enhanced Tool Registry
    "ToolRegistry",
    "register_tool",
    "global_registry",
    "discover_tools_in_module",
    "discover_tools_in_package",
    
    # Collaborative Agents
    "CollaborativeAgent",
    "CollaborationManager",
    "CollaborationContext",
    "global_collaboration_manager",
    
    # Dynamic Routing
    "AdaptiveRouter",
    "PerformanceMetrics",
    
    # Tool Marketplace
    "ToolMarketplace",
    "ToolRepository",
    "ToolMetadata",
    "global_marketplace"
]
