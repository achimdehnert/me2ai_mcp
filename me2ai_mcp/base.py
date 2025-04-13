"""
Base classes for ME2AI MCP servers.

This module provides base classes for creating standardized ME2AI MCP servers
that extend the functionality of the official MCP package.
"""
from typing import Dict, List, Any, Optional, Union, Callable, Type, TypeVar
import logging
import asyncio
import inspect
import json
from dataclasses import dataclass, field
from functools import wraps
import os
from pathlib import Path

# Import the official MCP package
from mcp import MCPServer as OfficialMCPServer
from mcp import register_tool as official_register_tool
from mcp import MCPToolInput

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

T = TypeVar("T", bound="ME2AIMCPServer")


class ME2AIMCPServer(OfficialMCPServer):
    """Base class for all ME2AI MCP servers with enhanced functionality."""

    def __init__(
        self,
        server_name: str,
        description: str = "",
        version: str = "0.1.0",
        debug: bool = False
    ) -> None:
        """Initialize a ME2AI MCP server.
        
        Args:
            server_name: Unique name of the MCP server
            description: Human-readable description of the server
            version: Server version
            debug: Whether to enable debug logging
        """
        super().__init__(server_name)
        
        # Additional properties
        self.description = description
        self.version = version
        self.metadata = {
            "server_name": server_name,
            "description": description,
            "version": version,
            "framework": "ME2AI MCP"
        }
        
        # Configure logging
        self.logger = logging.getLogger(f"me2ai-mcp-{server_name}")
        
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
            
        self.logger.info(f"Initializing ME2AI MCP Server: {server_name} v{version}")
        
        # Statistics tracking
        self.stats = {
            "requests": 0,
            "errors": 0,
            "tool_calls": {}
        }
        
        # Register built-in tools
        self._register_builtin_tools()
        
    def _register_builtin_tools(self) -> None:
        """Register built-in tools for all ME2AI MCP servers."""
        # Register server info tool
        @official_register_tool
        async def server_info(self) -> Dict[str, Any]:
            """Get information about this MCP server."""
            self.stats["requests"] += 1
            self.stats["tool_calls"]["server_info"] = self.stats["tool_calls"].get("server_info", 0) + 1
            
            return {
                "success": True,
                "server": self.metadata,
                "stats": self.stats,
                "tools": list(self.tools.keys())
            }
    
    @classmethod
    def from_config(cls: Type[T], config_path: Union[str, Path]) -> T:
        """Create an MCP server from a configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configured MCP server instance
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, "r") as f:
            config = json.load(f)
            
        server_name = config.get("server_name", "unnamed-server")
        description = config.get("description", "")
        version = config.get("version", "0.1.0")
        debug = config.get("debug", False)
        
        server = cls(
            server_name=server_name,
            description=description,
            version=version,
            debug=debug
        )
        
        return server
        
    async def start(self) -> None:
        """Start the MCP server with enhanced logging and error handling."""
        try:
            self.logger.info(f"Starting {self.server_name} MCP server (v{self.version})")
            self.logger.info(f"Available tools: {', '.join(self.tools.keys())}")
            
            # Start the official MCP server
            await super().start()
            
        except Exception as e:
            self.logger.error(f"Error starting MCP server: {str(e)}")
            self.stats["errors"] += 1
            raise


@dataclass
class BaseTool:
    """Base class for ME2AI MCP tools with enhanced functionality."""
    
    name: str
    description: str = ""
    enabled: bool = True
    stats: Dict[str, Any] = field(default_factory=lambda: {
        "calls": 0,
        "errors": 0,
        "last_call": None
    })
    
    def __post_init__(self) -> None:
        """Initialize the tool after all fields are set."""
        self.logger = logging.getLogger(f"me2ai-mcp-tool-{self.name}")
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with the given parameters.
        
        This method should be overridden by subclasses.
        
        Args:
            params: Tool parameters
            
        Returns:
            Tool execution result
        """
        raise NotImplementedError("Tool execution not implemented")


def register_tool(func: Optional[Callable] = None, *, name: Optional[str] = None) -> Callable:
    """Enhanced decorator for registering tools with ME2AI MCP servers.
    
    This extends the official register_tool decorator with additional
    functionality like automatic error handling and logging.
    
    Args:
        func: Function to register as a tool
        name: Custom name for the tool (optional)
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        # Get the function's signature for better error messages
        sig = inspect.signature(func)
        
        @wraps(func)
        async def wrapper(self, *args, **kwargs) -> Dict[str, Any]:
            tool_name = name or func.__name__
            
            # Update stats
            if hasattr(self, "stats") and isinstance(self.stats, dict):
                self.stats["requests"] += 1
                self.stats["tool_calls"][tool_name] = self.stats["tool_calls"].get(tool_name, 0) + 1
            
            # Log the call
            if hasattr(self, "logger"):
                params_str = str(kwargs) if kwargs else str(args)
                self.logger.info(f"Tool call: {tool_name}({params_str[:100]}{'...' if len(params_str) > 100 else ''})")
            
            try:
                # Execute the tool function
                result = await func(self, *args, **kwargs)
                
                # Ensure the result is a dictionary with success status
                if isinstance(result, dict) and "success" not in result:
                    result["success"] = True
                    
                return result
                
            except Exception as e:
                # Handle errors consistently
                error_message = f"Error executing {tool_name}: {str(e)}"
                
                if hasattr(self, "logger"):
                    self.logger.error(error_message, exc_info=True)
                    
                if hasattr(self, "stats") and isinstance(self.stats, dict):
                    self.stats["errors"] += 1
                
                return {
                    "success": False,
                    "error": error_message,
                    "exception_type": type(e).__name__
                }
        
        # Register with the official decorator
        official_register_tool(wrapper)
        
        return wrapper
    
    # Handle both @register_tool and @register_tool(name="custom_name") forms
    if func is None:
        return decorator
    else:
        return decorator(func)
