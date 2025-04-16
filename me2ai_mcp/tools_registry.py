"""
Enhanced tool registry for ME2AI MCP.

This module provides a central registry for all MCP tools
with advanced discovery, categorization, and management capabilities.
"""
from typing import Dict, List, Any, Optional, Union, Callable, Type, Set, Tuple
import logging
import inspect
import importlib
import pkgutil
import os
from pathlib import Path

from .agents import ToolCategory, BaseAgent
from .base import BaseTool


# Configure logging
logger = logging.getLogger("me2ai-mcp-tool-registry")


class ToolRegistry:
    """Central registry for all available tools."""
    
    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        self.tools: Dict[str, Callable] = {}
        self.tool_metadata: Dict[str, Dict[str, Any]] = {}
        self.categories: Dict[str, List[str]] = {}
        self.logger = logging.getLogger("me2ai-mcp-tool-registry")
    
    def register_tool(
        self, 
        tool_name: str, 
        tool_func: Callable, 
        categories: List[str] = None,
        description: str = "",
        parameters: Dict[str, Any] = None,
        returns: Dict[str, Any] = None
    ) -> None:
        """
        Register a new tool with the system.
        
        Args:
            tool_name: Name of the tool
            tool_func: Function implementing the tool
            categories: List of category names for this tool
            description: Description of what the tool does
            parameters: Dictionary describing the parameters
            returns: Dictionary describing the return values
        """
        if tool_name in self.tools:
            self.logger.warning(f"Tool '{tool_name}' is being re-registered")
        
        self.tools[tool_name] = tool_func
        
        # Extract metadata from function if not provided
        tool_doc = inspect.getdoc(tool_func) or ""
        tool_signature = inspect.signature(tool_func)
        
        # Store metadata
        self.tool_metadata[tool_name] = {
            "name": tool_name,
            "description": description or tool_doc,
            "parameters": parameters or {
                param.name: {
                    "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                    "required": param.default == inspect.Parameter.empty,
                    "default": None if param.default == inspect.Parameter.empty else param.default
                }
                for param in tool_signature.parameters.values()
            },
            "returns": returns or {},
            "categories": categories or ["general"]
        }
        
        # Update category mappings
        for category in self.tool_metadata[tool_name]["categories"]:
            if category not in self.categories:
                self.categories[category] = []
            if tool_name not in self.categories[category]:
                self.categories[category].append(tool_name)
        
        self.logger.info(f"Tool '{tool_name}' registered in categories: {categories or ['general']}")
    
    def unregister_tool(self, tool_name: str) -> bool:
        """
        Unregister a tool from the registry.
        
        Args:
            tool_name: Name of the tool to unregister
            
        Returns:
            Whether the tool was unregistered
        """
        if tool_name not in self.tools:
            self.logger.warning(f"Attempted to unregister non-existent tool '{tool_name}'")
            return False
        
        # Remove from categories
        categories = self.tool_metadata[tool_name]["categories"]
        for category in categories:
            if category in self.categories and tool_name in self.categories[category]:
                self.categories[category].remove(tool_name)
                # Clean up empty categories
                if not self.categories[category]:
                    del self.categories[category]
        
        # Remove tool and metadata
        del self.tools[tool_name]
        del self.tool_metadata[tool_name]
        
        self.logger.info(f"Tool '{tool_name}' unregistered")
        return True
    
    def get_tool(self, tool_name: str) -> Optional[Callable]:
        """
        Get a tool by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            The tool function or None if not found
        """
        return self.tools.get(tool_name)
    
    def get_tool_metadata(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Dictionary of metadata or None if not found
        """
        return self.tool_metadata.get(tool_name)
    
    def get_tools_by_category(self, category: str) -> Dict[str, Callable]:
        """
        Get all tools in a specific category.
        
        Args:
            category: Category name
            
        Returns:
            Dictionary of tool_name -> tool_function
        """
        if category not in self.categories:
            return {}
        
        return {name: self.tools[name] for name in self.categories[category] if name in self.tools}
    
    def get_categories(self) -> List[str]:
        """
        Get all available categories.
        
        Returns:
            List of category names
        """
        return list(self.categories.keys())
    
    def search_tools(self, query: str) -> Dict[str, Callable]:
        """
        Search for tools matching a query.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary of matching tool_name -> tool_function
        """
        query = query.lower()
        results = {}
        
        for tool_name, metadata in self.tool_metadata.items():
            if (
                query in tool_name.lower() or
                query in metadata["description"].lower() or
                any(query in category.lower() for category in metadata["categories"])
            ):
                results[tool_name] = self.tools[tool_name]
        
        return results
    
    def get_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all registered tools with metadata.
        
        Returns:
            Dictionary of tool_name -> tool_metadata
        """
        return self.tool_metadata.copy()


# Global instance for convenience
global_registry = ToolRegistry()


def register_tool(
    tool_name: Optional[str] = None,
    categories: List[str] = None,
    description: str = "",
    parameters: Dict[str, Any] = None,
    returns: Dict[str, Any] = None
) -> Callable:
    """
    Decorator to register a function as a tool.
    
    Args:
        tool_name: Name of the tool (defaults to function name)
        categories: List of category names
        description: Description of the tool
        parameters: Dictionary describing parameters
        returns: Dictionary describing return values
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        name = tool_name or func.__name__
        global_registry.register_tool(
            tool_name=name,
            tool_func=func,
            categories=categories,
            description=description or inspect.getdoc(func) or "",
            parameters=parameters,
            returns=returns
        )
        return func
    
    return decorator


def discover_tools_in_module(module_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Discover all tools in a module.
    
    Args:
        module_path: Dot notation path to the module
        
    Returns:
        Dictionary of discovered tools
    """
    try:
        module = importlib.import_module(module_path)
        discovered = {}
        
        # Look for functions with register_tool decorator
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isfunction(obj) and 
                hasattr(obj, "_is_mcp_tool") and 
                obj._is_mcp_tool is True
            ):
                # The tool is already registered via decorator
                tool_name = getattr(obj, "_tool_name", name)
                discovered[tool_name] = global_registry.get_tool_metadata(tool_name)
        
        return discovered
        
    except ImportError as e:
        logger.error(f"Error importing module {module_path}: {str(e)}")
        return {}


def discover_tools_in_package(package_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Recursively discover all tools in a package.
    
    Args:
        package_path: Dot notation path to the package
        
    Returns:
        Dictionary of discovered tools
    """
    try:
        package = importlib.import_module(package_path)
        all_tools = {}
        
        # First, check the package itself
        package_tools = discover_tools_in_module(package_path)
        all_tools.update(package_tools)
        
        # Then, recursive discovery for all modules
        if hasattr(package, "__path__"):
            for _, name, is_pkg in pkgutil.iter_modules(package.__path__):
                full_name = f"{package_path}.{name}"
                
                if is_pkg:
                    # Recursive for subpackages
                    subpackage_tools = discover_tools_in_package(full_name)
                    all_tools.update(subpackage_tools)
                else:
                    # Direct for modules
                    module_tools = discover_tools_in_module(full_name)
                    all_tools.update(module_tools)
        
        return all_tools
        
    except ImportError as e:
        logger.error(f"Error importing package {package_path}: {str(e)}")
        return {}
