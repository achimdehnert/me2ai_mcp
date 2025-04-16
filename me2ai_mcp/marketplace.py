"""
Tool Marketplace for ME2AI MCP.

This module provides a marketplace for discovering, sharing, and loading tools
between different ME2AI MCP instances and applications.
"""
from typing import Dict, List, Any, Optional, Union, Callable, Type, Set, Tuple
import logging
import json
import os
import re
import importlib
import inspect
import pkgutil
import tempfile
import shutil
import hashlib
from pathlib import Path
import urllib.request
import urllib.error

from .base import BaseTool
from .agents import BaseAgent, SpecializedAgent
from .tools_registry import global_registry, ToolRegistry


# Configure logging
logger = logging.getLogger("me2ai-mcp-marketplace")


class ToolMetadata:
    """Metadata for a tool package in the marketplace."""
    
    def __init__(
        self,
        tool_id: str,
        name: str,
        version: str,
        description: str,
        author: str = "Unknown",
        categories: List[str] = None,
        dependencies: Dict[str, str] = None,
        repository_url: Optional[str] = None,
        documentation_url: Optional[str] = None,
        examples: List[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize tool metadata.
        
        Args:
            tool_id: Unique identifier for the tool
            name: Display name of the tool
            version: Version string (semantic versioning)
            description: Description of the tool
            author: Tool author
            categories: List of categories the tool belongs to
            dependencies: Dictionary of dependencies (name -> version)
            repository_url: URL to the tool's repository
            documentation_url: URL to the tool's documentation
            examples: List of example usages
        """
        self.tool_id = tool_id
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.categories = categories or ["general"]
        self.dependencies = dependencies or {}
        self.repository_url = repository_url
        self.documentation_url = documentation_url
        self.examples = examples or []
        
        # Additional properties
        self.installed = False
        self.install_path = None
        self.checksum = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolMetadata':
        """
        Create a ToolMetadata instance from a dictionary.
        
        Args:
            data: Dictionary representation of tool metadata
            
        Returns:
            ToolMetadata instance
        """
        return cls(
            tool_id=data.get("tool_id", ""),
            name=data.get("name", ""),
            version=data.get("version", "0.0.1"),
            description=data.get("description", ""),
            author=data.get("author", "Unknown"),
            categories=data.get("categories"),
            dependencies=data.get("dependencies"),
            repository_url=data.get("repository_url"),
            documentation_url=data.get("documentation_url"),
            examples=data.get("examples")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "categories": self.categories,
            "dependencies": self.dependencies,
            "repository_url": self.repository_url,
            "documentation_url": self.documentation_url,
            "examples": self.examples,
            "installed": self.installed,
            "install_path": self.install_path,
            "checksum": self.checksum
        }


class ToolRepository:
    """Repository of tools available in the marketplace."""
    
    def __init__(
        self,
        repository_url: str = "https://raw.githubusercontent.com/achimdehnert/me2ai_mcp/main/tool_repository",
        cache_dir: Optional[str] = None
    ) -> None:
        """
        Initialize the tool repository.
        
        Args:
            repository_url: URL to the tool repository
            cache_dir: Directory to cache repository data
        """
        self.repository_url = repository_url
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".me2ai_mcp" / "tool_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_cache_path = self.cache_dir / "repository_index.json"
        self.index: Dict[str, ToolMetadata] = {}
        self.logger = logging.getLogger("me2ai-mcp-repository")
        
        # Initialize
        self._load_cached_index()
    
    def _load_cached_index(self) -> None:
        """Load the cached repository index."""
        if self.index_cache_path.exists():
            try:
                with open(self.index_cache_path, "r", encoding="utf-8") as f:
                    index_data = json.load(f)
                
                for tool_id, tool_data in index_data.items():
                    self.index[tool_id] = ToolMetadata.from_dict(tool_data)
                
                self.logger.info(f"Loaded {len(self.index)} tools from cached index")
            except Exception as e:
                self.logger.error(f"Error loading cached index: {str(e)}")
    
    def update_index(self, force: bool = False) -> bool:
        """
        Update the repository index from the remote repository.
        
        Args:
            force: Whether to force update even if cache is recent
            
        Returns:
            Whether the update was successful
        """
        # Check if cache is recent (less than 1 day old) and not forced
        if not force and self.index_cache_path.exists():
            cache_age = (
                Path.cwd().stat().st_mtime - self.index_cache_path.stat().st_mtime
            )
            if cache_age < 86400:  # 24 hours in seconds
                self.logger.info("Using cached repository index (less than 1 day old)")
                return True
        
        # Fetch the index from the remote repository
        index_url = f"{self.repository_url}/index.json"
        try:
            with urllib.request.urlopen(index_url) as response:
                index_data = json.loads(response.read().decode("utf-8"))
            
            # Update the index
            self.index = {}
            for tool_id, tool_data in index_data.items():
                self.index[tool_id] = ToolMetadata.from_dict(tool_data)
            
            # Update installed status
            for tool_id, metadata in self.index.items():
                tool_dir = self.cache_dir / tool_id
                metadata.installed = tool_dir.exists()
                if metadata.installed:
                    metadata.install_path = str(tool_dir)
            
            # Save to cache
            with open(self.index_cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {k: v.to_dict() for k, v in self.index.items()},
                    f,
                    indent=2
                )
            
            self.logger.info(f"Updated repository index with {len(self.index)} tools")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating repository index: {str(e)}")
            return False
    
    def get_tool_metadata(self, tool_id: str) -> Optional[ToolMetadata]:
        """
        Get metadata for a tool.
        
        Args:
            tool_id: Tool ID
            
        Returns:
            ToolMetadata or None if not found
        """
        return self.index.get(tool_id)
    
    def list_tools(
        self, 
        category: Optional[str] = None, 
        installed_only: bool = False
    ) -> Dict[str, ToolMetadata]:
        """
        List available tools.
        
        Args:
            category: Filter by category
            installed_only: Whether to only list installed tools
            
        Returns:
            Dictionary of tool_id -> ToolMetadata
        """
        filtered_tools = {}
        
        for tool_id, metadata in self.index.items():
            # Filter by installed status
            if installed_only and not metadata.installed:
                continue
            
            # Filter by category
            if category and category not in metadata.categories:
                continue
            
            filtered_tools[tool_id] = metadata
        
        return filtered_tools
    
    def search_tools(
        self, 
        query: str, 
        installed_only: bool = False
    ) -> Dict[str, ToolMetadata]:
        """
        Search for tools.
        
        Args:
            query: Search query
            installed_only: Whether to only search installed tools
            
        Returns:
            Dictionary of tool_id -> ToolMetadata
        """
        query = query.lower()
        search_results = {}
        
        for tool_id, metadata in self.index.items():
            # Filter by installed status
            if installed_only and not metadata.installed:
                continue
            
            # Search in various fields
            if (
                query in tool_id.lower() or
                query in metadata.name.lower() or
                query in metadata.description.lower() or
                any(query in category.lower() for category in metadata.categories)
            ):
                search_results[tool_id] = metadata
        
        return search_results
    
    def download_tool(self, tool_id: str) -> Optional[Path]:
        """
        Download a tool from the repository.
        
        Args:
            tool_id: Tool ID
            
        Returns:
            Path to the downloaded tool directory or None if failed
        """
        metadata = self.get_tool_metadata(tool_id)
        if not metadata:
            self.logger.error(f"Tool {tool_id} not found in repository")
            return None
        
        # Create a temporary directory for download
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Download the tool package
            package_url = f"{self.repository_url}/tools/{tool_id}/{tool_id}-{metadata.version}.zip"
            zip_path = temp_path / f"{tool_id}.zip"
            
            try:
                self.logger.info(f"Downloading tool {tool_id} from {package_url}")
                urllib.request.urlretrieve(package_url, zip_path)
                
                # Verify checksum if available
                if metadata.checksum:
                    with open(zip_path, "rb") as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()
                    
                    if file_hash != metadata.checksum:
                        self.logger.error(f"Checksum verification failed for {tool_id}")
                        return None
                
                # Create tool directory
                tool_dir = self.cache_dir / tool_id
                if tool_dir.exists():
                    shutil.rmtree(tool_dir)
                
                tool_dir.mkdir(parents=True)
                
                # Extract the zip file
                import zipfile
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(tool_dir)
                
                # Update metadata
                metadata.installed = True
                metadata.install_path = str(tool_dir)
                
                # Update index cache
                with open(self.index_cache_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {k: v.to_dict() for k, v in self.index.items()},
                        f,
                        indent=2
                    )
                
                self.logger.info(f"Tool {tool_id} downloaded and installed to {tool_dir}")
                return tool_dir
                
            except Exception as e:
                self.logger.error(f"Error downloading tool {tool_id}: {str(e)}")
                return None


class ToolMarketplace:
    """Marketplace for discovering and loading new tools."""
    
    def __init__(
        self,
        repository_url: str = "https://raw.githubusercontent.com/achimdehnert/me2ai_mcp/main/tool_repository",
        registry: Optional[ToolRegistry] = None,
        cache_dir: Optional[str] = None
    ) -> None:
        """
        Initialize the tool marketplace.
        
        Args:
            repository_url: URL to the tool repository
            registry: Tool registry to use (defaults to global registry)
            cache_dir: Directory to cache repository data
        """
        self.repository = ToolRepository(repository_url, cache_dir)
        self.registry = registry or global_registry
        self.logger = logging.getLogger("me2ai-mcp-marketplace")
        
        # Update the repository index
        self.repository.update_index()
    
    def list_available_tools(
        self, 
        category: Optional[str] = None,
        installed_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List tools available in the marketplace.
        
        Args:
            category: Filter by category
            installed_only: Whether to only list installed tools
            
        Returns:
            List of tool metadata dictionaries
        """
        tools = self.repository.list_tools(category, installed_only)
        return [metadata.to_dict() for metadata in tools.values()]
    
    def search_tools(
        self, 
        query: str,
        installed_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for tools in the marketplace.
        
        Args:
            query: Search query
            installed_only: Whether to only search installed tools
            
        Returns:
            List of tool metadata dictionaries
        """
        tools = self.repository.search_tools(query, installed_only)
        return [metadata.to_dict() for metadata in tools.values()]
    
    def install_tool(
        self, 
        tool_id: str,
        agent: Optional[SpecializedAgent] = None
    ) -> Dict[str, Any]:
        """
        Install a tool from the marketplace.
        
        Args:
            tool_id: Tool ID
            agent: Agent to install the tool for (optional)
            
        Returns:
            Installation result
        """
        self.logger.info(f"Installing tool {tool_id}")
        
        # Get tool metadata
        metadata = self.repository.get_tool_metadata(tool_id)
        if not metadata:
            return {
                "success": False,
                "error": f"Tool {tool_id} not found in repository"
            }
        
        # Download the tool
        tool_dir = self.repository.download_tool(tool_id)
        if not tool_dir:
            return {
                "success": False,
                "error": f"Failed to download tool {tool_id}"
            }
        
        # Add the tool directory to Python path
        import sys
        if str(tool_dir) not in sys.path:
            sys.path.insert(0, str(tool_dir))
        
        # Import the tool module
        try:
            # Find the main module (assuming it's named like the tool_id)
            module_name = tool_id.replace("-", "_")
            tool_module = importlib.import_module(module_name)
            
            # Look for a register_tools function
            if hasattr(tool_module, "register_tools"):
                # Register tools with the registry
                if agent:
                    tool_module.register_tools(agent)
                else:
                    tool_module.register_tools(self.registry)
                
                self.logger.info(f"Tool {tool_id} registered successfully")
                return {
                    "success": True,
                    "message": f"Tool {tool_id} installed and registered successfully",
                    "metadata": metadata.to_dict()
                }
            else:
                self.logger.warning(f"No register_tools function found in {tool_id}")
                return {
                    "success": False,
                    "error": f"No register_tools function found in {tool_id}"
                }
            
        except Exception as e:
            self.logger.error(f"Error importing tool {tool_id}: {str(e)}")
            return {
                "success": False,
                "error": f"Error importing tool {tool_id}: {str(e)}"
            }
    
    def uninstall_tool(self, tool_id: str) -> Dict[str, Any]:
        """
        Uninstall a tool.
        
        Args:
            tool_id: Tool ID
            
        Returns:
            Uninstallation result
        """
        self.logger.info(f"Uninstalling tool {tool_id}")
        
        # Get tool metadata
        metadata = self.repository.get_tool_metadata(tool_id)
        if not metadata:
            return {
                "success": False,
                "error": f"Tool {tool_id} not found in repository"
            }
        
        if not metadata.installed:
            return {
                "success": False,
                "error": f"Tool {tool_id} is not installed"
            }
        
        # Unregister tools
        module_name = tool_id.replace("-", "_")
        try:
            # Look for registered tools with this module prefix
            tool_prefix = f"{module_name}."
            
            # Find tools to unregister
            tools_to_unregister = []
            for tool_name in list(self.registry.tools.keys()):
                if tool_name.startswith(tool_prefix):
                    tools_to_unregister.append(tool_name)
            
            # Unregister the tools
            for tool_name in tools_to_unregister:
                self.registry.unregister_tool(tool_name)
            
            # Remove the tool directory
            tool_dir = Path(metadata.install_path)
            if tool_dir.exists():
                shutil.rmtree(tool_dir)
            
            # Update metadata
            metadata.installed = False
            metadata.install_path = None
            
            # Update index cache
            with open(self.repository.index_cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {k: v.to_dict() for k, v in self.repository.index.items()},
                    f,
                    indent=2
                )
            
            self.logger.info(f"Tool {tool_id} uninstalled successfully")
            return {
                "success": True,
                "message": f"Tool {tool_id} uninstalled successfully",
                "unregistered_tools": tools_to_unregister
            }
            
        except Exception as e:
            self.logger.error(f"Error uninstalling tool {tool_id}: {str(e)}")
            return {
                "success": False,
                "error": f"Error uninstalling tool {tool_id}: {str(e)}"
            }
    
    def create_tool_package(
        self,
        tool_id: str,
        name: str,
        version: str,
        description: str,
        author: str,
        tool_functions: List[Callable],
        output_dir: str,
        categories: List[str] = None
    ) -> Dict[str, Any]:
        """
        Create a tool package for distribution.
        
        Args:
            tool_id: Unique identifier for the tool
            name: Display name of the tool
            version: Version string (semantic versioning)
            description: Description of the tool
            author: Tool author
            tool_functions: List of tool functions to include
            output_dir: Directory to output the package
            categories: List of categories the tool belongs to
            
        Returns:
            Package creation result
        """
        self.logger.info(f"Creating tool package for {tool_id}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create package directory
        package_dir = output_path / tool_id
        if package_dir.exists():
            shutil.rmtree(package_dir)
        
        package_dir.mkdir()
        
        # Create module name (replace hyphens with underscores)
        module_name = tool_id.replace("-", "_")
        
        # Create __init__.py
        with open(package_dir / "__init__.py", "w", encoding="utf-8") as f:
            f.write(f'"""\n{name}\n\n{description}\n"""\n\n')
            f.write(f'__version__ = "{version}"\n\n')
            f.write('from .tools import register_tools\n\n')
            f.write(f'__all__ = ["register_tools"]\n')
        
        # Create tools.py with the functions
        with open(package_dir / "tools.py", "w", encoding="utf-8") as f:
            f.write(f'"""\nTool implementations for {name}.\n"""\n')
            f.write('from typing import Dict, List, Any, Optional, Union, Callable\n\n')
            
            # Import needed modules
            imports = set()
            for func in tool_functions:
                func_source = inspect.getsource(func)
                
                # Extract imports using regex (simplistic approach)
                import_matches = re.findall(r'import\s+([^\n]+)', func_source)
                for match in import_matches:
                    imports.add(f"import {match}\n")
                
                from_matches = re.findall(r'from\s+([^\n]+)', func_source)
                for match in from_matches:
                    imports.add(f"from {match}\n")
            
            # Write imports
            for import_line in sorted(imports):
                f.write(import_line)
            
            f.write('\n\n')
            
            # Write tool functions
            for func in tool_functions:
                f.write(inspect.getsource(func))
                f.write('\n\n')
            
            # Create registration function
            f.write('def register_tools(registry_or_agent):\n')
            f.write('    """Register tools with a registry or agent."""\n')
            f.write('    # Determine if we\'re registering with a registry or an agent\n')
            f.write('    if hasattr(registry_or_agent, "register_tool"):\n')
            f.write('        # It\'s a registry\n')
            for func in tool_functions:
                func_name = func.__name__
                category_str = str(categories or ["general"])
                f.write(f'        registry_or_agent.register_tool(\n')
                f.write(f'            tool_name="{module_name}.{func_name}",\n')
                f.write(f'            tool_func={func_name},\n')
                f.write(f'            categories={category_str},\n')
                f.write(f'            description="""{func.__doc__ or ""}\"""\n')
                f.write(f'        )\n')
            
            f.write('    elif hasattr(registry_or_agent, "tools"):\n')
            f.write('        # It\'s an agent\n')
            for func in tool_functions:
                func_name = func.__name__
                f.write(f'        registry_or_agent.tools["{func_name}"] = {func_name}\n')
            
            f.write('    return True\n')
        
        # Create metadata.json
        metadata = {
            "tool_id": tool_id,
            "name": name,
            "version": version,
            "description": description,
            "author": author,
            "categories": categories or ["general"],
            "tools": [func.__name__ for func in tool_functions],
            "created_at": datetime.now().isoformat()
        }
        
        with open(package_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        # Create a zip file
        import zipfile
        zip_path = output_path / f"{tool_id}-{version}.zip"
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for file_path in package_dir.glob("**/*"):
                if file_path.is_file():
                    zipf.write(
                        file_path,
                        file_path.relative_to(package_dir.parent)
                    )
        
        # Calculate checksum
        with open(zip_path, "rb") as f:
            checksum = hashlib.sha256(f.read()).hexdigest()
        
        self.logger.info(f"Tool package created at {zip_path}")
        return {
            "success": True,
            "message": f"Tool package created successfully",
            "package_path": str(zip_path),
            "metadata": metadata,
            "checksum": checksum
        }


# Global instance for convenience
global_marketplace = ToolMarketplace()
