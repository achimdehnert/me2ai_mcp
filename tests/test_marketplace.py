"""
Tests for the marketplace module of the ME2AI MCP package.

This test suite validates the functionality of the marketplace components:
- ToolMetadata
- ToolRepository
- ToolMarketplace
"""
import os
import json
import tempfile
import shutil
import hashlib
import unittest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from me2ai_mcp.marketplace import (
    ToolMetadata, 
    ToolRepository, 
    ToolMarketplace,
    global_marketplace
)
from me2ai_mcp.tools_registry import ToolRegistry, global_registry


class TestToolMetadata(unittest.TestCase):
    """Test cases for the ToolMetadata class."""
    
    def test_init(self):
        """Test initialization with full parameters."""
        metadata = ToolMetadata(
            tool_id="test-tool",
            name="Test Tool",
            version="1.0.0",
            description="A test tool",
            author="Test Author",
            categories=["test", "example"],
            dependencies={"numpy": ">=1.18.0"},
            repository_url="https://example.com/repo",
            documentation_url="https://example.com/docs",
            examples=[{"name": "Example 1", "code": "print('hello')"}]
        )
        
        self.assertEqual(metadata.tool_id, "test-tool")
        self.assertEqual(metadata.name, "Test Tool")
        self.assertEqual(metadata.version, "1.0.0")
        self.assertEqual(metadata.description, "A test tool")
        self.assertEqual(metadata.author, "Test Author")
        self.assertEqual(metadata.categories, ["test", "example"])
        self.assertEqual(metadata.dependencies, {"numpy": ">=1.18.0"})
        self.assertEqual(metadata.repository_url, "https://example.com/repo")
        self.assertEqual(metadata.documentation_url, "https://example.com/docs")
        self.assertEqual(metadata.examples, [{"name": "Example 1", "code": "print('hello')"}])
        self.assertFalse(metadata.installed)
        self.assertIsNone(metadata.install_path)
        self.assertIsNone(metadata.checksum)
    
    def test_init_with_defaults(self):
        """Test initialization with default values."""
        metadata = ToolMetadata(
            tool_id="test-tool",
            name="Test Tool",
            version="1.0.0",
            description="A test tool"
        )
        
        self.assertEqual(metadata.tool_id, "test-tool")
        self.assertEqual(metadata.name, "Test Tool")
        self.assertEqual(metadata.version, "1.0.0")
        self.assertEqual(metadata.description, "A test tool")
        self.assertEqual(metadata.author, "Unknown")
        self.assertEqual(metadata.categories, ["general"])
        self.assertEqual(metadata.dependencies, {})
        self.assertIsNone(metadata.repository_url)
        self.assertIsNone(metadata.documentation_url)
        self.assertEqual(metadata.examples, [])
    
    def test_from_dict(self):
        """Test creating metadata from a dictionary."""
        data = {
            "tool_id": "test-tool",
            "name": "Test Tool",
            "version": "1.0.0",
            "description": "A test tool",
            "author": "Test Author",
            "categories": ["test", "example"],
            "dependencies": {"numpy": ">=1.18.0"},
            "repository_url": "https://example.com/repo",
            "documentation_url": "https://example.com/docs",
            "examples": [{"name": "Example 1", "code": "print('hello')"}]
        }
        
        metadata = ToolMetadata.from_dict(data)
        
        self.assertEqual(metadata.tool_id, "test-tool")
        self.assertEqual(metadata.name, "Test Tool")
        self.assertEqual(metadata.version, "1.0.0")
        self.assertEqual(metadata.description, "A test tool")
        self.assertEqual(metadata.author, "Test Author")
        self.assertEqual(metadata.categories, ["test", "example"])
        self.assertEqual(metadata.dependencies, {"numpy": ">=1.18.0"})
        self.assertEqual(metadata.repository_url, "https://example.com/repo")
        self.assertEqual(metadata.documentation_url, "https://example.com/docs")
        self.assertEqual(metadata.examples, [{"name": "Example 1", "code": "print('hello')"}])
    
    def test_from_dict_with_minimal_data(self):
        """Test creating metadata from a minimal dictionary."""
        data = {
            "tool_id": "test-tool",
            "name": "Test Tool",
            "version": "1.0.0",
            "description": "A test tool"
        }
        
        metadata = ToolMetadata.from_dict(data)
        
        self.assertEqual(metadata.tool_id, "test-tool")
        self.assertEqual(metadata.name, "Test Tool")
        self.assertEqual(metadata.version, "1.0.0")
        self.assertEqual(metadata.description, "A test tool")
        self.assertEqual(metadata.author, "Unknown")
        self.assertEqual(metadata.categories, ["general"])
        self.assertEqual(metadata.dependencies, {})
        self.assertIsNone(metadata.repository_url)
        self.assertIsNone(metadata.documentation_url)
        self.assertEqual(metadata.examples, [])
    
    def test_to_dict(self):
        """Test converting metadata to a dictionary."""
        metadata = ToolMetadata(
            tool_id="test-tool",
            name="Test Tool",
            version="1.0.0",
            description="A test tool",
            author="Test Author",
            categories=["test", "example"],
            dependencies={"numpy": ">=1.18.0"},
            repository_url="https://example.com/repo",
            documentation_url="https://example.com/docs",
            examples=[{"name": "Example 1", "code": "print('hello')"}]
        )
        
        # Set additional properties
        metadata.installed = True
        metadata.install_path = "/tmp/tools/test-tool"
        metadata.checksum = "abc123"
        
        data = metadata.to_dict()
        
        self.assertEqual(data["tool_id"], "test-tool")
        self.assertEqual(data["name"], "Test Tool")
        self.assertEqual(data["version"], "1.0.0")
        self.assertEqual(data["description"], "A test tool")
        self.assertEqual(data["author"], "Test Author")
        self.assertEqual(data["categories"], ["test", "example"])
        self.assertEqual(data["dependencies"], {"numpy": ">=1.18.0"})
        self.assertEqual(data["repository_url"], "https://example.com/repo")
        self.assertEqual(data["documentation_url"], "https://example.com/docs")
        self.assertEqual(data["examples"], [{"name": "Example 1", "code": "print('hello')"}])
        self.assertTrue(data["installed"])
        self.assertEqual(data["install_path"], "/tmp/tools/test-tool")
        self.assertEqual(data["checksum"], "abc123")


class TestToolRepository(unittest.TestCase):
    """Test cases for the ToolRepository class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.temp_dir, "tool_cache")
        
        # Create the repository
        self.repository = ToolRepository(
            repository_url="https://example.com/repo",
            cache_dir=self.cache_dir
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test initialization of the repository."""
        self.assertEqual(self.repository.repository_url, "https://example.com/repo")
        self.assertEqual(str(self.repository.cache_dir), self.cache_dir)
        self.assertEqual(str(self.repository.index_cache_path), 
                         os.path.join(self.cache_dir, "repository_index.json"))
        self.assertEqual(self.repository.index, {})
    
    @patch("me2ai_mcp.marketplace.urllib.request.urlopen")
    def test_update_index(self, mock_urlopen):
        """Test updating the repository index."""
        # Mock the response from the repository
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "test-tool": {
                "tool_id": "test-tool",
                "name": "Test Tool",
                "version": "1.0.0",
                "description": "A test tool"
            }
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        # Update the index
        result = self.repository.update_index(force=True)
        
        # Verify the result
        self.assertTrue(result)
        self.assertEqual(len(self.repository.index), 1)
        self.assertIn("test-tool", self.repository.index)
        
        # Verify the cache file was created
        self.assertTrue(os.path.exists(self.repository.index_cache_path))
    
    @patch("me2ai_mcp.marketplace.urllib.request.urlopen")
    def test_update_index_with_error(self, mock_urlopen):
        """Test updating the repository index with an error."""
        # Mock an error
        mock_urlopen.side_effect = Exception("Test error")
        
        # Update the index
        result = self.repository.update_index(force=True)
        
        # Verify the result
        self.assertFalse(result)
        self.assertEqual(self.repository.index, {})
    
    def test_get_tool_metadata(self):
        """Test getting tool metadata."""
        # Add a tool to the index
        self.repository.index = {
            "test-tool": ToolMetadata.from_dict({
                "tool_id": "test-tool",
                "name": "Test Tool",
                "version": "1.0.0",
                "description": "A test tool"
            })
        }
        
        # Get the metadata
        metadata = self.repository.get_tool_metadata("test-tool")
        
        # Verify the metadata
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.tool_id, "test-tool")
        self.assertEqual(metadata.name, "Test Tool")
        
        # Get non-existent metadata
        metadata = self.repository.get_tool_metadata("non-existent")
        
        # Verify the result
        self.assertIsNone(metadata)
    
    def test_list_tools(self):
        """Test listing tools."""
        # Add tools to the index
        self.repository.index = {
            "test-tool-1": ToolMetadata.from_dict({
                "tool_id": "test-tool-1",
                "name": "Test Tool 1",
                "version": "1.0.0",
                "description": "A test tool",
                "categories": ["test"]
            }),
            "test-tool-2": ToolMetadata.from_dict({
                "tool_id": "test-tool-2",
                "name": "Test Tool 2",
                "version": "1.0.0",
                "description": "Another test tool",
                "categories": ["example"]
            })
        }
        
        # Set one tool as installed
        self.repository.index["test-tool-1"].installed = True
        
        # List all tools
        tools = self.repository.list_tools()
        
        # Verify the result
        self.assertEqual(len(tools), 2)
        self.assertIn("test-tool-1", tools)
        self.assertIn("test-tool-2", tools)
        
        # List installed tools
        tools = self.repository.list_tools(installed_only=True)
        
        # Verify the result
        self.assertEqual(len(tools), 1)
        self.assertIn("test-tool-1", tools)
        
        # List tools by category
        tools = self.repository.list_tools(category="test")
        
        # Verify the result
        self.assertEqual(len(tools), 1)
        self.assertIn("test-tool-1", tools)
    
    def test_search_tools(self):
        """Test searching for tools."""
        # Add tools to the index
        self.repository.index = {
            "test-tool-1": ToolMetadata.from_dict({
                "tool_id": "test-tool-1",
                "name": "Search Test Tool",
                "version": "1.0.0",
                "description": "A test tool for searching",
                "categories": ["test", "search"]
            }),
            "test-tool-2": ToolMetadata.from_dict({
                "tool_id": "test-tool-2",
                "name": "Another Tool",
                "version": "1.0.0",
                "description": "Another test tool",
                "categories": ["example"]
            })
        }
        
        # Set one tool as installed
        self.repository.index["test-tool-1"].installed = True
        
        # Search for tools
        tools = self.repository.search_tools("search")
        
        # Verify the result
        self.assertEqual(len(tools), 1)
        self.assertIn("test-tool-1", tools)
        
        # Search for installed tools
        tools = self.repository.search_tools("tool", installed_only=True)
        
        # Verify the result
        self.assertEqual(len(tools), 1)
        self.assertIn("test-tool-1", tools)
    
    @patch("me2ai_mcp.marketplace.urllib.request.urlretrieve")
    @patch("me2ai_mcp.marketplace.zipfile.ZipFile")
    def test_download_tool(self, mock_zipfile, mock_urlretrieve):
        """Test downloading a tool."""
        # Set up the repository
        self.repository.index = {
            "test-tool": ToolMetadata.from_dict({
                "tool_id": "test-tool",
                "name": "Test Tool",
                "version": "1.0.0",
                "description": "A test tool"
            })
        }
        
        # Mock the ZipFile
        mock_zip = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip
        
        # Download the tool
        tool_dir = self.repository.download_tool("test-tool")
        
        # Verify the result
        self.assertIsNotNone(tool_dir)
        self.assertTrue(os.path.exists(tool_dir))
        
        # Verify the metadata was updated
        metadata = self.repository.get_tool_metadata("test-tool")
        self.assertTrue(metadata.installed)
        self.assertEqual(metadata.install_path, str(tool_dir))
        
        # Download a non-existent tool
        tool_dir = self.repository.download_tool("non-existent")
        
        # Verify the result
        self.assertIsNone(tool_dir)
    
    @patch("me2ai_mcp.marketplace.urllib.request.urlretrieve")
    def test_download_tool_with_error(self, mock_urlretrieve):
        """Test downloading a tool with an error."""
        # Set up the repository
        self.repository.index = {
            "test-tool": ToolMetadata.from_dict({
                "tool_id": "test-tool",
                "name": "Test Tool",
                "version": "1.0.0",
                "description": "A test tool"
            })
        }
        
        # Mock an error
        mock_urlretrieve.side_effect = Exception("Test error")
        
        # Download the tool
        tool_dir = self.repository.download_tool("test-tool")
        
        # Verify the result
        self.assertIsNone(tool_dir)


class TestToolMarketplace(unittest.TestCase):
    """Test cases for the ToolMarketplace class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.temp_dir, "tool_cache")
        
        # Create a mock ToolRepository
        self.mock_repository = MagicMock()
        
        # Create a registry
        self.registry = ToolRegistry()
        
        # Create the marketplace with mocked repository
        self.marketplace = ToolMarketplace(
            repository_url="https://example.com/repo",
            registry=self.registry,
            cache_dir=self.cache_dir
        )
        
        # Replace the repository with the mock
        self.marketplace.repository = self.mock_repository
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_list_available_tools(self):
        """Test listing available tools."""
        # Mock the repository's list_tools method
        self.mock_repository.list_tools.return_value = {
            "test-tool-1": ToolMetadata.from_dict({
                "tool_id": "test-tool-1",
                "name": "Test Tool 1",
                "version": "1.0.0",
                "description": "A test tool"
            }),
            "test-tool-2": ToolMetadata.from_dict({
                "tool_id": "test-tool-2",
                "name": "Test Tool 2",
                "version": "1.0.0",
                "description": "Another test tool"
            })
        }
        
        # List all tools
        tools = self.marketplace.list_available_tools()
        
        # Verify the result
        self.assertEqual(len(tools), 2)
        self.mock_repository.list_tools.assert_called_with(None, False)
        
        # List installed tools
        tools = self.marketplace.list_available_tools(installed_only=True)
        
        # Verify the result
        self.mock_repository.list_tools.assert_called_with(None, True)
        
        # List tools by category
        tools = self.marketplace.list_available_tools(category="test")
        
        # Verify the result
        self.mock_repository.list_tools.assert_called_with("test", False)
    
    def test_search_tools(self):
        """Test searching for tools."""
        # Mock the repository's search_tools method
        self.mock_repository.search_tools.return_value = {
            "test-tool-1": ToolMetadata.from_dict({
                "tool_id": "test-tool-1",
                "name": "Test Tool 1",
                "version": "1.0.0",
                "description": "A test tool"
            })
        }
        
        # Search for tools
        tools = self.marketplace.search_tools("test")
        
        # Verify the result
        self.assertEqual(len(tools), 1)
        self.mock_repository.search_tools.assert_called_with("test", False)
        
        # Search for installed tools
        tools = self.marketplace.search_tools("test", installed_only=True)
        
        # Verify the result
        self.mock_repository.search_tools.assert_called_with("test", True)
    
    def test_install_tool(self):
        """Test installing a tool."""
        # Mock the repository methods
        tool_id = "test-tool"
        
        self.mock_repository.get_tool_metadata.return_value = ToolMetadata.from_dict({
            "tool_id": tool_id,
            "name": "Test Tool",
            "version": "1.0.0",
            "description": "A test tool"
        })
        
        self.mock_repository.download_tool.return_value = Path(os.path.join(self.cache_dir, tool_id))
        
        # Create a mock agent
        mock_agent = MagicMock()
        
        # Mock the module
        with patch("me2ai_mcp.marketplace.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_import.return_value = mock_module
            
            # Add register_tools function to the module
            mock_module.register_tools = MagicMock()
            
            # Install the tool
            result = self.marketplace.install_tool(tool_id, mock_agent)
            
            # Verify the result
            self.assertTrue(result["success"])
            self.mock_repository.get_tool_metadata.assert_called_with(tool_id)
            self.mock_repository.download_tool.assert_called_with(tool_id)
            mock_import.assert_called_once()
            mock_module.register_tools.assert_called_with(mock_agent)
    
    def test_install_tool_not_found(self):
        """Test installing a non-existent tool."""
        # Mock the repository methods
        tool_id = "non-existent"
        
        self.mock_repository.get_tool_metadata.return_value = None
        
        # Install the tool
        result = self.marketplace.install_tool(tool_id)
        
        # Verify the result
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], f"Tool {tool_id} not found in repository")
    
    def test_install_tool_download_failed(self):
        """Test installing a tool with download failure."""
        # Mock the repository methods
        tool_id = "test-tool"
        
        self.mock_repository.get_tool_metadata.return_value = ToolMetadata.from_dict({
            "tool_id": tool_id,
            "name": "Test Tool",
            "version": "1.0.0",
            "description": "A test tool"
        })
        
        self.mock_repository.download_tool.return_value = None
        
        # Install the tool
        result = self.marketplace.install_tool(tool_id)
        
        # Verify the result
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], f"Failed to download tool {tool_id}")
    
    def test_install_tool_no_register_function(self):
        """Test installing a tool with no register_tools function."""
        # Mock the repository methods
        tool_id = "test-tool"
        
        self.mock_repository.get_tool_metadata.return_value = ToolMetadata.from_dict({
            "tool_id": tool_id,
            "name": "Test Tool",
            "version": "1.0.0",
            "description": "A test tool"
        })
        
        self.mock_repository.download_tool.return_value = Path(os.path.join(self.cache_dir, tool_id))
        
        # Mock the module
        with patch("me2ai_mcp.marketplace.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_import.return_value = mock_module
            
            # No register_tools function
            
            # Install the tool
            result = self.marketplace.install_tool(tool_id)
            
            # Verify the result
            self.assertFalse(result["success"])
            self.assertEqual(result["error"], f"No register_tools function found in {tool_id}")
    
    def test_install_tool_import_error(self):
        """Test installing a tool with import error."""
        # Mock the repository methods
        tool_id = "test-tool"
        
        self.mock_repository.get_tool_metadata.return_value = ToolMetadata.from_dict({
            "tool_id": tool_id,
            "name": "Test Tool",
            "version": "1.0.0",
            "description": "A test tool"
        })
        
        self.mock_repository.download_tool.return_value = Path(os.path.join(self.cache_dir, tool_id))
        
        # Mock the module import to raise an error
        with patch("me2ai_mcp.marketplace.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("Test error")
            
            # Install the tool
            result = self.marketplace.install_tool(tool_id)
            
            # Verify the result
            self.assertFalse(result["success"])
            self.assertTrue("Error importing tool" in result["error"])
    
    def test_uninstall_tool(self):
        """Test uninstalling a tool."""
        # Mock the repository methods
        tool_id = "test-tool"
        module_name = tool_id.replace("-", "_")
        
        metadata = ToolMetadata.from_dict({
            "tool_id": tool_id,
            "name": "Test Tool",
            "version": "1.0.0",
            "description": "A test tool"
        })
        metadata.installed = True
        metadata.install_path = os.path.join(self.cache_dir, tool_id)
        
        self.mock_repository.get_tool_metadata.return_value = metadata
        self.mock_repository.index = {tool_id: metadata}
        self.mock_repository.index_cache_path = os.path.join(self.cache_dir, "repository_index.json")
        
        # Add tools to the registry
        self.registry.tools = {
            f"{module_name}.tool1": MagicMock(),
            f"{module_name}.tool2": MagicMock(),
            "other_module.tool": MagicMock()
        }
        
        # Create the tool directory
        os.makedirs(metadata.install_path, exist_ok=True)
        
        # Uninstall the tool
        with patch("me2ai_mcp.marketplace.open", mock_open()) as mock_file:
            result = self.marketplace.uninstall_tool(tool_id)
            
            # Verify the result
            self.assertTrue(result["success"])
            self.assertEqual(len(result["unregistered_tools"]), 2)
            self.assertIn(f"{module_name}.tool1", result["unregistered_tools"])
            self.assertIn(f"{module_name}.tool2", result["unregistered_tools"])
            
            # Verify the registry was updated
            self.assertEqual(len(self.registry.tools), 1)
            self.assertIn("other_module.tool", self.registry.tools)
            
            # Verify the metadata was updated
            self.assertFalse(metadata.installed)
            self.assertIsNone(metadata.install_path)
    
    def test_uninstall_tool_not_found(self):
        """Test uninstalling a non-existent tool."""
        # Mock the repository methods
        tool_id = "non-existent"
        
        self.mock_repository.get_tool_metadata.return_value = None
        
        # Uninstall the tool
        result = self.marketplace.uninstall_tool(tool_id)
        
        # Verify the result
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], f"Tool {tool_id} not found in repository")
    
    def test_uninstall_tool_not_installed(self):
        """Test uninstalling a tool that is not installed."""
        # Mock the repository methods
        tool_id = "test-tool"
        
        metadata = ToolMetadata.from_dict({
            "tool_id": tool_id,
            "name": "Test Tool",
            "version": "1.0.0",
            "description": "A test tool"
        })
        metadata.installed = False
        
        self.mock_repository.get_tool_metadata.return_value = metadata
        
        # Uninstall the tool
        result = self.marketplace.uninstall_tool(tool_id)
        
        # Verify the result
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], f"Tool {tool_id} is not installed")
    
    def test_uninstall_tool_error(self):
        """Test uninstalling a tool with an error."""
        # Mock the repository methods
        tool_id = "test-tool"
        
        metadata = ToolMetadata.from_dict({
            "tool_id": tool_id,
            "name": "Test Tool",
            "version": "1.0.0",
            "description": "A test tool"
        })
        metadata.installed = True
        metadata.install_path = os.path.join(self.cache_dir, tool_id)
        
        self.mock_repository.get_tool_metadata.return_value = metadata
        
        # Mock the registry to raise an error
        self.registry.unregister_tool = MagicMock(side_effect=Exception("Test error"))
        
        # Uninstall the tool
        result = self.marketplace.uninstall_tool(tool_id)
        
        # Verify the result
        self.assertFalse(result["success"])
        self.assertTrue("Error uninstalling tool" in result["error"])


if __name__ == "__main__":
    unittest.main()
