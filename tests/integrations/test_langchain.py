"""
Tests for the LangChain integration module.

This module contains tests for the LangChain integration functionality
in the ME2AI MCP package.
"""

import pytest
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock, AsyncMock

from me2ai_mcp.base import BaseTool
from me2ai_mcp.integrations.langchain import (
    LangChainToolAdapter,
    create_langchain_tools,
    LangChainToolFactory
)


class DummyTool(BaseTool):
    """Dummy tool for testing purposes."""
    
    name: str = "dummy_tool"
    description: str = "A dummy tool for testing"
    
    async def execute(self, param1: str, param2: int = 42) -> Dict[str, Any]:
        """Execute the dummy tool.
        
        Args:
            param1: First parameter
            param2: Second parameter
            
        Returns:
            Dict[str, Any]: Result
        """
        return {
            "success": True,
            "param1": param1,
            "param2": param2
        }


class TestLangChainToolAdapter:
    """Tests for the LangChainToolAdapter class."""
    
    def test_should_create_adapter_from_mcp_tool(self):
        """Test that an adapter can be created from an MCP tool."""
        # Arrange
        mcp_tool = DummyTool()
        
        # Act
        adapter = LangChainToolAdapter(mcp_tool)
        
        # Assert
        assert adapter.name == "dummy_tool"
        assert adapter.description == "A dummy tool for testing"
        assert adapter.mcp_tool == mcp_tool
        assert adapter.args_schema is not None
    
    def test_should_override_name_and_description(self):
        """Test that name and description can be overridden."""
        # Arrange
        mcp_tool = DummyTool()
        
        # Act
        adapter = LangChainToolAdapter(
            mcp_tool=mcp_tool,
            name="custom_name",
            description="Custom description"
        )
        
        # Assert
        assert adapter.name == "custom_name"
        assert adapter.description == "Custom description"
    
    def test_should_create_args_schema(self):
        """Test that an args schema is created correctly."""
        # Arrange
        mcp_tool = DummyTool()
        
        # Act
        adapter = LangChainToolAdapter(mcp_tool)
        schema = adapter.args_schema
        
        # Assert
        assert schema is not None
        
        # Check that schema has the right fields
        assert hasattr(schema, "__annotations__")
        annotations = schema.__annotations__
        assert "param1" in annotations
        assert "param2" in annotations
        
        # Check default values
        schema_instance = schema()
        assert schema_instance.param2 == 42  # Default value from DummyTool
    
    @pytest.mark.asyncio
    async def test_should_run_tool(self):
        """Test that the tool can be run."""
        # Arrange
        mcp_tool = DummyTool()
        adapter = LangChainToolAdapter(mcp_tool)
        
        # Mock execute method to track calls
        mcp_tool.execute = AsyncMock(return_value={
            "success": True,
            "param1": "test",
            "param2": 42
        })
        
        # Act
        result = adapter._run(param1="test", param2=42)
        
        # Assert
        mcp_tool.execute.assert_called_once_with(param1="test", param2=42)
        assert result["success"] is True
        assert result["param1"] == "test"
        assert result["param2"] == 42
    
    @pytest.mark.asyncio
    async def test_should_handle_json_string_input(self):
        """Test that the tool can handle JSON string input."""
        # Arrange
        mcp_tool = DummyTool()
        adapter = LangChainToolAdapter(mcp_tool)
        
        # Mock execute method to track calls
        mcp_tool.execute = AsyncMock(return_value={
            "success": True,
            "param1": "test",
            "param2": 99
        })
        
        # Act - Pass a JSON string
        result = adapter._run('{"param1": "test", "param2": 99}')
        
        # Assert
        mcp_tool.execute.assert_called_once_with(param1="test", param2=99)
        assert result["success"] is True
        assert result["param1"] == "test"
        assert result["param2"] == 99
    
    @pytest.mark.asyncio
    async def test_should_handle_error(self):
        """Test that the tool handles errors correctly."""
        # Arrange
        mcp_tool = DummyTool()
        adapter = LangChainToolAdapter(mcp_tool)
        
        # Mock execute method to raise an exception
        mcp_tool.execute = AsyncMock(side_effect=Exception("Tool execution failed"))
        
        # Act & Assert
        with pytest.raises(Exception):
            adapter._run(param1="test")


class TestToolHelpers:
    """Tests for the tool helper functions."""
    
    def test_should_create_langchain_tools(self):
        """Test that LangChain tools can be created from MCP tools."""
        # Arrange
        mcp_tools = [
            DummyTool(),
            DummyTool(name="another_tool", description="Another dummy tool")
        ]
        
        # Act
        lc_tools = create_langchain_tools(mcp_tools)
        
        # Assert
        assert len(lc_tools) == 2
        assert isinstance(lc_tools[0], LangChainToolAdapter)
        assert isinstance(lc_tools[1], LangChainToolAdapter)
        assert lc_tools[0].name == "dummy_tool"
        assert lc_tools[1].name == "another_tool"
    
    @patch("me2ai_mcp.integrations.langchain.create_langchain_tools")
    def test_should_create_postgres_tools(self, mock_create_tools):
        """Test that PostgreSQL tools can be created."""
        # Arrange
        mock_create_tools.return_value = ["postgres_tool1", "postgres_tool2"]
        
        # Act
        with patch("me2ai_mcp.integrations.langchain.ExecuteQueryTool"):
            with patch("me2ai_mcp.integrations.langchain.ListTablesTool"):
                with patch("me2ai_mcp.integrations.langchain.GetTableColumnsTool"):
                    with patch("me2ai_mcp.integrations.langchain.GetPLZDetailsTool"):
                        result = LangChainToolFactory.create_postgres_tools()
        
        # Assert
        assert mock_create_tools.called
        assert len(result) == 2
        assert result[0] == "postgres_tool1"
        assert result[1] == "postgres_tool2"
    
    @patch("me2ai_mcp.integrations.langchain.create_langchain_tools")
    def test_should_create_mysql_tools(self, mock_create_tools):
        """Test that MySQL tools can be created."""
        # Arrange
        mock_create_tools.return_value = ["mysql_tool1", "mysql_tool2"]
        
        # Act
        with patch("me2ai_mcp.integrations.langchain.ExecuteQueryTool"):
            with patch("me2ai_mcp.integrations.langchain.ListTablesTool"):
                with patch("me2ai_mcp.integrations.langchain.GetTableColumnsTool"):
                    with patch("me2ai_mcp.integrations.langchain.GetDatabaseInfoTool"):
                        result = LangChainToolFactory.create_mysql_tools()
        
        # Assert
        assert mock_create_tools.called
        assert len(result) == 2
        assert result[0] == "mysql_tool1"
        assert result[1] == "mysql_tool2"
    
    @patch("me2ai_mcp.integrations.langchain.LangChainToolFactory.create_postgres_tools")
    @patch("me2ai_mcp.integrations.langchain.LangChainToolFactory.create_mysql_tools")
    def test_should_create_database_tools(self, mock_mysql, mock_postgres):
        """Test that database tools can be created based on type."""
        # Arrange
        mock_postgres.return_value = ["postgres_tool"]
        mock_mysql.return_value = ["mysql_tool"]
        
        # Act - PostgreSQL
        result_pg = LangChainToolFactory.create_database_tools("postgres")
        
        # Act - MySQL
        result_mysql = LangChainToolFactory.create_database_tools("mysql")
        
        # Assert
        mock_postgres.assert_called_once()
        mock_mysql.assert_called_once()
        
        assert result_pg == ["postgres_tool"]
        assert result_mysql == ["mysql_tool"]
