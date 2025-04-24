"""
Tests for the PostgreSQL database connection and tools.

This module contains tests for the PostgreSQL database functionality
in the ME2AI MCP package.
"""

import os
import json
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from me2ai_mcp.db.postgres import (
    PostgreSQLConnection,
    PostgreSQLError,
    QueryError,
    SchemaError
)
from me2ai_mcp.tools.postgres import (
    ExecuteQueryTool,
    ListTablesTool,
    GetTableColumnsTool,
    GetPLZDetailsTool
)


class TestPostgreSQLConnection:
    """Tests for the PostgreSQLConnection class."""
    
    @patch("me2ai_mcp.db.postgres.get_credentials")
    @patch("me2ai_mcp.db.postgres.psycopg2.connect")
    def test_should_initialize_connection(self, mock_connect, mock_get_credentials):
        """Test that the connection is initialized correctly."""
        # Arrange
        mock_credentials = MagicMock()
        mock_credentials.host = "test-host"
        mock_credentials.port = 5432
        mock_credentials.username = "test-user"
        mock_credentials.password = "test-pass"
        
        mock_get_credentials.return_value = mock_credentials
        
        # Act
        connection = PostgreSQLConnection(
            env_prefix="TEST",
            default_schema="test_schema",
            connect_timeout=5
        )
        
        # Assert
        mock_get_credentials.assert_called_once_with(
            env_prefix="TEST",
            credential_file=None,
            connection_name=None
        )
        
        mock_connect.assert_called_once()
        # Verify connection parameters
        args, kwargs = mock_connect.call_args
        assert kwargs["host"] == "test-host"
        assert kwargs["port"] == 5432
        assert kwargs["user"] == "test-user"
        assert kwargs["password"] == "test-pass"
        assert kwargs["connect_timeout"] == 5
        
        assert connection._conn is not None
    
    @patch("me2ai_mcp.db.postgres.get_credentials")
    @patch("me2ai_mcp.db.postgres.psycopg2.connect")
    def test_should_handle_connection_error(self, mock_connect, mock_get_credentials):
        """Test that connection errors are handled correctly."""
        # Arrange
        mock_credentials = MagicMock()
        mock_get_credentials.return_value = mock_credentials
        
        # Simulate connection error
        mock_connect.side_effect = Exception("Connection failed")
        
        # Act & Assert
        with pytest.raises(PostgreSQLError):
            PostgreSQLConnection(env_prefix="TEST")
    
    @patch("me2ai_mcp.db.postgres.get_credentials")
    @patch("me2ai_mcp.db.postgres.psycopg2.connect")
    def test_should_validate_schema(self, mock_connect, mock_get_credentials):
        """Test that schema validation works correctly."""
        # Arrange
        mock_credentials = MagicMock()
        mock_get_credentials.return_value = mock_credentials
        
        # Mock successful connection
        mock_connect.return_value = MagicMock()
        
        connection = PostgreSQLConnection(
            allowed_schemas=["valid_schema", "poco", "poco-test"]
        )
        
        # Act & Assert
        # Valid schema
        connection.validate_schema("valid_schema")
        
        # Valid schema with hyphen
        connection.validate_schema("poco-test")
        
        # Invalid schema (not in allowed list)
        with pytest.raises(SchemaError):
            connection.validate_schema("invalid_schema")
        
        # Invalid schema format
        with pytest.raises(SchemaError):
            connection.validate_schema("invalid;schema")
    
    @patch("me2ai_mcp.db.postgres.get_credentials")
    @patch("me2ai_mcp.db.postgres.psycopg2.connect")
    def test_should_execute_query(self, mock_connect, mock_get_credentials):
        """Test that queries are executed correctly."""
        # Arrange
        mock_credentials = MagicMock()
        mock_get_credentials.return_value = mock_credentials
        
        # Mock connection and cursor
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_connection
        
        # Set up fetch results
        mock_cursor.fetchall.return_value = [
            {"id": 1, "name": "Test 1"},
            {"id": 2, "name": "Test 2"}
        ]
        
        # Create connection instance
        connection = PostgreSQLConnection()
        
        # Act
        result = connection.execute_query(
            query="SELECT * FROM test_table",
            schema="test_schema"
        )
        
        # Assert
        mock_cursor.execute.assert_any_call('SET search_path TO "test_schema";')
        mock_cursor.execute.assert_any_call("SELECT * FROM test_table", None)
        mock_cursor.fetchall.assert_called_once()
        
        assert result["success"] is True
        assert len(result["rows"]) == 2
        assert result["rowCount"] == 2
        assert result["schema"] == "test_schema"
        assert "executionTime" in result
    
    @patch("me2ai_mcp.db.postgres.get_credentials")
    @patch("me2ai_mcp.db.postgres.psycopg2.connect")
    def test_should_handle_query_error(self, mock_connect, mock_get_credentials):
        """Test that query errors are handled correctly."""
        # Arrange
        mock_credentials = MagicMock()
        mock_get_credentials.return_value = mock_credentials
        
        # Mock connection and cursor
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_connection
        
        # Simulate query error
        mock_cursor.execute.side_effect = Exception("Query failed")
        
        # Create connection instance
        connection = PostgreSQLConnection()
        
        # Act & Assert
        with pytest.raises(QueryError):
            connection.execute_query("SELECT * FROM test_table", "test_schema")


class TestPostgreSQLTools:
    """Tests for the PostgreSQL tools."""
    
    @pytest.fixture
    def mock_postgres_connection(self):
        """Fixture for mocked PostgreSQL connection."""
        with patch("me2ai_mcp.tools.postgres.PostgreSQLConnection") as mock:
            # Set up mock instance
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            
            yield mock_instance
    
    async def test_should_execute_query_tool(self, mock_postgres_connection):
        """Test that ExecuteQueryTool works correctly."""
        # Arrange
        mock_postgres_connection.execute_query.return_value = {
            "success": True,
            "rows": [{"id": 1, "name": "Test"}],
            "rowCount": 1,
            "schema": "test_schema"
        }
        
        tool = ExecuteQueryTool()
        
        # Act
        result = await tool.execute(
            query="SELECT * FROM test_table",
            schema="test_schema"
        )
        
        # Assert
        mock_postgres_connection.execute_query.assert_called_once_with(
            query="SELECT * FROM test_table",
            params=None,
            schema="test_schema",
            max_rows=50
        )
        
        assert result["success"] is True
        assert "query" in result
        assert len(result["rows"]) == 1
        assert result["rowCount"] == 1
    
    async def test_should_handle_query_error(self, mock_postgres_connection):
        """Test that ExecuteQueryTool handles errors correctly."""
        # Arrange
        mock_postgres_connection.execute_query.side_effect = QueryError("Query failed")
        
        tool = ExecuteQueryTool()
        
        # Act
        result = await tool.execute(
            query="SELECT * FROM test_table",
            schema="test_schema"
        )
        
        # Assert
        assert result["success"] is False
        assert "error" in result
        assert "Query failed" in result["error"]
    
    async def test_should_list_tables(self, mock_postgres_connection):
        """Test that ListTablesTool works correctly."""
        # Arrange
        mock_postgres_connection.list_tables.return_value = {
            "success": True,
            "schema": "test_schema",
            "tables": [
                {"name": "table1", "type": "TABLE", "columns": 5},
                {"name": "table2", "type": "VIEW", "columns": 3}
            ],
            "count": 2
        }
        
        tool = ListTablesTool()
        
        # Act
        result = await tool.execute(
            schema="test_schema",
            include_views=True
        )
        
        # Assert
        mock_postgres_connection.list_tables.assert_called_once_with(
            schema="test_schema",
            include_views=True
        )
        
        assert result["success"] is True
        assert result["schema"] == "test_schema"
        assert len(result["tables"]) == 2
        assert result["count"] == 2
    
    async def test_should_get_table_columns(self, mock_postgres_connection):
        """Test that GetTableColumnsTool works correctly."""
        # Arrange
        mock_postgres_connection.get_table_columns.return_value = {
            "success": True,
            "schema": "test_schema",
            "table": "test_table",
            "columns": [
                {"name": "id", "data_type": "integer", "is_nullable": "NO"},
                {"name": "name", "data_type": "character varying", "is_nullable": "YES"}
            ],
            "primaryKey": "id",
            "count": 2
        }
        
        tool = GetTableColumnsTool()
        
        # Act
        result = await tool.execute(
            table="test_table",
            schema="test_schema"
        )
        
        # Assert
        mock_postgres_connection.get_table_columns.assert_called_once_with(
            table="test_table",
            schema="test_schema"
        )
        
        assert result["success"] is True
        assert result["schema"] == "test_schema"
        assert result["table"] == "test_table"
        assert len(result["columns"]) == 2
        assert result["primaryKey"] == "id"
        assert result["count"] == 2
    
    async def test_should_get_plz_details(self, mock_postgres_connection):
        """Test that GetPLZDetailsTool works correctly."""
        # Arrange
        mock_postgres_connection.get_plz_details.return_value = {
            "success": True,
            "plz": "12345",
            "details": {
                "plz": "12345",
                "ort": "Test City",
                "bundesland": "Test State"
            }
        }
        
        tool = GetPLZDetailsTool()
        
        # Act
        result = await tool.execute(
            plz="12345",
            schema="poco"
        )
        
        # Assert
        mock_postgres_connection.get_plz_details.assert_called_once_with(
            plz="12345",
            schema="poco"
        )
        
        assert result["success"] is True
        assert result["plz"] == "12345"
        assert result["details"]["plz"] == "12345"
        assert result["details"]["ort"] == "Test City"
