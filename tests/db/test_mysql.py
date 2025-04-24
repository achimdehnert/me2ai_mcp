"""
Tests for the MySQL database connection and tools.

This module contains tests for the MySQL database functionality
in the ME2AI MCP package.
"""

import os
import json
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from me2ai_mcp.db.mysql import (
    MySQLConnection,
    ConnectionError,
    QueryError,
    SchemaError
)
from me2ai_mcp.tools.mysql import (
    ExecuteQueryTool,
    ListTablesTool,
    GetTableColumnsTool,
    GetDatabaseInfoTool
)


class TestMySQLConnection:
    """Tests for the MySQLConnection class."""
    
    @patch("me2ai_mcp.db.mysql.get_credentials")
    @patch("me2ai_mcp.db.mysql.pooling.MySQLConnectionPool")
    def test_should_initialize_connection_pool(self, mock_pool, mock_get_credentials):
        """Test that the connection pool is initialized correctly."""
        # Arrange
        mock_credentials = MagicMock()
        mock_credentials.host = "test-host"
        mock_credentials.port = 3306
        mock_credentials.username = "test-user"
        mock_credentials.password = "test-pass"
        
        mock_get_credentials.return_value = mock_credentials
        
        # Act
        connection = MySQLConnection(
            env_prefix="TEST",
            default_schema="test_db",
            pool_size=3
        )
        
        # Assert
        mock_get_credentials.assert_called_once_with(
            env_prefix="TEST",
            credential_file=None,
            connection_name=None
        )
        
        mock_pool.assert_called_once()
        # Verify pool configuration
        args, kwargs = mock_pool.call_args
        assert kwargs["pool_name"] == "me2ai_mysql_pool"
        assert kwargs["pool_size"] == 3
        assert kwargs["host"] == "test-host"
        assert kwargs["port"] == 3306
        assert kwargs["user"] == "test-user"
        assert kwargs["password"] == "test-pass"
        assert kwargs["database"] == "test_db"
        
        assert connection._pool is not None
    
    @patch("me2ai_mcp.db.mysql.get_credentials")
    @patch("me2ai_mcp.db.mysql.pooling.MySQLConnectionPool")
    def test_should_handle_connection_error(self, mock_pool, mock_get_credentials):
        """Test that connection errors are handled correctly."""
        # Arrange
        mock_credentials = MagicMock()
        mock_get_credentials.return_value = mock_credentials
        
        # Simulate connection error
        mock_pool.side_effect = Exception("Connection failed")
        
        # Act & Assert
        with pytest.raises(ConnectionError):
            MySQLConnection(env_prefix="TEST")
    
    @patch("me2ai_mcp.db.mysql.get_credentials")
    @patch("me2ai_mcp.db.mysql.pooling.MySQLConnectionPool")
    def test_should_validate_schema(self, mock_pool, mock_get_credentials):
        """Test that schema validation works correctly."""
        # Arrange
        mock_credentials = MagicMock()
        mock_get_credentials.return_value = mock_credentials
        
        connection = MySQLConnection(
            allowed_schemas=["valid_schema"]
        )
        
        # Act & Assert
        # Valid schema
        connection._validate_schema("valid_schema")
        
        # Invalid schema (not in allowed list)
        with pytest.raises(SchemaError):
            connection._validate_schema("invalid_schema")
        
        # Invalid schema format
        with pytest.raises(SchemaError):
            connection._validate_schema("invalid;schema")
    
    @patch("me2ai_mcp.db.mysql.get_credentials")
    @patch("me2ai_mcp.db.mysql.pooling.MySQLConnectionPool")
    def test_should_execute_query(self, mock_pool, mock_get_credentials):
        """Test that queries are executed correctly."""
        # Arrange
        mock_credentials = MagicMock()
        mock_get_credentials.return_value = mock_credentials
        
        # Mock connection and cursor
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        
        # Set up fetch results
        mock_cursor.with_rows = True
        mock_cursor.fetchall.return_value = [
            {"id": 1, "name": "Test 1"},
            {"id": 2, "name": "Test 2"}
        ]
        
        # Set up pool to return our mock connection
        mock_pool_instance = MagicMock()
        mock_pool_instance.get_connection.return_value = mock_connection
        mock_pool.return_value = mock_pool_instance
        
        # Create connection instance
        connection = MySQLConnection()
        
        # Act
        result = connection.execute_query(
            query="SELECT * FROM test_table",
            schema="test_schema"
        )
        
        # Assert
        mock_cursor.execute.assert_any_call("USE `test_schema`")
        mock_cursor.execute.assert_any_call("SELECT * FROM test_table", params=None)
        mock_cursor.fetchall.assert_called_once()
        mock_connection.commit.assert_called_once()
        mock_cursor.close.assert_called_once()
        mock_connection.close.assert_called_once()
        
        assert result["success"] is True
        assert len(result["rows"]) == 2
        assert result["rowCount"] == 2
        assert result["schema"] == "test_schema"
        assert "executionTime" in result
    
    @patch("me2ai_mcp.db.mysql.get_credentials")
    @patch("me2ai_mcp.db.mysql.pooling.MySQLConnectionPool")
    def test_should_handle_query_error(self, mock_pool, mock_get_credentials):
        """Test that query errors are handled correctly."""
        # Arrange
        mock_credentials = MagicMock()
        mock_get_credentials.return_value = mock_credentials
        
        # Mock connection and cursor
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        
        # Simulate query error
        mock_cursor.execute.side_effect = Exception("Query failed")
        
        # Set up pool to return our mock connection
        mock_pool_instance = MagicMock()
        mock_pool_instance.get_connection.return_value = mock_connection
        mock_pool.return_value = mock_pool_instance
        
        # Create connection instance
        connection = MySQLConnection()
        
        # Act & Assert
        with pytest.raises(QueryError):
            connection.execute_query("SELECT * FROM test_table")
        
        # Verify rollback was called
        mock_connection.rollback.assert_called_once()


class TestMySQLTools:
    """Tests for the MySQL tools."""
    
    @pytest.fixture
    def mock_mysql_connection(self):
        """Fixture for mocked MySQL connection."""
        with patch("me2ai_mcp.tools.mysql.MySQLConnection") as mock:
            # Set up mock instance
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            
            yield mock_instance
    
    async def test_should_execute_query_tool(self, mock_mysql_connection):
        """Test that ExecuteQueryTool works correctly."""
        # Arrange
        mock_mysql_connection.execute_query.return_value = {
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
        mock_mysql_connection.execute_query.assert_called_once_with(
            query="SELECT * FROM test_table",
            params=None,
            schema="test_schema",
            max_rows=50
        )
        
        assert result["success"] is True
        assert "query" in result
        assert len(result["rows"]) == 1
        assert result["rowCount"] == 1
    
    async def test_should_handle_query_error(self, mock_mysql_connection):
        """Test that ExecuteQueryTool handles errors correctly."""
        # Arrange
        mock_mysql_connection.execute_query.side_effect = QueryError("Query failed")
        
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
    
    async def test_should_list_tables(self, mock_mysql_connection):
        """Test that ListTablesTool works correctly."""
        # Arrange
        mock_mysql_connection.list_tables.return_value = {
            "success": True,
            "schema": "test_schema",
            "tables": [
                {"name": "table1", "type": "BASE TABLE", "columnCount": 5},
                {"name": "table2", "type": "VIEW", "columnCount": 3}
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
        mock_mysql_connection.list_tables.assert_called_once_with(
            schema="test_schema",
            include_views=True
        )
        
        assert result["success"] is True
        assert result["schema"] == "test_schema"
        assert len(result["tables"]) == 2
        assert result["count"] == 2
    
    async def test_should_get_table_columns(self, mock_mysql_connection):
        """Test that GetTableColumnsTool works correctly."""
        # Arrange
        mock_mysql_connection.get_table_columns.return_value = {
            "success": True,
            "schema": "test_schema",
            "table": "test_table",
            "columns": [
                {"name": "id", "dataType": "int", "isNullable": "NO"},
                {"name": "name", "dataType": "varchar", "isNullable": "YES"}
            ],
            "primaryKeys": ["id"],
            "count": 2
        }
        
        tool = GetTableColumnsTool()
        
        # Act
        result = await tool.execute(
            table="test_table",
            schema="test_schema"
        )
        
        # Assert
        mock_mysql_connection.get_table_columns.assert_called_once_with(
            table="test_table",
            schema="test_schema"
        )
        
        assert result["success"] is True
        assert result["schema"] == "test_schema"
        assert result["table"] == "test_table"
        assert len(result["columns"]) == 2
        assert result["primaryKeys"] == ["id"]
        assert result["count"] == 2
    
    async def test_should_get_database_info(self, mock_mysql_connection):
        """Test that GetDatabaseInfoTool works correctly."""
        # Arrange
        mock_mysql_connection.get_database_info.return_value = {
            "success": True,
            "version": "8.0.28",
            "databases": ["db1", "db2"],
            "defaultDatabase": "db1",
            "connection": {
                "host": "localhost",
                "port": 3306,
                "user": "testuser"
            }
        }
        
        tool = GetDatabaseInfoTool()
        
        # Act
        result = await tool.execute()
        
        # Assert
        mock_mysql_connection.get_database_info.assert_called_once()
        
        assert result["success"] is True
        assert result["version"] == "8.0.28"
        assert "db1" in result["databases"]
        assert result["defaultDatabase"] == "db1"
