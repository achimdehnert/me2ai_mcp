"""
MySQL tools for ME2AI MCP.

This module provides tools for interacting with MySQL databases
based on the flexible database connection system in ME2AI MCP.
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Set
import logging
import os
import json
from pathlib import Path
from dataclasses import dataclass

from me2ai_mcp.base import BaseTool
from me2ai_mcp.db.mysql import MySQLConnection, MySQLError, QueryError, SchemaError


@dataclass
class ExecuteQueryTool(BaseTool):
    """Tool for executing SQL queries on MySQL databases.
    
    This tool provides a flexible interface for executing
    queries on MySQL databases with schema validation
    and safe parameter handling.
    """
    
    name: str = "execute_query"
    description: str = "Execute a SQL query on a MySQL database"
    
    def __init__(
        self,
        env_prefix: str = "MYSQL",
        credential_file: Optional[Union[str, Path]] = None,
        connection_name: Optional[str] = None,
        allowed_schemas: Optional[List[str]] = None,
        default_schema: Optional[str] = None,
        max_rows: int = 50
    ) -> None:
        """Initialize the query execution tool.
        
        Args:
            env_prefix: Prefix for environment variables
            credential_file: Path to JSON credentials file
            connection_name: Connection name to use in credentials file
            allowed_schemas: List of allowed schemas
            default_schema: Default schema to use
            max_rows: Maximum number of rows to return
        """
        super().__init__()
        
        # Create database connection
        self.db = MySQLConnection(
            env_prefix=env_prefix,
            credential_file=credential_file,
            connection_name=connection_name,
            allowed_schemas=allowed_schemas,
            default_schema=default_schema
        )
        
        self.max_rows = max_rows
        
        # Set up logging
        self.logger = logging.getLogger(f"me2ai-mcp-tools-mysql-{self.name}")
    
    async def execute(
        self,
        query: str,
        schema: Optional[str] = None,
        params: Optional[Union[List, Tuple, Dict]] = None
    ) -> Dict[str, Any]:
        """Execute a SQL query.
        
        Args:
            query: SQL query to execute
            schema: Database schema to use
            params: Query parameters
            
        Returns:
            Dict[str, Any]: Query results with success status
            
        Raises:
            QueryError: If query execution fails
        """
        self.logger.info(f"Executing query on schema '{schema or self.db.default_schema}'")
        
        try:
            # Execute query
            result = self.db.execute_query(
                query=query,
                params=params,
                schema=schema,
                max_rows=self.max_rows
            )
            
            # Add query for reference
            result["query"] = query
            
            return result
            
        except SchemaError as e:
            self.logger.error(f"Schema error: {str(e)}")
            return {
                "success": False,
                "error": f"Schema error: {str(e)}"
            }
            
        except QueryError as e:
            self.logger.error(f"Query error: {str(e)}")
            return {
                "success": False,
                "error": f"Query error: {str(e)}"
            }
            
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }


@dataclass
class ListTablesTool(BaseTool):
    """Tool for listing tables in a MySQL schema.
    
    This tool provides information about tables and views
    in a specified MySQL schema.
    """
    
    name: str = "list_tables"
    description: str = "List all tables in a MySQL schema"
    
    def __init__(
        self,
        env_prefix: str = "MYSQL",
        credential_file: Optional[Union[str, Path]] = None,
        connection_name: Optional[str] = None,
        allowed_schemas: Optional[List[str]] = None,
        default_schema: Optional[str] = None
    ) -> None:
        """Initialize the list tables tool.
        
        Args:
            env_prefix: Prefix for environment variables
            credential_file: Path to JSON credentials file
            connection_name: Connection name to use in credentials file
            allowed_schemas: List of allowed schemas
            default_schema: Default schema to use
        """
        super().__init__()
        
        # Create database connection
        self.db = MySQLConnection(
            env_prefix=env_prefix,
            credential_file=credential_file,
            connection_name=connection_name,
            allowed_schemas=allowed_schemas,
            default_schema=default_schema
        )
        
        # Set up logging
        self.logger = logging.getLogger(f"me2ai-mcp-tools-mysql-{self.name}")
    
    async def execute(
        self,
        schema: Optional[str] = None,
        include_views: bool = True
    ) -> Dict[str, Any]:
        """List tables in a schema.
        
        Args:
            schema: Schema to list tables from
            include_views: Whether to include views in results
            
        Returns:
            Dict[str, Any]: List of tables with success status
            
        Raises:
            SchemaError: If schema is invalid
        """
        self.logger.info(
            f"Listing tables in schema '{schema or self.db.default_schema}'"
        )
        
        try:
            # Get tables
            result = self.db.list_tables(
                schema=schema,
                include_views=include_views
            )
            
            return result
            
        except SchemaError as e:
            self.logger.error(f"Schema error: {str(e)}")
            return {
                "success": False,
                "error": f"Schema error: {str(e)}"
            }
            
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }


@dataclass
class GetTableColumnsTool(BaseTool):
    """Tool for getting column information for a table.
    
    This tool provides detailed information about columns
    in a specified MySQL table.
    """
    
    name: str = "get_table_columns"
    description: str = "Get column information for a MySQL table"
    
    def __init__(
        self,
        env_prefix: str = "MYSQL",
        credential_file: Optional[Union[str, Path]] = None,
        connection_name: Optional[str] = None,
        allowed_schemas: Optional[List[str]] = None,
        default_schema: Optional[str] = None
    ) -> None:
        """Initialize the table columns tool.
        
        Args:
            env_prefix: Prefix for environment variables
            credential_file: Path to JSON credentials file
            connection_name: Connection name to use in credentials file
            allowed_schemas: List of allowed schemas
            default_schema: Default schema to use
        """
        super().__init__()
        
        # Create database connection
        self.db = MySQLConnection(
            env_prefix=env_prefix,
            credential_file=credential_file,
            connection_name=connection_name,
            allowed_schemas=allowed_schemas,
            default_schema=default_schema
        )
        
        # Set up logging
        self.logger = logging.getLogger(f"me2ai-mcp-tools-mysql-{self.name}")
    
    async def execute(
        self,
        table: str,
        schema: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get columns for a table.
        
        Args:
            table: Table name
            schema: Schema name
            
        Returns:
            Dict[str, Any]: Table columns information
            
        Raises:
            SchemaError: If schema is invalid
        """
        self.logger.info(
            f"Getting columns for table '{table}' in schema '{schema or self.db.default_schema}'"
        )
        
        try:
            # Get columns
            result = self.db.get_table_columns(
                table=table,
                schema=schema
            )
            
            return result
            
        except SchemaError as e:
            self.logger.error(f"Schema error: {str(e)}")
            return {
                "success": False,
                "error": f"Schema error: {str(e)}"
            }
            
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }


@dataclass
class GetDatabaseInfoTool(BaseTool):
    """Tool for getting information about the MySQL database.
    
    This tool provides metadata about the MySQL server,
    available schemas, and connection details.
    """
    
    name: str = "get_database_info"
    description: str = "Get information about the MySQL database server and available schemas"
    
    def __init__(
        self,
        env_prefix: str = "MYSQL",
        credential_file: Optional[Union[str, Path]] = None,
        connection_name: Optional[str] = None,
        allowed_schemas: Optional[List[str]] = None,
        default_schema: Optional[str] = None
    ) -> None:
        """Initialize the database info tool.
        
        Args:
            env_prefix: Prefix for environment variables
            credential_file: Path to JSON credentials file
            connection_name: Connection name to use in credentials file
            allowed_schemas: List of allowed schemas
            default_schema: Default schema to use
        """
        super().__init__()
        
        # Create database connection
        self.db = MySQLConnection(
            env_prefix=env_prefix,
            credential_file=credential_file,
            connection_name=connection_name,
            allowed_schemas=allowed_schemas,
            default_schema=default_schema
        )
        
        # Set up logging
        self.logger = logging.getLogger(f"me2ai-mcp-tools-mysql-{self.name}")
    
    async def execute(self) -> Dict[str, Any]:
        """Get information about the database.
        
        Returns:
            Dict[str, Any]: Database information
            
        Raises:
            QueryError: If query execution fails
        """
        self.logger.info("Getting database information")
        
        try:
            # Get database info
            result = self.db.get_database_info()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
