"""
PostgreSQL tools for ME2AI MCP.

This module provides tools for interacting with PostgreSQL databases
based on the flexible database connection system in ME2AI MCP.
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Set
import logging
import os
import json
from pathlib import Path
from dataclasses import dataclass

from me2ai_mcp.base import BaseTool
from me2ai_mcp.db.postgres import PostgreSQLConnection, PostgreSQLError, QueryError, SchemaError


@dataclass
class ExecuteQueryTool(BaseTool):
    """Tool for executing SQL queries on PostgreSQL databases.
    
    This tool provides a flexible interface for executing
    queries on PostgreSQL databases with schema validation
    and safe parameter handling.
    """
    
    name: str = "execute_query"
    description: str = "Execute a SQL query on a PostgreSQL database"
    
    def __init__(
        self,
        env_prefix: str = "POSTGRES",
        credential_file: Optional[Union[str, Path]] = None,
        connection_name: Optional[str] = None,
        allowed_schemas: Optional[List[str]] = None,
        default_schema: str = "poco",
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
        self.db = PostgreSQLConnection(
            env_prefix=env_prefix,
            credential_file=credential_file,
            connection_name=connection_name,
            allowed_schemas=allowed_schemas,
            default_schema=default_schema
        )
        
        self.max_rows = max_rows
        
        # Set up logging
        self.logger = logging.getLogger(f"me2ai-mcp-tools-postgres-{self.name}")
    
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
    """Tool for listing tables in a PostgreSQL schema.
    
    This tool provides information about tables and views
    in a specified PostgreSQL schema.
    """
    
    name: str = "list_tables"
    description: str = "List all tables in a PostgreSQL schema"
    
    def __init__(
        self,
        env_prefix: str = "POSTGRES",
        credential_file: Optional[Union[str, Path]] = None,
        connection_name: Optional[str] = None,
        allowed_schemas: Optional[List[str]] = None,
        default_schema: str = "poco"
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
        self.db = PostgreSQLConnection(
            env_prefix=env_prefix,
            credential_file=credential_file,
            connection_name=connection_name,
            allowed_schemas=allowed_schemas,
            default_schema=default_schema
        )
        
        # Set up logging
        self.logger = logging.getLogger(f"me2ai-mcp-tools-postgres-{self.name}")
    
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
    in a specified PostgreSQL table.
    """
    
    name: str = "get_table_columns"
    description: str = "Get column information for a PostgreSQL table"
    
    def __init__(
        self,
        env_prefix: str = "POSTGRES",
        credential_file: Optional[Union[str, Path]] = None,
        connection_name: Optional[str] = None,
        allowed_schemas: Optional[List[str]] = None,
        default_schema: str = "poco"
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
        self.db = PostgreSQLConnection(
            env_prefix=env_prefix,
            credential_file=credential_file,
            connection_name=connection_name,
            allowed_schemas=allowed_schemas,
            default_schema=default_schema
        )
        
        # Set up logging
        self.logger = logging.getLogger(f"me2ai-mcp-tools-postgres-{self.name}")
    
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
class GetPLZDetailsTool(BaseTool):
    """Tool for getting postal code (PLZ) details.
    
    This tool uses the PLZ lookup functionality in the ME2AI
    database, with fallback mechanisms if the function is unavailable.
    """
    
    name: str = "get_plz_details"
    description: str = "Get details for a specific postal code (PLZ)"
    
    def __init__(
        self,
        env_prefix: str = "POSTGRES",
        credential_file: Optional[Union[str, Path]] = None,
        connection_name: Optional[str] = None,
        allowed_schemas: Optional[List[str]] = None,
        default_schema: str = "poco"
    ) -> None:
        """Initialize the PLZ details tool.
        
        Args:
            env_prefix: Prefix for environment variables
            credential_file: Path to JSON credentials file
            connection_name: Connection name to use in credentials file
            allowed_schemas: List of allowed schemas
            default_schema: Default schema to use
        """
        super().__init__()
        
        # Create database connection
        self.db = PostgreSQLConnection(
            env_prefix=env_prefix,
            credential_file=credential_file,
            connection_name=connection_name,
            allowed_schemas=allowed_schemas,
            default_schema=default_schema
        )
        
        # Set up logging
        self.logger = logging.getLogger(f"me2ai-mcp-tools-postgres-{self.name}")
    
    async def execute(
        self,
        plz: str,
        schema: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get details for a postal code (PLZ).
        
        Args:
            plz: Postal code to look up
            schema: Schema to use
            
        Returns:
            Dict[str, Any]: PLZ details
            
        Raises:
            ValueError: If PLZ is invalid
            SchemaError: If schema is invalid
        """
        self.logger.info(
            f"Looking up PLZ '{plz}' in schema '{schema or self.db.default_schema}'"
        )
        
        try:
            # Validate PLZ
            if not plz.isdigit():
                raise ValueError("PLZ must be numeric")
            
            # Get PLZ details
            result = self.db.get_plz_details(
                plz=plz,
                schema=schema
            )
            
            return result
            
        except ValueError as e:
            self.logger.error(f"Invalid PLZ: {str(e)}")
            return {
                "success": False,
                "error": f"Invalid PLZ: {str(e)}"
            }
            
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
