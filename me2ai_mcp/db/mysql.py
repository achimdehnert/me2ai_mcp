"""
MySQL connection and query management for ME2AI MCP.

This module provides flexible MySQL database connection
and query management with support for multiple credential
sources and enhanced error handling.
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Set
import logging
import os
import json
from pathlib import Path
import re
import time
from contextlib import contextmanager

try:
    import mysql.connector
    from mysql.connector import pooling
    from mysql.connector.cursor import MySQLCursor
    from mysql.connector.errors import Error as MySQLError
except ImportError:
    raise ImportError(
        "MySQL Connector not installed. Install with: pip install mysql-connector-python"
    )

from me2ai_mcp.db.credentials import (
    DatabaseCredentials,
    CredentialSourceType,
    get_credentials
)


class MySQLError(Exception):
    """Base class for MySQL-related errors."""
    pass


class ConnectionError(MySQLError):
    """Error establishing a database connection."""
    pass


class QueryError(MySQLError):
    """Error executing a database query."""
    pass


class SchemaError(MySQLError):
    """Error related to database schema operations."""
    pass


class MySQLConnection:
    """Flexible MySQL database connection manager.
    
    This class provides a robust interface for connecting to
    MySQL databases with support for multiple credential sources,
    connection pooling, and enhanced error handling.
    
    Attributes:
        env_prefix: Prefix for environment variables
        credential_file: Path to JSON credentials file
        connection_name: Connection name to use in credentials file
        allowed_schemas: List of allowed schemas
        default_schema: Default schema to use
        pool_size: Connection pool size
        pool_name: Connection pool name
        connect_timeout: Connection timeout in seconds
        logger: Logger instance
    """
    
    def __init__(
        self,
        env_prefix: str = "MYSQL",
        credential_file: Optional[Union[str, Path]] = None,
        connection_name: Optional[str] = None,
        allowed_schemas: Optional[List[str]] = None,
        default_schema: Optional[str] = None,
        pool_size: int = 5,
        pool_name: str = "me2ai_mysql_pool",
        connect_timeout: int = 10
    ) -> None:
        """Initialize the MySQL connection manager.
        
        Args:
            env_prefix: Prefix for environment variables
            credential_file: Path to JSON credentials file
            connection_name: Connection name to use in credentials file
            allowed_schemas: List of allowed schemas
            default_schema: Default schema to use
            pool_size: Connection pool size
            pool_name: Connection pool name
            connect_timeout: Connection timeout in seconds
            
        Raises:
            ConnectionError: If connection initialization fails
        """
        # Configure connection parameters
        self.env_prefix = env_prefix
        self.credential_file = credential_file
        self.connection_name = connection_name
        self.allowed_schemas = allowed_schemas or []
        self.default_schema = default_schema
        self.pool_size = pool_size
        self.pool_name = pool_name
        self.connect_timeout = connect_timeout
        
        # Set up logging
        self.logger = logging.getLogger("me2ai-mcp-mysql")
        
        # Initialize connection pool
        self._pool = None
        self._init_connection_pool()
    
    def _init_connection_pool(self) -> None:
        """Initialize the connection pool.
        
        Raises:
            ConnectionError: If pool initialization fails
        """
        # Get database credentials
        try:
            self.credentials = get_credentials(
                env_prefix=self.env_prefix,
                credential_file=self.credential_file,
                connection_name=self.connection_name
            )
            
            # Log credential source
            self.logger.info(
                f"Using credentials from {self.credentials.source_type.name}"
            )
            
            # Prepare connection config
            config = {
                "host": self.credentials.host,
                "port": self.credentials.port,
                "user": self.credentials.username,
                "password": self.credentials.password,
                "connect_timeout": self.connect_timeout
            }
            
            # Add database if provided
            if self.default_schema:
                config["database"] = self.default_schema
            
            # Initialize connection pool
            self._pool = pooling.MySQLConnectionPool(
                pool_name=self.pool_name,
                pool_size=self.pool_size,
                **config
            )
            
            self.logger.info(
                f"Initialized MySQL connection pool ({self.pool_size} connections)"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize connection pool: {str(e)}")
            raise ConnectionError(f"Failed to initialize connection pool: {str(e)}")
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool.
        
        Yields:
            mysql.connector.connection.MySQLConnection: Database connection
            
        Raises:
            ConnectionError: If connection acquisition fails
        """
        connection = None
        
        try:
            # Get connection from pool
            connection = self._pool.get_connection()
            
            # Yield connection to caller
            yield connection
            
        except MySQLError as e:
            self.logger.error(f"Database connection error: {str(e)}")
            raise ConnectionError(f"Database connection error: {str(e)}")
            
        finally:
            # Return connection to pool
            if connection:
                try:
                    connection.close()
                except:
                    pass
    
    @contextmanager
    def get_cursor(self, schema: Optional[str] = None, dictionary: bool = True):
        """Get a database cursor.
        
        Args:
            schema: Schema to use
            dictionary: Whether to return results as dictionaries
            
        Yields:
            MySQLCursor: Database cursor
            
        Raises:
            ConnectionError: If connection acquisition fails
            SchemaError: If schema is invalid
        """
        # Validate schema if provided
        if schema:
            self._validate_schema(schema)
        
        # Use context manager for connection
        with self.get_connection() as connection:
            cursor = None
            
            try:
                # Create cursor
                cursor = connection.cursor(dictionary=dictionary)
                
                # Set schema if provided
                if schema:
                    cursor.execute(f"USE `{schema}`")
                
                # Yield cursor to caller
                yield cursor
                
                # Commit any pending transactions
                connection.commit()
                
            except MySQLError as e:
                # Rollback on error
                if connection:
                    connection.rollback()
                
                self.logger.error(f"Database cursor error: {str(e)}")
                raise QueryError(f"Database cursor error: {str(e)}")
                
            finally:
                # Close cursor
                if cursor:
                    try:
                        cursor.close()
                    except:
                        pass
    
    def _validate_schema(self, schema: str) -> None:
        """Validate a schema name.
        
        Args:
            schema: Schema name to validate
            
        Raises:
            SchemaError: If schema is invalid
        """
        # Check for valid schema name
        if not re.match(r'^[a-zA-Z0-9_-]+$', schema):
            raise SchemaError(f"Invalid schema name: {schema}")
        
        # Check if schema is allowed
        if self.allowed_schemas and schema not in self.allowed_schemas:
            allowed = ", ".join(self.allowed_schemas)
            raise SchemaError(
                f"Schema '{schema}' not allowed. Allowed schemas: {allowed}"
            )
    
    def execute_query(
        self,
        query: str,
        params: Optional[Union[List, Tuple, Dict]] = None,
        schema: Optional[str] = None,
        max_rows: int = 50
    ) -> Dict[str, Any]:
        """Execute a SQL query.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            schema: Schema to use
            max_rows: Maximum number of rows to return
            
        Returns:
            Dict[str, Any]: Query results
            
        Raises:
            QueryError: If query execution fails
            SchemaError: If schema is invalid
        """
        # Start timer for performance tracking
        start_time = time.time()
        
        try:
            # Execute query
            with self.get_cursor(schema=schema) as cursor:
                cursor.execute(query, params=params)
                
                # Check if query returns results
                if cursor.with_rows:
                    # Fetch results
                    results = cursor.fetchall()
                    
                    # Check if results were truncated
                    truncated = len(results) > max_rows
                    
                    # Truncate results if needed
                    if truncated:
                        results = results[:max_rows]
                    
                    # Calculate execution time
                    execution_time = time.time() - start_time
                    
                    # Return results
                    return {
                        "success": True,
                        "rows": results,
                        "rowCount": len(results),
                        "truncated": truncated,
                        "maxRows": max_rows,
                        "executionTime": round(execution_time, 3),
                        "schema": schema or self.default_schema
                    }
                else:
                    # No results, but might be an update/insert/delete
                    rows_affected = cursor.rowcount
                    
                    # Calculate execution time
                    execution_time = time.time() - start_time
                    
                    # Return affected rows
                    return {
                        "success": True,
                        "rowsAffected": rows_affected,
                        "executionTime": round(execution_time, 3),
                        "schema": schema or self.default_schema
                    }
                    
        except SchemaError:
            # Re-raise schema errors
            raise
            
        except Exception as e:
            self.logger.error(f"Query execution error: {str(e)}")
            raise QueryError(f"Query execution error: {str(e)}")
    
    def list_tables(
        self,
        schema: Optional[str] = None,
        include_views: bool = True
    ) -> Dict[str, Any]:
        """List tables in a schema.
        
        Args:
            schema: Schema to list tables from
            include_views: Whether to include views
            
        Returns:
            Dict[str, Any]: List of tables
            
        Raises:
            QueryError: If query execution fails
            SchemaError: If schema is invalid
        """
        # Use schema if provided, otherwise use default
        target_schema = schema or self.default_schema
        
        # Validate schema
        if target_schema:
            self._validate_schema(target_schema)
        else:
            raise SchemaError("No schema specified and no default schema set")
        
        try:
            # Build query based on include_views flag
            if include_views:
                query = """
                SELECT 
                    TABLE_NAME as name, 
                    TABLE_TYPE as type,
                    (
                        SELECT COUNT(*) 
                        FROM INFORMATION_SCHEMA.COLUMNS 
                        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = name
                    ) as columnCount
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = %s
                ORDER BY TABLE_NAME
                """
            else:
                query = """
                SELECT 
                    TABLE_NAME as name, 
                    TABLE_TYPE as type,
                    (
                        SELECT COUNT(*) 
                        FROM INFORMATION_SCHEMA.COLUMNS 
                        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = name
                    ) as columnCount
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = %s AND TABLE_TYPE = 'BASE TABLE'
                ORDER BY TABLE_NAME
                """
            
            # Execute query
            with self.get_cursor() as cursor:
                cursor.execute(query, (target_schema, target_schema))
                tables = cursor.fetchall()
                
                # Return tables
                return {
                    "success": True,
                    "schema": target_schema,
                    "tables": tables,
                    "count": len(tables)
                }
                
        except SchemaError:
            # Re-raise schema errors
            raise
            
        except Exception as e:
            self.logger.error(f"Failed to list tables: {str(e)}")
            raise QueryError(f"Failed to list tables: {str(e)}")
    
    def get_table_columns(
        self,
        table: str,
        schema: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get columns for a table.
        
        Args:
            table: Table name
            schema: Schema name
            
        Returns:
            Dict[str, Any]: Table columns
            
        Raises:
            QueryError: If query execution fails
            SchemaError: If schema is invalid
        """
        # Use schema if provided, otherwise use default
        target_schema = schema or self.default_schema
        
        # Validate schema
        if target_schema:
            self._validate_schema(target_schema)
        else:
            raise SchemaError("No schema specified and no default schema set")
        
        try:
            # Build query for column information
            query = """
            SELECT 
                COLUMN_NAME as name, 
                DATA_TYPE as dataType,
                CHARACTER_MAXIMUM_LENGTH as maxLength,
                IS_NULLABLE as isNullable,
                COLUMN_DEFAULT as defaultValue,
                COLUMN_KEY as keyType,
                COLUMN_COMMENT as description
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            ORDER BY ORDINAL_POSITION
            """
            
            # Execute query
            with self.get_cursor() as cursor:
                cursor.execute(query, (target_schema, table))
                columns = cursor.fetchall()
                
                # Check if table exists
                if not columns:
                    return {
                        "success": False,
                        "error": f"Table '{table}' not found in schema '{target_schema}'"
                    }
                
                # Get primary key information
                pk_query = """
                SELECT 
                    k.COLUMN_NAME
                FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS t
                JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE k
                ON t.CONSTRAINT_NAME = k.CONSTRAINT_NAME
                WHERE t.CONSTRAINT_TYPE = 'PRIMARY KEY'
                AND t.TABLE_SCHEMA = %s
                AND t.TABLE_NAME = %s
                """
                
                cursor.execute(pk_query, (target_schema, table))
                primary_keys = [row["COLUMN_NAME"] for row in cursor.fetchall()]
                
                # Return columns
                return {
                    "success": True,
                    "schema": target_schema,
                    "table": table,
                    "columns": columns,
                    "primaryKeys": primary_keys,
                    "count": len(columns)
                }
                
        except SchemaError:
            # Re-raise schema errors
            raise
            
        except Exception as e:
            self.logger.error(f"Failed to get table columns: {str(e)}")
            raise QueryError(f"Failed to get table columns: {str(e)}")
            
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the database.
        
        Returns:
            Dict[str, Any]: Database information
            
        Raises:
            QueryError: If query execution fails
        """
        try:
            # Execute query
            with self.get_cursor() as cursor:
                # Get MySQL version
                cursor.execute("SELECT VERSION() as version")
                version = cursor.fetchone()["version"]
                
                # Get database list
                cursor.execute("SHOW DATABASES")
                databases = [row["Database"] for row in cursor.fetchall()]
                
                # Filter databases if allowed_schemas is set
                if self.allowed_schemas:
                    databases = [db for db in databases if db in self.allowed_schemas]
                
                # Return database information
                return {
                    "success": True,
                    "version": version,
                    "databases": databases,
                    "defaultDatabase": self.default_schema,
                    "connection": {
                        "host": self.credentials.host,
                        "port": self.credentials.port,
                        "user": self.credentials.username
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get database info: {str(e)}")
            raise QueryError(f"Failed to get database info: {str(e)}")
