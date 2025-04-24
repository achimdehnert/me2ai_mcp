"""
PostgreSQL database integration for ME2AI MCP.

This module provides flexible PostgreSQL database connectivity
and utility functions for ME2AI MCP servers.
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Set
from pathlib import Path
import os
import time
import logging
from datetime import datetime, timedelta
import psycopg2
import psycopg2.extras
from psycopg2.extensions import connection as PgConnection
from psycopg2.extensions import cursor as PgCursor
from contextlib import contextmanager

from me2ai_mcp.db.credentials import PostgreSQLCredentialManager


class PostgreSQLError(Exception):
    """Base exception for PostgreSQL errors."""
    pass


class ConnectionError(PostgreSQLError):
    """Exception raised when connection fails."""
    pass


class QueryError(PostgreSQLError):
    """Exception raised when query execution fails."""
    pass


class SchemaError(PostgreSQLError):
    """Exception raised when schema validation fails."""
    pass


class PostgreSQLConnection:
    """Flexible connection manager for PostgreSQL databases.
    
    This class manages connections to PostgreSQL databases with support
    for credential management, connection pooling, automatic reconnection,
    and schema validation.
    
    Attributes:
        credential_manager: Manager for PostgreSQL credentials
        connection: Active database connection
        last_connection_time: Timestamp of last successful connection
        last_activity_time: Timestamp of last database activity
        allowed_schemas: Set of schemas that can be accessed
        default_schema: Default schema to use
        logger: Logger instance
    """
    
    def __init__(
        self,
        env_prefix: str = "POSTGRES",
        credential_file: Optional[Union[str, Path]] = None,
        connection_name: Optional[str] = None,
        allowed_schemas: Optional[List[str]] = None,
        default_schema: str = "public",
        reconnect_timeout: int = 60,
        idle_timeout: int = 300
    ) -> None:
        """Initialize the PostgreSQL connection manager.
        
        Args:
            env_prefix: Prefix for environment variables
            credential_file: Path to JSON credentials file
            connection_name: Connection name to use in credentials file
            allowed_schemas: List of allowed schemas (if None, no restriction)
            default_schema: Default schema to use
            reconnect_timeout: Seconds to wait before attempting reconnection
            idle_timeout: Seconds of inactivity before closing connection
        """
        # Set up logging
        self.logger = logging.getLogger("me2ai-mcp-postgres")
        
        # Set up credential manager
        self.credential_manager = PostgreSQLCredentialManager(
            env_prefix=env_prefix,
            credential_file=credential_file,
            connection_name=connection_name
        )
        
        # Initialize connection state
        self.connection = None
        self.last_connection_time = None
        self.last_activity_time = None
        self.last_error = None
        
        # Connection settings
        self.reconnect_timeout = reconnect_timeout
        self.idle_timeout = idle_timeout
        
        # Schema settings
        self.allowed_schemas = set(allowed_schemas) if allowed_schemas else None
        self.default_schema = self._validate_schema_name(default_schema)
        
        # Log initialization
        self.logger.info(
            f"Initialized PostgreSQL connection manager with "
            f"default schema '{self.default_schema}'"
        )
        if self.allowed_schemas:
            self.logger.info(
                f"Restricted to schemas: {', '.join(self.allowed_schemas)}"
            )
    
    def connect(self) -> bool:
        """Connect to the PostgreSQL database.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        # Check if we need to reconnect
        if self.connection and not self._should_reconnect():
            return True
        
        # Close existing connection if any
        if self.connection:
            self._close_connection()
        
        # Get connection parameters
        connection_params = self.credential_manager.get_connection_params()
        
        try:
            # Create new connection
            self.connection = psycopg2.connect(**connection_params)
            
            # Update connection timestamps
            now = datetime.now()
            self.last_connection_time = now
            self.last_activity_time = now
            
            # Log success
            db_name = connection_params.get("database", "unknown")
            host = connection_params.get("host", "unknown")
            self.logger.info(f"Connected to {db_name} at {host}")
            
            return True
            
        except Exception as e:
            self.last_error = str(e)
            self.logger.error(f"Connection failed: {str(e)}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the PostgreSQL database."""
        if self.connection:
            self._close_connection()
            self.logger.info("Disconnected from database")
    
    def _close_connection(self) -> None:
        """Close the database connection."""
        if self.connection:
            try:
                self.connection.close()
                self.connection = None
            except Exception as e:
                self.logger.warning(f"Error closing connection: {str(e)}")
                self.connection = None
    
    def _should_reconnect(self) -> bool:
        """Check if reconnection is needed.
        
        Returns:
            bool: True if reconnection is needed
        """
        # Always reconnect if no connection
        if not self.connection:
            return True
        
        # Check if connection is closed
        try:
            # Simple test query to check connection
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            
            # Update last activity time
            self.last_activity_time = datetime.now()
            
            # Check if idle timeout is exceeded
            if self.idle_timeout and self.last_activity_time:
                idle_seconds = (datetime.now() - self.last_activity_time).total_seconds()
                if idle_seconds > self.idle_timeout:
                    self.logger.info(
                        f"Connection idle for {idle_seconds}s, "
                        f"exceeding timeout of {self.idle_timeout}s"
                    )
                    return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Connection test failed: {str(e)}")
            
            # Check if reconnect timeout is exceeded
            if self.last_connection_time:
                time_since_last = (datetime.now() - self.last_connection_time).total_seconds()
                if time_since_last < self.reconnect_timeout:
                    self.logger.info(
                        f"Waiting {self.reconnect_timeout - time_since_last}s "
                        f"before reconnection attempt"
                    )
                    return False
            
            return True
    
    def _validate_schema_name(self, schema: str) -> str:
        """Validate a schema name.
        
        Args:
            schema: Schema name to validate
            
        Returns:
            str: Validated schema name
            
        Raises:
            SchemaError: If schema name is invalid
        """
        # Basic validation - remove any quotes and check for SQL injection
        schema = schema.replace('"', '').replace("'", '')
        
        # Check if schema is allowed
        if self.allowed_schemas and schema not in self.allowed_schemas:
            allowed = ", ".join(self.allowed_schemas)
            raise SchemaError(
                f"Schema '{schema}' is not allowed. "
                f"Allowed schemas: {allowed}"
            )
        
        # Check schema name validity
        if not schema.replace('-', '').replace('_', '').isalnum():
            raise SchemaError(f"Invalid schema name: {schema}")
        
        return schema
    
    @contextmanager
    def cursor(
        self, 
        schema: Optional[str] = None,
        cursor_factory: Any = None
    ) -> PgCursor:
        """Get a cursor for executing queries.
        
        Args:
            schema: Schema to use (if None, uses default_schema)
            cursor_factory: Optional cursor factory to use
            
        Yields:
            PgCursor: Database cursor
            
        Raises:
            ConnectionError: If connection fails
        """
        # Ensure we have a connection
        if not self.connection:
            if not self.connect():
                raise ConnectionError(
                    f"Could not connect to database: {self.last_error}"
                )
        
        # Get schema to use
        use_schema = schema or self.default_schema
        
        # Validate schema name
        use_schema = self._validate_schema_name(use_schema)
        
        try:
            # Create cursor
            if cursor_factory:
                cursor = self.connection.cursor(cursor_factory=cursor_factory)
            else:
                cursor = self.connection.cursor()
            
            # Set schema
            cursor.execute(f'SET search_path TO "{use_schema}";')
            
            # Update last activity time
            self.last_activity_time = datetime.now()
            
            try:
                # Yield cursor for use
                yield cursor
                
                # Commit if all went well
                self.connection.commit()
                
            except Exception as e:
                # Rollback on error
                self.connection.rollback()
                raise
                
            finally:
                # Always close cursor
                cursor.close()
                
        except Exception as e:
            self.logger.error(f"Database error: {str(e)}")
            
            # Try to reconnect on next operation
            self._close_connection()
            
            raise
    
    def execute_query(
        self, 
        query: str, 
        params: Optional[Union[List, Tuple, Dict]] = None,
        schema: Optional[str] = None,
        fetch_all: bool = True,
        as_dict: bool = True,
        max_rows: int = 50
    ) -> Dict[str, Any]:
        """Execute a SQL query.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            schema: Schema to use
            fetch_all: Whether to fetch all results
            as_dict: Whether to return results as dictionaries
            max_rows: Maximum number of rows to return
            
        Returns:
            Dict[str, Any]: Query results
            
        Raises:
            QueryError: If query execution fails
            SchemaError: If schema name is invalid
        """
        # Get cursor factory based on as_dict parameter
        cursor_factory = psycopg2.extras.RealDictCursor if as_dict else None
        
        try:
            # Get cursor
            with self.cursor(schema=schema, cursor_factory=cursor_factory) as cursor:
                # Execute query
                query_start = time.time()
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                query_time = time.time() - query_start
                
                # Handle results
                if cursor.description:
                    # Query returned results
                    if fetch_all:
                        results = cursor.fetchall()
                        
                        # Convert to list of dicts if using standard cursor
                        if not as_dict and results:
                            columns = [desc[0] for desc in cursor.description]
                            results = [
                                dict(zip(columns, row))
                                for row in results
                            ]
                        
                        # Limit results
                        truncated = len(results) > max_rows
                        limited_results = results[:max_rows] if truncated else results
                        
                        return {
                            "success": True,
                            "count": len(results),
                            "truncated": truncated,
                            "max_rows": max_rows,
                            "execution_time": query_time,
                            "results": limited_results
                        }
                    else:
                        # Don't fetch, just return metadata
                        return {
                            "success": True,
                            "has_results": True,
                            "execution_time": query_time
                        }
                else:
                    # Non-query statement (INSERT, UPDATE, etc.)
                    affected_rows = cursor.rowcount
                    
                    return {
                        "success": True,
                        "affected_rows": affected_rows,
                        "execution_time": query_time,
                        "message": f"Query executed successfully. {affected_rows} rows affected."
                    }
                
        except SchemaError as e:
            # Re-raise schema errors
            raise
            
        except Exception as e:
            raise QueryError(f"Query execution failed: {str(e)}")
    
    def list_tables(
        self, 
        schema: Optional[str] = None,
        include_views: bool = True
    ) -> Dict[str, Any]:
        """List tables in a schema.
        
        Args:
            schema: Schema to list tables from
            include_views: Whether to include views in results
            
        Returns:
            Dict[str, Any]: List of tables
            
        Raises:
            QueryError: If query execution fails
        """
        # Determine schema to use
        use_schema = schema or self.default_schema
        
        # SQL to get tables
        if include_views:
            table_types = "'BASE TABLE', 'VIEW'"
        else:
            table_types = "'BASE TABLE'"
            
        query = f"""
            SELECT
                table_name,
                table_type,
                (
                    SELECT count(*)
                    FROM information_schema.columns
                    WHERE table_schema = t.table_schema
                    AND table_name = t.table_name
                ) as column_count
            FROM
                information_schema.tables t
            WHERE
                table_schema = %s
                AND table_type IN ({table_types})
            ORDER BY
                table_name
        """
        
        try:
            # Execute query
            result = self.execute_query(
                query=query,
                params=[use_schema],
                schema="information_schema"
            )
            
            # Add schema to result
            result["schema"] = use_schema
            
            return result
            
        except Exception as e:
            raise QueryError(f"Could not list tables: {str(e)}")
    
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
            Dict[str, Any]: Table columns information
            
        Raises:
            QueryError: If query execution fails
        """
        # Determine schema to use
        use_schema = schema or self.default_schema
        
        # SQL to get columns
        query = """
            SELECT
                column_name,
                data_type,
                is_nullable,
                column_default,
                character_maximum_length,
                numeric_precision,
                numeric_scale
            FROM
                information_schema.columns
            WHERE
                table_schema = %s
                AND table_name = %s
            ORDER BY
                ordinal_position
        """
        
        try:
            # Execute query
            result = self.execute_query(
                query=query,
                params=[use_schema, table],
                schema="information_schema"
            )
            
            # Add table and schema to result
            result["table"] = table
            result["schema"] = use_schema
            
            return result
            
        except Exception as e:
            raise QueryError(f"Could not get table columns: {str(e)}")
    
    def get_plz_details(
        self,
        plz: str,
        schema: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get postal code (PLZ) details.
        
        Args:
            plz: Postal code to look up
            schema: Schema to use
            
        Returns:
            Dict[str, Any]: PLZ details
            
        Raises:
            QueryError: If query execution fails
            ValueError: If PLZ is invalid
        """
        # Validate PLZ
        if not plz.isdigit():
            raise ValueError("PLZ must be numeric")
        
        # Determine schema to use
        use_schema = schema or self.default_schema
        
        try:
            # Try to use the PLZ function if available
            try:
                # Call function with schema prefix
                query = f'SELECT * FROM "{use_schema}".fn_plz_details(%s)'
                result = self.execute_query(
                    query=query,
                    params=[plz],
                    schema=use_schema
                )
                
                # Add PLZ to result
                result["plz"] = plz
                result["method"] = "function"
                
                return result
                
            except QueryError:
                # Try to query the PLZ table directly
                self.logger.info(
                    f"PLZ function not available, falling back to table lookup"
                )
                
                # Try to find a PLZ table
                for table_name in ["poc_markt_plz", "plz", "postal_codes"]:
                    try:
                        # Check if table exists
                        columns = self.get_table_columns(
                            table=table_name,
                            schema=use_schema
                        )
                        
                        # If we get here, table exists
                        query = f'SELECT * FROM "{use_schema}"."{table_name}" WHERE plz = %s'
                        result = self.execute_query(
                            query=query,
                            params=[plz],
                            schema=use_schema
                        )
                        
                        # Add PLZ to result
                        result["plz"] = plz
                        result["method"] = f"table_{table_name}"
                        
                        return result
                        
                    except Exception:
                        # Table doesn't exist or error, try next one
                        pass
                
                # If we got here, no PLZ data found
                raise QueryError(f"Could not find PLZ data for {plz}")
                
        except Exception as e:
            raise QueryError(f"PLZ lookup failed: {str(e)}")
    
    def execute_raw_query(
        self,
        query: str,
        params: Optional[Union[List, Tuple, Dict]] = None,
        schema: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a raw SQL query with full control.
        
        This is a wrapper around execute_query that provides the
        same interface but with more descriptive naming to indicate
        that it's a raw, unfiltered query. Use with caution!
        
        Args:
            query: SQL query to execute
            params: Query parameters
            schema: Schema to use
            
        Returns:
            Dict[str, Any]: Query results
        """
        return self.execute_query(
            query=query,
            params=params,
            schema=schema
        )
