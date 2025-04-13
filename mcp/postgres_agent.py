"""Postgres database agent for ME2AI MCP server.

This module provides a Model Context Protocol server that exposes
Postgres database operations as tools that can be used by AI assistants.
"""
import os
import logging
import psycopg2
import psycopg2.extras
from typing import Dict, Any, List, Optional, Union, Callable
from .server import ME2AIMCPServer

logger = logging.getLogger("me2ai-mcp.postgres")

class PostgresDatabaseAgent(ME2AIMCPServer):
    """MCP server that provides Postgres database functionality.
    
    This agent connects to a PostgreSQL database and provides tools for
    executing queries, listing tables, describing table structures,
    and listing functions.
    """
    
    def __init__(self) -> None:
        """Initialize the Postgres database agent.
        
        Connects to the PostgreSQL database using environment variables and
        sets up the available tools.
        """
        super().__init__()
        self.connection: Optional[psycopg2.extensions.connection] = None
        self.connect_to_db()
    
    def connect_to_db(self) -> None:
        """Connect to the Postgres database using environment variables.
        
        Retrieves connection parameters from environment variables and
        establishes a connection to the PostgreSQL database. Supports both
        connection URI and individual parameters.
        """
        # First check if a URI/URL is provided
        uri = os.getenv("POSTGRES_URI") or os.getenv("POSTGRES_URL")
        
        if uri:
            # Connect using URI
            logger.info(f"Connecting to PostgreSQL using URI")
            try:
                self.connection = psycopg2.connect(uri)
                db_info = self.connection.info.dbname
                host_info = self.connection.info.host
                logger.info(f"Successfully connected to PostgreSQL database: {db_info} at {host_info}")
                return
            except Exception as e:
                logger.error(f"Error connecting to PostgreSQL using URI: {str(e)}")
                # Fall back to individual parameters
        
        # Get individual connection parameters
        host = os.getenv("POSTGRES_HOST")
        port = os.getenv("POSTGRES_PORT")
        user = os.getenv("POSTGRES_USER")
        password = os.getenv("POSTGRES_PASSWORD")
        database = os.getenv("POSTGRES_DATABASE")
        
        # Validate required parameters
        if not all([host, database, user, password]):
            error_msg = "Missing required database connection parameters"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            self.connection = psycopg2.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database
            )
            logger.info(f"Successfully connected to PostgreSQL database: {database} at {host}")
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL database: {str(e)}")
            # Don't expose detailed error message outside this function
            raise ConnectionError("Failed to connect to PostgreSQL database. Check your credentials.")
    
    def _setup_tools(self) -> None:
        """Register database tools with the MCP server.
        
        Sets up the available database tools that can be used by AI assistants
        through the Model Context Protocol.
        """
        # Register query execution tool
        self.register_tool(
            "execute_query",
            "Execute a SQL query on the Postgres database",
            self.execute_query,
            {
                "query": {"type": "string", "description": "SQL query to execute"},
                "schema": {"type": "string", "description": "Database schema to use (e.g., 'poco' or 'poco-test')"}
            }
        )
        
        # Register table listing tool
        self.register_tool(
            "list_tables",
            "List all tables in a specific schema",
            self.list_tables,
            {
                "schema": {"type": "string", "description": "Database schema to list tables from (e.g., 'poco' or 'poco-test')"}
            }
        )
        
        # Register table structure tool
        self.register_tool(
            "describe_table",
            "Describe the structure of a table",
            self.describe_table,
            {
                "schema": {"type": "string", "description": "Database schema (e.g., 'poco' or 'poco-test')"},
                "table": {"type": "string", "description": "Name of the table to describe"}
            }
        )
        
        # Register function listing tool
        self.register_tool(
            "list_functions",
            "List all functions in a specific schema",
            self.list_functions,
            {
                "schema": {"type": "string", "description": "Database schema to list functions from (e.g., 'poco' or 'poco-test')"}
            }
        )
    
    def execute_query(self, query: str, schema: str) -> Dict[str, Any]:
        """Execute a SQL query on the database.
        
        Args:
            query: SQL query to execute
            schema: Database schema to use (e.g., 'poco' or 'poco-test')
            
        Returns:
            Query results with the following structure:
            - success: Boolean indicating if the query was successful
            - count/affected_rows: Number of rows returned or affected
            - results: List of query results (for SELECT queries)
            - error: Error message if query failed
        """
        if not self.connection:
            return {"error": "No database connection available"}
        
        # Sanitize the schema name to prevent SQL injection
        if not schema.replace('-', '').replace('_', '').isalnum():
            return {
                "success": False,
                "error": "Invalid schema name format"
            }
            
        # Set the search path to the specified schema
        schema_query = f'SET search_path TO "{schema}";'
        
        try:
            # Create cursor with dictionary result
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Execute schema setting and user query
            cursor.execute(schema_query)
            cursor.execute(query)
            
            # Check if query returns results
            if cursor.description:
                results = cursor.fetchall()
                # Convert to list of dictionaries
                results_list = [dict(row) for row in results]
                
                self.connection.commit()
                cursor.close()
                
                # Limit result size to prevent issues
                max_results = 100
                truncated = len(results_list) > max_results
                
                return {
                    "success": True,
                    "count": len(results_list),
                    "truncated": truncated,
                    "results": results_list[:max_results] if truncated else results_list
                }
            else:
                # Query executed successfully but no results (e.g., INSERT, UPDATE)
                affected_rows = cursor.rowcount
                self.connection.commit()
                cursor.close()
                
                return {
                    "success": True,
                    "affected_rows": affected_rows,
                    "message": f"Query executed successfully. {affected_rows} rows affected."
                }
                
        except Exception as e:
            self.connection.rollback()
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_tables(self, schema: str) -> Dict[str, Any]:
        """List all tables in a specific schema.
        
        Args:
            schema: Database schema to list tables from
            
        Returns:
            List of tables
        """
        if not self.connection:
            return {"error": "No database connection available"}
        
        query = """
        SELECT table_name, table_type
        FROM information_schema.tables
        WHERE table_schema = %s
        ORDER BY table_name;
        """
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(query, (schema,))
            tables = cursor.fetchall()
            cursor.close()
            
            return {
                "success": True,
                "schema": schema,
                "tables": [dict(table) for table in tables]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def describe_table(self, schema: str, table: str) -> Dict[str, Any]:
        """Describe the structure of a table.
        
        Args:
            schema: Database schema
            table: Name of the table to describe
            
        Returns:
            Table structure description
        """
        if not self.connection:
            return {"error": "No database connection available"}
        
        column_query = """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position;
        """
        
        constraint_query = """
        SELECT con.conname as constraint_name, 
               con.contype as constraint_type,
               pg_get_constraintdef(con.oid) as constraint_definition
        FROM pg_constraint con
        JOIN pg_class rel ON rel.oid = con.conrelid
        JOIN pg_namespace nsp ON nsp.oid = rel.relnamespace
        WHERE nsp.nspname = %s AND rel.relname = %s;
        """
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get columns
            cursor.execute(column_query, (schema, table))
            columns = cursor.fetchall()
            
            # Get constraints
            cursor.execute(constraint_query, (schema, table))
            constraints = cursor.fetchall()
            
            cursor.close()
            
            return {
                "success": True,
                "schema": schema,
                "table": table,
                "columns": [dict(col) for col in columns],
                "constraints": [dict(con) for con in constraints]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_functions(self, schema: str) -> Dict[str, Any]:
        """List all functions in a specific schema.
        
        Args:
            schema: Database schema to list functions from
            
        Returns:
            List of functions
        """
        if not self.connection:
            return {"error": "No database connection available"}
        
        query = """
        SELECT 
            p.proname as function_name,
            pg_get_function_arguments(p.oid) as arguments,
            t.typname as return_type,
            d.description as description
        FROM pg_proc p
        JOIN pg_namespace n ON p.pronamespace = n.oid
        JOIN pg_type t ON p.prorettype = t.oid
        LEFT JOIN pg_description d ON d.objoid = p.oid
        WHERE n.nspname = %s
        ORDER BY p.proname;
        """
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(query, (schema,))
            functions = cursor.fetchall()
            cursor.close()
            
            return {
                "success": True,
                "schema": schema,
                "functions": [dict(func) for func in functions]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def close(self) -> None:
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            logger.info("PostgreSQL database connection closed.")
