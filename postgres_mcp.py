"""Standalone PostgreSQL MCP server for ME2AI.

This script runs a Model Context Protocol (MCP) server that provides
access to a PostgreSQL database for AI assistants like Cascade.
"""
import os
import sys
import asyncio
import signal
import logging
import traceback
from dotenv import load_dotenv

# Configure robust logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("postgres-mcp")

try:
    # Import MCP and PostgreSQL libraries
    from mcp.server import MCPServer
    from mcp.server.schema import ToolDefinition, FunctionDefinition
    import psycopg2
    import psycopg2.extras
    logger.info("Successfully imported required libraries")
except ImportError as e:
    logger.error(f"Failed to import required libraries: {e}")
    logger.error("Please install required packages: pip install mcp psycopg2-binary")
    sys.exit(1)


class PostgreSQLMCPServer(MCPServer):
    """MCP server for PostgreSQL database access."""
    
    def __init__(self):
        """Initialize the PostgreSQL MCP server."""
        super().__init__()
        self.connection = None
        self._register_tools()
        logger.info("PostgreSQL MCP server initialized")
    
    def _register_tools(self):
        """Register database tools with the MCP server."""
        # Query execution tool
        query_tool = ToolDefinition(
            name="execute_query",
            description="Execute a SQL query on the PostgreSQL database",
            function=FunctionDefinition(
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string", 
                            "description": "SQL query to execute"
                        },
                        "schema": {
                            "type": "string", 
                            "description": "Database schema to use (e.g., 'poco' or 'poco-test')"
                        }
                    },
                    "required": ["query", "schema"]
                }
            )
        )
        
        # List tables tool
        tables_tool = ToolDefinition(
            name="list_tables",
            description="List all tables in a specific schema",
            function=FunctionDefinition(
                parameters={
                    "type": "object",
                    "properties": {
                        "schema": {
                            "type": "string", 
                            "description": "Database schema to list tables from (e.g., 'poco' or 'poco-test')"
                        }
                    },
                    "required": ["schema"]
                }
            )
        )
        
        # PLZ details tool - specific to the system based on memory
        plz_tool = ToolDefinition(
            name="get_plz_details",
            description="Get details for a specific postal code (PLZ)",
            function=FunctionDefinition(
                parameters={
                    "type": "object",
                    "properties": {
                        "plz": {
                            "type": "string",
                            "description": "Postal code (PLZ) to look up"
                        },
                        "schema": {
                            "type": "string",
                            "description": "Database schema to use (e.g., 'poco' or 'poco-test')"
                        }
                    },
                    "required": ["plz", "schema"]
                }
            )
        )
        
        # Register all tools
        self.registry.register_tool(query_tool, self.execute_query)
        self.registry.register_tool(tables_tool, self.list_tables)
        self.registry.register_tool(plz_tool, self.get_plz_details)
        
        logger.info(f"Registered {len(self.registry.list_tools())} database tools")
    
    async def connect_to_database(self):
        """Establish connection to the PostgreSQL database.
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        # Load environment variables
        load_dotenv()
        logger.info("Environment variables loaded")
        
        # First try connection string if available
        uri = os.getenv("POSTGRES_URI") or os.getenv("POSTGRES_URL")
        
        if uri:
            try:
                logger.info("Attempting connection using URI...")
                self.connection = psycopg2.connect(uri)
                logger.info("✓ Successfully connected to database using URI")
                return True
            except Exception as e:
                logger.error(f"URI connection failed: {str(e)}")
                # Fall back to individual parameters
        
        # Try connection with individual parameters
        logger.info("Attempting connection using individual parameters...")
        host = os.getenv("POSTGRES_HOST")
        port = os.getenv("POSTGRES_PORT")
        user = os.getenv("POSTGRES_USER")
        password = os.getenv("POSTGRES_PASSWORD")
        database = os.getenv("POSTGRES_DATABASE")
        
        # Check if we have all required parameters
        if not all([host, user, password, database]):
            logger.error("Missing required database connection parameters")
            logger.error(f"Host: {host or 'MISSING'}")
            logger.error(f"User: {user or 'MISSING'}")
            logger.error(f"Database: {database or 'MISSING'}")
            logger.error(f"Password: {'PROVIDED' if password else 'MISSING'}")
            return False
        
        try:
            # Connect using individual parameters
            self.connection = psycopg2.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database
            )
            logger.info(f"✓ Successfully connected to {database} at {host}")
            
            # Test schema access
            cursor = self.connection.cursor()
            for schema in ['poco', 'poco-test']:
                try:
                    cursor.execute(f'SET search_path TO "{schema}";')
                    cursor.execute(
                        "SELECT COUNT(*) FROM information_schema.tables "
                        f"WHERE table_schema = '{schema}'"
                    )
                    count = cursor.fetchone()[0]
                    logger.info(f"✓ Access to '{schema}' schema confirmed ({count} tables)")
                except Exception as e:
                    logger.warning(f"Could not access '{schema}' schema: {str(e)}")
            
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            return False
    
    async def execute_query(self, query, schema):
        """Execute a SQL query on the database.
        
        Args:
            query: SQL query to execute
            schema: Database schema to use
            
        Returns:
            Query results dictionary
        """
        if not self.connection:
            return {"error": "No database connection available"}
        
        # Validate schema name
        if not schema.replace('-', '').replace('_', '').isalnum():
            return {"error": "Invalid schema name format"}
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Set schema and execute query
            cursor.execute(f'SET search_path TO "{schema}";')
            cursor.execute(query)
            
            # Handle results
            if cursor.description:
                results = [dict(row) for row in cursor.fetchall()]
                
                # Limit results to prevent huge responses
                max_results = 50
                truncated = len(results) > max_results
                
                self.connection.commit()
                cursor.close()
                
                return {
                    "success": True,
                    "count": len(results),
                    "truncated": truncated,
                    "results": results[:max_results] if truncated else results
                }
            else:
                # For non-SELECT queries (INSERT, UPDATE, etc.)
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
            error_message = str(e)
            logger.error(f"Query execution error: {error_message}")
            return {
                "success": False,
                "error": error_message
            }
    
    async def list_tables(self, schema):
        """List all tables in a specific schema.
        
        Args:
            schema: Database schema to list tables from
            
        Returns:
            Dictionary containing list of tables
        """
        if not self.connection:
            return {"error": "No database connection available"}
        
        # Validate schema name
        if not schema.replace('-', '').replace('_', '').isalnum():
            return {"error": "Invalid schema name format"}
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(
                "SELECT table_name, table_type "
                "FROM information_schema.tables "
                "WHERE table_schema = %s "
                "ORDER BY table_name",
                (schema,)
            )
            
            tables = [dict(row) for row in cursor.fetchall()]
            cursor.close()
            
            return {
                "success": True,
                "schema": schema,
                "count": len(tables),
                "tables": tables
            }
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error listing tables: {error_message}")
            return {
                "success": False,
                "error": error_message
            }
    
    async def get_plz_details(self, plz, schema):
        """Get details for a specific postal code (PLZ).
        
        Args:
            plz: Postal code to look up
            schema: Database schema to use
            
        Returns:
            Dictionary containing PLZ details
        """
        if not self.connection:
            return {"error": "No database connection available"}
        
        # Validate schema name
        if not schema.replace('-', '').replace('_', '').isalnum():
            return {"error": "Invalid schema name format"}
        
        # Validate PLZ format (should be numeric)
        if not plz.isdigit():
            return {"error": "PLZ should be numeric"}
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Set schema
            cursor.execute(f'SET search_path TO "{schema}";')
            
            # Try method 1: Call the function with schema prefix
            try:
                query = f'SELECT * FROM "{schema}".fn_plz_details(%s)'
                logger.info(f"Executing PLZ query with schema prefix: {query}")
                cursor.execute(query, (plz,))
                
                results = [dict(row) for row in cursor.fetchall()]
                cursor.close()
                
                return {
                    "success": True,
                    "plz": plz,
                    "count": len(results),
                    "results": results
                }
                
            except Exception as fn_error:
                logger.warning(f"First PLZ function call failed: {str(fn_error)}")
                
                # Try method 2: Look up from plz tables directly
                try:
                    # Try to find PLZ in poc_markt_plz table which we know exists
                    cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                    cursor.execute(f'SET search_path TO "{schema}";')
                    
                    plz_query = f'''
                        SELECT * FROM "{schema}".poc_markt_plz 
                        WHERE plz = %s 
                        LIMIT 50
                    '''
                    
                    logger.info(f"Executing alternative PLZ query from tables: {plz_query}")
                    cursor.execute(plz_query, (plz,))
                    
                    plz_results = [dict(row) for row in cursor.fetchall()]
                    cursor.close()
                    
                    return {
                        "success": True,
                        "method": "direct_table_lookup",
                        "plz": plz,
                        "count": len(plz_results),
                        "results": plz_results,
                        "note": "PLZ function unavailable, using direct table lookup instead"
                    }
                    
                except Exception as table_error:
                    logger.error(f"Alternative PLZ lookup failed: {str(table_error)}")
                    return {
                        "success": False,
                        "error": f"Primary PLZ function error: {str(fn_error)}\nAlternative lookup error: {str(table_error)}"
                    }
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error getting PLZ details: {error_message}")
            return {
                "success": False,
                "error": error_message
            }
    
    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")


async def main():
    """Run the PostgreSQL MCP server."""
    try:
        logger.info("===== Starting PostgreSQL MCP Server =====")
        logger.info(f"Current working directory: {os.getcwd()}")
        
        # Create server
        server = PostgreSQLMCPServer()
        
        # Connect to database
        connection_success = await server.connect_to_database()
        if not connection_success:
            logger.error("Failed to connect to database, exiting")
            return
        
        # Register signal handlers
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, shutting down")
            asyncio.create_task(shutdown(server))
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start server
        await server.start()
        logger.info("✓ MCP Server started successfully")
        
        # Log available tools
        tools = server.registry.list_tools()
        logger.info(f"Available tools ({len(tools)}):")
        for tool in tools:
            logger.info(f"- {tool.name}: {tool.description}")
        
        logger.info("Waiting for tool calls...")
        
        # Keep server running indefinitely
        while True:
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


async def shutdown(server):
    """Shut down the server gracefully."""
    logger.info("Shutting down MCP Server")
    server.close()
    await server.stop()
    logger.info("Server stopped")


if __name__ == "__main__":
    # Run the main function and keep the script alive
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Script terminated by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(traceback.format_exc())
