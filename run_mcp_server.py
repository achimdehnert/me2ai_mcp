"""Standalone script to run the ME2AI MCP PostgreSQL server.

This script provides a simple way to start the MCP server
with database integration for PostgreSQL.
"""
import os
import sys
import asyncio
import logging
import signal
from dotenv import load_dotenv
from modelcontextprotocol import MCPServer
from modelcontextprotocol.server import ToolDefinition, FunctionDefinition
import psycopg2
import psycopg2.extras


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("mcp-server")


class PostgresMCPServer(MCPServer):
    """MCP server for PostgreSQL database access."""
    
    def __init__(self) -> None:
        """Initialize the server."""
        super().__init__()
        self.connection = None
        self._register_tools()
        
    def _register_tools(self) -> None:
        """Register database tools with the MCP server."""
        # Register query tool
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
        
        # Register list tables tool
        list_tables_tool = ToolDefinition(
            name="list_tables",
            description="List all tables in a specific schema",
            function=FunctionDefinition(
                parameters={
                    "type": "object",
                    "properties": {
                        "schema": {
                            "type": "string", 
                            "description": "Database schema to list tables from"
                        }
                    },
                    "required": ["schema"]
                }
            )
        )
        
        # Register them with the MCP server
        self.registry.register_tool(query_tool, self.execute_query)
        self.registry.register_tool(list_tables_tool, self.list_tables)
        
    async def connect(self) -> bool:
        """Connect to the PostgreSQL database.
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        # Load environment variables
        load_dotenv()
        
        # Try connection string first
        uri = os.getenv("POSTGRES_URI") or os.getenv("POSTGRES_URL")
        
        if uri:
            logger.info("Attempting connection using connection string...")
            try:
                self.connection = psycopg2.connect(uri)
                logger.info("Successfully connected using connection string")
                return True
            except Exception as e:
                logger.error(f"Connection string method failed: {str(e)}")
        
        # Try individual parameters
        logger.info("Attempting connection using individual parameters...")
        host = os.getenv("POSTGRES_HOST")
        port = os.getenv("POSTGRES_PORT")
        user = os.getenv("POSTGRES_USER")
        password = os.getenv("POSTGRES_PASSWORD")
        database = os.getenv("POSTGRES_DATABASE")
        
        if not all([host, database, user, password]):
            logger.error("Missing required database connection parameters")
            return False
        
        try:
            self.connection = psycopg2.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database
            )
            logger.info(f"Successfully connected to database: {database}")
            return True
        except Exception as e:
            logger.error(f"Individual parameters method failed: {str(e)}")
            return False
    
    async def execute_query(self, query: str, schema: str) -> dict:
        """Execute a SQL query on the database.
        
        Args:
            query: SQL query to execute
            schema: Database schema to use
            
        Returns:
            Query results
        """
        if not self.connection:
            return {"error": "No database connection"}
        
        # Sanitize the schema name
        if not schema.replace('-', '').replace('_', '').isalnum():
            return {"error": "Invalid schema name format"}
        
        # Set search path and execute query
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Set schema
            cursor.execute(f'SET search_path TO "{schema}";')
            
            # Execute user query
            cursor.execute(query)
            
            # Check if query returns results
            if cursor.description:
                results = cursor.fetchall()
                results_list = [dict(row) for row in results]
                
                self.connection.commit()
                cursor.close()
                
                return {
                    "success": True,
                    "count": len(results_list),
                    "results": results_list[:100]  # Limit to 100 rows
                }
            else:
                # Query executed successfully but no results
                affected_rows = cursor.rowcount
                self.connection.commit()
                cursor.close()
                
                return {
                    "success": True,
                    "affected_rows": affected_rows,
                    "message": f"Query executed successfully"
                }
                
        except Exception as e:
            self.connection.rollback()
            return {
                "success": False,
                "error": str(e)
            }
    
    async def list_tables(self, schema: str) -> dict:
        """List all tables in a specific schema.
        
        Args:
            schema: Database schema to list tables from
            
        Returns:
            List of tables
        """
        if not self.connection:
            return {"error": "No database connection"}
        
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
    
    def close(self) -> None:
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")


async def main() -> None:
    """Run the MCP server with PostgreSQL integration."""
    logger.info("Starting ME2AI PostgreSQL MCP Server")
    
    # Create server
    server = PostgresMCPServer()
    
    # Connect to database
    connection_success = await server.connect()
    if not connection_success:
        logger.error("Failed to connect to database, exiting")
        return
    
    # Set up signal handlers
    def handle_shutdown(sig, frame):
        logger.info(f"Received signal {sig}, shutting down")
        loop = asyncio.get_event_loop()
        loop.create_task(shutdown(server))
    
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Start server
    await server.start()
    logger.info("MCP Server started successfully")
    
    # Show available tools
    tools = server.registry.list_tools()
    logger.info(f"Available tools ({len(tools)}):")
    for tool in tools:
        logger.info(f"- {tool.name}: {tool.description}")
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await shutdown(server)


async def shutdown(server: PostgresMCPServer) -> None:
    """Shut down the server gracefully."""
    logger.info("Shutting down MCP Server")
    server.close()
    await server.stop()
    logger.info("Server stopped")


if __name__ == "__main__":
    asyncio.run(main())
