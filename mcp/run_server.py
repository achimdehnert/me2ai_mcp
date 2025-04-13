"""Run the ME2AI MCP server with Postgres database agent."""
import asyncio
import os
import sys
import signal
import logging
from typing import Any, Optional
from dotenv import load_dotenv

# Use relative imports when within the package
from .postgres_agent import PostgresDatabaseAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("me2ai-mcp")

async def main() -> None:
    """Run the MCP server."""
    # Load environment variables
    load_dotenv()
    
    # Check if required environment variables are present
    required_vars = [
        "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_USER", 
        "POSTGRES_PASSWORD", "POSTGRES_DATABASE"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these variables in your .env file or environment")
        sys.exit(1)
    
    logger.info("Starting ME2AI Postgres MCP Server...")
    logger.info(f"Using database: {os.getenv('POSTGRES_DATABASE')} at {os.getenv('POSTGRES_HOST')}")
    logger.info(f"Schema: poco, poco-test")
    
    # Create and start Postgres database agent
    server = PostgresDatabaseAgent()
    
    # Set up signal handlers for graceful shutdown
    def handle_shutdown(sig, frame):
        print(f"\nReceived signal {sig}, shutting down gracefully...")
        loop = asyncio.get_event_loop()
        loop.create_task(shutdown(server))
    
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Start the server and keep it running
    try:
        await server.start()
        logger.info("MCP Server started successfully.")
        logger.info("Available tools:")
        for tool in server.registry.list_tools():
            logger.info(f"- {tool.name}: {tool.description}")
    except Exception as e:
        logger.error(f"Failed to start MCP server: {str(e)}")
        await shutdown(server)
        sys.exit(1)
    
    # Keep the server running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await shutdown(server)

async def shutdown(server: PostgresDatabaseAgent) -> None:
    """Shut down the server gracefully."""
    print("Shutting down MCP Server...")
    server.close()  # Close database connection
    await server.stop()
    print("Server stopped. Goodbye!")
    sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())
