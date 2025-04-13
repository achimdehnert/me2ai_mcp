"""Entry point for the MCP module when run as a package."""
import asyncio
from mcp.run_server import main

if __name__ == "__main__":
    asyncio.run(main())
