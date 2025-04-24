#!/usr/bin/env python3
"""
ME2AI PostgreSQL MCP Server

A Model Context Protocol (MCP) server for PostgreSQL database integration
with enhanced features, flexible credential management, and LangChain compatibility.

This server provides tools for:
- Executing SQL queries on specific schemas
- Listing tables in a schema
- Getting column information for tables
- Looking up postal code (PLZ) details

Usage:
    python postgres_mcp_server.py
"""

import os
import sys
import logging
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import ME2AI MCP components
from me2ai_mcp.base import ME2AIMCPServer
from me2ai_mcp.tools.postgres import (
    ExecuteQueryTool,
    ListTablesTool,
    GetTableColumnsTool,
    GetPLZDetailsTool
)


def setup_logging(level: int = logging.INFO) -> None:
    """Set up logging configuration.
    
    Args:
        level: Logging level to use
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="ME2AI PostgreSQL MCP Server")
    
    parser.add_argument(
        "--env-prefix",
        type=str,
        default="POSTGRES",
        help="Environment variable prefix for database credentials"
    )
    
    parser.add_argument(
        "--credential-file",
        type=str,
        default=None,
        help="Path to JSON file with database credentials"
    )
    
    parser.add_argument(
        "--connection-name",
        type=str,
        default=None,
        help="Connection name to use in credential file"
    )
    
    parser.add_argument(
        "--default-schema",
        type=str,
        default="poco",
        help="Default schema to use for queries"
    )
    
    parser.add_argument(
        "--allowed-schemas",
        type=str,
        nargs="+",
        default=["poco", "poco-test"],
        help="List of allowed schemas"
    )
    
    parser.add_argument(
        "--max-rows",
        type=int,
        default=50,
        help="Maximum number of rows to return"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


def main() -> None:
    """Run the PostgreSQL MCP server."""
    # Parse arguments
    args = parse_arguments()
    
    # Set up logging
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger("postgres-mcp-server")
    
    # Create MCP server
    logger.info("Starting PostgreSQL MCP server")
    logger.info(f"Default schema: {args.default_schema}")
    logger.info(f"Allowed schemas: {args.allowed_schemas}")
    
    server = ME2AIMCPServer(
        server_name="postgres",
        description="PostgreSQL database integration for ME2AI",
        version="1.0.0",
        debug=args.debug
    )
    
    # Create credential file path if provided
    credential_file = None
    if args.credential_file:
        credential_file = Path(args.credential_file)
        logger.info(f"Using credential file: {credential_file}")
    
    # Register tools with shared connection parameters
    common_params = {
        "env_prefix": args.env_prefix,
        "credential_file": credential_file,
        "connection_name": args.connection_name,
        "allowed_schemas": args.allowed_schemas,
        "default_schema": args.default_schema
    }
    
    # Create and register tools
    server.register_tool(ExecuteQueryTool(max_rows=args.max_rows, **common_params))
    server.register_tool(ListTablesTool(**common_params))
    server.register_tool(GetTableColumnsTool(**common_params))
    server.register_tool(GetPLZDetailsTool(**common_params))
    
    # Start server
    logger.info("Server initialized, starting to serve requests")
    server.serve()


if __name__ == "__main__":
    main()
