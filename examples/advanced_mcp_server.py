#!/usr/bin/env python
"""
Advanced ME2AI MCP Server Example

This example demonstrates a production-ready ME2AI MCP server with:
- Multiple authentication providers
- Custom error handling
- Extended tool registration
- Asynchronous operation support
- Environment configuration
- Comprehensive logging

Run this example with:
    python advanced_mcp_server.py
"""
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import os
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from me2ai_mcp.base import ME2AIMCPServer
from me2ai_mcp.auth import AuthManager, APIKeyAuth, TokenAuth
from me2ai_mcp.utils import sanitize_input, format_response


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("advanced_mcp_server.log")
    ]
)
logger = logging.getLogger("advanced-mcp-server")


class AdvancedMCPServer(ME2AIMCPServer):
    """
    Advanced ME2AI MCP Server with extended functionality.
    
    This server demonstrates best practices for ME2AI MCP implementations
    with comprehensive error handling, authentication, and logging.
    """
    
    def __init__(
        self, 
        server_name: str = "advanced-mcp",
        description: str = "Advanced ME2AI MCP Server Example",
        version: str = "0.0.6",
        config_path: Optional[str] = None
    ) -> None:
        """
        Initialize the advanced MCP server.
        
        Args:
            server_name: Name of the server
            description: Server description
            version: Server version
            config_path: Path to configuration file (optional)
        """
        super().__init__(server_name, description, version)
        self.logger = logging.getLogger(f"me2ai-{server_name}")
        self.config = self._load_config(config_path)
        self.setup_authentication()
        self.start_time = datetime.now()
        
        # Register tools
        self.register_system_tools()
        self.register_data_tools()
        self.register_processing_tools()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary containing configuration
        """
        default_config = {
            "max_requests_per_minute": 60,
            "max_tokens_per_request": 8000,
            "allowed_origins": ["https://me2ai.dev"],
            "log_level": "INFO",
            "cache_enabled": True,
            "cache_ttl_seconds": 3600
        }
        
        if not config_path:
            self.logger.info("Using default configuration")
            return default_config
        
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                self.logger.warning(f"Config file {config_path} not found, using defaults")
                return default_config
                
            with open(config_file, "r") as f:
                config = json.load(f)
                self.logger.info(f"Loaded configuration from {config_path}")
                return {**default_config, **config}  # Merge with defaults
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            return default_config
    
    def setup_authentication(self) -> None:
        """
        Configure authentication providers for the server.
        
        This method sets up multiple authentication options including:
        - API Key authentication
        - Token-based authentication
        - Environment variable configuration
        """
        load_dotenv()
        
        # Create auth manager with multiple providers
        self.auth_manager = AuthManager()
        
        # API Key authentication
        api_key_env_vars = [
            "ME2AI_API_KEY", 
            "ADVANCED_MCP_API_KEY",
            "MCP_API_KEY"
        ]
        for env_var in api_key_env_vars:
            if os.getenv(env_var):
                self.auth_manager.add_provider(APIKeyAuth(env_var_name=env_var))
                self.logger.info(f"Added API key auth provider from {env_var}")
        
        # Token authentication
        token_env_vars = [
            "ME2AI_ACCESS_TOKEN",
            "ADVANCED_MCP_TOKEN"
        ]
        for env_var in token_env_vars:
            if os.getenv(env_var):
                self.auth_manager.add_provider(TokenAuth(env_var_name=env_var))
                self.logger.info(f"Added token auth provider from {env_var}")
        
        # GitHub token (if available)
        try:
            github_auth = AuthManager.from_github_token()
            if github_auth.providers:
                for provider in github_auth.providers:
                    self.auth_manager.add_provider(provider)
                self.logger.info("Added GitHub token auth provider")
        except Exception as e:
            self.logger.warning(f"Could not initialize GitHub token auth: {str(e)}")
    
    def register_system_tools(self) -> None:
        """Register system-level tools for server management."""
        
        @self.register_tool(name="get_server_status")
        def get_server_status() -> Dict[str, Any]:
            """
            Get server status information.
            
            Returns:
                Dictionary with server status details
            """
            uptime = datetime.now() - self.start_time
            return {
                "status": "running",
                "version": self.version,
                "uptime_seconds": uptime.total_seconds(),
                "request_count": self.request_count,
                "error_count": self.error_count
            }
        
        @self.register_tool(name="get_server_config")
        def get_server_config() -> Dict[str, Any]:
            """
            Get server configuration.
            
            Returns:
                Dictionary with sanitized configuration
            """
            # Return only safe configuration values (no secrets)
            safe_config = {
                k: v for k, v in self.config.items() 
                if not any(secret in k.lower() for secret in ["key", "token", "password", "secret"])
            }
            return safe_config
    
    def register_data_tools(self) -> None:
        """Register data processing and storage tools."""
        
        self._data_store: Dict[str, Any] = {}
        
        @self.register_tool(name="store_data")
        def store_data(key: str, value: Any) -> Dict[str, str]:
            """
            Store data in the server's memory.
            
            Args:
                key: Unique identifier for the data
                value: Any JSON-serializable data to store
                
            Returns:
                Status message
            """
            try:
                # Sanitize and validate input
                safe_key = sanitize_input(key)
                if not safe_key:
                    raise ValueError("Invalid key")
                    
                # Store the value (ensure it's JSON serializable)
                json.dumps(value)  # This will raise an error if not serializable
                self._data_store[safe_key] = value
                
                return {
                    "status": "success",
                    "message": f"Data stored with key: {safe_key}"
                }
            except Exception as e:
                self.error_count += 1
                logger.error(f"Error storing data: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Failed to store data: {str(e)}"
                }
        
        @self.register_tool(name="retrieve_data")
        def retrieve_data(key: str) -> Dict[str, Any]:
            """
            Retrieve data from the server's memory.
            
            Args:
                key: Unique identifier for the data
                
            Returns:
                Data associated with the key or error message
            """
            try:
                # Sanitize input
                safe_key = sanitize_input(key)
                
                # Retrieve the value
                if safe_key in self._data_store:
                    return {
                        "status": "success",
                        "data": self._data_store[safe_key]
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"No data found for key: {safe_key}"
                    }
            except Exception as e:
                self.error_count += 1
                logger.error(f"Error retrieving data: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Failed to retrieve data: {str(e)}"
                }
    
    def register_processing_tools(self) -> None:
        """Register text and data processing tools."""
        
        @self.register_tool(name="process_text")
        def process_text(text: str, operations: List[str]) -> Dict[str, Any]:
            """
            Process text with specified operations.
            
            Args:
                text: Text to process
                operations: List of operations to perform
                
            Returns:
                Dictionary with processed text
            """
            result = text
            applied_ops = []
            
            try:
                for op in operations:
                    if op == "uppercase":
                        result = result.upper()
                        applied_ops.append(op)
                    elif op == "lowercase":
                        result = result.lower()
                        applied_ops.append(op)
                    elif op == "capitalize":
                        result = result.capitalize()
                        applied_ops.append(op)
                    elif op == "reverse":
                        result = result[::-1]
                        applied_ops.append(op)
                    elif op == "trim":
                        result = result.strip()
                        applied_ops.append(op)
                    else:
                        logger.warning(f"Unknown operation: {op}")
                
                return {
                    "status": "success",
                    "original": text,
                    "processed": result,
                    "applied_operations": applied_ops
                }
            except Exception as e:
                self.error_count += 1
                logger.error(f"Error processing text: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Failed to process text: {str(e)}"
                }
        
        @self.register_tool(name="analyze_text")
        async def analyze_text(text: str) -> Dict[str, Any]:
            """
            Analyze text and return statistics.
            
            Args:
                text: Text to analyze
                
            Returns:
                Dictionary with text statistics
            """
            try:
                # Simulate async processing
                await asyncio.sleep(0.5)
                
                # Perform analysis
                word_count = len(text.split())
                char_count = len(text)
                line_count = len(text.splitlines())
                
                # Calculate word frequency
                words = text.lower().split()
                word_freq = {}
                for word in words:
                    word = word.strip(".,!?;:\"'()[]{}").lower()
                    if word:
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                # Get top 5 words
                top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
                
                return {
                    "status": "success",
                    "statistics": {
                        "word_count": word_count,
                        "character_count": char_count,
                        "line_count": line_count,
                        "top_words": dict(top_words)
                    }
                }
            except Exception as e:
                self.error_count += 1
                logger.error(f"Error analyzing text: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Failed to analyze text: {str(e)}"
                }


def run_server() -> None:
    """Initialize and run the advanced MCP server."""
    logger.info("Starting Advanced ME2AI MCP Server")
    
    # Create server instance
    server = AdvancedMCPServer()
    
    # Print available tools
    tools = server.list_tools()
    logger.info(f"Registered {len(tools)} tools:")
    for tool in tools:
        logger.info(f" - {tool}")
    
    # Test a tool
    status = server.get_server_status()
    logger.info(f"Server status: {json.dumps(status, indent=2)}")
    
    logger.info("Advanced ME2AI MCP Server is ready")
    return server


if __name__ == "__main__":
    run_server()
