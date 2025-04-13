"""
ME2AI MCP - Model Context Protocol server extensions for ME2AI.

This package extends the official `mcp` package with custom utilities
and abstractions specific to the ME2AI project.
"""
from .base import ME2AIMCPServer, BaseTool
from .auth import AuthManager, APIKeyAuth, TokenAuth
from .utils import sanitize_input, format_response, extract_text

__version__ = "0.0.5"

__all__ = [
    "ME2AIMCPServer",
    "BaseTool",
    "AuthManager", 
    "APIKeyAuth",
    "TokenAuth",
    "sanitize_input",
    "format_response",
    "extract_text"
]
