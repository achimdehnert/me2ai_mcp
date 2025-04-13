# ME2AI MCP (v0.0.5)

Enhanced Model Context Protocol framework for ME2AI agents and services.

## Overview

ME2AI MCP is an extension package that builds on top of the official [MCP (Model Context Protocol)](https://pypi.org/project/mcp/) package. It provides standardized base classes, utilities, and tools designed specifically for building robust MCP server implementations.

## Key Features

- **Enhanced Base Classes**: Extends the official MCP package with ME2AI-specific functionality
- **Standardized Error Handling**: Consistent error handling and response formatting
- **Authentication Utilities**: Built-in support for API keys and tokens
- **Common Tool Implementations**: Reusable tools for web scraping, filesystem operations, and more
- **Logging and Statistics**: Automatic logging and request statistics

## Installation

```bash
# Install from PyPI (recommended)
pip install me2ai_mcp

# Install with all optional dependencies
pip install me2ai_mcp[all]

# Install with specific feature set
pip install me2ai_mcp[web]
```

Alternatively, install directly from GitHub:

```bash
# Install from GitHub
pip install git+https://github.com/achimdehnert/me2ai.git#subdirectory=me2ai_mcp
```

See [INSTALLATION.md](INSTALLATION.md) for detailed installation options.

## Quick Start

```python
from me2ai_mcp import ME2AIMCPServer, register_tool

class MyMCPServer(ME2AIMCPServer):
    def __init__(self):
        super().__init__(
            server_name="my-server",
            description="My custom MCP server",
            version="0.1.0"
        )
    
    @register_tool
    async def my_tool(self, param1: str, param2: int = 10) -> dict:
        """Example tool with automatic error handling."""
        # Your tool implementation here
        result = f"Processed {param1} with value {param2}"
        
        return {
            "content": result
            # No need to add success=True, it's added automatically
        }

# Start the server
if __name__ == "__main__":
    import asyncio
    
    server = MyMCPServer()
    asyncio.run(server.start())
```

## Using Authentication

```python
from me2ai_mcp import ME2AIMCPServer
from me2ai_mcp.auth import AuthManager, APIKeyAuth, TokenAuth

# Create server with GitHub authentication
class GitHubMCPServer(ME2AIMCPServer):
    def __init__(self):
        super().__init__("github")
        
        # Set up authentication from environment variables
        self.auth = AuthManager.from_github_token()
```

## Using Built-in Tools

```python
from me2ai_mcp import ME2AIMCPServer, register_tool
from me2ai_mcp.tools.web import WebFetchTool
from me2ai_mcp.tools.filesystem import FileReaderTool

class ToolsServer(ME2AIMCPServer):
    def __init__(self):
        super().__init__("tools-server")
        
        # Initialize tools
        self.web_fetch_tool = WebFetchTool()
        self.file_reader_tool = FileReaderTool()
        
    @register_tool
    async def fetch_webpage(self, url: str) -> dict:
        """Fetch a webpage and return its content."""
        return await self.web_fetch_tool.execute({"url": url})
```

## Architecture

The ME2AI MCP package is organized as follows:

```text
me2ai_mcp/
├── __init__.py        # Package exports
├── base.py            # Base classes extending MCP
├── auth.py            # Authentication utilities
├── utils.py           # General utilities
└── tools/             # Reusable tool implementations
    ├── web.py         # Web-related tools
    ├── filesystem.py  # Filesystem tools
    └── github.py      # GitHub API tools
```

## Testing

ME2AI MCP comes with a comprehensive test suite that includes:

- Unit tests for all components
- Integration tests for cross-component functionality
- Performance tests for scalability verification

Run the tests using the provided test runner:

```bash
# Run all tests with coverage report
python run_mcp_tests.py --html --cov

# Run only unit tests
python run_mcp_tests.py --unit-only

# Run without performance tests
python run_mcp_tests.py --skip-performance
```

## License

MIT
