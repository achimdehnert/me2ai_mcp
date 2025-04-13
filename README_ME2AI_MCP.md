# ME2AI MCP Framework

A comprehensive extension of the Model Context Protocol (MCP) framework for building robust, standardized MCP servers with enhanced error handling, authentication, and tool management.

[![PyPI version](https://badge.fury.io/py/me2ai-mcp.svg)](https://badge.fury.io/py/me2ai-mcp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

ME2AI MCP extends the official [MCP (Model Context Protocol)](https://pypi.org/project/mcp/) package with enhanced functionality, standardized patterns, and reusable components for building robust MCP servers. It provides a consistent framework for error handling, authentication, and tool management, making it easier to develop and maintain MCP servers.

## Key Features

- **Extended Base Classes**: Enhanced MCP server base classes with automatic error handling, logging, and statistics tracking
- **Standardized Authentication**: Built-in support for API key and token-based authentication
- **Reusable Tools**: Common tool implementations for web content fetching, file operations, and GitHub integration
- **Consistent Patterns**: Standardized response formats and error handling across all tools
- **Type Safety**: Comprehensive type hints throughout the codebase
- **Flexible Configuration**: Support for configuration files and environment variables
- **Extensive Testing**: Comprehensive test suite with Robot Framework and Selenium integration

## Installation

### Quick Install

```bash
pip install me2ai-mcp
```

### Install with Optional Features

```bash
# Install with web tools (BeautifulSoup, etc.)
pip install me2ai-mcp[web]

# Install with GitHub tools
pip install me2ai-mcp[github]

# Install all optional dependencies
pip install me2ai-mcp[all]
```

### Development Installation

For development, it's recommended to install in editable mode with development dependencies:

```bash
# Clone the repository
git clone https://github.com/achimdehnert/me2ai.git
cd me2ai

# Install in development mode with all extras
python install_me2ai_mcp.py --all --dev --editable
```

Or use the installation script with options:

```bash
python install_me2ai_mcp.py --help
```

## Quick Start

### Basic MCP Server

```python
from me2ai_mcp import ME2AIMCPServer, register_tool

class MyMCPServer(ME2AIMCPServer):
    def __init__(self):
        super().__init__(
            server_name="my-mcp-server",
            description="My custom MCP server",
            version="1.0.0"
        )
    
    @register_tool
    async def my_tool(self, param1: str, param2: int = 10) -> dict:
        """Example tool with automatic error handling."""
        result = f"Processed {param1} with value {param2}"
        return {"content": result}

# Run the server
if __name__ == "__main__":
    import asyncio
    server = MyMCPServer()
    asyncio.run(server.start())
```

### Using Built-in Tools

```python
from me2ai_mcp import ME2AIMCPServer
from me2ai_mcp.tools.web import WebFetchTool
from me2ai_mcp.tools.filesystem import FileReaderTool

class MyToolsServer(ME2AIMCPServer):
    def __init__(self):
        super().__init__("tools-server")
        
        # Initialize tools
        self.web_fetch = WebFetchTool()
        self.file_reader = FileReaderTool()
    
    @register_tool
    async def fetch_webpage(self, url: str) -> dict:
        """Fetch a webpage and return its content."""
        return await self.web_fetch.execute({"url": url})
    
    @register_tool
    async def read_file(self, file_path: str) -> dict:
        """Read a file and return its content."""
        return await self.file_reader.execute({"file_path": file_path})
```

### Adding Authentication

```python
from me2ai_mcp import ME2AIMCPServer
from me2ai_mcp.auth import AuthManager, APIKeyAuth

class SecureMCPServer(ME2AIMCPServer):
    def __init__(self):
        super().__init__("secure-server")
        
        # Set up authentication
        self.auth = AuthManager()
        self.auth.add_provider(APIKeyAuth(api_key="your-api-key"))
        
        # Register protected tools
        self._register_protected_tools()
```

## Available Tools

### Web Tools

- **WebFetchTool**: Fetch web content with error handling and size limits
- **HTMLParserTool**: Parse HTML content and extract structured data
- **URLUtilsTool**: URL manipulation and processing utilities

### Filesystem Tools

- **FileReaderTool**: Read file content with safety checks
- **FileWriterTool**: Write content to files with directory creation
- **DirectoryListerTool**: List directory contents with filtering options

### GitHub Tools

- **GitHubRepositoryTool**: Repository search and metadata operations
- **GitHubCodeTool**: Code search and file content retrieval
- **GitHubIssuesTool**: Issues management and listing

## Testing

The package includes comprehensive tests using pytest and Robot Framework:

```bash
# Run unit tests
pytest tests/me2ai_mcp/

# Run Robot Framework API tests
robot -d reports tests/robot/tests/api/

# Run Robot Framework UI tests (requires Selenium setup)
robot -d reports tests/robot/tests/ui/
```

See [tests/robot/README.md](tests/robot/README.md) for detailed testing instructions.

## Examples

Check out the [examples](examples/) directory for complete example implementations:

- [custom_mcp_server.py](examples/custom_mcp_server.py): Basic MCP server with custom tools
- See [examples/README.md](examples/README.md) for more details

## Windsurf Integration

To use ME2AI MCP servers with Windsurf, add configurations to `~/.codeium/windsurf/mcp_config.json`:

```json
"my_server": {
  "command": "python",
  "args": ["path/to/my_server.py"],
  "cwd": "/path/to/working/directory"
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
