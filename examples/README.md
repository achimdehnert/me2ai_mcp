# ME2AI MCP Examples

This directory contains example implementations of MCP servers built with the ME2AI MCP framework.

## Overview

The ME2AI MCP framework extends the official MCP (Model Context Protocol) package with enhanced functionality, standardized patterns, and reusable components. These examples demonstrate how to build custom MCP servers using the framework.

## Examples

### Basic Example Server

[`custom_mcp_server.py`](./custom_mcp_server.py) - A simple MCP server demonstrating:
- Basic server setup and configuration
- Custom tool implementation
- Authentication integration
- Tool composition (combining multiple tools into higher-level operations)
- Using built-in tools from the framework

### Running the Examples

1. Install the ME2AI MCP package:
   ```bash
   # From the project root
   pip install -e .
   
   # Or with all optional dependencies
   pip install -e .[all]
   ```

2. Run an example server:
   ```bash
   python examples/custom_mcp_server.py
   ```

3. The server will start on the default port (typically 8080).

4. You can connect to the server using the MCP client library or the Windsurf configuration.

## MCP Configuration

To use these MCP servers with Windsurf, add the following to your `~/.codeium/windsurf/mcp_config.json`:

```json
"example": {
  "command": "python",
  "args": ["examples/custom_mcp_server.py"],
  "cwd": "/path/to/me2ai/repo"
}
```

## Creating Your Own MCP Server

Follow these steps to create your own custom MCP server:

1. Import the necessary components:
   ```python
   from me2ai_mcp import ME2AIMCPServer, register_tool
   ```

2. Create a server class extending ME2AIMCPServer:
   ```python
   class MyCustomServer(ME2AIMCPServer):
       def __init__(self):
           super().__init__(
               server_name="my-custom-server",
               description="My custom MCP server",
               version="1.0.0"
           )
   ```

3. Add custom tools with the @register_tool decorator:
   ```python
   @register_tool
   async def my_custom_tool(self, parameter1: str, parameter2: int = 10) -> dict:
       """Tool documentation."""
       # Tool implementation
       return {
           "result": f"Processed {parameter1} with value {parameter2}"
       }
   ```

4. Run the server:
   ```python
   async def main():
       server = MyCustomServer()
       await server.start()

   if __name__ == "__main__":
       import asyncio
       asyncio.run(main())
   ```

## Best Practices

Follow these guidelines when developing MCP servers:

1. **Documentation**: Thoroughly document each tool with clear descriptions, parameter details, and return values.
2. **Error Handling**: Use try/except blocks and return clear error messages with the `success: False` pattern.
3. **Authentication**: Implement appropriate authentication for any tools accessing external services.
4. **Input Validation**: Validate all user inputs before processing.
5. **Tool Composition**: Create higher-level tools that combine existing tools for complex operations.
6. **Statistics**: Track usage statistics for performance analysis.

## Additional Resources

- [ME2AI MCP Documentation](../me2ai_mcp/README.md)
- [Official MCP Package](https://pypi.org/project/mcp/)
- [Windsurf Configuration Guide](https://github.com/achimdehnert/me2ai/wiki/Windsurf-Configuration)
