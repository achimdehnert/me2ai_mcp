"""
Basic ME2AI MCP Server Example

This example demonstrates how to create a simple MCP server
with basic tool functionality using the ME2AI MCP framework.
"""

import os
import json
from typing import Dict, Any, List, Optional

from me2ai_mcp.base import ME2AIMCPServer
from me2ai_mcp.auth import AuthManager


def main():
    """Run a basic ME2AI MCP server demonstration."""
    # Create a new MCP server instance
    server = ME2AIMCPServer(
        server_name="example_server",
        description="Basic Example ME2AI MCP Server",
        version="0.0.5"
    )
    
    print(f"Starting {server.description} (v{server.version})")
    
    # Register a simple text processing tool
    @server.register_tool
    def process_text(text: str, operation: str = "uppercase") -> Dict[str, Any]:
        """
        Process text with various operations.
        
        Args:
            text: The input text to process
            operation: The operation to perform (uppercase, lowercase, reverse, count)
            
        Returns:
            Dict containing the processed result
        """
        if operation == "uppercase":
            result = text.upper()
        elif operation == "lowercase":
            result = text.lower()
        elif operation == "reverse":
            result = text[::-1]
        elif operation == "count":
            result = str(len(text))
        else:
            return {"error": f"Unknown operation: {operation}"}
        
        return {
            "original": text,
            "operation": operation,
            "result": result
        }
    
    # Register a data aggregation tool
    @server.register_tool
    def aggregate_data(items: List[Dict[str, Any]], 
                       field: str, 
                       operation: str = "sum") -> Dict[str, Any]:
        """
        Aggregate data from a list of items.
        
        Args:
            items: List of dictionary items to process
            field: The field to aggregate on
            operation: The aggregation operation (sum, avg, min, max, count)
            
        Returns:
            Dict containing the aggregation result
        """
        if not items:
            return {"error": "No items provided"}
            
        if not all(field in item for item in items):
            return {"error": f"Field '{field}' not present in all items"}
            
        values = [item[field] for item in items if field in item]
        
        if not values:
            return {"error": "No valid values found to aggregate"}
            
        # Ensure numeric values
        try:
            numeric_values = [float(v) for v in values]
        except ValueError:
            return {"error": f"Non-numeric values found in field '{field}'"}
            
        if operation == "sum":
            result = sum(numeric_values)
        elif operation == "avg":
            result = sum(numeric_values) / len(numeric_values)
        elif operation == "min":
            result = min(numeric_values)
        elif operation == "max":
            result = max(numeric_values)
        elif operation == "count":
            result = len(numeric_values)
        else:
            return {"error": f"Unknown operation: {operation}"}
            
        return {
            "field": field,
            "operation": operation,
            "result": result,
            "count": len(numeric_values)
        }
    
    # Demo execution with some example inputs
    print("\n--- Demo Tool Execution ---")
    
    print("\nExecuting 'process_text' tool:")
    text_result = server.execute_tool("process_text", {
        "text": "Hello from ME2AI MCP!",
        "operation": "uppercase"
    })
    print(f"Result: {json.dumps(text_result, indent=2)}")
    
    print("\nExecuting 'aggregate_data' tool:")
    data = [
        {"id": 1, "value": 10, "name": "Item 1"},
        {"id": 2, "value": 20, "name": "Item 2"},
        {"id": 3, "value": 30, "name": "Item 3"},
        {"id": 4, "value": 40, "name": "Item 4"},
    ]
    agg_result = server.execute_tool("aggregate_data", {
        "items": data,
        "field": "value",
        "operation": "avg"
    })
    print(f"Result: {json.dumps(agg_result, indent=2)}")
    
    # Demonstrate error handling
    print("\nDemonstrating error handling:")
    error_result = server.execute_tool("process_text", {
        "text": "Error test",
        "operation": "invalid_operation"
    })
    print(f"Error Result: {json.dumps(error_result, indent=2)}")
    
    # List available tools
    print("\nAvailable tools:")
    tools = server.get_tools()
    for tool in tools:
        print(f"- {tool['name']}: {tool['description']}")


if __name__ == "__main__":
    main()
