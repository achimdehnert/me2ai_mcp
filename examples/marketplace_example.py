"""
Example demonstrating the Tool Marketplace functionality in ME2AI MCP.

This example shows how to:
1. Initialize the marketplace
2. Search for tools
3. Install tools from the repository
4. Use installed tools in agents

Run with: python -m examples.marketplace_example
"""
import sys
import os
import logging
from typing import Dict, Any

# Add parent directory to path to import me2ai_mcp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from me2ai_mcp.agents import SpecializedAgent
from me2ai_mcp.marketplace import ToolMarketplace, global_marketplace
from me2ai_mcp.tools_registry import global_registry
from me2ai_mcp.server import MCPServer
from me2ai_mcp.routing import RoutingRule, MCPRouter
from me2ai_mcp.dynamic_routing import AdaptiveRouter
from me2ai_mcp.collaborative_agent import CollaborativeAgent, CollaborationManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("marketplace-example")


# Define a simple function to act as a local tool
def simple_calculator(a: float, b: float, operation: str = "add") -> Dict[str, Any]:
    """
    A simple calculator tool that performs basic operations.
    
    Args:
        a: First number
        b: Second number
        operation: Operation to perform (add, subtract, multiply, divide)
        
    Returns:
        Result of the operation
    """
    result = 0
    
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            return {"error": "Division by zero"}
        result = a / b
    else:
        return {"error": f"Unknown operation: {operation}"}
    
    return {
        "operation": operation,
        "a": a,
        "b": b,
        "result": result
    }


class MarketplaceExampleAgent(CollaborativeAgent):
    """Example agent that demonstrates marketplace integration."""
    
    def __init__(self, agent_id: str, marketplace: ToolMarketplace = None):
        """
        Initialize the example agent.
        
        Args:
            agent_id: Agent ID
            marketplace: Tool marketplace to use
        """
        super().__init__(
            agent_id=agent_id,
            name="Marketplace Example Agent",
            description="Demonstrates tool marketplace functionality",
            server=None  # Will be set later
        )
        
        self.marketplace = marketplace or global_marketplace
        
        # Register a local tool
        self.tools["calculator"] = simple_calculator
    
    def process_request(self, request: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Process a request using marketplace tools.
        
        Args:
            request: Request string
            **kwargs: Additional arguments
            
        Returns:
            Response dictionary
        """
        # Parse the request to determine what operation to perform
        request_lower = request.lower()
        
        if "list tools" in request_lower:
            # List available tools in the marketplace
            return self._list_tools()
            
        elif "search tools" in request_lower:
            # Extract search query
            import re
            query_match = re.search(r'search tools for "([^"]+)"', request_lower)
            if query_match:
                query = query_match.group(1)
                return self._search_tools(query)
            else:
                return {"error": "Please specify a search query"}
            
        elif "install tool" in request_lower:
            # Extract tool ID
            import re
            tool_match = re.search(r'install tool "([^"]+)"', request_lower)
            if tool_match:
                tool_id = tool_match.group(1)
                return self._install_tool(tool_id)
            else:
                return {"error": "Please specify a tool ID"}
            
        elif "uninstall tool" in request_lower:
            # Extract tool ID
            import re
            tool_match = re.search(r'uninstall tool "([^"]+)"', request_lower)
            if tool_match:
                tool_id = tool_match.group(1)
                return self._uninstall_tool(tool_id)
            else:
                return {"error": "Please specify a tool ID"}
            
        elif "calculate" in request_lower:
            # Extract calculation parameters
            import re
            calc_match = re.search(
                r'calculate (\d+\.?\d*) (\w+) (\d+\.?\d*)', 
                request_lower
            )
            if calc_match:
                a = float(calc_match.group(1))
                operation = calc_match.group(2)
                b = float(calc_match.group(3))
                
                op_map = {
                    "plus": "add",
                    "minus": "subtract",
                    "times": "multiply",
                    "divided": "divide"
                }
                
                if operation in op_map:
                    operation = op_map[operation]
                
                return self.tools["calculator"](a, b, operation)
            else:
                return {"error": "Invalid calculation format. Use: calculate [number] [operation] [number]"}
        
        elif "help" in request_lower:
            # Return help information
            return {
                "message": "Available commands:",
                "commands": [
                    "list tools",
                    "search tools for \"[query]\"",
                    "install tool \"[tool_id]\"",
                    "uninstall tool \"[tool_id]\"",
                    "calculate [number] [plus|minus|times|divided] [number]",
                    "help"
                ]
            }
        
        else:
            # Try to find a suitable tool
            for tool_name, tool_func in self.tools.items():
                if tool_name in request_lower:
                    # Found a matching tool
                    return {"message": f"Using tool: {tool_name}", "result": "Tool execution would happen here"}
            
            return {"message": "I don't understand that request. Try 'help' for available commands."}
    
    def _list_tools(self) -> Dict[str, Any]:
        """List available tools in the marketplace."""
        available_tools = self.marketplace.list_available_tools()
        installed_tools = self.marketplace.list_available_tools(installed_only=True)
        
        return {
            "message": f"Found {len(available_tools)} available tools, {len(installed_tools)} installed",
            "available_tools": available_tools,
            "installed_tools": installed_tools
        }
    
    def _search_tools(self, query: str) -> Dict[str, Any]:
        """
        Search for tools in the marketplace.
        
        Args:
            query: Search query
            
        Returns:
            Search results
        """
        search_results = self.marketplace.search_tools(query)
        
        return {
            "message": f"Found {len(search_results)} tools matching '{query}'",
            "results": search_results
        }
    
    def _install_tool(self, tool_id: str) -> Dict[str, Any]:
        """
        Install a tool from the marketplace.
        
        Args:
            tool_id: Tool ID
            
        Returns:
            Installation result
        """
        result = self.marketplace.install_tool(tool_id, self)
        
        if result["success"]:
            return {
                "message": f"Tool '{tool_id}' installed successfully",
                "details": result
            }
        else:
            return {
                "message": f"Failed to install tool '{tool_id}'",
                "error": result["error"]
            }
    
    def _uninstall_tool(self, tool_id: str) -> Dict[str, Any]:
        """
        Uninstall a tool.
        
        Args:
            tool_id: Tool ID
            
        Returns:
            Uninstallation result
        """
        result = self.marketplace.uninstall_tool(tool_id)
        
        if result["success"]:
            return {
                "message": f"Tool '{tool_id}' uninstalled successfully",
                "details": result
            }
        else:
            return {
                "message": f"Failed to uninstall tool '{tool_id}'",
                "error": result["error"]
            }


def main():
    """Run the marketplace example."""
    logger.info("Initializing Tool Marketplace example")
    
    # Create MCP server
    server = MCPServer()
    
    # Create the marketplace
    marketplace = ToolMarketplace()
    
    # Create and register the agent
    agent = MarketplaceExampleAgent("marketplace_agent", marketplace)
    agent.server = server  # Set the server reference
    
    # Register the agent with the server
    server.register_agent(agent)
    
    # Create a collaboration manager
    collab_manager = CollaborationManager(server)
    
    # Create a router with adaptive routing
    router = AdaptiveRouter()
    
    # Add routing rules
    router.add_rule(RoutingRule(
        pattern=r"marketplace|tools|install|uninstall",
        agent_id="marketplace_agent",
        priority=100,
        description="Route marketplace-related requests to the marketplace agent"
    ))
    
    # Set the router
    server.router = router
    
    # Create a mock tool repository if one doesn't exist
    create_mock_repository()
    
    # Simulate some requests
    logger.info("\n==== Example 1: Listing available tools ====")
    response = server.process_request("list tools")
    logger.info(f"Response: {response}")
    
    logger.info("\n==== Example 2: Searching for tools ====")
    response = server.process_request('search tools for "calculator"')
    logger.info(f"Response: {response}")
    
    logger.info("\n==== Example 3: Installing a tool ====")
    response = server.process_request('install tool "advanced-calculator"')
    logger.info(f"Response: {response}")
    
    logger.info("\n==== Example 4: Using a tool ====")
    response = server.process_request("calculate 10.5 plus 20.3")
    logger.info(f"Response: {response}")
    
    logger.info("\n==== Example 5: Uninstalling a tool ====")
    response = server.process_request('uninstall tool "advanced-calculator"')
    logger.info(f"Response: {response}")
    
    logger.info("\nMarketplace example completed")


def create_mock_repository():
    """Create a mock tool repository for demonstration purposes."""
    import json
    import os
    from pathlib import Path
    
    # Create repository directory
    repo_dir = Path.home() / ".me2ai_mcp" / "tool_cache"
    repo_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock index.json if it doesn't exist
    index_path = repo_dir / "repository_index.json"
    
    if not index_path.exists():
        mock_index = {
            "advanced-calculator": {
                "tool_id": "advanced-calculator",
                "name": "Advanced Calculator",
                "version": "1.0.0",
                "description": "Advanced calculator with scientific functions",
                "author": "ME2AI",
                "categories": ["math", "utility"],
                "dependencies": {},
                "repository_url": "https://example.com/tools/advanced-calculator",
                "documentation_url": "https://example.com/docs/advanced-calculator",
                "examples": [
                    {
                        "name": "Basic Addition",
                        "description": "Adding two numbers",
                        "code": "calculate 10 plus 20"
                    }
                ],
                "installed": False,
                "install_path": None,
                "checksum": None
            },
            "text-analyzer": {
                "tool_id": "text-analyzer",
                "name": "Text Analyzer",
                "version": "1.0.0",
                "description": "Analyzes text for sentiment, entities, and more",
                "author": "ME2AI",
                "categories": ["nlp", "utility"],
                "dependencies": {},
                "repository_url": "https://example.com/tools/text-analyzer",
                "documentation_url": "https://example.com/docs/text-analyzer",
                "examples": [
                    {
                        "name": "Sentiment Analysis",
                        "description": "Analyzing sentiment of text",
                        "code": "analyze sentiment of 'I love this product!'"
                    }
                ],
                "installed": False,
                "install_path": None,
                "checksum": None
            }
        }
        
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(mock_index, f, indent=2)


if __name__ == "__main__":
    main()
