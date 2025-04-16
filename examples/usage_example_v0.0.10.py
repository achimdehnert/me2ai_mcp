"""
ME2AI MCP v0.0.10 Usage Example

This example demonstrates how to use the enhanced ME2AI MCP framework
with all the new features:
1. Tool Registry System
2. Collaborative Agent System
3. Dynamic Routing
4. Tool Marketplace

Installation:
    pip install --upgrade me2ai_mcp==0.0.10
"""
import logging
import sys
from typing import Dict, Any, List, Optional

from me2ai_mcp import (
    # Core components
    ME2AIMCPServer,
    BaseTool,
    
    # Agent abstractions
    BaseAgent,
    SpecializedAgent,
    CollaborativeAgent,
    
    # Routing and collaboration
    AdaptiveRouter,
    RoutingRule,
    CollaborationManager,
    
    # Tool registry and marketplace
    ToolRegistry,
    global_registry,
    ToolMarketplace,
    global_marketplace
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("me2ai-mcp-example")


# Custom tools
def calculate_sum(a: float, b: float) -> Dict[str, Any]:
    """
    Calculate the sum of two numbers.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Dictionary with result
    """
    return {
        "operation": "sum",
        "result": a + b
    }


def search_knowledge_base(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search a knowledge base for information.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        Dictionary with search results
    """
    # Mock implementation
    return {
        "query": query,
        "results": [
            {"title": f"Result {i}", "snippet": f"Information about {query} - {i}"} 
            for i in range(1, max_results + 1)
        ]
    }


# Specialized agent for math operations
class MathAgent(SpecializedAgent):
    """Agent specializing in mathematical operations."""
    
    def __init__(self, agent_id: str):
        super().__init__(
            agent_id=agent_id,
            name="Math Agent",
            description="Performs mathematical operations",
            server=None  # Will be set later
        )
        
        # Register tools
        self.tools["calculate_sum"] = calculate_sum
        
        # Register with global registry
        global_registry.register_tool(
            tool_name="math.calculate_sum",
            tool_func=calculate_sum,
            categories=["math", "calculation"],
            description="Calculate the sum of two numbers."
        )
    
    def process_request(self, request: str, **kwargs: Any) -> Dict[str, Any]:
        """Process a request related to math operations."""
        request_lower = request.lower()
        
        if "add" in request_lower or "sum" in request_lower:
            # Extract numbers using simple parsing
            import re
            numbers = re.findall(r'\d+\.?\d*', request)
            
            if len(numbers) >= 2:
                return calculate_sum(float(numbers[0]), float(numbers[1]))
            else:
                return {"error": "Could not find two numbers in the request"}
        
        return {"message": "I don't understand that math operation"}


# Research agent with collaboration capabilities
class ResearchAgent(CollaborativeAgent):
    """Agent specializing in research with collaboration capabilities."""
    
    def __init__(self, agent_id: str, collaboration_manager: Optional[CollaborationManager] = None):
        super().__init__(
            agent_id=agent_id,
            name="Research Agent",
            description="Performs research and information retrieval",
            server=None,  # Will be set later
            collaboration_manager=collaboration_manager
        )
        
        # Register tools
        self.tools["search_kb"] = search_knowledge_base
        
        # Register with global registry
        global_registry.register_tool(
            tool_name="research.search_knowledge_base",
            tool_func=search_knowledge_base,
            categories=["research", "information"],
            description="Search a knowledge base for information."
        )
    
    def process_request(self, request: str, **kwargs: Any) -> Dict[str, Any]:
        """Process a research-related request."""
        request_lower = request.lower()
        
        # Check for collaboration requests
        if "collaborate" in request_lower:
            # Create a collaboration context
            collab_id = self.create_collaboration(
                participants=["math_agent", "research_agent"],
                topic="Research with calculations",
                metadata={"initial_request": request}
            )
            
            # Share information with the math agent
            self.send_message(
                recipient_id="math_agent",
                collaboration_id=collab_id,
                message="Starting a collaborative research task",
                data={"query": request}
            )
            
            return {
                "message": "Collaboration started",
                "collaboration_id": collab_id
            }
        
        if "search" in request_lower or "find" in request_lower:
            # Extract query
            query = request.replace("search", "").replace("find", "").strip()
            return search_knowledge_base(query)
        
        return {"message": "I don't understand that research request"}


# Example of using the marketplace
def marketplace_example():
    """Demonstrate tool marketplace functionality."""
    logger.info("Initializing Tool Marketplace example")
    
    # Check for available tools
    available_tools = global_marketplace.list_available_tools()
    logger.info(f"Found {len(available_tools)} available tools in the marketplace")
    
    # Search for math tools
    math_tools = global_marketplace.search_tools("math")
    logger.info(f"Found {len(math_tools)} math-related tools")
    
    # The following would actually install a tool if it existed in the repository
    # result = global_marketplace.install_tool("advanced-calculator")
    # logger.info(f"Tool installation result: {result['success']}")


def main():
    """Run the ME2AI MCP example."""
    logger.info("Initializing ME2AI MCP v0.0.10 example")
    
    # Create the MCP server
    server = ME2AIMCPServer()
    
    # Create a collaboration manager
    collaboration_manager = CollaborationManager(server)
    
    # Create agents
    math_agent = MathAgent("math_agent")
    research_agent = ResearchAgent("research_agent", collaboration_manager)
    
    # Set server references
    math_agent.server = server
    research_agent.server = server
    
    # Register agents with the server
    server.register_agent(math_agent)
    server.register_agent(research_agent)
    
    # Create an adaptive router
    router = AdaptiveRouter()
    
    # Add routing rules
    router.add_rule(RoutingRule(
        pattern=r"\badd\b|\bsum\b|\bcalculate\b",
        agent_id="math_agent",
        priority=100,
        description="Route math operations to math agent"
    ))
    
    router.add_rule(RoutingRule(
        pattern=r"\bsearch\b|\bfind\b|\bresearch\b",
        agent_id="research_agent",
        priority=90,
        description="Route research queries to research agent"
    ))
    
    # Set the router
    server.router = router
    
    # Process some example requests
    logger.info("\n==== Example 1: Math Operation ====")
    response = server.process_request("Add 42 and 58")
    logger.info(f"Response: {response}")
    
    logger.info("\n==== Example 2: Research Query ====")
    response = server.process_request("Search for artificial intelligence")
    logger.info(f"Response: {response}")
    
    logger.info("\n==== Example 3: Collaboration ====")
    response = server.process_request("Collaborate on research about math algorithms")
    logger.info(f"Response: {response}")
    
    logger.info("\n==== Example 4: Tool Marketplace ====")
    marketplace_example()
    
    logger.info("\nExample completed successfully")


if __name__ == "__main__":
    main()
