"""
Routing layer for ME2AI MCP.

This module provides components for automatic routing
of requests between agents and tools.
"""
from typing import Dict, List, Any, Optional, Union, Callable, Type, Set, Tuple
import logging
import re
import json
from dataclasses import dataclass, field

from .base import ME2AIMCPServer, BaseTool
from .agents import BaseAgent, RoutingAgent, SpecializedAgent, ToolCategory
from .utils import sanitize_input, format_response


# Configure logging
logger = logging.getLogger("me2ai-mcp-routing")


@dataclass
class RoutingRule:
    """Rule for routing requests to specific agents."""
    
    pattern: str  # Regex pattern to match
    agent_id: str  # ID of agent to route to
    priority: int = 0  # Higher numbers take precedence
    description: str = ""  # Human-readable description
    
    def __post_init__(self) -> None:
        """Compile the regex pattern."""
        self.compiled_pattern = re.compile(self.pattern, re.IGNORECASE)
    
    def matches(self, request: str) -> bool:
        """
        Check if this rule matches the request.
        
        Args:
            request: User request text
            
        Returns:
            Whether this rule matches
        """
        return bool(self.compiled_pattern.search(request))


class AgentRegistry:
    """Registry for managing multiple agents."""
    
    def __init__(self) -> None:
        """Initialize the agent registry."""
        self.agents: Dict[str, BaseAgent] = {}
        self.routing_rules: List[RoutingRule] = []
        self.default_agent_id: Optional[str] = None
        self.logger = logging.getLogger("me2ai-mcp-agent-registry")
    
    def register_agent(
        self, 
        agent: BaseAgent,
        make_default: bool = False
    ) -> None:
        """
        Register an agent with the registry.
        
        Args:
            agent: The agent to register
            make_default: Whether to make this the default agent
        """
        self.agents[agent.agent_id] = agent
        self.logger.info(f"Registered agent: {agent.agent_id} ({agent.name})")
        
        if make_default or not self.default_agent_id:
            self.default_agent_id = agent.agent_id
            self.logger.info(f"Set default agent to: {agent.agent_id}")
    
    def add_routing_rule(self, rule: RoutingRule) -> None:
        """
        Add a routing rule.
        
        Args:
            rule: The routing rule to add
        """
        self.routing_rules.append(rule)
        # Sort rules by priority (highest first)
        self.routing_rules.sort(key=lambda r: -r.priority)
        self.logger.info(f"Added routing rule: {rule.description} (priority: {rule.priority})")
    
    def route_request(self, request: str) -> Tuple[BaseAgent, str]:
        """
        Route a request to the appropriate agent.
        
        Args:
            request: User request text
            
        Returns:
            Tuple of (selected agent, sanitized request)
        """
        if not self.agents:
            raise ValueError("No agents registered")
        
        # Sanitize input
        clean_request = sanitize_input(request)
        
        # Find matching rule
        for rule in self.routing_rules:
            if rule.matches(clean_request):
                agent_id = rule.agent_id
                if agent_id in self.agents:
                    self.logger.info(
                        f"Routing to agent {agent_id} based on rule: {rule.description}"
                    )
                    return self.agents[agent_id], clean_request
        
        # Fall back to default agent
        if self.default_agent_id and self.default_agent_id in self.agents:
            self.logger.info(f"Using default agent: {self.default_agent_id}")
            return self.agents[self.default_agent_id], clean_request
        
        # If no default, use the first registered agent
        agent_id = next(iter(self.agents.keys()))
        self.logger.warning(f"No matching rule or default, using first agent: {agent_id}")
        return self.agents[agent_id], clean_request


class MCPRouter:
    """
    Central router that connects agents and tools.
    
    This router manages a collection of agents and an MCP server,
    routing requests to the appropriate agent and collecting responses.
    """
    
    def __init__(
        self,
        server: ME2AIMCPServer,
        registry: Optional[AgentRegistry] = None
    ) -> None:
        """
        Initialize the MCP router.
        
        Args:
            server: MCP server with registered tools
            registry: Agent registry (optional, will create if not provided)
        """
        self.server = server
        self.registry = registry or AgentRegistry()
        self.logger = logging.getLogger("me2ai-mcp-router")
        self.request_history: List[Dict[str, Any]] = []
    
    def register_agent(
        self, 
        agent: BaseAgent,
        make_default: bool = False
    ) -> None:
        """
        Register an agent and connect it to the server.
        
        Args:
            agent: The agent to register
            make_default: Whether to make this the default agent
        """
        # Connect agent to server
        agent.connect_to_server(self.server)
        
        # Register with the registry
        self.registry.register_agent(agent, make_default)
    
    def add_routing_rule(self, rule: RoutingRule) -> None:
        """
        Add a routing rule.
        
        Args:
            rule: The routing rule to add
        """
        self.registry.add_routing_rule(rule)
    
    def process_request(self, request: str, **kwargs) -> Dict[str, Any]:
        """
        Process a request by routing to appropriate agent.
        
        Args:
            request: User request text
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary
        """
        try:
            # Route to appropriate agent
            agent, clean_request = self.registry.route_request(request)
            
            # Process with selected agent
            response = agent.process_request(clean_request, **kwargs)
            
            # Record in history
            self.request_history.append({
                "request": request,
                "agent_id": agent.agent_id,
                "response": response
            })
            
            # Add routing metadata to response
            if isinstance(response, dict):
                response["_routing"] = {
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name
                }
            
            return response
        
        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            return format_response(
                None,
                success=False,
                error=f"Error processing request: {str(e)}"
            )
    
    def get_agent_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all registered agents.
        
        Returns:
            Dictionary of agent statistics
        """
        stats = {}
        for agent_id, agent in self.registry.agents.items():
            stats[agent_id] = {
                "name": agent.name,
                "description": agent.description,
                "request_count": agent.request_count,
                "error_count": agent.error_count
            }
        return stats
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about routing decisions.
        
        Returns:
            Dictionary of routing statistics
        """
        if not self.request_history:
            return {"total_requests": 0}
        
        agent_counts = {}
        for record in self.request_history:
            agent_id = record.get("agent_id", "unknown")
            agent_counts[agent_id] = agent_counts.get(agent_id, 0) + 1
        
        return {
            "total_requests": len(self.request_history),
            "agent_distribution": agent_counts
        }


# Common routing rules
def create_default_rules() -> List[RoutingRule]:
    """
    Create a set of default routing rules.
    
    Returns:
        List of default routing rules
    """
    return [
        RoutingRule(
            pattern=r"github|repo|commit|pull|issue",
            agent_id="github_agent",
            priority=100,
            description="GitHub related requests"
        ),
        RoutingRule(
            pattern=r"text|analyze|summarize|translate",
            agent_id="text_agent",
            priority=90,
            description="Text processing requests"
        ),
        RoutingRule(
            pattern=r"data|fetch|retrieve|store",
            agent_id="data_agent",
            priority=80,
            description="Data retrieval and storage requests"
        ),
        RoutingRule(
            pattern=r"system|status|config|health",
            agent_id="system_agent",
            priority=70,
            description="System administration requests"
        )
    ]
