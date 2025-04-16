"""
Dynamic routing capabilities for ME2AI MCP.

This module extends the routing layer with adaptive routing capabilities,
learning from request patterns and agent performance.
"""
from typing import Dict, List, Any, Optional, Union, Callable, Type, Set, Tuple
import logging
import re
import json
import random
from datetime import datetime
from dataclasses import dataclass, field

from .base import ME2AIMCPServer
from .agents import BaseAgent, RoutingAgent, SpecializedAgent
from .routing import MCPRouter, RoutingRule, AgentRegistry
from .utils import sanitize_input, format_response


# Configure logging
logger = logging.getLogger("me2ai-mcp-dynamic-routing")


@dataclass
class PerformanceMetrics:
    """Performance metrics for an agent."""
    
    agent_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update(self, success: bool, response_time: float) -> None:
        """
        Update metrics with a new request result.
        
        Args:
            success: Whether the request was successful
            response_time: Response time in seconds
        """
        self.total_requests += 1
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Update average response time using rolling average
        if self.total_requests == 1:
            self.average_response_time = response_time
        else:
            self.average_response_time = (
                (self.average_response_time * (self.total_requests - 1) + response_time) / 
                self.total_requests
            )
        
        self.last_updated = datetime.now()
    
    @property
    def success_rate(self) -> float:
        """
        Calculate the success rate.
        
        Returns:
            Success rate as a float between 0 and 1
        """
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests


class AdaptiveRouter(MCPRouter):
    """Router with adaptive routing capabilities."""
    
    def __init__(
        self,
        server: ME2AIMCPServer,
        registry: Optional[AgentRegistry] = None,
        learning_rate: float = 0.1,
        exploration_rate: float = 0.05,
        enable_adaptive_routing: bool = True
    ) -> None:
        """
        Initialize the adaptive router.
        
        Args:
            server: MCP server with registered tools
            registry: Agent registry (optional, will create if not provided)
            learning_rate: Rate at which to update routing weights
            exploration_rate: Probability of random agent selection for exploration
            enable_adaptive_routing: Whether to use adaptive routing
        """
        super().__init__(server, registry)
        
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.enable_adaptive_routing = enable_adaptive_routing
        
        # Performance metrics for each agent
        self.metrics: Dict[str, PerformanceMetrics] = {}
        
        # Routing weights for each pattern/agent combination
        self.routing_weights: Dict[str, Dict[str, float]] = {}
        
        # Request history for learning
        self.request_history: List[Dict[str, Any]] = []
        
        self.logger = logging.getLogger("me2ai-mcp-adaptive-router")
    
    def register_agent(self, agent: BaseAgent, make_default: bool = False) -> None:
        """
        Register an agent and initialize its metrics.
        
        Args:
            agent: The agent to register
            make_default: Whether to make this the default agent
        """
        super().register_agent(agent, make_default)
        
        # Initialize metrics
        self.metrics[agent.agent_id] = PerformanceMetrics(agent_id=agent.agent_id)
        
        # Initialize routing weights if needed
        for rule in self.rules:
            if rule.agent_id == agent.agent_id:
                pattern_key = rule.pattern
                if pattern_key not in self.routing_weights:
                    self.routing_weights[pattern_key] = {}
                self.routing_weights[pattern_key][agent.agent_id] = 1.0
    
    def add_routing_rule(self, rule: RoutingRule) -> None:
        """
        Add a routing rule and initialize its weights.
        
        Args:
            rule: The routing rule to add
        """
        super().add_routing_rule(rule)
        
        # Initialize routing weights
        pattern_key = rule.pattern
        if pattern_key not in self.routing_weights:
            self.routing_weights[pattern_key] = {}
        
        # Add weight for the agent in this rule
        self.routing_weights[pattern_key][rule.agent_id] = 1.0
    
    def process_request(self, request: str, **kwargs) -> Dict[str, Any]:
        """
        Process a request with adaptive routing.
        
        Args:
            request: User request text
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary
        """
        sanitized_request = sanitize_input(request)
        start_time = datetime.now()
        
        # Select an agent
        agent_id = self._select_agent_adaptively(sanitized_request)
        
        # Process with the selected agent
        self.logger.info(f"Routing request to agent: {agent_id}")
        result = self._process_with_agent(agent_id, sanitized_request, **kwargs)
        
        # Calculate response time
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        
        # Update metrics
        success = result.get("success", False)
        self._update_metrics(agent_id, success, response_time)
        
        # Update routing weights if adaptive routing is enabled
        if self.enable_adaptive_routing:
            self._update_routing_weights(sanitized_request, agent_id, success)
        
        # Add to request history
        request_entry = {
            "timestamp": start_time.isoformat(),
            "request": sanitized_request,
            "agent_id": agent_id,
            "success": success,
            "response_time": response_time
        }
        self.request_history.append(request_entry)
        
        return result
    
    def _select_agent_adaptively(self, request: str) -> str:
        """
        Select an agent using adaptive routing.
        
        Args:
            request: User request text
            
        Returns:
            Agent ID
        """
        if not self.enable_adaptive_routing:
            # Fall back to standard routing if adaptive routing is disabled
            return self._select_agent(request)
        
        # Exploration: occasionally choose a random agent to explore
        if random.random() < self.exploration_rate:
            agent_ids = list(self.registry.agents.keys())
            if agent_ids:
                random_agent_id = random.choice(agent_ids)
                self.logger.info(f"Exploration: randomly selected agent {random_agent_id}")
                return random_agent_id
        
        # Find matching rules and get their weights
        matching_agents = {}
        highest_priority = -float('inf')
        
        for rule in sorted(self.rules, key=lambda r: r.priority, reverse=True):
            if rule.matches(request):
                # Get the weights for this pattern
                pattern_key = rule.pattern
                weights = self.routing_weights.get(pattern_key, {})
                agent_weight = weights.get(rule.agent_id, 0.0)
                
                # Update weights based on agent performance
                if rule.agent_id in self.metrics:
                    # Adjust weight based on success rate
                    success_rate = self.metrics[rule.agent_id].success_rate
                    performance_boost = success_rate * 2.0  # Scale the success rate
                    adjusted_weight = agent_weight * (1.0 + performance_boost)
                else:
                    adjusted_weight = agent_weight
                
                # Consider rule priority
                if rule.priority > highest_priority:
                    # Higher priority rule, clear previous matches
                    matching_agents = {rule.agent_id: adjusted_weight}
                    highest_priority = rule.priority
                elif rule.priority == highest_priority:
                    # Same priority, add to candidates
                    matching_agents[rule.agent_id] = adjusted_weight
        
        if matching_agents:
            # Choose agent probabilistically based on weights
            agent_ids = list(matching_agents.keys())
            weights = list(matching_agents.values())
            
            # Ensure weights are positive
            min_weight = min(weights) if weights else 0.0
            if min_weight < 0:
                weights = [w - min_weight + 0.1 for w in weights]
            
            # Normalize weights to sum to 1
            total_weight = sum(weights) if weights else 1.0
            normalized_weights = [w / total_weight for w in weights]
            
            # Select agent using weights
            selected_agent_id = random.choices(agent_ids, weights=normalized_weights, k=1)[0]
            return selected_agent_id
        
        # Default to original selection method if no matches
        return self._select_agent(request)
    
    def _update_metrics(self, agent_id: str, success: bool, response_time: float) -> None:
        """
        Update performance metrics for an agent.
        
        Args:
            agent_id: Agent ID
            success: Whether the request was successful
            response_time: Response time in seconds
        """
        if agent_id in self.metrics:
            self.metrics[agent_id].update(success, response_time)
        else:
            # Create metrics if they don't exist
            self.metrics[agent_id] = PerformanceMetrics(agent_id=agent_id)
            self.metrics[agent_id].update(success, response_time)
    
    def _update_routing_weights(self, request: str, agent_id: str, success: bool) -> None:
        """
        Update routing weights based on request results.
        
        Args:
            request: User request text
            agent_id: Agent ID that processed the request
            success: Whether the request was successful
        """
        # Find matching rules
        for rule in self.rules:
            if rule.matches(request):
                pattern_key = rule.pattern
                
                # Initialize weights if needed
                if pattern_key not in self.routing_weights:
                    self.routing_weights[pattern_key] = {}
                
                # Update weights for this pattern and agent
                if agent_id not in self.routing_weights[pattern_key]:
                    self.routing_weights[pattern_key][agent_id] = 0.0
                
                # Apply reinforcement: increase weight if successful, decrease if failed
                reward = 1.0 if success else -0.5
                self.routing_weights[pattern_key][agent_id] += self.learning_rate * reward
                
                # Ensure weight stays positive
                self.routing_weights[pattern_key][agent_id] = max(0.1, self.routing_weights[pattern_key][agent_id])
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """
        Get detailed performance metrics for all agents.
        
        Returns:
            Dictionary of detailed metrics
        """
        return {
            agent_id: {
                "total_requests": metrics.total_requests,
                "successful_requests": metrics.successful_requests,
                "failed_requests": metrics.failed_requests,
                "success_rate": metrics.success_rate,
                "average_response_time": metrics.average_response_time,
                "last_updated": metrics.last_updated.isoformat()
            }
            for agent_id, metrics in self.metrics.items()
        }
    
    def get_routing_weights(self) -> Dict[str, Any]:
        """
        Get the current routing weights.
        
        Returns:
            Dictionary of routing weights
        """
        return {pattern: dict(weights) for pattern, weights in self.routing_weights.items()}
    
    def reset_adaptive_learning(self) -> None:
        """Reset all adaptive learning data."""
        # Reset performance metrics
        for agent_id in self.metrics:
            self.metrics[agent_id] = PerformanceMetrics(agent_id=agent_id)
        
        # Reset routing weights
        for pattern in self.routing_weights:
            for agent_id in self.routing_weights[pattern]:
                self.routing_weights[pattern][agent_id] = 1.0
        
        # Clear request history
        self.request_history = []
        
        self.logger.info("Adaptive learning data has been reset")
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about routing decisions.
        
        Returns:
            Dictionary of routing statistics
        """
        stats = super().get_routing_stats()
        
        # Add adaptive routing stats
        stats.update({
            "adaptive_routing_enabled": self.enable_adaptive_routing,
            "learning_rate": self.learning_rate,
            "exploration_rate": self.exploration_rate,
        })
        
        return stats
