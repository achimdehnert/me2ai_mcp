"""Multi-agent optimization system."""

from typing import Dict, List, Set, Optional
import networkx as nx
from dataclasses import dataclass
import json

@dataclass
class AgentInteraction:
    """Record of interaction between agents."""
    source_agent: str
    target_agent: str
    message: str
    latency: float
    success: bool
    
@dataclass
class TeamMetrics:
    """Metrics for a team of agents."""
    total_latency: float
    success_rate: float
    message_count: int
    bottlenecks: List[str]
    
class MultiAgentOptimizer:
    """Optimizes performance of multiple agents working together."""
    
    def __init__(self):
        self.interaction_graph = nx.DiGraph()
        self.interactions: List[AgentInteraction] = []
        self.team_configurations: Dict[str, Dict] = {}
        
    def record_interaction(self, interaction: AgentInteraction) -> None:
        """Record an interaction between agents."""
        self.interactions.append(interaction)
        
        # Update interaction graph
        if not self.interaction_graph.has_edge(
            interaction.source_agent, 
            interaction.target_agent
        ):
            self.interaction_graph.add_edge(
                interaction.source_agent,
                interaction.target_agent,
                messages=0,
                total_latency=0,
                failures=0
            )
            
        edge = self.interaction_graph.edges[
            interaction.source_agent, 
            interaction.target_agent
        ]
        edge["messages"] += 1
        edge["total_latency"] += interaction.latency
        if not interaction.success:
            edge["failures"] += 1
            
    def analyze_team(self) -> TeamMetrics:
        """Analyze team performance metrics."""
        if not self.interactions:
            raise ValueError("No interactions recorded")
            
        # Calculate team-wide metrics
        total_latency = sum(i.latency for i in self.interactions)
        success_count = sum(1 for i in self.interactions if i.success)
        message_count = len(self.interactions)
        
        # Find bottlenecks (nodes with high incoming traffic or high latency)
        bottlenecks = []
        for node in self.interaction_graph.nodes():
            in_edges = self.interaction_graph.in_edges(node, data=True)
            if not in_edges:
                continue
                
            avg_latency = sum(
                e[2]["total_latency"] / e[2]["messages"] 
                for e in in_edges
            ) / len(in_edges)
            
            if avg_latency > 1.0:  # threshold in seconds
                bottlenecks.append(node)
                
        return TeamMetrics(
            total_latency=total_latency,
            success_rate=success_count / message_count if message_count > 0 else 0,
            message_count=message_count,
            bottlenecks=bottlenecks
        )
        
    def optimize_team(self) -> Dict:
        """Generate optimization recommendations for the team."""
        metrics = self.analyze_team()
        
        recommendations = {
            "configuration_changes": {},
            "workflow_changes": [],
            "critical_paths": [],
            "scaling_suggestions": []
        }
        
        # Analyze communication patterns
        for node in self.interaction_graph.nodes():
            out_edges = self.interaction_graph.out_edges(node, data=True)
            if not out_edges:
                continue
                
            # Check for agents with too many outgoing connections
            if len(out_edges) > 5:  # threshold
                recommendations["workflow_changes"].append(
                    f"Consider splitting {node}'s responsibilities "
                    "into multiple specialized agents"
                )
                
        # Find critical paths (longest latency paths)
        try:
            critical_path = nx.dag_longest_path(
                self.interaction_graph,
                weight=lambda u, v, d: d["total_latency"] / d["messages"]
            )
            recommendations["critical_paths"].append({
                "path": critical_path,
                "suggestion": "Optimize these agents for lower latency"
            })
        except nx.NetworkXError:
            pass  # Graph might have cycles
            
        # Generate scaling suggestions
        for bottleneck in metrics.bottlenecks:
            recommendations["scaling_suggestions"].append({
                "agent": bottleneck,
                "suggestion": "Consider running multiple instances"
            })
            
        # Configuration changes based on interaction patterns
        for node in self.interaction_graph.nodes():
            node_edges = self.interaction_graph.edges(node, data=True)
            if not node_edges:
                continue
                
            avg_latency = sum(
                e[2]["total_latency"] / e[2]["messages"] 
                for e in node_edges
            ) / len(node_edges)
            
            if avg_latency > 1.0:
                recommendations["configuration_changes"][node] = {
                    "max_tokens": "Reduce by 20%",
                    "temperature": "Reduce to 0.7",
                    "presence_penalty": "Reduce to 0.0"
                }
                
        return recommendations
    
    def save_team_config(self, team_id: str, config: Dict) -> None:
        """Save a team configuration for future reference."""
        self.team_configurations[team_id] = config
        
    def load_team_config(self, team_id: str) -> Optional[Dict]:
        """Load a team configuration."""
        return self.team_configurations.get(team_id)
    
    def export_analysis(self, filepath: str) -> None:
        """Export team analysis to a file."""
        analysis = {
            "team_metrics": self.analyze_team().__dict__,
            "optimization_recommendations": self.optimize_team(),
            "interaction_stats": {
                "total_interactions": len(self.interactions),
                "unique_agents": len(self.interaction_graph.nodes()),
                "connection_patterns": [
                    {
                        "source": e[0],
                        "target": e[1],
                        "stats": e[2]
                    }
                    for e in self.interaction_graph.edges(data=True)
                ]
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2)
