"""Team coordinator for managing groups of agents."""

from typing import Dict, List, Optional, Any
from ...base import BaseAgent
from me2ai.llms.base import LLMProvider

class TeamCoordinator(BaseAgent):
    """Coordinates a team of agents working together."""
    
    def __init__(self, role: str = "coordinator", system_prompt: str = "",
                 llm_provider: Optional[LLMProvider] = None):
        super().__init__(role=role, system_prompt=system_prompt,
                        llm_provider=llm_provider)
        self.team_members: Dict[str, BaseAgent] = {}
        self.workflows: Dict[str, List[str]] = {}
        
    def add_team_member(self, role: str, agent: BaseAgent) -> None:
        """Add a team member with a specific role."""
        self.team_members[role] = agent
        
    def remove_team_member(self, role: str) -> None:
        """Remove a team member."""
        if role in self.team_members:
            del self.team_members[role]
            
    def define_workflow(self, name: str, steps: List[str]) -> None:
        """Define a workflow with ordered steps assigned to team members."""
        # Validate all steps have assigned team members
        for step in steps:
            if step not in self.team_members:
                raise ValueError(f"No team member assigned to role: {step}")
        self.workflows[name] = steps
        
    def execute_workflow(self, name: str, input_data: Any) -> Dict[str, Any]:
        """Execute a defined workflow."""
        if name not in self.workflows:
            raise ValueError(f"Workflow not found: {name}")
            
        results = {}
        current_data = input_data
        
        for step in self.workflows[name]:
            agent = self.team_members[step]
            result = agent.respond(str(current_data))
            results[step] = result
            current_data = result
            
        return results
        
    def get_team_member(self, role: str) -> Optional[BaseAgent]:
        """Get a team member by role."""
        return self.team_members.get(role)
        
    def list_workflows(self) -> List[str]:
        """List all defined workflows."""
        return list(self.workflows.keys())
        
    def get_workflow_steps(self, name: str) -> Optional[List[str]]:
        """Get the steps for a specific workflow."""
        return self.workflows.get(name)
