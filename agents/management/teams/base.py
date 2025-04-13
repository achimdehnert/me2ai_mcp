"""Base team functionality."""

from typing import Dict, List, Optional, Any
from ...base import BaseAgent
from me2ai.llms.base import LLMProvider
from me2ai.memory import ConversationMemory

class TeamAgent(BaseAgent):
    """Base class for team-based agents."""
    
    def __init__(self, role: str, system_prompt: str, 
                 llm_provider: LLMProvider,
                 memory: Optional[ConversationMemory] = None,
                 team_members: Optional[List[BaseAgent]] = None,
                 coordinator: Optional[BaseAgent] = None):
        super().__init__(role=role, system_prompt=system_prompt,
                        llm_provider=llm_provider, memory=memory)
        self.team_members = team_members or []
        self.coordinator = coordinator
        
    def add_team_member(self, role: str, agent: BaseAgent) -> None:
        """Add a team member."""
        self.team_members.append(agent)
        
    def get_team_member(self, role: str) -> Optional[BaseAgent]:
        """Get a team member by role."""
        for member in self.team_members:
            if member.role == role:
                return member
        return None
        
    def list_team_members(self) -> List[str]:
        """List all team member roles."""
        return [member.role for member in self.team_members]

    async def respond(self, user_input: str) -> str:
        """Generate team response by coordinating team members.
        
        Args:
            user_input: User input to respond to
            
        Returns:
            Coordinated team response
        """
        # Get responses from team members
        member_responses = []
        for member in self.team_members:
            response = await member.respond(user_input)
            member_responses.append(response)
            
        # If we have a coordinator, use it to combine responses
        if self.coordinator:
            coordinator_input = f"User input: {user_input}\n\nTeam responses:\n"
            for i, response in enumerate(member_responses):
                coordinator_input += f"\nTeam member {i+1}:\n{response}\n"
            
            return await self.coordinator.respond(coordinator_input)
            
        # Otherwise, just concatenate responses
        return "\n\n".join(member_responses)
