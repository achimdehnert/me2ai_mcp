"""Relationship team implementation."""
from typing import Optional
from me2ai.llms.base import LLMProvider
from me2ai.agents.base import BaseAgent
from me2ai.agents.management.teams.base import TeamAgent
from me2ai.agents.individual.dating_expert import DatingExpert
from me2ai.agents.coaching.moderator import ModeratorAgent
from me2ai.agents.coaching.life_coach import LifeCoachAgent
from me2ai.memory import ConversationMemory

class RelationshipTeam(TeamAgent):
    """Relationship advice and guidance team."""
    
    def __init__(self, llm_provider: LLMProvider, memory: Optional[ConversationMemory] = None):
        """Initialize relationship team.
        
        Args:
            llm_provider: LLM provider to use
            memory: Optional conversation memory to use
        """
        shared_memory = memory or ConversationMemory()
        
        # Create specialized team members
        dating_expert = DatingExpert(llm_provider, shared_memory)
        life_coach = LifeCoachAgent(
            role="Life Coach",
            system_prompt="""You are an empathetic life coach focusing on personal growth in relationships. Your role is to:
            1. Help people develop emotional intelligence
            2. Guide personal development in relationship contexts
            3. Provide strategies for self-improvement
            4. Support healthy boundary setting
            
            Work with your team members to provide comprehensive relationship guidance.""",
            llm_provider=llm_provider,
            memory=shared_memory
        )
        
        # Create team coordinator
        coordinator = ModeratorAgent(
            role="Relationship Team Coordinator",
            system_prompt="""You are coordinating a relationship advice team. Your role is to:
            1. Synthesize insights from relationship experts
            2. Ensure advice is practical and actionable
            3. Create structured relationship plans
            4. Maintain a supportive and empathetic environment
            
            When combining team perspectives:
            - Focus on healthy relationship strategies
            - Address key relationship challenges
            - Present clear action items
            - Emphasize both immediate improvements and long-term growth""",
            llm_provider=llm_provider,
            memory=shared_memory
        )
        
        super().__init__(
            role="Relationship Team",
            system_prompt="Relationship advice and guidance team",
            llm_provider=llm_provider,
            memory=shared_memory,
            team_members=[dating_expert, life_coach],
            coordinator=coordinator
        )
