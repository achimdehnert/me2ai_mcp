"""Business team implementation."""
from typing import Optional
from me2ai.llms.base import LLMProvider
from me2ai.agents.base import BaseAgent
from me2ai.agents.management.teams.base import TeamAgent
from me2ai.agents.individual.data_analyst import DataAnalystAgent as DataAnalyst
from me2ai.agents.individual.content_strategist import ContentStrategistAgent as ContentStrategist
from me2ai.agents.coaching.moderator import ModeratorAgent
from me2ai.memory import ConversationMemory

class BusinessTeam(TeamAgent):
    """Business strategy and analysis team."""
    
    def __init__(self, llm_provider: LLMProvider, memory: Optional[ConversationMemory] = None):
        """Initialize business team.
        
        Args:
            llm_provider: LLM provider to use
            memory: Optional conversation memory to use
        """
        shared_memory = memory or ConversationMemory()
        
        # Create specialized team members
        data_analyst = DataAnalyst(llm_provider, shared_memory)
        content_strategist = ContentStrategist(llm_provider, shared_memory)
        
        # Create team coordinator
        coordinator = ModeratorAgent(
            role="Business Team Coordinator",
            system_prompt="""You are coordinating a business strategy team. Your role is to:
            1. Synthesize insights from business experts
            2. Ensure advice is practical and actionable
            3. Create structured business plans
            4. Maintain a professional and strategic focus
            
            When combining team perspectives:
            - Focus on data-driven decision making
            - Address key business challenges
            - Present clear action items
            - Emphasize both short-term wins and long-term strategy""",
            llm_provider=llm_provider,
            memory=shared_memory
        )
        
        super().__init__(
            role="Business Team",
            system_prompt="Business strategy and analysis team",
            llm_provider=llm_provider,
            memory=shared_memory,
            team_members=[data_analyst, content_strategist],
            coordinator=coordinator
        )
