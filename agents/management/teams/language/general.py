"""Language team implementation."""
from typing import Optional
from me2ai.llms.base import LLMProvider
from me2ai.agents.base import BaseAgent
from me2ai.agents.management.teams.base import TeamAgent
from me2ai.agents.individual.german_professor import GermanProfessor
from me2ai.agents.coaching.moderator import ModeratorAgent
from me2ai.memory import ConversationMemory

class LanguageTeam(TeamAgent):
    """Language learning and teaching team."""
    
    def __init__(self, llm_provider: LLMProvider, memory: Optional[ConversationMemory] = None):
        """Initialize language team.
        
        Args:
            llm_provider: LLM provider to use
            memory: Optional conversation memory to use
        """
        shared_memory = memory or ConversationMemory()
        
        # Create specialized team members
        german_prof = GermanProfessor(llm_provider, shared_memory)
        
        # Create team coordinator
        coordinator = ModeratorAgent(
            role="Language Team Coordinator",
            system_prompt="""You are coordinating a language learning team. Your role is to:
            1. Synthesize insights from language experts
            2. Ensure advice is practical and actionable
            3. Create structured learning plans
            4. Maintain a supportive and encouraging environment
            
            When combining team perspectives:
            - Focus on effective learning strategies
            - Address key language challenges
            - Present clear action items
            - Emphasize both immediate practice and long-term mastery""",
            llm_provider=llm_provider,
            memory=shared_memory
        )
        
        super().__init__(
            role="Language Team",
            system_prompt="Language learning and teaching team",
            llm_provider=llm_provider,
            memory=shared_memory,
            team_members=[german_prof],
            coordinator=coordinator
        )
