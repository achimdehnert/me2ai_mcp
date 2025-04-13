"""SEO team implementation."""
from typing import Optional
from me2ai.llms.base import LLMProvider
from me2ai.agents.base import BaseAgent
from me2ai.agents.management.teams.base import TeamAgent
from me2ai.agents.individual.seo_expert import SEOExpert
from me2ai.agents.individual.content_strategist import ContentStrategistAgent
from me2ai.agents.coaching.moderator import ModeratorAgent
from me2ai.memory import ConversationMemory

class SEOTeam(TeamAgent):
    """SEO optimization and strategy team."""
    
    def __init__(self, llm_provider: LLMProvider, memory: Optional[ConversationMemory] = None):
        """Initialize SEO team.
        
        Args:
            llm_provider: LLM provider to use
            memory: Optional conversation memory to use
        """
        shared_memory = memory or ConversationMemory()
        
        # Create specialized team members
        seo_expert = SEOExpert(llm_provider, shared_memory)
        content_strategist = ContentStrategistAgent(llm_provider, shared_memory)
        
        # Create team coordinator
        coordinator = ModeratorAgent(
            role="SEO Team Coordinator",
            system_prompt="""You are coordinating an SEO optimization team. Your role is to:
            1. Synthesize insights from SEO experts
            2. Ensure strategies are practical and actionable
            3. Create structured optimization plans
            4. Maintain a data-driven approach
            
            When combining team perspectives:
            - Focus on effective SEO strategies
            - Address key optimization challenges
            - Present clear action items
            - Emphasize both quick wins and long-term growth""",
            llm_provider=llm_provider,
            memory=shared_memory
        )
        
        super().__init__(
            role="SEO Team",
            system_prompt="SEO optimization and strategy team",
            llm_provider=llm_provider,
            memory=shared_memory,
            team_members=[seo_expert, content_strategist],
            coordinator=coordinator
        )
