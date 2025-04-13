"""Local SEO team composition for business optimization."""
from typing import Optional
from langchain_core.memory import BaseMemory
from me2ai.llms.base import LLMProvider
from me2ai.agents.base import BaseAgent
from me2ai.agents.individual.seo_expert import SEOExpert
from me2ai.agents.individual.data_analyst import DataAnalystAgent
from me2ai.agents.individual.content_strategist import ContentStrategistAgent
from me2ai.agents.coaching.moderator import ModeratorAgent
from me2ai.memory import ConversationMemory
from me2ai.agents.management.teams.base import TeamAgent
from me2ai.tools.seo.local_tools import (
    GMBOptimizationTool,
    CitationAnalyzerTool,
    LocalRankTrackingTool,
    ReviewManagementTool
)

class LocalSEOExpert(SEOExpert):
    """Local SEO specialist."""
    
    def __init__(self, llm_provider: LLMProvider, memory: Optional[BaseMemory] = None):
        super().__init__(llm_provider, memory)
        self.tools.extend([
            GMBOptimizationTool(),
            CitationAnalyzerTool(),
            LocalRankTrackingTool(),
            ReviewManagementTool()
        ])
        self.system_prompt += """
        Additional expertise in:
        1. Google My Business optimization
        2. Local citation management
        3. Review management
        4. Local competitive analysis
        """

def create_local_seo_team(llm_provider: LLMProvider) -> TeamAgent:
    """Create a Local SEO optimization team."""
    shared_memory = ConversationMemory()
    
    # Create specialized team members
    local_seo_expert = LocalSEOExpert(llm_provider, shared_memory)
    data_analyst = DataAnalystAgent(llm_provider, shared_memory)
    content_strategist = ContentStrategistAgent(llm_provider, shared_memory)
    
    # Create team coordinator
    coordinator = ModeratorAgent(
        role="Local SEO Team Coordinator",
        system_prompt="""You are coordinating a local SEO team. Your role is to:
        1. Synthesize insights from specialists:
           - Local SEO Expert: GMB and citation optimization
           - Data Analyst: Local performance metrics
           - Content Strategist: Local content and reviews
           
        2. Create optimization plans focusing on:
           - Local search visibility
           - Google My Business optimization
           - Review management
           - Local content strategy
           
        3. Ensure recommendations:
           - Drive local visibility
           - Improve local rankings
           - Build local authority
           - Enhance local reputation
        """,
        llm_provider=llm_provider,
        memory=shared_memory
    )
    
    # Create team
    team = TeamAgent(
        role="Local SEO Team",
        system_prompt="Local SEO optimization team",
        llm_provider=llm_provider,
        memory=shared_memory
    )
    
    # Add team members
    team.add_team_member("coordinator", coordinator)
    team.add_team_member("seo_expert", local_seo_expert)
    team.add_team_member("data_analyst", data_analyst)
    team.add_team_member("content_strategist", content_strategist)
    
    return team
