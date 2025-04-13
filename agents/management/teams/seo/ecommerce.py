"""E-commerce SEO team composition for online store optimization."""
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
from me2ai.tools.seo.ecommerce_tools import (
    ProductOptimizationTool,
    CategoryStructureTool,
    ConversionOptimizationTool,
    InventoryOptimizationTool
)

class EcommerceSEOExpert(SEOExpert):
    """E-commerce SEO specialist."""
    
    def __init__(self, llm_provider: LLMProvider, memory: Optional[BaseMemory] = None):
        super().__init__(llm_provider, memory)
        self.tools.extend([
            ProductOptimizationTool(),
            CategoryStructureTool(),
            ConversionOptimizationTool(),
            InventoryOptimizationTool()
        ])
        self.system_prompt += """
        Additional expertise in:
        1. Product page optimization
        2. E-commerce site architecture
        3. Conversion optimization
        4. Inventory management SEO
        """

def create_ecommerce_seo_team(llm_provider: LLMProvider) -> TeamAgent:
    """Create an E-commerce SEO optimization team."""
    shared_memory = ConversationMemory()
    
    # Create specialized team members
    ecommerce_seo_expert = EcommerceSEOExpert(llm_provider, shared_memory)
    data_analyst = DataAnalystAgent(llm_provider, shared_memory)
    content_strategist = ContentStrategistAgent(llm_provider, shared_memory)
    
    # Create team coordinator
    coordinator = ModeratorAgent(
        role="E-commerce SEO Team Coordinator",
        system_prompt="""You are coordinating an e-commerce SEO team. Your role is to:
        1. Synthesize insights from specialists:
           - E-commerce SEO Expert: Technical and product optimization
           - Data Analyst: Performance metrics and conversion data
           - Content Strategist: Product descriptions and category content
           
        2. Create optimization plans focusing on:
           - Product page optimization
           - Category structure improvements
           - Conversion rate optimization
           - Content strategy for products
           
        3. Ensure recommendations:
           - Drive sales and revenue
           - Improve search visibility
           - Enhance user experience
           - Scale across product catalog
        """,
        llm_provider=llm_provider,
        memory=shared_memory
    )
    
    # Create team
    team = TeamAgent(
        role="E-commerce SEO Team",
        system_prompt="E-commerce SEO optimization team",
        llm_provider=llm_provider,
        memory=shared_memory
    )
    
    # Add team members
    team.add_team_member("coordinator", coordinator)
    team.add_team_member("seo_expert", ecommerce_seo_expert)
    team.add_team_member("data_analyst", data_analyst)
    team.add_team_member("content_strategist", content_strategist)
    
    return team
