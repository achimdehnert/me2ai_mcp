"""Factory functions for creating agents."""
import os
from typing import Dict, Optional

from llms.base import LLMProvider
from llms.openai_provider import OpenAIProvider
from llms.groq_provider import GroqProvider
from agents.base import BaseAgent
from agents.individual import (
    GermanProfessor,
    DatingExpert,
    SEOExpert,
    Researcher,
    ContentStrategistAgent,
    DataAnalystAgent
)
from agents.management.teams.business.general import BusinessTeam
from agents.management.teams.language.general import LanguageTeam
from agents.management.teams.relationship.general import RelationshipTeam
from agents.management.teams.seo.general import SEOTeam
from agents.management.routing.router import RouterAgent
from memory import ConversationMemory

class AgentFactory:
    """Factory class for creating agents."""
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """Initialize the factory.
        
        Args:
            llm_provider: Optional LLM provider. If not provided, will use default based on environment
        """
        self.llm_provider = llm_provider or (
            GroqProvider() if os.getenv("GROQ_API_KEY") else OpenAIProvider()
        )
    
    async def create_expert_agent(
        self,
        agent_type: str,
        memory: Optional[ConversationMemory] = None
    ) -> BaseAgent:
        """Create an expert agent of the specified type.
        
        Args:
            agent_type: Type of expert agent to create
            memory: Optional conversation memory to use
            
        Returns:
            BaseAgent: The created expert agent
            
        Raises:
            ValueError: If agent_type is not recognized
        """
        # Create appropriate agent based on type
        if agent_type == "german_professor":
            return GermanProfessor(llm_provider=self.llm_provider, memory=memory)
        elif agent_type == "dating_expert":
            return DatingExpert(llm_provider=self.llm_provider, memory=memory)
        elif agent_type == "seo_expert":
            return SEOExpert(llm_provider=self.llm_provider, memory=memory)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    async def create_router_agent(
        self,
        agents: Dict[str, BaseAgent],
        memory: Optional[ConversationMemory] = None
    ) -> RouterAgent:
        """Create a router agent.
        
        Args:
            agents: Dictionary of available agents to route between
            memory: Optional conversation memory to use
            
        Returns:
            RouterAgent: The created router agent
        """
        return RouterAgent(
            agents=agents,
            llm_provider=self.llm_provider,
            memory=memory
        )
        
    async def create_business_team(self) -> BusinessTeam:
        """Create a business team with relevant experts."""
        return BusinessTeam(llm_provider=self.llm_provider)

    async def create_language_team(self) -> LanguageTeam:
        """Create a language team with relevant experts."""
        return LanguageTeam(llm_provider=self.llm_provider)

    async def create_relationship_team(self) -> RelationshipTeam:
        """Create a relationship team with relevant experts."""
        return RelationshipTeam(llm_provider=self.llm_provider)

    async def create_seo_team(self) -> SEOTeam:
        """Create an SEO team with relevant experts."""
        return SEOTeam(llm_provider=self.llm_provider)

    async def create_all_agents(self) -> Dict[str, BaseAgent]:
        """Create all available agents.
        
        Returns:
            Dict[str, BaseAgent]: Dictionary of all created agents
        """
        # Create individual expert agents
        experts = {
            "german_professor": await self.create_expert_agent("german_professor"),
            "dating_expert": await self.create_expert_agent("dating_expert"),
            "seo_expert": await self.create_expert_agent("seo_expert")
        }
        
        # Create router and teams
        agents = {
            "router": await self.create_router_agent(experts),
            "business_team": await self.create_business_team(),
            "language_team": await self.create_language_team(),
            "relationship_team": await self.create_relationship_team(),
            "seo_team": await self.create_seo_team()
        }
        
        # Add experts to the pool
        agents.update(experts)
        return agents
