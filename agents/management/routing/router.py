"""Router agent that directs queries to appropriate expert agents."""
from typing import Dict, Tuple, Optional
from me2ai.agents.base import BaseAgent
from me2ai.llms.base import LLMProvider
from me2ai.memory import ConversationMemory

class RouterAgent(BaseAgent):
    """Agent that routes queries to appropriate expert agents based on content."""

    def __init__(self, agents: Dict[str, BaseAgent], llm_provider: LLMProvider, memory: Optional[ConversationMemory] = None):
        """Initialize router agent.
        
        Args:
            agents: Dictionary of available agents to route between
            llm_provider: LLM provider to use
            memory: Optional conversation memory
        """
        system_prompt = """You are a router agent that directs queries to appropriate expert agents. You analyze user queries and determine which expert would be best suited to handle them. If no expert is clearly suitable, you will attempt to help the user directly."""
        super().__init__(
            role="Router",
            system_prompt=system_prompt,
            llm_provider=llm_provider,
            memory=memory
        )
        self.agents = agents
        self.routing_rules = [
            ("german", "german_professor"),
            ("dating", "dating_expert"),
            ("seo", "seo_expert")
        ]

    async def get_agent(self, query: str) -> Tuple[BaseAgent, str]:
        """Get appropriate agent for a query.
        
        Args:
            query: User query to route
            
        Returns:
            Tuple of (selected agent, reason for selection)
        """
        # Convert query to lowercase for case-insensitive matching
        query = query.lower()
        
        # Check each routing rule
        for keyword, agent_name in self.routing_rules:
            if keyword in query:
                if agent_name in self.agents:
                    return self.agents[agent_name], f"Query contains '{keyword}' which matches {agent_name}"
        
        # If no match found, return self
        return self, "No specific expert found for this query, I'll try to help"

    async def respond(self, query: str) -> str:
        """Generate a response to the query.
        
        Args:
            query: User query to respond to
            
        Returns:
            Response string
        """
        # Get appropriate agent
        agent, reason = await self.get_agent(query)
        
        if agent == self:
            # Handle query ourselves
            return f"I'm not sure which expert would be best for this query ({reason}). Let me try to help: " + \
                   await super().respond(query)
        else:
            # Forward to expert
            return await agent.respond(query)
