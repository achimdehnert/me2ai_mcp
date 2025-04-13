"""Router agent for directing queries to appropriate experts."""
from typing import Dict, Tuple, Any
from agents.base import BaseAgent
from agents.individual.german_professor import GermanProfessor
from agents.individual.dating_expert import DatingExpert
from agents.individual.seo_expert import SEOExpert
from memory import ConversationMemory
from llms.base import LLMProvider

class RouterAgent(BaseAgent):
    """Agent that routes queries to appropriate expert agents."""

    def __init__(self, experts: Dict[str, BaseAgent], llm_provider: LLMProvider):
        """Initialize the router agent with available experts."""
        super().__init__("router", llm_provider)
        self.experts = experts
        self.memory = ConversationMemory()

    async def get_agent(self, query: str) -> Tuple[BaseAgent, str]:
        """Determine the most appropriate expert for the query."""
        # For now, use simple keyword matching
        query = query.lower()
        
        if "german" in query or "deutsch" in query:
            return self.experts["german_professor"], "Query is about German language"
        elif "date" in query or "relationship" in query or "love" in query:
            return self.experts["dating_expert"], "Query is about dating or relationships"
        elif "seo" in query or "search engine" in query or "website" in query:
            return self.experts["seo_expert"], "Query is about SEO or website optimization"
        
        # Default to router if no match found
        return self, "No specific expert match found"

    async def respond(self, query: str) -> str:
        """Generate a response to the query."""
        # Store the query in memory
        self.memory.add_user_message(query)
        
        # Get appropriate expert
        expert, reason = await self.get_agent(query)
        
        # Get response from expert
        response = await expert.respond(query)
        
        # Store response in memory
        self.memory.add_ai_message(response)
        
        return response
