"""SEO expert agent."""
from typing import Optional

from me2ai.llms.base import LLMProvider
from me2ai.memory import ConversationMemory
from me2ai.agents.base import BaseAgent
from me2ai.tools.seo.technical_tools import TechnicalSEOTools
from me2ai.tools.web_tools import WebSearchTool

class SEOExpert(BaseAgent):
    """SEO expert agent."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        memory: Optional[ConversationMemory] = None
    ):
        """Initialize the agent.
        
        Args:
            llm_provider: LLM provider to use
            memory: Optional conversation memory
        """
        system_prompt = """You are an SEO expert.
        You help people optimize their websites and content for search engines.
        
        When analyzing websites:
        - Evaluate technical SEO factors
        - Review content quality and relevance
        - Assess backlink profiles
        - Check mobile optimization
        
        When providing recommendations:
        - Prioritize high-impact changes
        - Consider user experience
        - Follow latest SEO best practices
        - Provide actionable steps
        
        Always maintain a data-driven and strategic approach.
        """
        
        tools = [
            TechnicalSEOTools(),
            WebSearchTool()
        ]
        
        super().__init__(
            role="SEO Expert",
            system_prompt=system_prompt,
            llm_provider=llm_provider,
            memory=memory,
            tools=tools
        )
        
    async def respond(self, message: str, context: Optional[str] = None) -> str:
        """Generate a response to the user's message.
        
        Args:
            message: The input message to respond to
            context: Optional additional context
            
        Returns:
            str: The generated response
        """
        return await self._generate_response(message)
