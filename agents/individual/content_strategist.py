"""Content strategy specialist agent."""
from typing import Optional
from langchain_core.memory import BaseMemory
from ..base import BaseAgent
from me2ai.llms.base import LLMProvider
from me2ai.tools.content_tools import (
    KeywordResearchTool,
    ContentAnalyzerTool,
    TopicClusteringTool,
    ContentGapAnalyzerTool
)

class ContentStrategistAgent(BaseAgent):
    """Content strategy and optimization specialist."""
    
    def __init__(self, llm_provider: LLMProvider, memory: Optional[BaseMemory] = None):
        """Initialize the content strategist agent.
        
        Args:
            llm_provider: The LLM provider to use
            memory: Optional memory instance
        """
        super().__init__(
            role="Content Strategist",
            system_prompt="""You are a content strategy expert who:
            1. Develops content strategies
            2. Optimizes content for search
            3. Plans content hierarchies
            4. Identifies content opportunities
            
            You can use tools to:
            - Research keywords and topics
            - Analyze content performance
            - Create topic clusters
            - Identify content gaps
            
            Create comprehensive content strategies that align with SEO goals.""",
            llm_provider=llm_provider,
            memory=memory,
            tools=[
                KeywordResearchTool(),
                ContentAnalyzerTool(),
                TopicClusteringTool(),
                ContentGapAnalyzerTool()
            ]
        )

    async def respond(self, query: str) -> str:
        """Generate a response to a user query.
        
        Args:
            query: User's query
            
        Returns:
            Agent's response
        """
        messages = self.memory.get_messages() if self.memory else []
        messages.append({"role": "user", "content": query})
        response = await self.llm_provider.generate_response(messages)
        if self.memory:
            self.memory.add_message("assistant", response)
        return response
