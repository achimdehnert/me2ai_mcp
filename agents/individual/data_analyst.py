"""Data analysis specialist agent."""
from typing import Optional
from langchain_core.memory import BaseMemory
from ..base import BaseAgent
from me2ai.llms.base import LLMProvider
from me2ai.tools.analysis_tools import (
    DataVisualizationTool,
    StatisticalAnalysisTool,
    TrendAnalysisTool,
    CompetitorAnalysisTool
)

class DataAnalystAgent(BaseAgent):
    """Data analysis and insights specialist."""
    
    def __init__(self, llm_provider: LLMProvider, memory: Optional[BaseMemory] = None):
        """Initialize the data analyst agent.
        
        Args:
            llm_provider: The LLM provider to use
            memory: Optional memory instance
        """
        super().__init__(
            role="Data Analyst",
            system_prompt="""You are a data analysis expert who:
            1. Analyzes complex datasets
            2. Identifies patterns and trends
            3. Creates data visualizations
            4. Provides actionable insights
            
            You can use tools to:
            - Create data visualizations
            - Perform statistical analysis
            - Analyze market trends
            - Compare competitor data
            
            Provide clear, data-driven insights that inform SEO strategy.""",
            llm_provider=llm_provider,
            memory=memory,
            tools=[
                DataVisualizationTool(),
                StatisticalAnalysisTool(),
                TrendAnalysisTool(),
                CompetitorAnalysisTool()
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
